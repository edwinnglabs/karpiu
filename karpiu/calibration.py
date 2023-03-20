import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from copy import deepcopy
import logging
import matplotlib
import matplotlib.pyplot as plt
import math
from typing import Optional, Dict, Any, Tuple


from .explainability.attribution import Attributor
from .models import MMM
from .utils import get_logger


class PriorSolver:
    """Solving Regression Coefficient Prior from MMM by using Attribution logic"""

    def __init__(self, tests_df: pd.DataFrame):
        self.tests_df = tests_df.copy()

    def derive_prior(
        self,
        model: MMM,
        shuffle: bool = False,
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> pd.DataFrame:
        """Solving priors independently (and sequentially) based on each coefficients loc and scale input.

        Args:
            model (MMM): _description_
            shuffle (bool, optional): _description_. Defaults to False.
            debug (bool, optional): _description_. Defaults to False.

        Returns:
            pd.DataFrame: _description_
        """
        if logger is None:
            self.logger = get_logger("karpiu-calibration")

        input_df = model.raw_df.copy()
        if shuffle:
            tests_df = self.tests_df.sample(frac=1, ignore_index=True)
        else:
            tests_df = self.tests_df.copy()

        output_df = tests_df.copy()

        # initialize values
        output_df["coef_prior"] = np.nan
        output_df["sigma_prior"] = np.nan
        output_df["test_spend"] = np.nan
        output_df["test_lift"] = np.nan

        date_col = model.date_col

        if debug:
            return tests_df
        else:
            for idx, row in tests_df.iterrows():
                test_start = row["test_start"]
                test_end = row["test_end"]
                test_icac = row["test_icac"]
                test_channel = row["test_channel"]
                test_se = row["test_se"]
                logger.info("test channel:{}".format(test_channel))

                # derive test spend
                mask = (input_df[date_col] >= test_start) & (
                    input_df[date_col] <= test_end
                )
                sub_input_df = input_df[mask].reset_index(drop=True)
                test_spend = sub_input_df[test_channel].sum()
                # derive lift from spend data from model to ensure consistency
                test_lift = test_spend / test_icac
                if test_lift <= 1e-5 or test_spend <= 1e-5:
                    logger.info(
                        "Minimal lift or spend detected of the channel. Skip prior derivation."
                    )
                    continue

                test_lift_upper = test_spend / (test_icac - test_se)

                # create a callback used for scipy.optimize.fsolve
                attr_obj = Attributor(model, start=test_start, end=test_end)

                def attr_call_back(x, target):
                    attr_res = attr_obj.make_attribution(
                        new_coef_name=test_channel,
                        new_coef=x,
                        true_up=True,
                    )
                    _, spend_attr_df, _, _ = attr_res
                    mask = (spend_attr_df[date_col] >= test_start) & (
                        spend_attr_df[date_col] <= test_end
                    )
                    res = np.sum(spend_attr_df.loc[mask, test_channel].values)
                    loss = np.fabs(res - target)
                    return loss

                init_search_pt = model.get_coef_vector([test_channel])
                coef_prior = fsolve(
                    attr_call_back,
                    x0=init_search_pt,
                    args=test_lift,
                )[0]
                coef_prior_upper = fsolve(
                    attr_call_back,
                    x0=init_search_pt,
                    args=test_lift_upper,
                )[0]
                logger.info(
                    "test spend: {:.3f}, test lift: {:.3f}".format(
                        test_spend, test_lift
                    )
                )
                sigma_prior = coef_prior_upper - coef_prior
                sigma_prior = max(sigma_prior, 1e-3)

                logger.info(
                    "coef prior: {:.3f}, sigma prior: {:.3f}".format(
                        coef_prior, sigma_prior
                    )
                )

                # store derived result
                output_df.loc[idx, "coef_prior"] = coef_prior
                output_df.loc[idx, "sigma_prior"] = sigma_prior
                output_df.loc[idx, "test_spend"] = test_spend
                output_df.loc[idx, "test_lift"] = test_lift

            output_df = output_df.loc[np.isfinite(output_df["coef_prior"])].reset_index(
                drop=True
            )

            return output_df


def calibrate_model_with_test(
    prev_model: MMM,
    tests_df: pd.DataFrame,
    n_iter: int = 1,
    seed: Optional[int] = None,
    fit_args: Optional[Dict[str, Any]] = None,
    debug: bool = False,
    haircut_ratio: float = 0.75,
    logger: Optional[logging.Logger] = None,
) -> Tuple[MMM, pd.DataFrame]:
    """Generate a new model based on baseline model with extra priors; This function can be reiterated to generate final calibrated model

    Args:
        prev_model (MMM): _description_
        tests_df (pd.DataFrame): _description_
        n_iter (int, optional): _description_. Defaults to 1.
        seed (Optional[int], optional): _description_. Defaults to None.
        fit_args (Optional[Dict[str, Any]], optional): _description_. Defaults to None.
        debug (bool, optional): _description_. Defaults to False.
        logger (Optional[logging.Logger], optional): _description_. Defaults to None.

    Returns:
        Tuple[MMM, pd.DataFrame]: new_model: the final calibrated model calibration_report: the dataframe stored the statistics of
        the calibration process.
    """
    spend_cols = prev_model.get_spend_cols()
    kpi_col = prev_model.kpi_col
    date_col = prev_model.date_col
    spend_cols = prev_model.get_spend_cols()
    # TODO: should have a get function
    full_event_cols = deepcopy(prev_model.full_event_cols)
    seasonality = deepcopy(prev_model.seasonality)
    fs_orders = deepcopy(prev_model.fs_orders)
    adstock_df = prev_model.get_adstock_df()

    # make validation report
    validation_dfs_list = list()

    if logger is None:
        logger = get_logger("karpiu-calibration")

    # solve ab-test priors
    ps = PriorSolver(tests_df=tests_df)
    for n in range(n_iter):
        logger.info("{}/{} iteration:".format(n + 1, n_iter))
        # shuffle is not impacting as we solve all priors from same initial model
        curr_priors_full = ps.derive_prior(prev_model, shuffle=False, logger=logger)
        # curr_priors_full has row for each test
        # curr_priors_unique has row for each channel only (combining all tests within a channel)
        # len(validation_dfs_list) = number of runs
        # check condition if this is first run
        if len(validation_dfs_list) > 0:
            curr_priors_full = curr_priors_full.set_index("test_name")
            # enforce strictly decreasing sigma; sigma is the minimum of the current or the previous
            all_test_names = prev_validation_df["test_name"].values
            prev_sigma_prior = prev_validation_df["sigma_prior"].values
            curr_sigma_prior = curr_priors_full.loc[
                all_test_names, "sigma_prior"
            ].values
            applied_sigma_prior = np.minimum(prev_sigma_prior, curr_sigma_prior)
            curr_priors_full.loc[all_test_names, "sigma_prior"] = applied_sigma_prior

            # for any channel solver cost beyond +/- 1se from input cost, make hair cut of the priors
            haircut_mask = (
                prev_validation_df["solver_cost"]
                >= prev_validation_df["input_cost_upper_1se"]
            ) | (
                prev_validation_df["solver_cost"]
                <= prev_validation_df["input_cost_lower_1se"]
            )
            # if haircut channel exists
            if haircut_mask.sum() > 0:
                haircut_test_names = prev_validation_df.loc[
                    haircut_mask, "test_name"
                ].values
                tighter_sigma_prior = (
                    haircut_ratio
                    * curr_priors_full.loc[haircut_test_names, "sigma_prior"].values
                )
                logger.info(
                    "Reduce sigma priors ({:.3%}) for test(s):{}".format(
                        haircut_ratio, haircut_test_names
                    )
                )
                logger.info("Reduced sigma priors:{}".format(tighter_sigma_prior))
                curr_priors_full.loc[
                    haircut_test_names, "sigma_prior"
                ] = tighter_sigma_prior

            # reset index for proper usage in other stages
            curr_priors_full = curr_priors_full.reset_index()

        curr_priors_unique = (
            curr_priors_full.groupby(by=["test_channel"])[["coef_prior", "sigma_prior"]]
            .apply(
                np.mean,
                axis=0,
            )
            .reset_index()
        )

        if debug:
            continue
        else:
            # derive rest of the priors from pervious posteriors
            reg_coef_df = prev_model._model.get_regression_coefs()
            posteriors_carryover = {
                "test_channel": [],
                "sigma_prior": [],
                "coef_prior": [],
            }
            for ch in spend_cols:
                if ch not in curr_priors_unique["test_channel"].values:
                    posteriors_carryover["test_channel"].append(ch)
                    coef = reg_coef_df.loc[
                        reg_coef_df["regressor"] == ch, "coefficient"
                    ].values[0]
                    posteriors_carryover["coef_prior"].append(coef)
                    # use median x 0.01 as sigma to lock down previous posteriors as priors
                    posteriors_carryover["sigma_prior"].append(0.01 * coef)
            posteriors_carryover = pd.DataFrame(posteriors_carryover)
            extra_priors_input = pd.concat(
                [curr_priors_unique, posteriors_carryover], axis=0
            )

            new_model = MMM(
                kpi_col=kpi_col,
                date_col=date_col,
                spend_cols=spend_cols,
                event_cols=full_event_cols,
                seed=seed,
                seasonality=seasonality,
                fs_orders=fs_orders,
                adstock_df=adstock_df,
                logger=prev_model.get_logger(),
                # no market sigma constraint here
            )
            df = prev_model.get_raw_df()
            new_model.set_features(prev_model.get_event_cols())
            new_model.set_hyper_params(prev_model.best_params)
            new_model.set_saturation(prev_model.get_saturation())

            if fit_args is not None:
                new_model.fit(df, extra_priors=extra_priors_input, **fit_args)
            else:
                new_model.fit(df, extra_priors=extra_priors_input)

            validation_entries_list = list()
            coef_posteriors_df = new_model.get_regression_summary().set_index(
                "regressor"
            )

            for _, row in curr_priors_full.iterrows():
                test_start = row.loc["test_start"]
                test_end = row.loc["test_end"]
                test_icac = row.loc["test_icac"]
                test_channel = row.loc["test_channel"]
                test_name = row.loc["test_name"]
                coef_prior = row.loc["coef_prior"]
                sigma_prior = row.loc["sigma_prior"]
                test_lift = row.loc["test_lift"]
                test_se = row.loc["test_se"]

                if test_channel not in tests_df["test_channel"].values:
                    continue

                attr_obj = Attributor(new_model, start=test_start, end=test_end)
                attr_res = attr_obj.make_attribution()
                _, spend_attr_df, _, _ = attr_res
                mask = (spend_attr_df[date_col] >= test_start) & (
                    spend_attr_df[date_col] <= test_end
                )
                total_attr = np.sum(spend_attr_df.loc[mask, test_channel].values)
                mask = (df[date_col] >= test_start) & (df[date_col] <= test_end)
                total_spend = np.sum(df.loc[mask, test_channel].values)

                validation_entry = {
                    "iteration": n,
                    "channel": test_channel,
                    "test_name": test_name,
                    "test_start": test_start,
                    "test_end": test_end,
                    "coef_prior": coef_prior,
                    "sigma_prior": sigma_prior,
                    "coef_posterior": coef_posteriors_df.loc[test_channel, "coef_p50"],
                    "input_cost": test_icac,
                    "input_cost_upper_1se": test_icac + test_se,
                    "input_cost_lower_1se": test_icac - test_se,
                    "input_cost_se": test_se,
                    "input_lift": test_lift,
                    "solver_cost": total_spend / total_attr,
                    "solver_lift": total_attr,
                }
                validation_entries_list.append(validation_entry)
                # end of test channel iterations

            prev_validation_df = pd.DataFrame(validation_entries_list)
            validation_dfs_list.append(prev_validation_df)
            prev_model = deepcopy(new_model)
            # end of n trial iteration

    if debug:
        pass
    else:
        calibration_report = pd.concat(validation_dfs_list, axis=0, ignore_index=True)
        return new_model, calibration_report


def make_cost_calibration_plot(report_df: pd.DataFrame) -> matplotlib.axes:
    """Visualize cost calibration given report dataframe"""
    test_names = report_df["test_name"].unique().tolist()
    nrows = math.ceil(len(test_names) / 2)
    fig, axes = plt.subplots(nrows, 2, figsize=(20, 2 + 3.5 * nrows))
    axes = axes.flatten()
    for idx, name in enumerate(test_names):
        mask = report_df["test_name"] == name
        x = report_df.loc[mask, "iteration"].values
        y_input = report_df.loc[mask, "input_cost"].values
        y_solver = report_df.loc[mask, "solver_cost"].values
        y_input_upper = report_df.loc[mask, "input_cost_upper_1se"].values
        y_input_lower = report_df.loc[mask, "input_cost_lower_1se"].values
        axes[idx].set_title(name)
        axes[idx].plot(x, y_solver, label="solver", color="orange", alpha=0.5)
        axes[idx].plot(
            x, y_input, linestyle="--", label="input", color="dodgerblue", alpha=0.5
        )
        axes[idx].fill_between(
            x, y_input_lower, y_input_upper, alpha=0.2, color="dodgerblue"
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.0, 0.05, 1.0, 1.0),
        fancybox=True,
        shadow=True,
        ncols=2,
        fontsize=18,
    )
    plt.close()
    return axes


def make_coef_calibration_plot(report_df: pd.DataFrame) -> matplotlib.axes:
    """Visualize coefficients calibration given report dataframe"""
    test_names = report_df["test_name"].unique().tolist()
    nrows = math.ceil(len(test_names) / 2)
    fig, axes = plt.subplots(nrows, 2, figsize=(20, 2 + 3.5 * nrows))
    axes = axes.flatten()
    for idx, name in enumerate(test_names):
        mask = report_df["test_name"] == name
        x = report_df.loc[mask, "iteration"].values
        y_priors = report_df.loc[mask, "coef_prior"].values
        y_posteriors = report_df.loc[mask, "coef_posterior"].values
        y_priors_upper = y_priors + report_df.loc[mask, "sigma_prior"].values
        y_priors_lower = y_priors - report_df.loc[mask, "sigma_prior"].values
        axes[idx].set_title(name)
        axes[idx].plot(x, y_posteriors, label="posteriors", color="orange", alpha=0.5)
        axes[idx].plot(
            x, y_priors, linestyle="--", label="priors", color="dodgerblue", alpha=0.5
        )
        axes[idx].fill_between(
            x, y_priors_lower, y_priors_upper, alpha=0.2, color="dodgerblue"
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.0, 0.05, 1.0, 1.0),
        fancybox=True,
        shadow=True,
        ncols=2,
        fontsize=18,
    )
    plt.close()
    return axes
