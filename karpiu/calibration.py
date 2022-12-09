import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from .explainability import Attributor
from .models import MMM
from copy import deepcopy
import logging

logger = logging.getLogger("karpiu-mmm")

from typing import Optional, Dict, Any


class PriorSolver:
    """Solving Regression Coefficient Prior from MMM by using Attribution logic"""

    def __init__(self, tests_df: pd.DataFrame):
        self.tests_df = tests_df.copy()

    def derive_prior(self, model: MMM, shuffle: bool = False, debug: bool = False) -> pd.DataFrame:
        # TODO: shuffle is not useful for now as it always solves from the same initial model
        input_df = model.raw_df.copy()
        if shuffle:
            tests_df = self.tests_df .sample(frac=1, ignore_index=True)
        else:
            tests_df = self.tests_df .copy()
        output_df = tests_df.copy()
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

                # derive test spend
                mask = (input_df[date_col] >= test_start) & (input_df[date_col] <= test_end)
                sub_input_df = input_df[mask].reset_index(drop=True)
                test_spend = sub_input_df[test_channel].sum()
                # derive lift from spend data from model to ensure consistency
                test_lift = test_spend / test_icac
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
                logger.info("test channel:{}".format(test_channel))
                logger.info(
                    "test spend: {:.3f}, test lift: {:.3f}".format(test_spend, test_lift)
                )
                sigma_prior = (coef_prior_upper - coef_prior) * 0.3
                logger.info(
                    "coef prior: {:.3f}, sigma prior: {:.3f}".format(coef_prior, sigma_prior)
                )

                # store derived result
                output_df.loc[idx, "coef_prior"] = coef_prior
                # since model can be over-confident on empirical result and the non-linear relationship,
                # introduce a 0.3 haircut on the derive sigma here
                output_df.loc[idx, "sigma_prior"] = sigma_prior
                output_df.loc[idx, "test_spend"] = test_spend
                output_df.loc[idx, "test_lift"] = test_lift

            return output_df


def calibrate_model_with_test(
    prev_model: MMM,
    tests_df: pd.DataFrame,
    n_iter: int = 1,
    seed: Optional[int] = None,
    fit_args: Optional[Dict[str, Any]] = None,
    debug: bool = False,
) -> MMM:
    """Generate a new model based on baseline model with extra priors; This function can be reiterated to generate final calibrated model."""
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

    # solve ab-test priors
    ps = PriorSolver(tests_df=tests_df)
    for n in range(n_iter):
        logger.info("{}/{} iteration:".format(n + 1, n_iter))
        # shuffle is not impacting as we solve all priors from same initial model
        curr_priors_full = ps.derive_prior(prev_model, shuffle=False)
        # curr_priors_full has row for each test
        # curr_priors_unique has row for each channel only (combining all tests within a channel)
        if len(validation_dfs_list) > 0:
            tighter_prior_mask = (
                (prev_validation_df['solver_cost'] >= prev_validation_df['input_cost_upper_1se']) |
                (prev_validation_df['solver_cost'] <= prev_validation_df['input_cost_lower_1se'])
            )
            if tighter_prior_mask.sum() > 0:
                tighter_prior_tests = prev_validation_df.loc[tighter_prior_mask, "test_name"].values
                tighter_sigma_prior = prev_validation_df.loc[tighter_prior_mask, "sigma_prior"].values * 0.5
                logger.info("Reduce sigma priors (50%) for test(s):{}".format(tighter_prior_tests))
                logger.info("Reduced sigma priors:{}".format(tighter_sigma_prior))

                curr_priors_full = curr_priors_full.set_index("test_name") 
                curr_priors_full.loc[tighter_prior_tests, "sigma_prior"] = tighter_sigma_prior
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
            posteriors_carryover = {"test_channel": [], "sigma_prior": [], "coef_prior": []}
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
                # TODO: should have a get function
                event_cols=full_event_cols,
                seed=seed,
                seasonality=seasonality,
                fs_orders=fs_orders,
                adstock_df=adstock_df,
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
        return new_model, pd.concat(validation_dfs_list, axis=0, ignore_index=True)
