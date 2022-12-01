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
        self.tests_df = tests_df

    def derive_prior(self, model: MMM) -> pd.DataFrame:
        input_df = model.raw_df.copy()
        output_df = self.tests_df.copy()
        date_col = model.date_col

        for idx, row in self.tests_df.iterrows():
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

            logger.info("test channel:{}".format(test_channel))
            logger.info(
                "test spend: {:.3f}, test lift: {:.3f}".format(test_spend, test_lift)
            )

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

            # store derived result
            output_df.loc[idx, "coef_prior"] = coef_prior
            # since model can be over-confident on empirical result and the non-linear relationship,
            # introduce a 0.3 haircut on the derive sigma here
            output_df.loc[idx, "sigma_prior"] = (coef_prior_upper - coef_prior) * 0.3
            output_df.loc[idx, "test_spend"] = test_spend
            output_df.loc[idx, "test_lift"] = test_lift

        return output_df


def calibrate_model_with_test(
    prev_model: MMM,
    tests_df: pd.DataFrame,
    n_iter: int = 1,
    seed: Optional[int] = None,
    fit_args: Optional[Dict[str, Any]] = None,
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
    validation_entries_list = list()

    for n in range(n_iter):
        logger.info("{}/{} iteration:".format(n + 1, n_iter))
        # solve ab-test priors
        ps = PriorSolver(tests_df=tests_df)
        extra_priors = ps.derive_prior(prev_model)
        extra_priors_input = (
            extra_priors.groupby(by=["test_channel"])[["coef_prior", "sigma_prior"]]
            .apply(
                np.mean,
                axis=0,
            )
            .reset_index()
        )

        # derive rest of the priors from pervious posteriors
        reg_coef_df = prev_model._model.get_regression_coefs()
        posteriors_carryover = {"test_channel": [], "sigma_prior": [], "coef_prior": []}
        for ch in spend_cols:
            if ch not in extra_priors_input["test_channel"].values:
                posteriors_carryover["test_channel"].append(ch)
                coef = reg_coef_df.loc[
                    reg_coef_df["regressor"] == ch, "coefficient"
                ].values[0]
                posteriors_carryover["coef_prior"].append(coef)
                # use median x 0.01 as sigma to lock down previous posteriors as priors
                posteriors_carryover["sigma_prior"].append(0.01 * coef)
        posteriors_carryover = pd.DataFrame(posteriors_carryover)
        extra_priors_input = pd.concat(
            [extra_priors_input, posteriors_carryover], axis=0
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

        for idx, row in extra_priors.iterrows():
            test_start = row.loc["test_start"]
            test_end = row.loc["test_end"]
            test_icac = row.loc["test_icac"]
            test_channel = row.loc["test_channel"]
            test_prior = row.loc["coef_prior"]
            test_lift = row.loc["test_lift"]

            if test_channel not in tests_df["test_channel"].values:
                continue

            attr_obj = Attributor(new_model, start=test_start, end=test_end)
            attr_res = attr_obj.make_attribution()
            _, spend_attr_df, _, cost_df = attr_res
            mask = (spend_attr_df[date_col] >= test_start) & (
                spend_attr_df[date_col] <= test_end
            )
            total_attr = np.sum(spend_attr_df.loc[mask, test_channel].values)
            mask = (df[date_col] >= test_start) & (df[date_col] <= test_end)
            total_spend = np.sum(df.loc[mask, test_channel].values)

            # print("Channel: {}".format(test_channel))
            # print("Input test information. ICAC: {:.3f} Total Lift {:.3f}".format(test_icac, test_lift))
            # print("Solver Result. ICAC: {:.3f} Total Lift {:.3f}".format(total_spend / total_attr, total_attr))

            validation_entry = {
                "iteration": n,
                "channel": test_channel,
                "input_cost": test_icac,
                "input_lift": test_lift,
                "solver_cost": total_spend / total_attr,
                "solver_lift": total_attr,
            }
            validation_entries_list.append(validation_entry)
            # end of test channel iterations
        prev_model = deepcopy(new_model)
        # end of n trial iteration

    return new_model, pd.DataFrame(validation_entries_list)
