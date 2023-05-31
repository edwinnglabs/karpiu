import numpy as np
import pandas as pd
import scipy.optimize as optim
from copy import deepcopy
import logging
import matplotlib
import matplotlib.pyplot as plt
import math
from typing import Optional, Dict, Tuple, List

from ..explainability import AttributorBeta
from ..models import MMM
from ..utils import get_logger

#TODO: factor important keys and strings as ENUM / constants
class CalibrationProcess:
    def __init__(
        self,
        model: MMM,
        calib_tests_df: pd.DataFrame,
        sigma_prior_haircut: float = 1e-3,
        logger: Optional[logging.Logger] = None,
    ):
        if logger is None:
            self.logger = get_logger("karpiu-calibration")
        else:
            self.logger = logger

        self.base_model = deepcopy(model)
        self.curr_model = deepcopy(model)

        self.solver_results = list()
        self.calib_reports = list()

        extended_tests_df_base = self.compute_test_stat(model, calib_tests_df)
        self.sigma_prior_haircut = sigma_prior_haircut
        self.n_fit_steps = 0
        self.n_iter_steps = 0

        check_flag = extended_tests_df_base["test_spend"].values <= 1e-5
        if np.any(check_flag):
            drop_channels = extended_tests_df_base.loc[
                check_flag, "test_channel"
            ].values
            self.logger.info("Dropping zero spend channels: {}".format(drop_channels))
            extended_tests_df_base = extended_tests_df_base.loc[
                ~extended_tests_df_base["test_channel"].isin(drop_channels)
            ].reset_index(drop=True)

        # make sure there is no duplicate of channels
        channels_pool = set(extended_tests_df_base["test_channel"].to_list())
        assert len(channels_pool) == extended_tests_df_base.shape[0]

        # with test_channel as index
        self.calib_tests_df = extended_tests_df_base.set_index("test_channel")
        # a set
        self.channels_pool = channels_pool

    def calibrate(
        self,
        n_iters: int = 1,
        from_new: bool = False,
        fit_args: Optional[Dict] = None,
    ) -> None:
        # TODO: implement logic when from_new is True
        curr_model = deepcopy(self.curr_model)

        spend_cols = curr_model.get_spend_cols()
        kpi_col = curr_model.kpi_col
        date_col = curr_model.date_col
        spend_cols = curr_model.get_spend_cols()
        # TODO: should have a get function
        full_event_cols = deepcopy(curr_model.full_event_cols)
        seasonality = deepcopy(curr_model.seasonality)
        fs_orders = deepcopy(curr_model.fs_orders)
        adstock_df = curr_model.get_adstock_df()
        if fit_args is None:
            fit_args = dict()

        # n_step include each call of fit
        n_steps = 0
        for n in range(n_iters):
            # update and report attribution result based on the current model
            temp_reset_df = self.calib_tests_df.copy()
            temp_reset_df = temp_reset_df.reset_index()
            attr_report_df = self.report_attribution(curr_model, temp_reset_df)

            # pick up a channel based on the criteria
            sample_channels = self.select_calib_channels(attr_report_df, se_multiplier=0.1)
            if len(sample_channels) == 0:
                self.logger.info("Meet exit condition for all channels.")
                continue

            rest_channels = list(self.channels_pool - set(sample_channels))
            self.logger.info("Iterations: {}/{}".format(n + 1, n_iters))
            self.logger.info("Sampled channels: {}".format(sample_channels))
            self.logger.info("Rest of channels: {}".format(rest_channels))

            # derive priors
            for ch in sample_channels:
                self.logger.info("Calibrating channel: {}".format(ch))

                # access by test_channel as index
                # result_df should be single row
                test_start = self.calib_tests_df.loc[ch, "test_start"]
                test_end = self.calib_tests_df.loc[ch, "test_end"]
                test_lift = self.calib_tests_df.loc[ch, "test_lift"]
                test_lift_upper = self.calib_tests_df.loc[ch, "test_lift_upper"]

                coef_prior, sigma_prior = self._solve_one_channel(
                    curr_model,
                    ch,
                    test_lift,
                    test_lift_upper,
                    test_start,
                    test_end,
                    sigma_prior_haircut=1e-1,
                    fixed_intercept=False,
                )
                self.logger.info(
                    "Solved priors: ({},{})".format(coef_prior, sigma_prior)
                )

                result_df = pd.DataFrame(
                    {
                        "test_channel": ch,
                        "coef_prior": coef_prior,
                        "sigma_prior": sigma_prior,
                    },
                    index=[0],
                )

                # store result
                result_df["iter"] = self.n_iter_steps + n
                result_df["n_step"] = self.n_fit_steps + n_steps
                self.solver_results.append(result_df)

                # update and merge with previous priors
                new_priors_df = self._merge_prior(curr_model, result_df)

                new_model = MMM(
                    kpi_col=kpi_col,
                    date_col=date_col,
                    spend_cols=spend_cols,
                    event_cols=full_event_cols,
                    # seed=seed,
                    seasonality=seasonality,
                    fs_orders=fs_orders,
                    adstock_df=adstock_df,
                    logger=curr_model.get_logger(),
                    # no market sigma constraint here to allow flexibility
                )
                df = curr_model.get_raw_df()

                new_model.set_features(curr_model.get_event_cols())
                new_model.set_hyper_params(curr_model.best_params)
                new_model.set_saturation(curr_model.get_saturation())
                new_model.fit(
                    df,
                    extra_priors=new_priors_df,
                    **fit_args,
                )

                # replace model
                curr_model = deepcopy(new_model)

                # TODO: there is some overhead of the attribution when there is an exit here
                # but it is minimal; so ignore it for now
                temp_reset_df = self.calib_tests_df.copy()
                temp_reset_df = temp_reset_df.reset_index()
                # this report has extra info such as posteriors etc.
                calib_report = self.report_attribution(curr_model, temp_reset_df)
                calib_report = self.extra_model_info(curr_model, calib_report)
                calib_report["n_step"] = self.n_fit_steps + n_steps
                self.calib_reports.append(calib_report)

                n_steps += 1

        self.curr_model = deepcopy(curr_model)
        self.n_fit_steps += n_steps
        self.n_iter_steps += n_iters

    def get_calib_report(self):
        calib_report = pd.concat(self.calib_reports).reset_index(drop=True)
        calib_report["input_cost_upper_1se"] = (
            calib_report["test_icac"] + calib_report["test_se"]
        )
        calib_report["input_cost_lower_1se"] = (
            calib_report["test_icac"] - calib_report["test_se"]
        )
        return calib_report

    def get_solver_result(self):
        return pd.concat(self.solver_results).reset_index(drop=True)

    def get_curr_model(self):
        return deepcopy(self.curr_model)

    @staticmethod
    def plot_coef_calib(calib_report_df: pd.DataFrame, is_visible: bool = True):
        test_channels = calib_report_df["test_channel"].unique().tolist()
        nrows = math.ceil(len(test_channels) / 2)

        fig, axes = plt.subplots(nrows, 2, figsize=(20, 2 + 3.5 * nrows))
        axes = axes.flatten()
        for idx, ch in enumerate(test_channels):
            mask = calib_report_df['test_channel'] == ch
            x = calib_report_df.loc[mask, 'n_step'].values
            y_input = calib_report_df.loc[mask, 'test_icac'].values
            y_solver = calib_report_df.loc[mask, 'mmm_icac'].values
            y_input_upper = calib_report_df.loc[mask, 'input_cost_upper_1se'].values
            y_input_lower = calib_report_df.loc[mask, 'input_cost_lower_1se'].values
            axes[idx].set_title(ch)
            axes[idx].plot(x, y_solver, label='solver', color='orange', alpha=0.5)
            axes[idx].plot(x, y_input, linestyle='--', label='input', color='dodgerblue', alpha=0.5)
            axes[idx].fill_between(x, y_input_lower, y_input_upper, alpha=0.2, color='dodgerblue')

        fig.suptitle("Cost Calibration", y=0.92, fontsize=28, verticalalignment="center")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels, 
            loc='lower center',
            bbox_to_anchor=(0., 0.05, 1., 1.),
            fancybox=True, shadow=True, ncols=2,
            fontsize=18,
        )

        if is_visible:
            plt.show()
        else:
            plt.close()

        return axes

    @staticmethod
    def plot_cost_calib(calib_report_df: pd.DataFrame, is_visible: bool = True):
        test_channels = calib_report_df["test_channel"].unique().tolist()
        nrows = math.ceil(len(test_channels) / 2)

        fig, axes = plt.subplots(nrows, 2, figsize=(20, 2 + 3.5 * nrows))
        axes = axes.flatten()
        for idx, ch in enumerate(test_channels):
            mask = calib_report_df['test_channel'] == ch
            x = calib_report_df.loc[mask, 'n_step'].values
            y_priors = calib_report_df.loc[mask, 'coef_prior'].values
            y_posteriors = calib_report_df.loc[mask, 'coef_posterior'].values
            y_priors_upper = y_priors + calib_report_df.loc[mask, 'sigma_prior'].values
            y_priors_lower = y_priors - calib_report_df.loc[mask, 'sigma_prior'].values
            axes[idx].set_title(ch)
            axes[idx].plot(x, y_posteriors, label='posteriors', color='orange', alpha=0.5)
            axes[idx].plot(x, y_priors, linestyle='--', label='priors', color='dodgerblue', alpha=0.5)
            axes[idx].fill_between(x, y_priors_lower, y_priors_upper, alpha=0.2, color='dodgerblue')

        fig.suptitle("Coefficient Calibration", y=0.92, fontsize=24, verticalalignment="center")
        
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels, 
            loc='lower center',
            bbox_to_anchor=(0., 0.05, 1., 1.),
            fancybox=True, shadow=True, ncols=2,
            fontsize=18,
        )

        if is_visible:
            plt.show()
        else:
            plt.close()

        return axes
        

    def select_calib_channels(
        self,
        attr_report_df: pd.DataFrame,
        se_multiplier: float = 0.1,
    ) -> List[str]:
        """Given a attribution report dataframe, recommend and return
        the list of channels should be calibrated

        Args:
            attr_report_df (pd.DataFrame): attribution data frame
            se_multiplier (float, optional): Criteria to select channels. Defaults to 0.1.

        Returns:
            List[str]: recommended channels
        """
        result_df = attr_report_df.copy()
        select_flag = np.logical_not(
            np.abs(result_df["mmm_icac"] - result_df["test_icac"])
            <= result_df["test_se"] * se_multiplier
        )
        return result_df.loc[select_flag, "test_channel"].values

    def report_attribution(
        self,
        model: MMM,
        ab_tests_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """_summary_

        Args:
        model (MMM): _description_
        ab_tests_df (pd.DataFrame): _description_

        Returns:
        pd.DataFrame: _description_
        """
        new_ab_tests_df = ab_tests_df.copy()
        new_ab_tests_df["mmm_icac"] = -1
        new_ab_tests_df["mmm_attr_lift"] = -1

        for idx, row in new_ab_tests_df.iterrows():
            df = model.get_raw_df()
            test_start = row["test_start"]
            test_end = row["test_end"]
            test_channel = row["test_channel"]
            test_spend = row["test_spend"]

            attr_obj = AttributorBeta(model, df=df, start=test_start, end=test_end)
            # for reporting, always fix
            _, spend_attr_df, _, _ = attr_obj.make_attribution(fixed_intercept=False)
            attr_sum = np.sum(spend_attr_df.loc[:, test_channel].values)
            new_ab_tests_df.loc[idx, "mmm_attr_lift"] = attr_sum
            if attr_sum > 0:
                new_ab_tests_df.loc[idx, "mmm_icac"] = test_spend / attr_sum
            else:
                new_ab_tests_df.loc[idx, "mmm_icac"] = np.nan
        return new_ab_tests_df

    def extra_model_info(
        self, model: MMM, attr_report_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Given a fitted model and the target data frame, attach regression
        related information to the end result data frame and return it.

        Args:
            model (MMM): model object
            attr_report_df (pd.DataFrame): target data frame

        Returns:
            pd.DataFrame: _description_
        """
        regression_summary = model.get_regression_summary()[
            ["regressor", "coef_p50", "loc_prior", "scale_prior"]
        ]
        regression_summary = regression_summary.rename(
            columns={
                "regressor": "test_channel",
                "loc_prior": "coef_prior",
                "scale_prior": "sigma_prior",
                "coef_p50": "coef_posterior",
            }
        )

        attr_report_df = pd.merge(
            attr_report_df, regression_summary, how="inner", on="test_channel"
        )
        return attr_report_df

    def compute_test_stat(
        self,
        model: MMM,
        ab_tests_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """_summary_

        Args:
            model (MMM): _description_
            ab_tests_df (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """
        date_col = model.date_col
        df = model.get_raw_df()

        new_ab_tests_df = ab_tests_df.copy()
        new_ab_tests_df["test_lift"] = -1
        new_ab_tests_df["test_lift_upper"] = -1

        for idx, row in new_ab_tests_df.iterrows():
            test_start = row["test_start"]
            test_end = row["test_end"]
            test_icac = row["test_icac"]
            test_channel = row["test_channel"]
            test_se = row["test_se"]

            # derive test spend
            mask = (df[date_col] >= test_start) & (df[date_col] <= test_end)
            sub_input_df = df[mask].reset_index(drop=True)
            test_spend = sub_input_df[test_channel].sum()
            # derive lift from spend data from model to ensure consistency
            test_lift = test_spend / test_icac
            new_ab_tests_df.loc[idx, "test_lift"] = test_lift
            test_lift_upper = test_spend / (test_icac - test_se)
            new_ab_tests_df.loc[idx, "test_spend"] = test_spend
            new_ab_tests_df.loc[idx, "test_lift_upper"] = test_lift_upper

        return new_ab_tests_df

    # test_channel, coef_prior, sigma_prior are the reserved keywords in main model
    # TODO: consider make them consistent later
    def _merge_prior(
        self,
        model: MMM,
        new_ab_tests_df,
        # test_channel: str,
        # coef_prior: float,
        # sigma_prior: float,
    ) -> pd.DataFrame:
        spend_cols = model.get_spend_cols()
        # if test_channel not in spend_cols:
        #     raise Exception("Input channel is not included in the spend column(s) from the model.")

        new_prior_df = model.get_regression_summary()[
            ["regressor", "loc_prior", "scale_prior"]
        ]
        new_prior_df = new_prior_df.rename(
            columns={
                "regressor": "test_channel",
                "loc_prior": "coef_prior",
                "scale_prior": "sigma_prior",
            }
        )
        new_prior_df = new_prior_df[
            new_prior_df["test_channel"].isin(spend_cols)
        ].reset_index(drop=True)
        new_prior_df = new_prior_df.set_index("test_channel")
        # print(new_ab_tests_df.head(10))
        # print(new_prior_df.head(10))
        new_prior_df.loc[
            new_ab_tests_df["test_channel"].values, "coef_prior"
        ] = new_ab_tests_df["coef_prior"].values
        new_prior_df.loc[
            new_ab_tests_df["test_channel"].values, "sigma_prior"
        ] = new_ab_tests_df["sigma_prior"].values
        new_prior_df = new_prior_df.reset_index()
        # print(new_prior_df.head(10))
        return new_prior_df

    # test_channel, coef_prior, sigma_prior are the reserved keywords in main model
    # TODO: consider make them consistent later
    def _solve_one_channel(
        self,
        model: MMM,
        test_channel: str,
        test_lift: float,
        test_lift_upper: float,
        test_start: str,
        test_end: str,
        sigma_prior_haircut: float,
        fixed_intercept: bool,
    ) -> Tuple[float, float]:
        date_col = model.date_col
        attr_obj = AttributorBeta(model, start=test_start, end=test_end)

        def attr_call_back(x, target):
            attr_res = attr_obj.make_attribution(
                new_coef_name=test_channel,
                new_coef=x,
                true_up=True,
                fixed_intercept=fixed_intercept,
            )
            _, spend_attr_df, _, _ = attr_res
            mask = (spend_attr_df[date_col] >= test_start) & (
                spend_attr_df[date_col] <= test_end
            )
            res = np.sum(spend_attr_df.loc[mask, test_channel].values)
            loss = np.fabs(res - target)
            return loss

        # minimize approach
        # init_search_pt = max(
        #     model.get_coef_vector([test_channel]),
        #     1e-3,
        # )
        # bounds = optim.Bounds(
        #     np.ones(1) * 1e-3,
        #     np.ones(1) * np.inf,
        # )

        # sol = optim.minimize(
        #     attr_call_back,
        #     x0=init_search_pt,
        #     method="SLSQP",
        #     bounds=bounds,
        #     options={
        #         "disp": True,
        #         "maxiter": 1000,
        #         # "eps": 1.,
        #         # "ftol": 1.,
        #     },
        #     args=test_lift,
        # )
        # coef_prior = sol.x[0]

        # sol = optim.minimize(
        #     attr_call_back,
        #     x0=init_search_pt,
        #     method="SLSQP",
        #     bounds=bounds,
        #     options={
        #         "disp": True,
        #         "maxiter": 1000,
        #         # "eps": 1e-3.,
        #         # "ftol": 1.,
        #     },
        #     args=test_lift_upper,
        # )
        # coef_prior_upper = sol.x[0]

        # fsolve approach
        init_search_pt = max(
            model.get_coef_vector([test_channel]),
            1e-3,
        )

        sol = optim.fsolve(
            attr_call_back,
            x0=init_search_pt,
            args=test_lift,
        )
        coef_prior = sol[0]

        sol = optim.fsolve(
            attr_call_back,
            x0=init_search_pt,
            args=test_lift_upper,
        )
        coef_prior_upper = sol[0]

        sigma_prior = coef_prior_upper - coef_prior
        # prevent underflow
        sigma_prior = max(sigma_prior * sigma_prior_haircut, 1e-5)
        # non-negative coef
        coef_prior = max(coef_prior, 0.0)

        return coef_prior, sigma_prior
