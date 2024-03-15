import numpy as np
import pandas as pd
import scipy.optimize as optim
from copy import deepcopy
import logging
import matplotlib.pyplot as plt
import math
from typing import Optional, Dict, Tuple, List

from ..explainability import AttributorGamma
from ..models import MMM
from ..utils import get_logger


# TODO: factor important keys and strings as ENUM / constants
class CalibrationProcess:
    def __init__(
        self,
        model: MMM,
        calib_tests_df: pd.DataFrame,
        logger: Optional[logging.Logger] = None,
        se_tolerance: float = 1.0,
        silent: bool = False,
    ):
        if logger is None:
            self.logger = get_logger("karpiu-calibration")
            # only warning and error messages
            if silent:
                self.logger.setLevel(30)
        else:
            self.logger = logger

        self.base_mod = model.copy()
        self.curr_mod = model.copy()

        self.calib_reports = list()
        self.calibrated_channels = list()
        self.se_tolerance = se_tolerance
        self.init_tests_df = calib_tests_df.copy()
        init_report = CalibrationProcess.report_attribution(
            self.curr_mod, self.init_tests_df
        )
        init_report = CalibrationProcess.append_model_info(self.curr_mod, init_report)
        init_report["iter"] = 0

        self.calib_reports.append(init_report)
        self.full_calib_regressors = init_report.index

    def calibrate(
        self,
        max_iters: int = 1,
        fit_args: Optional[Dict] = None,
    ) -> None:
        # TODO: implement logic when from_new is True
        curr_mod = self.curr_mod.copy(suppress_adstock=False)

        base_fit_args = {
            "num_warmup": 2000,
            "num_sample": 2000,
            "chains": 4,
        }
        if fit_args is not None:
            base_fit_args.update(fit_args)
            fit_args = base_fit_args
        else:
            fit_args = base_fit_args

        iter = 0
        # sorted in descending order
        calibrating_channels = self.select_calib_channels(self.calib_reports[-1])
        self.logger.info(
            "{}/{} channels required to be calibrated:\n {}".format(
                len(calibrating_channels),
                self.calib_reports[0].shape[0],
                calibrating_channels,
            )
        )
        self.logger.info(
            "Total target distance: {}".format(
                self.calib_reports[-1]["target_dist"].sum()
            )
        )

        while (iter < max_iters) and len(calibrating_channels) > 0:
            self.logger.info(f"Iteration: {iter}/{max_iters}")

            # pick the largest distance channel
            target_channel = calibrating_channels[0]
            self.logger.info(f"Calibrating channel: {target_channel}")
            loc_prior = self._solve_channel_prior(
                curr_mod, target_channel, self.calib_reports[-1]
            )

            new_priors_df = CalibrationProcess.generate_prior(
                model=curr_mod,
                modify_channel=target_channel,
                loc_prior=loc_prior,
            )

            # calibrate new model
            new_mod = MMM(
                kpi_col=curr_mod.kpi_col,
                date_col=curr_mod.date_col,
                spend_cols=curr_mod.spend_cols,
                event_cols=curr_mod.full_event_cols,
                # seed=seed,
                seasonality=curr_mod.seasonality,
                fs_orders=curr_mod.fs_orders,
                adstock_df=curr_mod.adstock_df,
                logger=curr_mod.get_logger(),
                # no market sigma constraint here to allow flexibility
            )
            df = curr_mod.get_raw_df()

            new_mod.set_features(curr_mod.get_event_cols())
            new_mod.set_hyper_params(curr_mod.best_params)
            new_mod.set_saturation(curr_mod.get_saturation())

            new_mod.fit(
                df,
                extra_priors=new_priors_df,
                **fit_args,
            )
            curr_mod = new_mod
            iter += 1

            new_report = CalibrationProcess.report_attribution(
                curr_mod, self.init_tests_df
            )
            new_report = CalibrationProcess.append_model_info(curr_mod, new_report)
            new_report["iter"] = iter

            self.calib_reports.append(new_report)
            self.calibrated_channels.append(target_channel)

            calibrating_channels = self.select_calib_channels(new_report)
            self.logger.info(
                "{}/{} channels required to be calibrated:\n {}".format(
                    len(calibrating_channels),
                    self.calib_reports[0].shape[0],
                    calibrating_channels,
                )
            )
            self.logger.info(
                "Total target distance: {}".format(
                    self.calib_reports[-1]["target_dist"].sum()
                )
            )

        self.curr_mod = curr_mod

    def get_calib_model(self):
        return self.curr_mod.copy()

    @staticmethod
    def plot_cost_calib(
        calib_report_df: pd.DataFrame,
        calibrated_steps: pd.DataFrame,
        is_visible: bool = True,
    ):
        regressors = calib_report_df["regressor"].unique().tolist()
        nrows = math.ceil(len(regressors) / 2)

        fig, axes = plt.subplots(nrows, 2, figsize=(20, 2 + 3.5 * nrows))
        axes = axes.flatten()
        for idx, ch in enumerate(regressors):
            mask = calib_report_df["regressor"] == ch
            x = calib_report_df.loc[mask, "iter"].values
            y_target = calib_report_df.loc[mask, "test_icac"].values
            y_target_lower = y_target - calib_report_df.loc[mask, "test_se"].values
            y_target_upper = y_target + calib_report_df.loc[mask, "test_se"].values

            y_mmm = calib_report_df.loc[mask, "mmm_icac"].values

            axes[idx].set_title(ch)
            axes[idx].plot(x, y_mmm, label="mmm", color="orange", alpha=0.5)
            axes[idx].plot(
                x,
                y_target,
                linestyle="--",
                label="target",
                color="dodgerblue",
                alpha=0.5,
            )
            axes[idx].fill_between(
                x, y_target_lower, y_target_upper, alpha=0.2, color="dodgerblue"
            )

            mask = calibrated_steps["regressor"] == ch
            if np.any(mask):
                x_steps = calibrated_steps.loc[mask, "iter"]
                axes[idx].scatter(
                    x_steps, y_mmm[x_steps], color="orange", label="calibration"
                )

        fig.suptitle(
            "Cost Calibration", y=0.92, fontsize=28, verticalalignment="center"
        )

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.0, 0.05, 1.0, 1.0),
            fancybox=True,
            shadow=True,
            ncols=3,
            fontsize=18,
        )

        if is_visible:
            plt.show()
        else:
            plt.close()

        return axes

    @staticmethod
    def plot_coef_calib(
        calib_report_df: pd.DataFrame,
        calibrated_steps: pd.DataFrame,
        is_visible: bool = True,
    ):
        regressors = calib_report_df["regressor"].unique().tolist()
        nrows = math.ceil(len(regressors) / 2)

        fig, axes = plt.subplots(nrows, 2, figsize=(20, 2 + 3.5 * nrows))
        axes = axes.flatten()
        for idx, ch in enumerate(regressors):
            mask = calib_report_df["regressor"] == ch
            x = calib_report_df.loc[mask, "iter"].values
            y_priors = calib_report_df.loc[mask, "coef_prior"].values
            y_posteriors = calib_report_df.loc[mask, "coef_posterior"].values
            y_priors_upper = y_priors + calib_report_df.loc[mask, "sigma_prior"].values
            y_priors_lower = y_priors - calib_report_df.loc[mask, "sigma_prior"].values

            axes[idx].set_title(ch)
            axes[idx].plot(
                x, y_posteriors, label="posteriors", color="orange", alpha=0.5
            )
            axes[idx].plot(
                x,
                y_priors,
                linestyle="--",
                label="priors",
                color="dodgerblue",
                alpha=0.5,
            )
            axes[idx].fill_between(
                x, y_priors_lower, y_priors_upper, alpha=0.2, color="dodgerblue"
            )

            mask = calibrated_steps["regressor"] == ch
            if np.any(mask):
                x_steps = calibrated_steps.loc[mask, "iter"]
                axes[idx].scatter(
                    x_steps, y_posteriors[x_steps], color="orange", label="calibration"
                )

        fig.suptitle(
            "Coefficient Calibration", y=0.92, fontsize=24, verticalalignment="center"
        )

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.0, 0.05, 1.0, 1.0),
            fancybox=True,
            shadow=True,
            ncols=3,
            fontsize=18,
        )

        if is_visible:
            plt.show()
        else:
            plt.close()

        return axes

    def select_calib_channels(
        self,
        report_df: pd.DataFrame,
        se_tolerance: float = 1.0,
    ) -> List[str]:
        """Given a attribution report dataframe, recommend and return
        the list of channels should be calibrated

        Args:
            report_df (pd.DataFrame): attribution data frame
            se_tolerance (float, optional): Criteria to select channels. Defaults to 0.1.

        Returns:
            List[str]: recommended channels
        """
        # filter zero spend and within tolerance channels
        select_mask = (report_df["spend"] > 1e-5) & (
            np.abs(report_df["mmm_icac"] - report_df["test_icac"])
            > self.se_tolerance * report_df["test_se"]
        )
        reduced_ch_df = (
            report_df.loc[select_mask, :]
            .sort_values(by="target_dist", ascending=False)
            .reset_index()
        )
        return reduced_ch_df["regressor"].tolist()

    @staticmethod
    def append_model_info(model: MMM, report_df: pd.DataFrame) -> pd.DataFrame:
        """Given a fitted model and the target data frame, attach regression
        related information to the end result data frame and return it.

        Args:
            model (MMM): model object
            report_df (pd.DataFrame): target data frame

        Returns:
            pd.DataFrame: _description_
        """
        regression_summary = model.get_regression_summary()[
            ["regressor", "coef_p50", "loc_prior", "scale_prior"]
        ]
        regression_summary = regression_summary.rename(
            columns={
                "loc_prior": "coef_prior",
                "scale_prior": "sigma_prior",
                "coef_p50": "coef_posterior",
            }
        ).set_index("regressor")

        expended_report = pd.concat(
            [report_df, regression_summary], axis=1, join="inner"
        )
        return expended_report

    @staticmethod
    def report_attribution(
        model: MMM,
        ab_tests_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Use current model and ab tests input to generate a report

        Args:
            model (MMM): _description_
            ab_tests_df (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """
        # return new_ab_tests_df
        date_col = model.date_col
        logger = logging.getLogger("karpiu-planning-test")
        logger.setLevel(30)

        report_df = ab_tests_df.copy()
        report_df["mmm_icac"] = -1
        report_df["mmm_attr"] = -1
        report_df["spend"] = -1
        # report_df = report_df.rename(columns={'test_channel': 'regressor'}).set_index('regressor')

        for k in report_df.index:
            test_start = report_df.loc[k, "test_start"]
            test_end = report_df.loc[k, "test_end"]
            attributor = AttributorGamma(
                model, start=test_start, end=test_end, logger=logger
            )
            _, spend_attr, spend_df, _ = attributor.make_attribution()

            mask = (spend_attr[date_col] >= test_start) & (
                spend_attr[date_col] <= test_end
            )
            attr_sum = np.sum(spend_attr.loc[mask, k].values)
            mask = (spend_df[date_col] >= test_start) & (spend_df[date_col] <= test_end)
            cost_sum = np.sum(spend_df.loc[mask, k].values)
            if attr_sum > 0:
                report_df.loc[k, "mmm_icac"] = np.round(cost_sum / attr_sum, 3)
            else:
                report_df.loc[k, "mmm_icac"] = np.nan
            report_df.loc[k, "mmm_attr"] = np.round(attr_sum, 0)
            report_df.loc[k, "spend"] = np.round(cost_sum, 0)

        report_df["target_lift"] = np.round(
            report_df["spend"] / report_df["test_icac"], 0
        ).astype(int)
        report_df["target_dist"] = np.abs(
            report_df["target_lift"] - report_df["mmm_attr"]
        )

        return report_df

    @staticmethod
    def generate_prior(
        model,
        modify_channel: str = None,
        loc_prior: float = None,
        sigma_prior: float = None,
    ) -> pd.DataFrame:
        spend_cols = model.get_spend_cols()
        # if test_channel not in spend_cols:
        #     raise Exception("Input channel is not included in the spend column(s) from the model.")

        new_priors_df = model.get_regression_summary()[
            ["regressor", "loc_prior", "scale_prior"]
        ]
        new_priors_df = new_priors_df.set_index("regressor")
        new_priors_df = new_priors_df.loc[model.get_spend_cols()]
        if modify_channel is not None:
            new_priors_df.loc[modify_channel, "loc_prior"] = loc_prior
            if sigma_prior is not None:
                new_priors_df.loc[modify_channel, "scale_prior"] = sigma_prior
            else:
                new_priors_df.loc[modify_channel, "scale_prior"] = max(
                    loc_prior * 0.01, 1e-5
                )
        new_priors_df = new_priors_df.reset_index().rename(
            columns={
                "regressor": "test_channel",
                "loc_prior": "coef_prior",
                "scale_prior": "sigma_prior",
            }
        )
        return new_priors_df

    # regressor, coef_prior, sigma_prior are the reserved keywords in main model
    # TODO: consider make them consistent later
    def _solve_channel_prior(self, model, regressor, calib_report):
        # test_channel = 'facebook'
        # ab_tests_df is static so it is okay to assume to be global
        test_start = calib_report.loc[regressor, "test_start"]
        test_end = calib_report.loc[regressor, "test_end"]
        target = calib_report.loc[regressor, "target_lift"]

        # always suppress solver logger
        logger = logging.getLogger("karpiu-attr-solver")
        logger.setLevel(30)

        def attr_obj_func(x, target):
            attributor = AttributorGamma(
                model, start=test_start, end=test_end, logger=logger
            )
            attr_res = attributor.make_attribution(new_coef_name=regressor, new_coef=x)
            _, spend_attr, _, _ = attr_res
            mask = (spend_attr[model.date_col] >= test_start) & (
                spend_attr[model.date_col] <= test_end
            )
            res = np.sum(spend_attr.loc[mask, regressor].values)
            loss = np.fabs(res - target)
            return loss

        init_search_pt = max(
            model.get_coef_vector([regressor]),
            1e-5,
        )

        sol = optim.fsolve(
            attr_obj_func,
            x0=init_search_pt,
            args=target,
        )
        coef_prior = sol[0]
        return coef_prior

    def get_calib_report(self) -> pd.DataFrame:
        return pd.concat(
            [df.reset_index() for df in self.calib_reports], ignore_index=True
        )

    def get_calibrated_steps(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "iter": np.arange(1, len(self.calibrated_channels) + 1),
                "regressor": self.calibrated_channels,
            }
        )
