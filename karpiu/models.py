import numpy as np
import pandas as pd
from copy import deepcopy
import logging
from typing import Dict, List, Optional, Any

from orbit.models import DLT
from orbit.utils.features import make_fourier_series_df
from orbit.utils.params_tuning import grid_search_orbit

from .utils import adstock_process, non_zero_quantile, get_logger


class MMM:
    """The core class of building a MMM

    Attributes:
        kpi_col: column of the metrics to be used in final measurement
        response_col: column of the metric to be fitted; in the current prototype, it is the same as kpi_col
        date_col: date string column use to index observations on each time step
        spend_cols: columns indicate the channel investment
        full_event_cols:
        event_cols:
        control_feat_cols:
        fs_cols:
        fs_order:
        regressors: list of strings indicate all the regressors used in the model including fourier series terms
        regression_scheme: regression scheme used for model fitting
        saturation_df:
        adstock_df:
        best_params:
        tuning_df:
        _model:
    """

    def __init__(
        self,
        kpi_col: str,
        date_col: str,
        spend_cols: List[str],
        adstock_df: Optional[pd.DataFrame] = None,
        # scalability_df: Optional[pd.DataFrame] = None,
        control_feat_cols: Optional[List[str]] = None,
        event_cols: Optional[List[str]] = None,
        events_sigma_prior: float = 0.3,
        seasonality: Optional[List[int]] = None,
        fs_orders: Optional[List[int]] = None,
        total_market_sigma_prior: float = None,
        default_spend_sigma_prior: float = 0.1,
        logger: Optional[logging.Logger] = None,
        **kwargs
    ):
        """
        Args:
            kpi_col: column of the metrics to be used in final measurement
            date_col: date string column use to index observations on each time step
            spend_cols: columns indicate the channel investment
            adstock_df: dataframe in a specific format describing the adstock
            control_feat_cols: optional; features for control
            event_cols: optional; events to include for time-series forecast
            fs_orders: orders of fourier terms; used in seasonality
            **kwargs:
        """

        if logger is None:
            self.logger = get_logger("karpiu-mmm")
        else:
            self.logger = logger

        self.logger.info("Initialize model")
        self.kpi_col = kpi_col
        # for backtest purpose
        self.response_col = kpi_col
        self.date_col = date_col

        self.spend_cols = deepcopy(spend_cols)
        self.default_spend_sigma_prior = default_spend_sigma_prior
        self.spend_cols.sort()
        # self.scalability_df = scalability_df

        if event_cols is None:
            self.full_event_cols = list()
        else:
            self.full_event_cols = deepcopy(event_cols)
            self.full_event_cols.sort()

        if control_feat_cols is None:
            self.full_control_feat_cols = list()
        else:
            self.full_control_feat_cols = deepcopy(control_feat_cols)
            self.full_control_feat_cols.sort()

        # initialize as full columns
        self.event_cols = deepcopy(self.full_event_cols)
        self.events_sigma_prior = events_sigma_prior
        self.control_feat_cols = deepcopy(self.full_control_feat_cols)

        # complex seasonality
        self.fs_cols = list()
        self.total_market_sigma_prior = total_market_sigma_prior

        if seasonality is not None:
            self.seasonality = seasonality
        else:
            self.seasonality = list()

        if fs_orders is not None:
            self.fs_orders = fs_orders
        else:
            self.fs_orders = list()

        assert len(self.seasonality) == len(self.fs_orders)

        self.extra_priors = None
        self._model = None

        # for complex seasonality
        self.fs_cols_flatten = list()
        if len(self.fs_orders) > 0:
            for s, fs in zip(self.seasonality, self.fs_orders):
                fs_cols = list()
                for x in range(1, fs + 1):
                    fs_cols.append("s{}_fs_cos{}".format(s, x))
                    fs_cols.append("s{}_fs_sin{}".format(s, x))
                fs_cols.sort()
                self.fs_cols_flatten += fs_cols
                self.fs_cols.append(fs_cols)

        self.raw_df = None

        # for attribution purpose; keep tracking on different types of regressors
        self.regressors = (
            self.spend_cols
            + self.fs_cols_flatten
            + self.event_cols
            + self.control_feat_cols
        )
        self.regressors.sort()
        self.saturation_df = None
        self.adstock_df = adstock_df
        if adstock_df is not None:
            for x in spend_cols:
                if x not in self.adstock_df.index.tolist():
                    raise ("Spend channel {} is not included in adstock.".format(x))

        self.regression_scheme = None
        self.best_params = {}
        self.tuning_df = None

    def filter_features(
        self,
        df: pd.DataFrame,
        num_warmup: int = 400,
        num_sample: int = 100,
        chains: int = 1,
        **kwargs
    ):
        self.logger.info("Screen events by Pr(coef>=0) >= 0.9 or Pr(coef>=0) <= 0.1.")
        transform_df = df.copy()
        transform_df[self.kpi_col] = np.log(transform_df[self.kpi_col])
        transform_df[self.full_control_feat_cols] = np.log1p(
            transform_df[self.full_control_feat_cols]
        )

        if len(self.fs_orders) > 0:
            for s, fs_order in zip(self.seasonality, self.fs_orders):
                transform_df, _ = make_fourier_series_df(
                    transform_df, prefix="s{}_".format(s), period=s, order=fs_order
                )

        # exclude spend cols in the extra features screening
        regressor_cols = self.full_event_cols + self.full_control_feat_cols

        # in case we already select the best params
        kwargs.update(self.best_params)
        if "level_sm_input" not in kwargs:
            kwargs["level_sm_input"] = 0.001
        else:
            self.logger.info(
                "Using best params level smoothing input(={}) instead of default (=0.001)."
            ).format(kwargs["level_sm_input"])

        temp_dlt = DLT(
            # seasonality=7,
            response_col=self.kpi_col,
            regressor_col=regressor_cols,
            regressor_sigma_prior=[self.events_sigma_prior] * len(regressor_cols),
            date_col=self.date_col,
            # uses mcmc for high dimensional features selection
            estimator="stan-mcmc",
            # a safe setting for fast regression; will estimate this in final model
            # level_sm_input=0.001,
            # slope_sm_input=0.01,
            num_warmup=num_warmup,
            num_sample=num_sample,
            chains=chains,
            # use small sigma for global trend as this is a long-term daily model
            global_trend_sigma_prior=0.001,
            # **self.best_params,
            **kwargs,
        )
        temp_dlt.fit(transform_df)
        coef_df = temp_dlt.get_regression_coefs()
        mask = (coef_df["Pr(coef >= 0)"] >= 0.9) | (coef_df["Pr(coef >= 0)"] <= 0.1)
        selected_feats = coef_df.loc[mask, "regressor"].values
        self.event_cols = [x for x in selected_feats if x in self.full_event_cols]
        self.control_feat_cols = [
            x for x in selected_feats if x in self.full_control_feat_cols
        ]

        # re-order regressors after events filtering
        self.regressors = (
            self.spend_cols
            + self.fs_cols_flatten
            + self.event_cols
            + self.control_feat_cols
        )
        self.regressors.sort()

        self.logger.info(
            "Full features: {}".format(
                self.full_event_cols + self.full_control_feat_cols
            )
        )
        self.logger.info(
            "Selected features: {}".format(self.event_cols + self.control_feat_cols)
        )

    def set_features(self, features: List[str]):
        """Instead of features selection, user can use this function to directly set features for fitting"""
        self.event_cols = [x for x in features if x in self.full_event_cols]
        self.control_feat_cols = [
            x for x in features if x in self.full_control_feat_cols
        ]

        self.logger.info(
            "Full features: {}".format(
                self.full_event_cols + self.full_control_feat_cols
            )
        )
        self.logger.info(
            "Selected features: {}".format(self.event_cols + self.control_feat_cols)
        )

    def optim_hyper_params(
        self, df: pd.DataFrame, param_grid: Optional[Dict[str, Any]] = None, **kwargs
    ) -> None:
        self.logger.info(
            "Optimize smoothing params. Only events and seasonality are involved."
        )
        transform_df = df.copy()
        self.logger.info("Pre-process data.")
        transform_df[self.kpi_col] = np.log(transform_df[self.kpi_col])
        if len(self.fs_orders) > 0:
            for s, fs_order in zip(self.seasonality, self.fs_orders):
                transform_df, _ = make_fourier_series_df(
                    transform_df, prefix="s{}_".format(s), period=s, order=fs_order
                )

        # some choices for hyper-parameters gird search
        if param_grid is None:
            # half-life: 7d=0.9057, 28d=0.9755, 90d=0.9923, 180=0.9962
            param_grid = {
                "slope_sm_input": list(1 - np.linspace(0.9057, 0.9755, 3)),
                "level_sm_input": list(1 - np.linspace(0.9755, 0.9962, 5)),
                "damped_factor": list(np.linspace(0.9057, 0.9923, 3)),
            }
        regressors = self.fs_cols_flatten + self.event_cols + self.control_feat_cols
        dlt_proto = DLT(
            # seasonality=7,
            response_col=self.kpi_col,
            regressor_col=regressors,
            # only include events and seasonality in hyper-params screening
            regressor_sign=["="] * len(regressors),
            regressor_sigma_prior=[self.events_sigma_prior] * len(regressors),
            date_col=self.date_col,
            # 4 weeks
            forecast_horizon=28,
            estimator="stan-map",
            verbose=False,
            # use small sigma for global trend as this is a long-term daily model
            global_trend_sigma_prior=0.001,
            **kwargs,
        )

        best_params, tuning_df = grid_search_orbit(
            param_grid,
            model=dlt_proto,
            df=transform_df,
            eval_method="bic",
            # this does not matter
            forecast_len=28,
            verbose=False,
        )

        # you might end up multiple params; just pick the first set
        best_params = best_params[0]
        self.best_params.update(best_params)
        self.tuning_df = deepcopy(tuning_df)

        for k, v in self.best_params.items():
            self.logger.info("Best params {} set as {:.5f}".format(k, v))

    def set_hyper_params(self, params: Dict[str, List]) -> None:
        self.logger.info("Set hyper-parameters.")
        self.best_params.update(params)
        for k, v in self.best_params.items():
            self.logger.info("Best params {} set as {:.5f}".format(k, v))

    def derive_saturation(
        self,
        df: pd.DataFrame,
        q: Optional[float] = 0.1,
        scalability_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """A function used to derivation saturation constant based quantiles with a scalability dataframe

        Args:
            df (pd.DataFrame): spend data from input dataframe
            q (Optional[float], optional): quantile used to set saturation. Defaults to 0.1.
            scalability_df (Optional[pd.DataFrame], optional): extra information of the scalability which should
            contain two columns: ["regressor", "scalability"]. Defaults to None which means all scalability to
            be set as one. The final saturation constant will be the derived quantiles times scalability scalar.

        Raises:
            Exception: _description_
            Exception: _description_

        Returns:
            pd.DataFrame: saturation df
        """
        self.logger.info("Deriving saturation constants...")
        # rebuild a base every time it fits data
        self.saturation_df = (
            df[self.spend_cols].apply(non_zero_quantile, q=q).reset_index()
        ).rename(columns={"index": "regressor", 0: "saturation_base"})

        if scalability_df is not None:
            if set(self.spend_cols) != set(scalability_df["regressor"].tolist()):
                raise Exception(
                    "Regressors between saturation based and scalability df are not perfectly matching."
                )

            # left join to extract scalability condition and validate values
            self.saturation_df = pd.merge(
                self.saturation_df,
                scalability_df,
                on="regressor",
                how="left",
            )
        else:
            self.saturation_df["scalability"] = 1.0

        scalability = self.saturation_df["scalability"].values
        if np.any(scalability < 0):
            raise Exception("All spend scalability needs to be > 0.")

        # multiply the condition
        self.saturation_df["saturation"] = (
            self.saturation_df["saturation_base"] * scalability
        )
        self.saturation_df = self.saturation_df.set_index("regressor")
        self.logger.info("Derived saturation constants.")
        return deepcopy(self.saturation_df)

    def set_saturation(
        self,
        saturation_df,
    ) -> None:
        if not saturation_df.index.names == ["regressor"]:
            raise Exception("Saturation index must be set as: ['regressor'].")
        self.saturation_df = deepcopy(saturation_df)
        self.logger.info("Set saturation.")

    def _preprocess_df(
        self,
        df: pd.DataFrame,
        transform_response: bool = True,
    ) -> pd.DataFrame:
        self.logger.debug("Pre-process data.")

        # adstock transformed data-frame will lose first n_max_adstock observations
        n_max_adstock = self.get_max_adstock()
        adstock_matrix = self.get_adstock_matrix()
        # (n_steps - max_adstock, ...)
        transform_df = df[n_max_adstock:].reset_index(drop=True)
        transform_df[self.spend_cols] = adstock_process(
            regressor_matrix=df[self.spend_cols].values,
            adstock_matrix=adstock_matrix,
        )

        # log-log transformation
        # (n_regressor, )
        sat_array = self.get_saturation_vector(regressors=self.spend_cols)
        # (n_steps,  n_regressors)
        transform_df[self.spend_cols] = np.log1p(
            transform_df[self.spend_cols].values / sat_array
        )

        if transform_response:
            transform_df[self.kpi_col] = np.log(transform_df[self.kpi_col])

        if self.control_feat_cols:
            transform_df[self.control_feat_cols] = np.log1p(
                transform_df[self.control_feat_cols]
            )

        if len(self.fs_orders) > 0:
            for s, fs_order in zip(self.seasonality, self.fs_orders):
                transform_df, _ = make_fourier_series_df(
                    transform_df, prefix="s{}_".format(s), period=s, order=fs_order
                )

        return transform_df

    def fit(
        self,
        df: pd.DataFrame,
        extra_priors: Optional[pd.DataFrame] = None,
        num_warmup: int = 400,
        num_sample: int = 100,
        chains: int = 1,
        stan_mcmc_args: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        self.logger.info("Fit final model.")

        if self.saturation_df is None:
            self.derive_saturation(df)

        raw_df = df.copy()
        self.raw_df = df.copy()
        # self._derive_saturation(raw_df)
        transform_df = self._preprocess_df(raw_df)

        self.logger.info("Build a default regression scheme")
        reg_scheme = pd.DataFrame()
        # regressors should be in this order
        # spend_cols + fs_cols + event_cols + control_feat_cols
        reg_scheme["regressor"] = (
            self.spend_cols
            + self.fs_cols_flatten
            + self.event_cols
            + self.control_feat_cols
        )
        reg_scheme["regressor_sign"] = ["+"] * len(self.spend_cols) + ["="] * len(
            self.fs_cols_flatten + self.event_cols + self.control_feat_cols
        )
        reg_scheme["regressor_coef_prior"] = [0.0] * reg_scheme.shape[0]
        reg_scheme["regressor_sigma_prior"] = [self.default_spend_sigma_prior] * len(
            self.spend_cols
        ) + [self.events_sigma_prior] * len(
            self.fs_cols_flatten + self.event_cols + self.control_feat_cols
        )
        reg_scheme = reg_scheme.set_index("regressor")

        # replace original one with extra_priors if given
        if extra_priors is not None:
            self.extra_priors = deepcopy(extra_priors)

        # if extra_priors already set and doesn't get any from input, use the one attached in the object
        if self.extra_priors is not None:
            for idx, row in self.extra_priors.iterrows():
                test_channel = row["test_channel"]
                self.logger.info("Updating {} prior".format(test_channel))
                reg_scheme.loc[test_channel, "regressor_coef_prior"] = row["coef_prior"]
                reg_scheme.loc[test_channel, "regressor_sigma_prior"] = row[
                    "sigma_prior"
                ]

        self._model = DLT(
            response_col=self.kpi_col,
            regressor_col=reg_scheme.index.tolist(),
            regressor_sign=reg_scheme["regressor_sign"].tolist(),
            regressor_beta_prior=reg_scheme["regressor_coef_prior"].tolist(),
            regressor_sigma_prior=reg_scheme["regressor_sigma_prior"].tolist(),
            date_col=self.date_col,
            estimator="stan-mcmc",
            num_warmup=num_warmup,
            num_sample=num_sample,
            chains=chains,
            # use small sigma for global trend as this is a long-term daily model
            global_trend_sigma_prior=0.001,
            **self.best_params,
            **kwargs,
        )
        self._model.fit(transform_df, point_method="median")

        # run it again to use sigma constraint and weight by original coefficient size
        if self.total_market_sigma_prior is not None:
            reg_coef_dfs = self._model.get_regression_coefs().set_index("regressor")
            self.logger.info(
                "Build a regression scheme with total marketing sigma constraint {:.3f}".format(
                    self.total_market_sigma_prior
                )
            )
            reg_scheme = pd.DataFrame()
            # regressors should be in this order
            # spend_cols + fs_cols + event_cols + control_feat_cols
            reg_scheme["regressor"] = (
                self.spend_cols
                + self.fs_cols_flatten
                + self.event_cols
                + self.control_feat_cols
            )
            reg_scheme["regressor_sign"] = ["+"] * len(self.spend_cols) + ["="] * len(
                self.fs_cols_flatten + self.event_cols + self.control_feat_cols
            )
            reg_scheme["regressor_coef_prior"] = [0.0] * reg_scheme.shape[0]
            reg_scheme = reg_scheme.set_index("regressor")
            n_spend_cols = len(self.spend_cols)
            # make total market sigma weighted by original coef size
            reg_coefs = reg_coef_dfs.loc[self.spend_cols, "coefficient"].values
            spend_sigma_prior = list(
                self.total_market_sigma_prior * reg_coefs / np.sum(reg_coefs)
            )
            reg_scheme["regressor_sigma_prior"] = spend_sigma_prior + [
                self.events_sigma_prior
            ] * len(self.fs_cols_flatten + self.event_cols + self.control_feat_cols)

            if self.extra_priors is not None:
                for idx, row in self.extra_priors.iterrows():
                    test_channel = row["test_channel"]
                    self.logger.info("Updating {} prior".format(test_channel))
                    reg_scheme.loc[test_channel, "regressor_coef_prior"] = row[
                        "coef_prior"
                    ]
                    reg_scheme.loc[test_channel, "regressor_sigma_prior"] = row[
                        "sigma_prior"
                    ]

            self._model = DLT(
                response_col=self.kpi_col,
                regressor_col=reg_scheme.index.tolist(),
                regressor_sign=reg_scheme["regressor_sign"].tolist(),
                regressor_beta_prior=reg_scheme["regressor_coef_prior"].tolist(),
                regressor_sigma_prior=reg_scheme["regressor_sigma_prior"].tolist(),
                date_col=self.date_col,
                estimator="stan-mcmc",
                num_warmup=num_warmup,
                num_sample=num_sample,
                chains=chains,
                # use small sigma for global trend as this is a long-term daily model
                global_trend_sigma_prior=0.001,
                **self.best_params,
                **kwargs,
            )
            # self._model.fit(transform_df)
            self._model.fit(transform_df, point_method="median")

        self.regression_scheme = reg_scheme

        # validation
        regression_summary = self.get_regression_summary()
        spend_coefs = regression_summary.loc[
            regression_summary["regressor"].isin(self.spend_cols), "coef_p50"
        ].values
        spend_coefs_sum = np.sum(spend_coefs)
        if spend_coefs_sum > 1.0:
            self.logger.warning(
                "Spend channels regression coefficients sum ({}) exceeds 1.0. Users need to re-fit model with priors or saturations revised.".format(
                    spend_coefs_sum
                )
            )
        elif spend_coefs_sum > 0.8:
            self.logger.warning(
                "Spend channels regression coefficients sum ({}) exceeds 0.8. Common range is (0, 0.8]. Consider re-fit model with priors or saturations revised.".format(
                    spend_coefs_sum
                )
            )
        else:
            self.logger.info(
                "Spend channels regression coefficients sum ({}) is within common range (0, 0.8].".format(
                    spend_coefs_sum
                )
            )

    def predict(
        self, df: pd.DataFrame, decompose: bool = False, **kwargs
    ) -> pd.DataFrame:
        # TODO: can make transformation a module

        df = df.copy()
        transform_df = self._preprocess_df(df, transform_response=False)
        pred = self._model.predict(transform_df, decompose=decompose, **kwargs)

        # _5 and _95 probably won't exist with median prediction for current version
        pred_tr_col = [
            x
            for x in ["prediction_5", "prediction", "prediction_95"]
            if x in pred.columns
        ]
        pred[pred_tr_col] = pred[pred_tr_col].apply(np.exp)

        pred_base = df[[self.date_col]]
        # preserve the shape of original input; first n(=adstock) will have null values
        pred = pd.merge(pred_base, pred, on=[self.date_col], how="left")
        n_max_adstock = self.get_max_adstock()

        # unlike orbit, decompose the regression and seasonal regression here
        if decompose:
            pred = pred.drop(columns=["seasonality", "regression"])
            # (n_regressors, )
            coef_paid = self.get_coef_vector(self.spend_cols)
            # (n_steps, n_regressors)
            x_paid = transform_df[self.spend_cols].values
            reg_paid = np.concatenate(
                [
                    np.full(n_max_adstock, fill_value=np.nan),
                    np.sum(coef_paid * x_paid, -1),
                ]
            )
            pred["paid"] = reg_paid

            if self.event_cols:
                # (n_regressors, )
                coef_event = self.get_coef_vector(self.event_cols)
                # (n_steps, n_regressors)
                x_event = transform_df[self.event_cols].values
                # workaround with period that was unknown due to adstock
                reg_event = np.concatenate(
                    [
                        np.full(n_max_adstock, fill_value=np.nan),
                        np.sum(coef_event * x_event, -1),
                    ]
                )
                pred["events"] = reg_event

            if len(self.fs_cols) > 0:
                for s, fs_col in zip(self.seasonality, self.fs_cols):
                    # (n_regressors, )
                    coef_fs = self.get_coef_vector(fs_col)
                    # (n_steps, n_regressors)
                    x_fs = transform_df[fs_col].values
                    # workaround with period that was unknown due to adstock
                    reg_fs = np.concatenate(
                        [
                            np.full(n_max_adstock, fill_value=np.nan),
                            np.sum(coef_fs * x_fs, -1),
                        ]
                    )
                    pred["s-{} seasonality".format(s)] = reg_fs

            if len(self.control_feat_cols) > 0:
                # (n_regressors, )
                coef_control = self.get_coef_vector(self.control_feat_cols)
                # (n_steps, n_regressors)
                x_control = np.log1p(transform_df[self.control_feat_cols].values)
                # workaround with period that was unknown due to adstock
                reg_control = np.concatenate(
                    [
                        np.full(n_max_adstock, fill_value=np.nan),
                        np.sum(coef_control * x_control, -1),
                    ]
                )
                pred["control_features"] = reg_control

        return pred

    def get_regressors(self, exclude_fs_cols: bool = True) -> List[str]:
        """Return all regressors used in the model including fourier-series terms (optional), spend etc.

        Args:
            exclude_fs_cols : whether to return list including fourier-series columns

        Returns:
            list of regressors of the model

        """
        if exclude_fs_cols:
            res = deepcopy(self.spend_cols + self.event_cols + self.control_feat_cols)
        else:
            res = deepcopy(self.regressors)
        return res

    def get_spend_cols(self):
        return deepcopy(self.spend_cols)

    def get_adstock_df(self):
        return deepcopy(self.adstock_df)

    # add arg with spend_col
    def get_adstock_matrix(
        self,
        spend_cols: Optional[List[str]] = None,
    ) -> np.array:
        if spend_cols is None:
            spend_cols = self.get_spend_cols()

        if self.adstock_df is not None:
            adstock_df = self.get_adstock_df()
            adstock_matrix = adstock_df.loc[spend_cols, :].values
        else:
            adstock_matrix = np.ones((len(spend_cols), 1))
        return adstock_matrix

    def get_max_adstock(self):
        """Returns zero for now until we implement adstock"""
        adstock_matrix = self.get_adstock_matrix()
        return adstock_matrix.shape[1] - 1

    def get_regression_summary(self) -> pd.DataFrame:
        # by default, orbit uses lower=0.05, upper=0.95
        regression_coef_dfs = self._model.get_regression_coefs().rename(
            columns={
                "regressor_sign": "sign",
                "coefficient": "coef_p50",
                "coefficient_lower": "coef_p05",
                "coefficient_upper": "coef_p95",
            }
        )
        regression_scheme = (
            self.regression_scheme.copy()
            .rename(
                columns={
                    "regressor_coef_prior": "loc_prior",
                    "regressor_sigma_prior": "scale_prior",
                }
            )
            .drop(columns=["regressor_sign"])
        )
        out = pd.merge(
            left=regression_coef_dfs, right=regression_scheme, on=["regressor"]
        )
        return out

    def get_coef_vector(
        self,
        regressors: Optional[List[str]] = None,
    ) -> np.array:
        coef_df = self._model.get_regression_coefs()
        coef_df = coef_df.set_index("regressor")
        if regressors is not None:
            coef_array = coef_df.loc[regressors, "coefficient"].values
        else:
            coef_array = coef_df.loc[:, "coefficient"].values
        return coef_array

    def get_coef_matrix(
        self,
        date_array: np.array,
        regressors: Optional[List[str]] = None,
    ) -> np.array:
        """Right now we ignore date_array since this is static coef. model
        Args:
            date_array: user supplied date array for the regressors; right now this is dummy and just used for
            regressors: list of strings of regressors labels to return the coefficient matrix
            determining the output array length

        Returns:
            coefficient matrix
        """

        coef_df = self._model.get_regression_coefs()
        coef_df = coef_df.set_index("regressor")
        if regressors is not None:
            coef_matrix = coef_df.loc[regressors, "coefficient"].values
        else:
            coef_matrix = coef_df.loc[:, "coefficient"].values
        coef_matrix = np.tile(coef_matrix, (len(date_array), 1))
        return coef_matrix

    def get_saturation(self):
        # in the same order of spend
        return deepcopy(self.saturation_df)

    def get_saturation_vector(
        self,
        regressors: Optional[List[str]] = None,
    ) -> np.array:
        sat_df = self.get_saturation()
        if regressors is not None:
            sat_array = sat_df.loc[regressors, "saturation"].values
        else:
            sat_array = sat_df.loc[:, "saturation"].values
        return sat_array

    def get_extra_priors(self):
        return deepcopy(self.extra_priors)

    def get_event_cols(self):
        return deepcopy(self.event_cols)

    def get_control_feat_cols(self):
        return deepcopy(self.control_feat_cols)

    def get_raw_df(self):
        return deepcopy(self.raw_df)

    def get_logger(self):
        # not a deepcopy
        return self.logger
