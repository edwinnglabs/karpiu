import numpy as np
import pandas as pd
from copy import deepcopy
import logging

import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats

from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import durbin_watson

import arviz as az
from orbit.models import DLT
from orbit.utils.features import make_fourier_series_df
from orbit.utils.params_tuning import grid_search_orbit

from .utils import adstock_process, non_zero_quantile

logger = logging.getLogger("karpiu-mmm")


class OneStageModel(object):
    # TODO: add doc string here
    """
    """

    def __init__(self, kpi_col, date_col, spend_cols, adstock_df,
                 control_feat_cols=[], event_cols=[], fs_order=0, **kwargs):

        logger.info("Initialize model")
        self.kpi_col = kpi_col
        self.date_col = date_col
        self.full_event_cols = event_cols
        self.event_cols = deepcopy(event_cols)
        self.full_control_feat_cols = control_feat_cols
        self.control_feat_cols = deepcopy(control_feat_cols)
        self.fs_cols = list()
        self.fs_order = fs_order

        # for dual seasonality
        if self.fs_order > 0:
            for x in range(1, self.fs_order + 1):
                self.fs_cols.append('fs_cos{}'.format(x))
                self.fs_cols.append('fs_sin{}'.format(x))
            self.fs_cols.sort()

        self.spend_cols = spend_cols
        self.spend_cols.sort()

        # for backtest purpose
        self.response_col = kpi_col
        self.raw_df = None

        # for attribution purpose
        self.regressors = self.spend_cols + self.fs_cols + self.event_cols + self.control_feat_cols
        self.regressors.sort()

        self.saturation_df = None
        self.adstock_df = adstock_df
        if adstock_df is not None:
            for x in spend_cols:
                if x not in self.adstock_df.index.tolist():
                    raise ("Spend channel {} is not included in adstock.".format(x))

        self.regression_scheme = None

        self.model = None
        self.best_params = {}
        self.tuning_df = None

    def filter_features(self, df, **kwargs):
        logger.info("Screen events by Pr(coef>=0) >= 0.9 or Pr(coef>=0) <= 0.1.")
        transform_df = df.copy()
        transform_df[self.kpi_col] = np.log(transform_df[self.kpi_col])
        transform_df[self.full_control_feat_cols] = np.log(transform_df[self.full_control_feat_cols])

        if self.fs_order > 0:
            transform_df, _ = make_fourier_series_df(
                transform_df, period=365.25, order=self.fs_order
            )

        regressor_cols = self.fs_cols + self.full_event_cols + self.full_control_feat_cols

        temp_dlt = DLT(
            seasonality=7,
            response_col=self.kpi_col,
            regressor_col=regressor_cols,
            regressor_sigma_prior=[10.0] * len(regressor_cols),
            date_col=self.date_col,
            estimator='stan-mcmc', #estimator='stan-map',
            # just a dummy to do early fitting of event
            level_sm_input=0.01,
            slope_sm_input=0.01,
            num_warmup=4000,
            num_sample=1000,
            **self.best_params,
            **kwargs,
        )
        temp_dlt.fit(transform_df)
        coef_df = temp_dlt.get_regression_coefs()
        mask = (coef_df['Pr(coef >= 0)'] >= 0.9) | (coef_df['Pr(coef >= 0)'] <= 0.1)
        selected_feats = coef_df.loc[mask, 'regressor'].values
        self.event_cols = [x for x in selected_feats if x in self.full_event_cols]
        self.control_feat_cols = [x for x in selected_feats if x in self.full_control_feat_cols]

        # re-order regressors
        self.regressors = self.spend_cols + self.fs_cols + self.event_cols + self.control_feat_cols
        self.regressors.sort()

        logger.info("Full features: {}".format(self.full_event_cols + self.full_control_feat_cols))
        logger.info("Selected features: {}".format(self.event_cols + self.control_feat_cols))

    def optim_hyper_params(self, df, **kwargs):
        logger.info("Optimize smoothing params. Only events and seasonality are involved.")
        transform_df = df.copy()
        logger.info("Pre-process data.")
        transform_df[self.kpi_col] = np.log(transform_df[self.kpi_col])
        if self.fs_order > 0:
            transform_df, _ = make_fourier_series_df(
                transform_df, period=365.25, order=self.fs_order
            )

        # TODO: 
        # revisit this later, optimizing level smoothing yields better long-term results
        # leaving level smoothing to be trained later yields better result in avoiding auto-correlation
        # after all, it seems auto-correlation is a more severe problem. 
        # so let's optimize slope and seasonality smoothing only and train level smoothing in final fitting process
        param_grid = {
            # "level_sm_input": list(np.exp(np.linspace(-6.5, 0, 20))),
            # 5 steps per each param should be reasonably enough
            # in general we don't want the smoothing param too big
            # that makes the state memoryless and loss forestability
            # "level_sm_input": list(np.exp(np.linspace(-6.5, -0.1, 5))),
            "slope_sm_input": list(np.exp(np.linspace(-6.5, -1, 5))),
            "seasonality_sm_input": list(np.exp(np.linspace(-6.5, -1, 5))),
            # short and long term memory
            "damped_factor": [0.7, 0.8, 0.95],
        }

        regressors = self.fs_cols + self.event_cols + self.control_feat_cols
        dlt_proto = DLT(
            seasonality=7,
            response_col=self.kpi_col,
            regressor_col=regressors,
            # only include events and seasonality in hyper-params screening
            regressor_sign=["="] * len(regressors),
            regressor_sigma_prior=[10.0] * len(regressors),
            date_col=self.date_col,
            # 4 weeks
            forecast_horizon=28,
            estimator='stan-map',
            **kwargs,
        )

        best_params, tuning_df = grid_search_orbit(
            param_grid,
            model=dlt_proto,
            df=transform_df,
            eval_method="bic",
            # this does not matter
            forecast_len=28,
        )

        # you might end up multiple params; just pick the first set
        best_params = best_params[0]
        self.best_params.update(best_params)
        self.tuning_df = tuning_df

        for k, v in self.best_params.items():
            logger.info("Best params {} set as {}".format(k, v))

    def set_hyper_params(self, params):
        logger.info("Set hyper-parameters.")
        self.best_params.update(params)
        for k, v in self.best_params.items():
            logger.info("Best params {} set as {}".format(k, v))

    def derive_saturation(self, df):
        self.saturation_df = (
            df[self.spend_cols].apply(non_zero_quantile).reset_index()
        ).rename(columns={'index': 'regressor', 0: 'saturation'})
        self.saturation_df = self.saturation_df.set_index('regressor')

    def fit(self, df, extra_priors=None, **kwargs):
        logger.info("Fit final model.")
        self.raw_df = df.copy()
        transform_df = df.copy()
        logger.info("Pre-process data.")
        transform_df[self.kpi_col] = np.log(transform_df[self.kpi_col])
        transform_df[self.control_feat_cols] = np.log(transform_df[self.control_feat_cols])

        self.derive_saturation(transform_df)
        sat_array = self.saturation_df['saturation'].values

        # transformed data-frame would lose first n(=adstock size) observations due to adstock process
        adstock_matrix = self.get_adstock_matrix()
        max_adstock = self.get_max_adstock()

        new_transform_df = transform_df[max_adstock:].reset_index(drop=True)
        new_transform_df[self.spend_cols] = adstock_process(
            regressor_matrix=transform_df[self.spend_cols].values,
            adstock_matrix=adstock_matrix,
        )
        # remove the old df
        transform_df = new_transform_df.copy()

        # dim: time x num of regressors
        transform_df[self.spend_cols] = transform_df[self.spend_cols].values / sat_array.reshape(1, -1)
        transform_df[self.spend_cols] = np.log1p(transform_df[self.spend_cols])

        if self.fs_order > 0:
            transform_df, _ = make_fourier_series_df(
                transform_df, period=365.25, order=self.fs_order
            )

        logger.info("Build regression scheme")
        reg_scheme = pd.DataFrame()
        # in this order
        # self.spend_cols + self.fs_cols + self.event_cols + self.control_feat_cols 
        reg_scheme['regressor'] = self.spend_cols + self.fs_cols + self.event_cols + self.control_feat_cols
        reg_scheme['regressor_sign'] = \
            ["+"] * len(self.spend_cols) + ["="] * len(self.fs_cols + self.event_cols + self.control_feat_cols)
        reg_scheme['regressor_coef_prior'] = [0.0] * reg_scheme.shape[0]
        reg_scheme['regressor_sigma_prior'] = \
            [0.1] * len(self.spend_cols) + [10.0] * len(self.fs_cols + self.event_cols + self.control_feat_cols)
        reg_scheme = reg_scheme.set_index('regressor')

        if extra_priors is not None:
            for idx, row in extra_priors.iterrows():
                test_channel = row['test_channel']
                logger.info("Updating {} prior".format(test_channel))
                reg_scheme.loc[test_channel, 'regressor_coef_prior'] = row['coef_prior']
                reg_scheme.loc[test_channel, 'regressor_sigma_prior'] = row['sigma_prior']

        self.regression_scheme = reg_scheme

        self._model = DLT(
            seasonality=7,
            response_col=self.kpi_col,
            regressor_col=reg_scheme.index.tolist(),
            regressor_sign=reg_scheme['regressor_sign'].tolist(),
            regressor_beta_prior=reg_scheme['regressor_coef_prior'].tolist(),
            regressor_sigma_prior=reg_scheme['regressor_sigma_prior'].tolist(),
            date_col=self.date_col,
            estimator='stan-mcmc',
            num_warmup=8000,
            num_sample=4000,
            **self.best_params,
            **kwargs,
        )
        # self._model.fit(transform_df)
        self._model.fit(transform_df, point_method='median')

    def predict(self, df, **kwargs):
        # TODO: can make transformation a module
        transform_df = df.copy()
        sat_array = self.saturation_df['saturation'].values

        if self.fs_order > 0:
            transform_df, _ = make_fourier_series_df(
                transform_df, period=365.25, order=self.fs_order
            )

        # transformed data-frame would lose first n(=adstock size) observations due to adstock process
        adstock_matrix = self.get_adstock_matrix()
        max_adstock = self.get_max_adstock()

        new_transform_df = transform_df[max_adstock:].reset_index(drop=True)
        new_transform_df[self.spend_cols] = adstock_process(
            regressor_matrix=transform_df[self.spend_cols].values,
            adstock_matrix=adstock_matrix,
        )
        # remove the old df
        transform_df = new_transform_df

        # dim: time x num of regressors
        transform_df[self.spend_cols] = transform_df[self.spend_cols].values / sat_array.reshape(1, -1)
        transform_df[self.spend_cols] = np.log1p(transform_df[self.spend_cols])
        transform_df[self.control_feat_cols] = np.log(transform_df[self.control_feat_cols])

        pred = self._model.predict(transform_df, **kwargs)
        pred_tr_col = [x for x in ['prediction_5', 'prediction', 'prediction_95'] if x in pred.columns]
        # TODO: transform trend as well if we do decompose
        pred[pred_tr_col] = pred[pred_tr_col].apply(np.exp)
        # pred[pred_tr_col] = pred[pred_tr_col].apply(np.expm1).apply(np.clip, a_min=0, a_max=np.inf)

        return pred

    def get_regressors(self, exclude_fs_cols=True):
        """Return all regressors used in the model including fourier-series terms (optional), spend etc.

        Parameters
        ----------
        exclude_fs_cols : bool
            whether to return list including fourier-series columns 

        Returns
        -------
        list

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

    def get_adstock_matrix(self):
        if self.adstock_df is not None:
            adstock_df = self.get_adstock_df()
            adstock_matrix = adstock_df.values
        else:
            adstock_matrix = np.ones((len(self.spend_cols), 1))
        return adstock_matrix

    def get_max_adstock(self):
        """Returns zero for now until we implement adstock"""
        adstock_matrix = self.get_adstock_matrix()
        return adstock_matrix.shape[1] - 1

    def get_coef_matrix(self, regressors, date_array=None):
        """Right now we ignore date_array since this is static coef. model"""
        coef_df = self._model.get_regression_coefs()
        coef_df = coef_df.set_index('regressor')
        coef_array = coef_df.loc[regressors, 'coefficient'].values
        # coef_array = coef_array.reshape(1, -1)
        coef_array = np.tile(coef_array, (len(date_array), 1))
        return coef_array

    def get_saturation(self):
        # in the same order of spend
        return deepcopy(self.saturation_df)


def check_convergence(model):
    posetriors = model._model.get_posterior_samples(relabel=True, permute=False)
    spend_cols = model.get_spend_cols()

    az.style.use('arviz-darkgrid')
    az.plot_trace(
        posetriors,
        var_names=spend_cols,
        chain_prop={"color": ['r', 'b', 'g', 'y']},
        # figsize=(len(spend_cols), 30),
    )


# diagnostic tools
def check_residuals(model):
    # TODO:
    # assert model instance is the one stage model

    max_adstock = model.get_max_adstock()
    df = model.raw_df.copy()
    pred = model.predict(df, decompose=False)
    # degree of freedom param
    resid_dof = model._model.get_point_posteriors()['median']['nu']

    # skip the non-settlement period due to adstock
    residuals = np.log(df['signups'].values[max_adstock:]) - np.log(pred['prediction'].values)

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    axes[0, 0].plot(df['dt'].values[max_adstock:], residuals, 'o', markersize=2)
    axes[0, 0].set_title('residuals vs. time')
    axes[0, 1].hist(residuals, bins=25, edgecolor='black')
    axes[0, 1].set_title('residuals hist')
    axes[1, 0].plot(np.log(pred['prediction'].values), residuals, 'o', markersize=2)
    axes[1, 0].set_title('residuals vs. fitted')
    # t-dist qq-plot
    _ = stats.probplot(residuals, dist=stats.t, sparams=resid_dof, plot=axes[1, 1])
    # auto-correlations
    sm.graphics.tsa.plot_acf(residuals, lags=30, ax=axes[2, 0])
    sm.graphics.tsa.plot_pacf(residuals, lags=30, ax=axes[2, 1], method='ywm')

    fig.tight_layout();


def check_stationarity(model):
    # 1. Run [Augmented Dicker-Fuller test](https://en.wikipedia.org/wiki/Augmented_Dickey%E2%80%93Fuller_test), 
    # it needs to reject the null which means unit root is not present.
    # 2. Check [Durbin-Watson Stat](https://en.wikipedia.org/wiki/Durbin%E2%80%93Watson_statistic), 
    # the closer to `2`, the better.

    # TODO:
    # assert model instance is the one stage model
    max_adstock = model.get_max_adstock()
    df = model.raw_df.copy()
    pred = model.predict(df, decompose=False)
    # skip the non-settlement period due to adstock
    residuals = np.log(df['signups'].values[max_adstock:]) - np.log(pred['prediction'].values)

    adfuller_pval = adfuller(residuals)[1]
    print("Adfuller Test P-Val: {:.3f} Recommended Values:(x <= 0.05)".format(adfuller_pval))

    dw_stat = durbin_watson(residuals)
    print("Durbin-Watson Stat: {:.3f} Recommended Values:(|x - 2|>=1.0".format(dw_stat))
