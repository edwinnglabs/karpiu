import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats

from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import durbin_watson

""" Diagnostic tools for MMM model object
"""


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