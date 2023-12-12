import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats
import arviz as az
import logging
from copy import deepcopy

from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import durbin_watson

from tqdm.auto import tqdm

from typing import Tuple

from .models import MMM
from karpiu.explainability import AttributorGamma
from karpiu.model_shell import MMMShell


""" Diagnostic tools for MMM model object
"""


def check_residuals(model: MMM):
    max_adstock = model.get_max_adstock()
    df = model.raw_df.copy()
    pred = model.predict(df, decompose=False)
    # degree of freedom param
    resid_dof = model._model.get_point_posteriors()["median"]["nu"]

    # skip the non-settlement period due to adstock
    resid = np.log(df["signups"].values[max_adstock:]) - np.log(
        pred["prediction"].values[max_adstock:]
    )

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    axes[0, 0].plot(df["dt"].values[max_adstock:], resid, "o", markersize=2)
    axes[0, 0].set_title("residuals vs. time")
    axes[0, 1].hist(resid, bins=25, edgecolor="black")
    axes[0, 1].set_title("residuals hist")
    axes[1, 0].plot(
        np.log(pred["prediction"].values[max_adstock:]), resid, "o", markersize=2
    )
    axes[1, 0].set_title("residuals vs. fitted")
    # t-dist qq-plot
    _ = stats.probplot(resid, dist=stats.t, sparams=resid_dof, plot=axes[1, 1])
    # auto-correlations
    sm.graphics.tsa.plot_acf(resid, lags=30, ax=axes[2, 0])
    sm.graphics.tsa.plot_pacf(resid, lags=30, ax=axes[2, 1], method="ywm")

    fig.tight_layout()


def check_stationarity(model: MMM):
    # 1. Run [Augmented Dicker-Fuller test](https://en.wikipedia.org/wiki/Augmented_Dickey%E2%80%93Fuller_test),
    # it needs to reject the null which means unit root is not present.
    # 2. Check [Durbin-Watson Stat](https://en.wikipedia.org/wiki/Durbin%E2%80%93Watson_statistic),
    # the closer to `2`, the better.

    max_adstock = model.get_max_adstock()
    df = model.raw_df.copy()
    pred = model.predict(df, decompose=False)
    # skip the non-settlement period due to adstock
    resid = np.log(df["signups"].values[max_adstock:]) - np.log(
        pred["prediction"].values[max_adstock:]
    )

    adfuller_pval = adfuller(resid)[1]
    print(
        "Adfuller Test P-Val: {:.3f} Recommended Values:(x <= 0.05)".format(
            adfuller_pval
        )
    )

    dw_stat = durbin_watson(resid)
    print("Durbin-Watson Stat: {:.3f} Recommended Values:(|x - 2|>=1.0".format(dw_stat))


def check_convergence(model: MMM):
    posetriors = model._model.get_posterior_samples(relabel=True, permute=False)
    spend_cols = model.get_spend_cols()

    az.style.use("arviz-darkgrid")
    az.plot_trace(
        posetriors,
        var_names=spend_cols,
        chain_prop={"color": ["r", "b", "g", "y"]},
        # figsize=(len(spend_cols), 30),
    )


def two_steps_optim_check(
    model: MMM,
    budget_start: str,
    n_iters: int = 10,
    adstock_off: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = deepcopy(model)
    channels = model.get_spend_cols()
    n_channels = len(channels)
    date_col = model.date_col
    budget_start = pd.to_datetime(budget_start)
    budget_end = budget_start + pd.DateOffset(days=1)

    raw_df = model.get_raw_df()
    init_weight = np.mean(
        raw_df.loc[
            (raw_df[date_col] >= budget_start) & (raw_df[date_col] <= budget_end),
            channels,
        ].values,
        axis=0,
    )

    # arbitrary
    base_weight = np.ones((1, n_channels)) * init_weight
    # arbitrary
    ltv = np.random.random_integers(low=20, high=50, size=n_channels)

    total_response = np.empty(n_iters)
    revs = np.empty(n_iters)
    # for 2-steps
    budget_ratios = np.linspace(0, 1, n_iters)

    # suppress adstock for testing
    init_max_adstock = model.get_max_adstock()
    if adstock_off and model.get_max_adstock() > 0:
        model.adstock_df = None

    model.raw_df = model.raw_df.loc[init_max_adstock:, :].reset_index(drop=True)
    df = model.get_raw_df()

    # turn-off info
    logger = logging.getLogger("karpiu-planning-test")
    logger.setLevel(30)

    for idx, x in enumerate(tqdm(budget_ratios)):
        budget_vector = np.array([[x], [1 - x]])
        budget_matrix = budget_vector * base_weight
        # print(budget_matrix)
        spend_df = df.copy()
        spend_df.loc[
            (spend_df[date_col] >= budget_start) & (spend_df[date_col] <= budget_end),
            channels,
        ] = budget_matrix

        attributor = AttributorGamma(
            model=model,
            df=spend_df,
            start=budget_start,
            end=budget_end,
            logger=logger,
        )
        _, spend_attr, _, _ = attributor.make_attribution()
        revs[idx] = np.sum(spend_attr.loc[:, channels].values * ltv)

        pred_df = model.predict(spend_df, decompose=True)
        msh = MMMShell(model)
        # note that this is un-normalized comp; not equal to final marketing attribution
        paid_arr = pred_df.loc[
            (pred_df[date_col] >= budget_start) & (pred_df[date_col] <= budget_end),
            "paid",
        ].values
        organic_attr_arr = msh.attr_organic[
            (pred_df[date_col] >= budget_start) & (pred_df[date_col] <= budget_end)
        ]
        total_response[idx] = np.sum(organic_attr_arr * np.exp(paid_arr))

    return budget_ratios, revs, total_response
