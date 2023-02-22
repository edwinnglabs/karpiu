import pytest
import numpy as np
import pandas as pd

from karpiu.models import MMM
from karpiu.simulation import make_mmm_daily_data
from karpiu.planning import TargetMaximizer, calculate_marginal_cost
from karpiu.explainability import Attributor


def test_target_maximizer():
    # data_args
    seed = 2022
    n_steps = 365 * 3
    channels_coef = [0.03, 0.05, 0.028, 0.01, 0.03]
    channels = ["tv", "radio", "social", "promo", "search"]
    features_loc = np.array([10000, 5000, 3000, 2000, 850])
    features_scale = np.array([5000, 3000, 1000, 550, 500])
    scalability = np.array([1.1, 0.75, 1.3, 1.5, 0.9])
    start_date = "2019-01-01"
    adstock_args = {
        "n_steps": 28,
        "peak_step": np.array([10, 8, 5, 3, 2]),
        "left_growth": np.array([0.05, 0.08, 0.1, 0.5, 0.75]),
        "right_growth": np.array([-0.03, -0.6, -0.5, -0.1, -0.25]),
    }
    best_params = {
        "damped_factor": 0.9057,
        "level_sm_input": 0.0245,
        "slope_sm_input": 0.0943,
    }
    np.random.seed(seed)

    df, scalability_df, adstock_df, event_cols = make_mmm_daily_data(
        channels_coef=channels_coef,
        channels=channels,
        features_loc=features_loc,
        features_scale=features_scale,
        scalability=scalability,
        n_steps=n_steps,
        start_date=start_date,
        adstock_args=adstock_args,
        with_yearly_seasonality=True,
        with_weekly_seasonality=True,
        country="US",
    )

    mmm = MMM(
        kpi_col="sales",
        date_col="date",
        spend_cols=channels,
        event_cols=event_cols,
        seed=seed,
        adstock_df=adstock_df,
        seasonality=[7, 365.25],
        fs_orders=[2, 3],
    )
    mmm.derive_saturation(df=df, scalability_df=scalability_df)
    mmm.set_hyper_params(params=best_params)
    mmm.fit(df, num_warmup=1000, num_sample=1000, chains=4)
    budget_start = "2021-01-01"
    budget_end = "2021-01-31"
    optim_channels = mmm.get_spend_cols()
    # to be safe in beta version, use sorted list of channels
    optim_channels.sort()

    spend_scaler = 1e3

    # steal this code to get all the matrices in the background
    # and also report average cost of acq
    attr_obj = Attributor(
        mmm, attr_regressors=optim_channels, start=budget_start, end=budget_end
    )
    _, spend_attr_df, spend_df, _ = attr_obj.make_attribution()
    tot_attr_df = spend_attr_df[optim_channels].apply(np.sum, axis=0)
    tot_spend_df = spend_df[optim_channels].apply(np.sum, axis=0)
    avg_cost_df = tot_spend_df / tot_attr_df
    avg_cost_df = pd.DataFrame(avg_cost_df)
    avg_cost_df.index = avg_cost_df.index.rename("regressor")
    avg_cost_df = avg_cost_df.rename(columns={0: "avg_cost"})

    # more data massage
    # 1e4 is a fixed constant to reduce over-float of spend
    # not that they need to be consistent with the subsequent calculation of
    # post optimization metrics
    tot_spend_df = tot_spend_df / spend_scaler
    tot_spend_df = pd.DataFrame(tot_spend_df)
    tot_spend_df.index = tot_spend_df.index.rename("regressor")
    tot_spend_df = tot_spend_df.rename(columns={0: "pre_opt_spend"})

    mc_df = calculate_marginal_cost(
        mmm,
        channels=optim_channels,
        spend_start=budget_start,
        spend_end=budget_end,
    )

    pre_opt_report = pd.concat(
        [avg_cost_df, mc_df, tot_spend_df], axis=1, keys="regressor"
    )
    pre_opt_report.columns = [
        "pre-opt-avg-cost",
        "pre-opt-marginal-cost",
        "pre-opt-spend",
    ]

    maximizer = TargetMaximizer(
        model=mmm,
        budget_start=budget_start,
        budget_end=budget_end,
        optim_channel=optim_channels,
    )
    optim_spend_df = maximizer.optimize(maxiter=1000, eps=1e-3)

    optim_spend_matrix = maximizer.get_current_state()
    init_spend_matrix = maximizer.get_init_state()
    # always spend all budget in target maximization; assure the total preserves
    assert np.allclose(np.sum(optim_spend_matrix), np.sum(init_spend_matrix))

    post_mc_df = calculate_marginal_cost(
        mmm,
        channels=mmm.get_spend_cols(),
        spend_start=budget_start,
        spend_end=budget_end,
        spend_df=optim_spend_df,
    )

    attr_obj = Attributor(
        mmm,
        df=optim_spend_df,
        attr_regressors=optim_channels,
        start=budget_start,
        end=budget_end,
    )
    _, spend_attr_df, spend_df, _ = attr_obj.make_attribution()
    optim_tot_attr_df = spend_attr_df[optim_channels].apply(np.sum, axis=0)
    optim_tot_spend_df = spend_df[optim_channels].apply(np.sum, axis=0)
    post_avg_cost_df = optim_tot_spend_df / optim_tot_attr_df
    post_avg_cost_df = pd.DataFrame(post_avg_cost_df)
    post_avg_cost_df = post_avg_cost_df.rename(columns={0: "post-opt-avg-cost"})

    # more data massage
    optim_tot_spend_df = optim_tot_spend_df / spend_scaler
    optim_tot_spend_df = pd.DataFrame(optim_tot_spend_df)
    optim_tot_spend_df.index = optim_tot_spend_df.index.rename("regressor")
    optim_tot_spend_df = optim_tot_spend_df.rename(columns={0: "post-opt-spend"})

    post_opt_report = pd.concat(
        [post_avg_cost_df, post_mc_df, optim_tot_spend_df], axis=1
    )
    post_opt_report.columns = [
        "post-opt-avg-cost",
        "post-opt-marginal-cost",
        "post-opt-spend",
    ]

    opt_report = pd.concat([pre_opt_report, post_opt_report], axis=1)
    opt_report = opt_report[
        [
            "pre-opt-avg-cost",
            "post-opt-avg-cost",
            "pre-opt-marginal-cost",
            "post-opt-marginal-cost",
            "pre-opt-spend",
            "post-opt-spend",
        ]
    ]

    # check 1: all marginal cost should be close; within 10% of median
    post_mc = opt_report["post-opt-marginal-cost"].values
    abs_delta_perc = np.abs(post_mc / np.nanmean(post_mc) - 1.00)
    assert np.all(abs_delta_perc < 0.1)

    cv = np.nanstd(post_mc) / np.nanmean(post_mc)
    assert np.all(cv < 0.1)
    print(opt_report.head(5))

    # check2: total predicted response must be higher than current
    optim_pred = mmm.predict(optim_spend_df)
    init_pred = mmm.predict(df)
    measurement_mask = (df["date"] >= maximizer.calc_start) & (
        df["date"] <= maximizer.calc_end
    )
    total_optim_pred = np.sum(optim_pred.loc[measurement_mask, "prediction"].values)
    total_init_pred = np.sum(init_pred.loc[measurement_mask, "prediction"].values)
    assert total_optim_pred - total_init_pred > 0
