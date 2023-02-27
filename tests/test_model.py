import pytest
import numpy as np

from karpiu.models import MMM
from karpiu.simulation import make_mmm_daily_data
from orbit.diagnostics.metrics import smape


# test basic fit predict and all get functions
@pytest.mark.parametrize(
    "with_events",
    [True, False],
    ids=["w_events", "wo_events"],
)
@pytest.mark.parametrize(
    "adstock_args",
    [
        {
            "n_steps": 28,
            "peak_step": np.array([10, 8, 5, 3, 2]),
            "left_growth": np.array([0.05, 0.08, 0.1, 0.5, 0.75]),
            "right_growth": np.array([-0.03, -0.6, -0.5, -0.1, -0.25]),
        },
        None,
    ],
    ids=["w_adstock", "wo_adstock"],
)
def test_mmm_basic(with_events, adstock_args):
    # data_args
    seed = 2022
    n_steps = 365 * 3
    channels_coef = [0.053, 0.15, 0.19, 0.175, 0.15]
    channels = ["promo", "radio", "search", "social", "tv"]
    features_loc = np.array([2000, 5000, 3850, 3000, 7500])
    features_scale = np.array([550, 2500, 500, 1000, 3500])
    scalability = np.array([3.0, 1.25, 0.8, 1.3, 1.5])
    start_date = "2019-01-01"

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
        country="US" if with_events else None,
    )
    mmm = MMM(
        kpi_col="sales",
        date_col="date",
        spend_cols=channels,
        event_cols=event_cols,
        seed=seed,
        adstock_df=adstock_df,
    )
    mmm.derive_saturation(df=df, scalability_df=scalability_df)
    mmm.fit(df, num_warmup=400, num_sample=100, chains=4)
    saturation_df = mmm.get_saturation()
    assert saturation_df.index.name == "regressor"
    assert all(saturation_df.index == mmm.get_spend_cols())

    mmm.get_logger()
    mmm.get_raw_df()
    mmm.get_control_feat_cols()
    mmm_event_cols = mmm.get_event_cols()
    mmm.get_extra_priors()
    mmm.get_saturation_vector()
    mmm.get_saturation()
    mmm.get_coef_matrix(date_array=df["date"].values)
    mmm.get_coef_vector()
    mmm.get_regression_summary()
    n_max_adstock = mmm.get_max_adstock()
    if adstock_args is not None:
        assert n_max_adstock == adstock_args["n_steps"] - 1
    else:
        assert n_max_adstock == 0
    mmm.get_adstock_matrix()
    mmm.get_adstock_df()
    mmm.get_spend_cols()
    mmm.get_regressors()

    assert set(mmm_event_cols) == set(event_cols)

    pred_df = mmm.predict(df)
    # make sure it returns the same shape
    assert pred_df.shape[0] == n_steps
    prediction = pred_df["prediction"].values
    actaul = df["sales"]
    metrics = smape(actual=actaul, prediction=prediction)
    # make sure in-sample fitting is reasonable
    assert metrics <= 0.5


@pytest.mark.parametrize(
    "events_selection", ["direct_set", "data_driven"], ids=["direct_set", "data_driven"]
)
def test_events_selection(events_selection):
    # data_args
    seed = 2022
    n_steps = 365 * 3
    channels_coef = [0.053, 0.15, 0.19, 0.175, 0.15]
    channels = ["promo", "radio", "search", "social", "tv"]
    features_loc = np.array([2000, 5000, 3850, 3000, 7500])
    features_scale = np.array([550, 2500, 500, 1000, 3500])
    scalability = np.array([3.0, 1.25, 0.8, 1.3, 1.5])
    start_date = "2019-01-01"
    adstock_args = {
        "n_steps": 28,
        "peak_step": np.array([10, 8, 5, 3, 2]),
        "left_growth": np.array([0.05, 0.08, 0.1, 0.5, 0.75]),
        "right_growth": np.array([-0.03, -0.6, -0.5, -0.1, -0.25]),
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
        country="US",
    )

    mmm = MMM(
        kpi_col="sales",
        date_col="date",
        spend_cols=channels,
        event_cols=event_cols,
        seed=seed,
        adstock_df=adstock_df,
    )
    mmm.derive_saturation(df=df, scalability_df=scalability_df)
    if events_selection == "data_driven":
        mmm.filter_features(
            df,
            num_warmup=100,
            num_sample=100,
            chains=4,
        )
    elif events_selection == "direct_set":
        mmm.set_features(event_cols[1:3])

    mmm.fit(df, num_warmup=100, num_sample=100, chains=4)

    saturation_df = mmm.get_saturation()
    assert saturation_df.index.name == "regressor"
    assert all(saturation_df.index == mmm.get_spend_cols())
    mmm_event_cols = set(mmm.get_event_cols())
    mmm_full_event_cols = set(mmm.full_event_cols)
    assert mmm_event_cols.issubset(mmm_full_event_cols)
    assert set(event_cols).issubset(mmm_full_event_cols)

    pred_df = mmm.predict(df)
    # make sure it returns the same shape
    assert pred_df.shape[0] == n_steps
    prediction = pred_df["prediction"].values
    actaul = df["sales"]
    metrics = smape(actual=actaul, prediction=prediction)
    # make sure in-sample fitting is reasonable
    assert metrics <= 0.5


@pytest.mark.parametrize(
    "direct_set_hyper_param",
    [True, False],
    ids=["direct_set_hyper_param", "optim_hyper_param"],
)
def test_hyper_params_selection(direct_set_hyper_param):
    # data_args
    seed = 2023
    n_steps = 365 * 3
    channels_coef = [0.053, 0.15, 0.19, 0.175, 0.15]
    channels = ["promo", "radio", "search", "social", "tv"]
    features_loc = np.array([2000, 5000, 3850, 3000, 7500])
    features_scale = np.array([550, 2500, 500, 1000, 3500])
    scalability = np.array([3.0, 1.25, 0.8, 1.3, 1.5])
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
        country="US",
    )

    mmm = MMM(
        kpi_col="sales",
        date_col="date",
        spend_cols=channels,
        event_cols=event_cols,
        seed=seed,
        adstock_df=adstock_df,
    )
    mmm.derive_saturation(df=df, scalability_df=scalability_df)
    mmm.set_features(event_cols[1:3])
    if direct_set_hyper_param:
        mmm.set_hyper_params(best_params)
    else:
        mmm.optim_hyper_params(df)

    mmm.fit(df, num_warmup=100, num_sample=100, chains=4)

    saturation_df = mmm.get_saturation()
    assert saturation_df.index.name == "regressor"
    assert all(saturation_df.index == mmm.get_spend_cols())
    mmm_event_cols = set(mmm.get_event_cols())
    mmm_full_event_cols = set(mmm.full_event_cols)
    assert mmm_event_cols.issubset(mmm_full_event_cols)
    assert set(event_cols).issubset(mmm_full_event_cols)

    pred_df = mmm.predict(df)
    # make sure it returns the same shape
    assert pred_df.shape[0] == n_steps
    prediction = pred_df["prediction"].values
    actaul = df["sales"]
    metrics = smape(actual=actaul, prediction=prediction)
    # make sure in-sample fitting is reasonable
    assert metrics <= 0.5


# TODO: may check additional regression coefs from the Fourier terms
@pytest.mark.parametrize(
    "with_yearly_seasonality, with_weekly_seasonality, seasonality, fs_orders",
    [
        (False, True, [7], [3]),
        (True, False, [365.25], [5]),
        (True, True, [7, 365.25], [2, 3]),
    ],
    ids=["weekly_seas_only", "yearly_seas_only", "complex_seas"],
)
def test_seasonality(
    with_yearly_seasonality,
    with_weekly_seasonality,
    seasonality,
    fs_orders,
):
    # data_args
    seed = 2022
    n_steps = 365 * 3
    channels_coef = [0.053, 0.15, 0.19, 0.175, 0.15]
    channels = ["promo", "radio", "search", "social", "tv"]
    features_loc = np.array([2000, 5000, 3850, 3000, 7500])
    features_scale = np.array([550, 2500, 500, 1000, 3500])
    scalability = np.array([3.0, 1.25, 0.8, 1.3, 1.5])
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
        country="US",
        with_weekly_seasonality=with_weekly_seasonality,
        with_yearly_seasonality=with_yearly_seasonality,
    )

    df, scalability_df, adstock_df, event_cols = make_mmm_daily_data(
        channels_coef=channels_coef,
        channels=channels,
        features_loc=features_loc,
        features_scale=features_scale,
        scalability=scalability,
        n_steps=n_steps,
        start_date=start_date,
        adstock_args=adstock_args,
        country="US",
    )

    mmm = MMM(
        kpi_col="sales",
        date_col="date",
        spend_cols=channels,
        event_cols=event_cols,
        seed=seed,
        adstock_df=adstock_df,
        seasonality=seasonality,
        fs_orders=fs_orders,
    )
    mmm.derive_saturation(df=df, scalability_df=scalability_df)
    mmm.set_features(event_cols[1:3])
    mmm.set_hyper_params(params=best_params)
    mmm.fit(df, num_warmup=100, num_sample=100, chains=4)

    pred_df = mmm.predict(df)
    # make sure it returns the same shape
    assert pred_df.shape[0] == n_steps
    prediction = pred_df["prediction"].values
    actaul = df["sales"]
    metrics = smape(actual=actaul, prediction=prediction)
    # make sure in-sample fitting is reasonable
    assert metrics <= 0.5
