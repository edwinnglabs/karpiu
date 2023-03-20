import pytest
import numpy as np
from karpiu.models import MMM
from karpiu.simulation import make_mmm_daily_data
from karpiu.explainability.attribution import Attributor


@pytest.mark.parametrize(
    "with_events",
    [True, False],
    ids=["w_events", "wo_events"],
)
@pytest.mark.parametrize(
    "seasonality, fs_orders",
    [
        ([365.25], [3]),
        (None, None),
    ],
    ids=["w_seasonality", "wo_seasonality"],
)
def test_wo_adstock_attribution(with_events, seasonality, fs_orders):
    # data_args
    seed = 2022
    n_steps = 365 * 3
    channels_coef = [0.053, 0.08, 0.19, 0.125, 0.1]
    channels = ["promo", "radio", "search", "social", "tv"]
    features_loc = np.array([2000, 5000, 3850, 3000, 7500])
    features_scale = np.array([550, 2500, 500, 1000, 3500])
    scalability = np.array([3.0, 1.25, 0.8, 1.3, 1.5])
    start_date = "2019-01-01"
    best_params = {
        "damped_factor": 0.9057,
        "level_sm_input": 0.0245,
    }

    np.random.seed(seed)
    with_yearly_seasonality = False
    if seasonality is not None and len(seasonality) > 0:
        with_yearly_seasonality = True

    df, scalability_df, adstock_df, event_cols = make_mmm_daily_data(
        channels_coef=channels_coef,
        channels=channels,
        features_loc=features_loc,
        features_scale=features_scale,
        scalability=scalability,
        n_steps=n_steps,
        start_date=start_date,
        adstock_args=None,
        with_yearly_seasonality=with_yearly_seasonality,
        country="US" if with_events else None,
    )
    mmm = MMM(
        kpi_col="sales",
        date_col="date",
        spend_cols=channels,
        event_cols=event_cols,
        seasonality=seasonality,
        fs_orders=fs_orders,
        adstock_df=adstock_df,
        seed=seed,
    )
    mmm.derive_saturation(df=df, scalability_df=scalability_df)
    mmm.set_hyper_params(best_params)
    mmm.fit(df, num_warmup=100, num_sample=100, chains=4)

    channels_subsets = (channels, channels[0:2])

    for ch_subset in channels_subsets:
        attr_obj = Attributor(
            mmm,
            attr_regressors=ch_subset,
            start="2020-01-01",
            end="2020-01-31",
        )

        res = attr_obj.make_attribution()
        activities_attr_df, spend_attr_df, spend_df, cost_df = res

        assert activities_attr_df.shape == spend_attr_df.shape
        assert spend_attr_df.shape[0] == spend_df.shape[0]
        assert spend_attr_df.shape[1] - 1 == spend_df.shape[1]
        assert cost_df.shape[0] == spend_df.shape[0]

        # in no adstock case, activities and spend attribution should be identical
        assert np.all(
            np.equal(
                activities_attr_df[["organic"] + ch_subset].values,
                spend_attr_df[["organic"] + ch_subset].values,
            )
        )

    # in single step, compare with prediction delta to make sure the one-of is calculated
    # correctly
    attr_obj = Attributor(
        mmm,
        attr_regressors=channels,
        start="2020-03-01",
        end="2020-03-01",
    )
    # extract delta matrix
    res = attr_obj.make_attribution(debug=True)
    activities_attr_df, spend_attr_df, spend_df, cost_df = res
    delta_matrix = attr_obj.delta_matrix

    assert np.all(delta_matrix >= 0.0)

    # prediction delta
    full_pred = mmm.predict(df)
    full_comp = full_pred.loc[full_pred["date"] == "2020-03-01", "prediction"].values

    pred_delta = np.zeros(len(channels) + 1)
    all_off_df = df.copy()
    all_off_df.loc[all_off_df["date"] == "2020-03-01", channels] = 0.0
    all_off_pred = mmm.predict(all_off_df)
    all_off_comp = all_off_pred.loc[
        all_off_pred["date"] == "2020-03-01", "prediction"
    ].values
    pred_delta[0] = all_off_comp
    for idx, ch in enumerate(channels):
        one_off_df = df.copy()
        one_off_df.loc[one_off_df["date"] == "2020-03-01", ch] = 0.0
        one_off_pred = mmm.predict(one_off_df)
        one_off_comp = one_off_pred.loc[
            one_off_pred["date"] == "2020-03-01", "prediction"
        ].values
        pred_delta[idx + 1] = full_comp - one_off_comp

    assert np.allclose(delta_matrix[0, 0, ...], pred_delta)


# with adstock the logic to verify whether the decomp is calculated correctly is complex
# right now mainly relies on the non-adstock cases to lock down the logic
# TODO: a complex module is doable to sum over more terms of prediction delta
# TODO: in verifying the decomp
@pytest.mark.parametrize(
    "with_events",
    [True, False],
    ids=["w_events", "wo_events"],
)
@pytest.mark.parametrize(
    "seasonality, fs_orders",
    [
        ([365.25], [3]),
        (None, None),
    ],
    ids=["w_seasonality", "wo_seasonality"],
)
def test_w_adstock_attribution(with_events, seasonality, fs_orders):
    # data_args
    seed = 2022
    n_steps = 365 * 3
    channels_coef = [0.053, 0.08, 0.19, 0.125, 0.1]
    channels = ["promo", "radio", "search", "social", "tv"]
    features_loc = np.array([2000, 5000, 3850, 3000, 7500])
    features_scale = np.array([550, 2500, 500, 1000, 3500])
    scalability = np.array([3.0, 1.25, 0.8, 1.3, 1.5])
    start_date = "2019-01-01"
    best_params = {
        "damped_factor": 0.9057,
        "level_sm_input": 0.0245,
    }
    adstock_args = {
        "n_steps": 28,
        "peak_step": np.array([10, 8, 5, 3, 2]),
        "left_growth": np.array([0.05, 0.08, 0.1, 0.5, 0.75]),
        "right_growth": np.array([-0.03, -0.6, -0.5, -0.1, -0.25]),
    }

    np.random.seed(seed)
    with_yearly_seasonality = False
    if seasonality is not None and len(seasonality) > 0:
        with_yearly_seasonality = True

    df, scalability_df, adstock_df, event_cols = make_mmm_daily_data(
        channels_coef=channels_coef,
        channels=channels,
        features_loc=features_loc,
        features_scale=features_scale,
        scalability=scalability,
        n_steps=n_steps,
        start_date=start_date,
        adstock_args=adstock_args,
        with_yearly_seasonality=with_yearly_seasonality,
        country="US" if with_events else None,
    )
    mmm = MMM(
        kpi_col="sales",
        date_col="date",
        spend_cols=channels,
        event_cols=event_cols,
        seasonality=seasonality,
        fs_orders=fs_orders,
        adstock_df=adstock_df,
        seed=seed,
    )
    mmm.derive_saturation(df=df, scalability_df=scalability_df)
    mmm.set_hyper_params(best_params)
    mmm.fit(df, num_warmup=100, num_sample=100, chains=4)

    channels_subsets = (channels, channels[0:2])

    for ch_subset in channels_subsets:
        attr_obj = Attributor(
            mmm,
            attr_regressors=ch_subset,
            start="2020-01-01",
            end="2020-01-31",
        )

        res = attr_obj.make_attribution(debug=True)
        delta_matrix = attr_obj.delta_matrix
        # after the adstock period, all delta should be finite
        assert np.all(delta_matrix[mmm.get_max_adstock() :, ...] >= 0.0)
        activities_attr_df, spend_attr_df, spend_df, cost_df = res

        assert (
            activities_attr_df.shape[0]
            == spend_attr_df.shape[0] + mmm.get_max_adstock()
        )
        assert activities_attr_df.shape[1] == spend_attr_df.shape[1]
        assert spend_attr_df.shape[0] == spend_df.shape[0]
        assert spend_attr_df.shape[1] - 1 == spend_df.shape[1]
        assert cost_df.shape[0] == spend_df.shape[0]

    # test different ways to call attribution
    # TODO: parameterize this later
    # with regressors specified
    attr_obj = Attributor(
        mmm,
        attr_regressors=channels[0:2],
        start="2020-01-01",
        end="2020-01-31",
    )
    _, _, _, _ = attr_obj.make_attribution()
    # without regressors specified
    attr_obj = Attributor(
        mmm,
        start="2020-01-01",
        end="2020-01-31",
    )
    # with regressors specified
    attr_obj = Attributor(
        mmm,
        attr_regressors=channels[0:2],
        start="2020-01-01",
        end="2020-01-31",
    )
    _, _, _, _ = attr_obj.make_attribution()
    # without date-range specified
    attr_obj = Attributor(
        mmm,
    )
    _, _, _, _ = attr_obj.make_attribution()
    # with df specified
    attr_obj = Attributor(
        mmm,
        df=mmm.get_raw_df(),
        start="2020-01-01",
        end="2020-01-31",
    )
    _, _, _, _ = attr_obj.make_attribution()
