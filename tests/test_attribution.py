import pytest
import numpy as np
from karpiu.models import MMM
from karpiu.simulation import make_mmm_daily_data
from karpiu.explainability import Attributor


# test basic run attribution
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
@pytest.mark.parametrize(
    "seasonality, fs_orders",
    [
        ([365.25], [3]),
        (None, None),
    ],
    ids=["w_seasonality", "wo_seasonality"],
)
def test_attribution(with_events, adstock_args, seasonality, fs_orders):
    # data_args
    seed = 2022
    n_steps = 365 * 3
    channels_coef = [0.03, 0.05, 0.028, 0.01, 0.03]
    channels = ["tv", "radio", "social", "promo", "search"]
    features_loc = np.array([10000, 5000, 3000, 2000, 850])
    features_scale = np.array([5000, 3000, 1000, 550, 500])
    scalability = np.array([1.1, 0.75, 1.3, 1.5, 0.9])
    start_date = "2019-01-01"
    best_params = {
        "damped_factor": 0.9057,
        "level_sm_input": 0.0245,
        "slope_sm_input": 0.0943,
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
        seed=seed,
        adstock_df=adstock_df,
    )
    mmm.derive_saturation(df=df, scalability_df=scalability_df)
    mmm.set_hyper_params(best_params)
    mmm.fit(df, num_warmup=400, num_sample=100, chains=4)

    # FIXME: parameterize this later
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
