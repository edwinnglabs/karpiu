import pytest
import numpy as np
import pandas as pd
import pickle
import os
from karpiu.explainability import AttributorGamma
from karpiu.utils import insert_events, extend_ts_features

# TODO:
# test cases:
# 1. basic case in no-adosc
# spend and activities attr are always the same
# general cases:
# 1. organic is always unchanged with varying spend input [done]
# 2. larger spend leads to larger attribution
# 3. zero spend infer zero spend attribution [done]
# 4. attribution add up to original response in original spend
# 5. attribution sum is larger than the original response with increasing spend
# 6. diminishing returns of attribution [done]
# 7. capable of attributing future scenarios when spend is provided [separate a test module]


@pytest.mark.parametrize(
    "model_file",
    [
        "simple-model.pkl",
        "non-seasonal-model.pkl",
        "seasonal-model.pkl",
    ],
    ids=[
        "simple-model",
        "non-seaonal-adstock-model",
        "seasonal-adstock-model",
    ],
)
def test_attribution_core(model_file):
    model_path = os.path.join(os.path.dirname(__file__), "..", "resources/", model_file)
    model_path = os.path.abspath(model_path)

    with open(model_path, "rb") as f:
        mmm = pickle.load(f)

    attr_start = "2020-01-01"
    attr_end = "2020-01-15"
    duration = (pd.to_datetime(attr_end) - pd.to_datetime(attr_start)).days + 1

    # original df
    attributor = AttributorGamma(mmm, start=attr_start, end=attr_end, debug=True)
    activities_attr_df, spend_attr_df, spend_df, _ = attributor.make_attribution()
    max_adstock = attributor.max_adstock
    if max_adstock == 0:
        assert np.all(
            activities_attr_df[["organic"] + mmm.get_spend_cols()].sum()
            == spend_attr_df[["organic"] + mmm.get_spend_cols()].sum()
        )
    else:
        assert np.all(
            activities_attr_df[["organic"]].sum() == spend_attr_df[["organic"]].sum()
        )

    assert spend_attr_df.shape[0] == duration
    assert activities_attr_df.shape[0] == duration
    assert attributor.delta_matrix.shape == (
        duration + max_adstock * 2,
        max_adstock + 1,
        len(mmm.get_spend_cols()),
    )

    # increase spend df
    new_df = mmm.get_raw_df()
    new_df["search"] *= 10
    new_attributor = AttributorGamma(
        mmm, df=new_df, start=attr_start, end=attr_end, debug=True
    )
    (
        new_activities_attr_df,
        new_spend_attr_df,
        new_spend_df,
        _,
    ) = new_attributor.make_attribution()

    # larger spend leads to larger attribution
    assert new_activities_attr_df["search"].sum() >= activities_attr_df["search"].sum()
    assert new_spend_attr_df["search"].sum() > spend_attr_df["search"].sum()

    # zero spend leads to zero attribution
    assert np.allclose(spend_attr_df.loc[spend_df["social"] == 0, "social"].values, 0)
    assert np.allclose(spend_attr_df.loc[spend_df["search"] == 0, "search"].values, 0)

    # larger spend leads to larger attribution of marketing
    assert np.sum(new_attributor.attr_marketing) > np.sum(attributor.attr_marketing)

    # organic attribution always stay constant
    assert np.nansum(new_attributor.resid_df["resid"]) == np.nansum(
        attributor.resid_df["resid"]
    )
    assert np.allclose(
        new_attributor.attr_organic[max_adstock:], attributor.attr_organic[max_adstock:]
    )
    assert np.allclose(
        new_activities_attr_df.loc[max_adstock:, "organic"].values,
        activities_attr_df.loc[max_adstock:, "organic"].values,
    )

    # attribution add up to original response in original spend
    tot_attr = (
        activities_attr_df[["organic"] + mmm.get_spend_cols()]
        .apply(np.sum, axis=1)
        .values
    )
    raw_df = mmm.get_raw_df()
    response_in_attr_period = raw_df.loc[
        (raw_df["date"] >= attr_start) & (raw_df["date"] <= attr_end), "sales"
    ].values

    assert np.allclose(tot_attr, response_in_attr_period)
    # attribution sum is larger than the original response with increasing spend
    new_tot_attr = (
        new_activities_attr_df[["organic"] + mmm.get_spend_cols()]
        .apply(np.sum, axis=1)
        .values
    )
    assert np.all(np.round(new_tot_attr) >= response_in_attr_period)

    # diminishing returns property
    avg_cost_of_sales = spend_df["search"].sum() / spend_attr_df["search"].sum()
    new_avg_cost_of_sales = (
        new_spend_df["search"].sum() / new_spend_attr_df["search"].sum()
    )
    assert new_avg_cost_of_sales > avg_cost_of_sales


@pytest.mark.parametrize(
    "model_file",
    [
        "simple-model.pkl",
        "non-seasonal-model.pkl",
        "seasonal-model.pkl",
    ],
    ids=[
        "simple-model",
        "non-seaonal-adstock-model",
        "seasonal-adstock-model",
    ],
)
def test_future_attribution(model_file):
    model_path = os.path.join(os.path.dirname(__file__), "..", "resources/", model_file)
    model_path = os.path.abspath(model_path)

    with open(model_path, "rb") as f:
        mmm = pickle.load(f)

    raw_df = mmm.get_raw_df()

    n_periods = 360
    extended_df = raw_df[["date"] + mmm.get_spend_cols()]
    # spend + features
    extended_df = extend_ts_features(
        extended_df,
        n_periods=n_periods,
        date_col="date",
        rolling_window=n_periods,
    )
    extended_df, event_cols = insert_events(extended_df, date_col="date", country="US")
    missing_events = list(set(mmm.get_event_cols()) - set(event_cols))
    extended_df[missing_events] = 0

    # since df['date'].max() == '2021-12-30', we test with periods
    # 1. overlapping past and future
    # 2. future only
    # 3. past only
    # TODO: assert their organic should be the same
    # TODO: assert the attribution in the overlapping period should be the same

    past_and_future_attributor = AttributorGamma(
        model=mmm, df=extended_df, start="2021-12-30", end="2021-12-31", debug=True
    )
    (
        _,
        past_and_future_spend_attr_df,
        _,
        _,
    ) = past_and_future_attributor.make_attribution()

    future_only_attributor = AttributorGamma(
        model=mmm, df=extended_df, start="2021-12-31", end="2021-12-31", debug=True
    )
    _, future_only_spend_attr_df, _, _ = future_only_attributor.make_attribution()

    past_only_attributor = AttributorGamma(
        model=mmm, df=extended_df, start="2021-12-30", end="2021-12-30", debug=True
    )
    _, past_only_spend_attr_df, _, _ = past_only_attributor.make_attribution()

    max_adstock = past_and_future_attributor.max_adstock
    # organic attribution always stay constant
    assert np.nansum(
        past_and_future_attributor.resid_df["resid"].values[max_adstock:]
    ) == np.nansum(future_only_attributor.resid_df["resid"].values[max_adstock:])

    assert np.allclose(
        past_and_future_attributor.attr_organic[max_adstock:],
        future_only_attributor.attr_organic[max_adstock:],
    )

    assert np.nansum(
        past_and_future_attributor.resid_df["resid"].values[max_adstock:]
    ) == np.nansum(past_only_attributor.resid_df["resid"].values[max_adstock:])

    assert np.allclose(
        past_and_future_attributor.attr_organic[max_adstock:],
        past_only_attributor.attr_organic[max_adstock:],
    )

    # attribution should be unchanged
    assert np.allclose(
        past_and_future_spend_attr_df.loc[:, mmm.get_spend_cols()].values[1],
        future_only_spend_attr_df.loc[:, mmm.get_spend_cols()].values[0],
    )

    assert np.allclose(
        past_and_future_spend_attr_df.loc[:, mmm.get_spend_cols()].values[0],
        past_only_spend_attr_df.loc[:, mmm.get_spend_cols()].values[0],
    )
