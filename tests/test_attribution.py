import pytest
import numpy as np
import pandas as pd
import pickle

from karpiu.explainability import AttributorBeta


def test_wo_attribution():
    with open("./tests/resources/simple-model.pkl", "rb") as f:
        mmm = pickle.load(f)
    
    df = mmm.get_raw_df()
    channels = mmm.get_spend_cols()
    channels_subsets = (channels, channels[0:2])

    attr_start = "2020-01-01"
    attr_end = "2020-01-31"
    duration = (pd.to_datetime(attr_end) - pd.to_datetime(attr_start)).days + 1

    # check attribution works with full set and subset of channels
    for ch_subset in channels_subsets:
        attr_obj = AttributorBeta(
            mmm,
            attr_regressors=ch_subset,
            start="2020-01-01",
            end="2020-01-31",
        )

        res = attr_obj.make_attribution(fixed_intercept=False)
        activities_attr_df, spend_attr_df, spend_df, cost_df = res

        assert activities_attr_df.shape[0] == duration
        assert (activities_attr_df.shape[1] - 2) == len(ch_subset)
        assert spend_attr_df.shape[0] == duration
        assert (spend_attr_df.shape[1] - 2) == len(ch_subset)
        assert cost_df.shape[0] == duration

        # in no adstock case, activities and spend attribution should be identical
        assert np.all(
            np.equal(
                activities_attr_df[["organic"] + ch_subset].values,
                spend_attr_df[["organic"] + ch_subset].values,
            )
        )

    # in single step, compare with prediction delta to make sure the one-of is calculated
    # correctly
    attr_obj = AttributorBeta(
        mmm,
        attr_regressors=channels,
        start=attr_start,
        end=attr_end,
    )
    # extract delta matrix
    res = attr_obj.make_attribution(debug=True, fixed_intercept=False)
    activities_attr_df, spend_attr_df, spend_df, cost_df = res
    delta_matrix = attr_obj.delta_matrix

    assert np.all(delta_matrix >= 0.0)

    # prediction delta
    full_pred = mmm.predict(df)
    full_comp = full_pred.loc[full_pred["date"] == attr_start, "prediction"].values

    pred_delta = np.zeros(len(channels) + 1)
    all_off_df = df.copy()
    all_off_df.loc[all_off_df["date"] == attr_start, channels] = 0.0
    all_off_pred = mmm.predict(all_off_df)
    all_off_comp = all_off_pred.loc[
        all_off_pred["date"] == attr_start, "prediction"
    ].values
    pred_delta[0] = all_off_comp
    for idx, ch in enumerate(channels):
        one_off_df = df.copy()
        one_off_df.loc[one_off_df["date"] == attr_start, ch] = 0.0
        one_off_pred = mmm.predict(one_off_df)
        one_off_comp = one_off_pred.loc[
            one_off_pred["date"] == attr_start, "prediction"
        ].values
        pred_delta[idx + 1] = full_comp - one_off_comp

    assert np.allclose(delta_matrix[0, 0, ...], pred_delta)


# with adstock the logic to verify whether the decomp is calculated correctly is complex
# right now mainly relies on the non-adstock cases to lock down the logic
# TODO: a complex module is doable to sum over more terms of prediction delta
# TODO: in verifying the decomp
@pytest.mark.parametrize(
    "model_path",
    [
        "./tests/resources/seasonal-model.pkl",
        "./tests/resources/non-seasonal-model.pkl",
    ],
    ids=["full_case", "wo_seasonality_and_events"],
)
def test_w_adstock_attribution(model_path):
    with open(model_path, "rb") as f:
        mmm = pickle.load(f)

    channels = mmm.get_spend_cols()
    channels_subsets = (channels, channels[0:2])

    attr_start = "2020-01-01"
    attr_end = "2020-01-31"
    duration = (pd.to_datetime(attr_end) - pd.to_datetime(attr_start)).days + 1

    for ch_subset in channels_subsets:
        attr_obj = AttributorBeta(
            mmm,
            attr_regressors=ch_subset,
            start=attr_start,
            end=attr_end,
        )

        res = attr_obj.make_attribution(debug=True, fixed_intercept=False)
        delta_matrix = attr_obj.delta_matrix

        # after the adstock period, all delta should be finite
        assert np.all(delta_matrix[mmm.get_max_adstock() :, ...] >= 0.0)
        activities_attr_df, spend_attr_df, _, cost_df = res

        assert activities_attr_df.shape[0] == duration
        assert (activities_attr_df.shape[1] - 2) == len(ch_subset)
        assert spend_attr_df.shape[0] == duration
        assert (spend_attr_df.shape[1] - 2) == len(ch_subset)
        assert cost_df.shape[0] == duration

    # test different ways to call attribution
    # TODO: parameterize this later
    # with regressors specified
    attr_obj = AttributorBeta(
        mmm,
        attr_regressors=channels[0:2],
        start="2020-01-01",
        end="2020-01-31",
    )
    _, _, _, _ = attr_obj.make_attribution(fixed_intercept=False)
    # without regressors specified
    attr_obj = AttributorBeta(
        mmm,
        start="2020-01-01",
        end="2020-01-31",
    )
    # with regressors specified
    attr_obj = AttributorBeta(
        mmm,
        attr_regressors=channels[0:2],
        start="2020-01-01",
        end="2020-01-31",
    )
    _, _, _, _ = attr_obj.make_attribution(fixed_intercept=False)
    # without date-range specified
    attr_obj = AttributorBeta(
        mmm,
    )
    _, _, _, _ = attr_obj.make_attribution(fixed_intercept=False)
    # with df specified
    attr_obj = AttributorBeta(
        mmm,
        df=mmm.get_raw_df(),
        start="2020-01-01",
        end="2020-01-31",
    )
    _, _, _, _ = attr_obj.make_attribution(fixed_intercept=False)
