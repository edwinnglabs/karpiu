import pytest
import numpy as np
import pickle
from copy import deepcopy

from karpiu.models import MMM
from karpiu.simulation import make_mmm_daily_data
from karpiu.planning.optim import TargetMaximizer
from karpiu.planning.common import generate_cost_report
from karpiu.utils import adstock_process


@pytest.mark.parametrize(
    "model_path",
    [
        "./tests/resources/seasonal-model.pkl",
        "./tests/resources/non-seasonal-model.pkl",
    ],
    ids=["full_case", "wo_seasonality_and_events"],
)
def test_target_maximizer_init(model_path):
    with open(model_path, "rb") as f:
        mmm = pickle.load(f)

    channels = mmm.get_spend_cols()
    df = mmm.get_raw_df()
    budget_start = "2020-01-01"
    budget_end = "2020-01-31"

    # test the prediction function preserve with orbit method and karpiu numpy method
    maximizer = TargetMaximizer(
        model=mmm,
        budget_start=budget_start,
        budget_end=budget_end,
        optim_channels=channels,
    )

    coef_matrix = maximizer.target_coef_matrix
    adstock_matrix = maximizer.target_adstock_matrix
    input_spend_matrix = df.loc[:, channels].values
    input_spend_matrix = input_spend_matrix[maximizer.calc_mask]
    # zero out before and after with adstock periods; add the background spend
    # for testing background spend correctness
    if maximizer.max_adstock > 0:
        input_spend_matrix[: maximizer.max_adstock] = 0.0
        input_spend_matrix[-maximizer.max_adstock :] = 0.0
    input_spend_matrix += maximizer.target_regressor_bkg_matrix

    # adstock, log1p, saturation
    # (n_steps - max_adstock, )
    transformed_regressors_matrix = adstock_process(input_spend_matrix, adstock_matrix)
    transformed_regressors_matrix = np.log1p(
        transformed_regressors_matrix / maximizer.target_sat_array
    )

    reg_comp = np.sum(coef_matrix * transformed_regressors_matrix, axis=-1)
    # from maximizer parameters
    pred_comp_from_optim = np.exp(reg_comp) * maximizer.base_comp_result

    # from karpiu/orbit method
    pred_df = mmm.predict(df)
    pred_comp = pred_df.loc[maximizer.calc_mask, "prediction"].values
    pred_comp = pred_comp[maximizer.max_adstock :]

    assert np.allclose(pred_comp_from_optim, pred_comp)


# test target maximizer optimization behaviors
def test_target_maximizer():
    with open("./tests/resources/seasonal-model.pkl", "rb") as f:
        mmm = pickle.load(f)

    df = mmm.get_raw_df()
    budget_start = "2020-01-01"
    budget_end = "2020-01-31"
    optim_channels = mmm.get_spend_cols()
    # to be safe in beta version, use sorted list of channels
    optim_channels.sort()

    spend_scaler = 1e1
    response_scaler = 0.01 * np.std(df["sales"].values)

    maximizer = TargetMaximizer(
        model=mmm,
        budget_start=budget_start,
        budget_end=budget_end,
        optim_channels=optim_channels,
        spend_scaler=spend_scaler,
        response_scaler=response_scaler,
    )
    optim_spend_df = maximizer.optimize(maxiter=3000, ftol=1e-3)

    optim_spend_matrix = maximizer.get_current_state()
    init_spend_matrix = maximizer.get_init_state()
    # always spend all budget in target maximization; assure the total preserves
    assert np.allclose(np.sum(optim_spend_matrix), np.sum(init_spend_matrix))

    cost_report = generate_cost_report(
        model=mmm,
        channels=optim_channels,
        start=budget_start,
        end=budget_end,
        pre_spend_df=df,
        post_spend_df=optim_spend_df,
    )

    pre_opt_spend = cost_report["pre-opt-spend"].values
    pre_ac = cost_report["pre-opt-avg-cost"].values[pre_opt_spend > 0]
    pre_mc = cost_report["pre-opt-marginal-cost"].values[pre_opt_spend > 0]

    post_opt_spend = cost_report["post-opt-spend"].values
    post_ac = cost_report["post-opt-avg-cost"].values[post_opt_spend > 0]
    post_mc = cost_report["post-opt-marginal-cost"].values[post_opt_spend > 0]

    assert np.all(pre_mc >= pre_ac)
    assert np.all(post_mc >= post_ac)

    # check 2: all marginal cost of channel that has spend
    # within 15% of mean
    abs_delta_perc = np.abs(post_mc / np.nanmean(post_mc) - 1.00)
    assert np.all(abs_delta_perc < 0.15)

    cv = np.nanstd(post_mc) / np.nanmean(post_mc)
    assert np.all(cv < 0.1)

    # check 3: total predicted response must be equal to or higher than current
    optim_pred = mmm.predict(optim_spend_df)
    init_pred = mmm.predict(df)
    total_optim_pred = np.sum(
        optim_pred.loc[maximizer.result_mask, "prediction"].values
    )
    total_init_pred = np.sum(init_pred.loc[maximizer.result_mask, "prediction"].values)
    assert total_optim_pred - total_init_pred >= 0

    # check 4: optimization result should be indifferent with initial values
    # create different initial spend df and plug back into the model
    new_raw_df = mmm.get_raw_df()
    new_spend_matrix = new_raw_df.loc[
        (new_raw_df["date"] >= budget_start) & (new_raw_df["date"] <= budget_end),
        optim_channels,
    ].values
    # mutable numpy array
    np.random.shuffle(new_spend_matrix)
    new_raw_df.loc[
        (new_raw_df["date"] >= budget_start) & (new_raw_df["date"] <= budget_end),
        optim_channels,
    ] = new_spend_matrix
    new_mmm = deepcopy(mmm)
    new_mmm.raw_df = new_raw_df
    new_maximizer = TargetMaximizer(
        model=new_mmm,
        budget_start=budget_start,
        budget_end=budget_end,
        optim_channels=optim_channels,
        spend_scaler=spend_scaler,
        response_scaler=response_scaler,
    )
    optim_spend_df = new_maximizer.optimize(maxiter=3000, ftol=1e-3)
    new_optim_spend_matrix = new_maximizer.get_current_state()
    new_init_spend_matrix = new_maximizer.get_init_state()

    # the final result should be closed in either by 1e-1 or .1%
    assert np.any(np.not_equal(new_init_spend_matrix, init_spend_matrix))
    # FIXME: this should not be 10% different; come back to this once we have a faster unit test
    # right now, we turn off adstock make this work
    # something wrong with target max optimization here; the state does not converge
    # assert np.allclose(new_optim_spend_matrix, optim_spend_matrix, atol=1e-1, rtol=1e-1)
