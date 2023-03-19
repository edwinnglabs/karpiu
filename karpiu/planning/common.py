import pandas as pd
import numpy as np

from typing import List, Optional

from ...models import MMM
from ...utils import adstock_process
from ...explainability import Attributor


def calculate_marginal_cost(
    model: MMM,
    channels: List[str],
    spend_start: str,
    spend_end: str,
    spend_df: Optional[pd.DataFrame] = None,
    delta: float = 1e-1,
) -> pd.DataFrame:
    """Generate overall marginal cost per channel with given period [spend_start, spend_end]
    Args:
        model:
        channels:
        spend_df:
        spend_start:
        spend_end:
        delta:

    Returns:

    """
    if spend_df is None:
        df = model.get_raw_df()
    else:
        df = spend_df.copy()

    date_col = model.date_col
    max_adstock = model.get_max_adstock()

    spend_start = pd.to_datetime(spend_start)
    spend_end = pd.to_datetime(spend_end)
    mea_start = spend_start
    mea_end = spend_end + pd.Timedelta(days=max_adstock)
    calc_start = spend_start - pd.Timedelta(days=max_adstock)
    calc_end = spend_end + pd.Timedelta(days=max_adstock)

    spend_mask = (df[date_col] >= spend_start) & (df[date_col] <= spend_end)
    mea_mask = (df[date_col] >= mea_start) & (df[date_col] <= mea_end)
    calc_mask = (df[date_col] >= calc_start) & (df[date_col] <= calc_end)
    dt_arr = df.loc[mea_mask, date_col].values

    # base_comp calculation finished above
    zero_df = df.copy()
    zero_df[channels] = 0.0
    zero_pred_df = model.predict(df=zero_df)
    base_comp = zero_pred_df.loc[mea_mask, "prediction"].values

    baseline_spend_matrix = df.loc[calc_mask, channels].values
    sat_arr = model.get_saturation_vector(regressors=channels)
    adstock_matrix = model.get_adstock_matrix(spend_cols=channels)
    coef_matrix = model.get_coef_matrix(date_array=dt_arr, regressors=channels)

    # (mea_steps, n_regressors)
    transformed_spend_matrix = adstock_process(baseline_spend_matrix, adstock_matrix)
    transformed_spend_matrix = np.log1p(transformed_spend_matrix / sat_arr)
    reg_comp = np.sum(coef_matrix * transformed_spend_matrix, axis=-1)
    baseline_pred_comp = base_comp * np.exp(reg_comp)

    # the varying comp is computed below
    marginal_cost = np.empty(len(channels))
    for idx, ch in enumerate(channels):
        # (calc_steps, n_regressors)
        delta_matrix = np.zeros_like(baseline_spend_matrix)
        if max_adstock > 0:
            delta_matrix[max_adstock:-max_adstock, idx] = delta
        else:
            delta_matrix += delta

        # (calc_steps, n_regressors)
        new_spend_matrix = baseline_spend_matrix + delta_matrix
        new_transformed_spend_matrix = adstock_process(new_spend_matrix, adstock_matrix)
        new_transformed_spend_matrix = np.log1p(new_transformed_spend_matrix / sat_arr)
        new_reg_comp = np.sum(coef_matrix * new_transformed_spend_matrix, -1)

        new_pred_comp = base_comp * np.exp(new_reg_comp)
        marginal_cost[idx] = np.sum(delta_matrix) / np.sum(
            new_pred_comp - baseline_pred_comp
        )

    return pd.DataFrame(
        {
            "regressor": channels,
            "marginal_cost": marginal_cost,
        }
    ).set_index("regressor")


def generate_cost_report(
    model: MMM,
    channels: List[str],
    start: str,
    end: str,
    pre_spend_df: pd.DataFrame,
    post_spend_df: pd.DataFrame,
    spend_scaler: float = 1e3,
    delta: float = 1e-1,
):
    """A wrapper function combining calculation of average and marginal cost in pre and post optimization"""
    # report average and marginal cost
    # pre-opt result
    attr_obj = Attributor(model, attr_regressors=channels, start=start, end=end)
    _, spend_attr_df, spend_df, _ = attr_obj.make_attribution()
    tot_attr_df = spend_attr_df[channels].apply(np.sum, axis=0)
    tot_spend_df = spend_df[channels].apply(np.sum, axis=0)
    avg_cost_df = tot_spend_df / tot_attr_df
    avg_cost_df = pd.DataFrame(avg_cost_df)
    avg_cost_df.index = avg_cost_df.index.rename("regressor")
    avg_cost_df = avg_cost_df.rename(columns={0: "avg_cost"})

    tot_spend_df = tot_spend_df / spend_scaler
    tot_spend_df = pd.DataFrame(tot_spend_df)
    tot_spend_df.index = tot_spend_df.index.rename("regressor")
    tot_spend_df = tot_spend_df.rename(columns={0: "pre_opt_spend"})

    mc_df = calculate_marginal_cost(
        model,
        spend_df=pre_spend_df,
        channels=channels,
        spend_start=start,
        spend_end=end,
        delta=delta,
    )

    pre_opt_report = pd.concat(
        [avg_cost_df, mc_df, tot_spend_df], axis=1, keys="regressor"
    )
    pre_opt_report.columns = [
        "pre-opt-avg-cost",
        "pre-opt-marginal-cost",
        "pre-opt-spend",
    ]

    # post-opt result
    attr_obj = Attributor(
        model, df=post_spend_df, attr_regressors=channels, start=start, end=end
    )
    _, spend_attr_df, spend_df, _ = attr_obj.make_attribution()
    optim_tot_attr_df = spend_attr_df[channels].apply(np.sum, axis=0)
    optim_tot_spend_df = spend_df[channels].apply(np.sum, axis=0)
    post_avg_cost_df = optim_tot_spend_df / optim_tot_attr_df
    post_avg_cost_df = pd.DataFrame(post_avg_cost_df)
    post_avg_cost_df = post_avg_cost_df.rename(columns={0: "post-opt-avg-cost"})

    optim_tot_spend_df = optim_tot_spend_df / spend_scaler
    optim_tot_spend_df = pd.DataFrame(optim_tot_spend_df)
    optim_tot_spend_df.index = optim_tot_spend_df.index.rename("regressor")
    optim_tot_spend_df = optim_tot_spend_df.rename(columns={0: "post-opt-spend"})

    post_mc_df = calculate_marginal_cost(
        model,
        spend_df=post_spend_df,
        channels=channels,
        spend_start=start,
        spend_end=end,
    )

    post_opt_report = pd.concat(
        [post_avg_cost_df, post_mc_df, optim_tot_spend_df], axis=1
    )
    post_opt_report.columns = [
        "post-opt-avg-cost",
        "post-opt-marginal-cost",
        "post-opt-spend",
    ]

    report = pd.concat([pre_opt_report, post_opt_report], axis=1)
    report = report[
        [
            "pre-opt-avg-cost",
            "post-opt-avg-cost",
            "pre-opt-marginal-cost",
            "post-opt-marginal-cost",
            "pre-opt-spend",
            "post-opt-spend",
        ]
    ]
    return report
