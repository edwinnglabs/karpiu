import scipy.optimize as optim
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from typing import Optional
import math
import matplotlib.pyplot as plt

from .models import MMM


class Optimizer:
    """Perform optimization with a given Marketing Mix Model"""


def generate_cost_curves(
        model: MMM,
        spend_df: Optional[pd.DataFrame] = None,
        spend_start: Optional[str] = None,
        spend_end: Optional[str] = None,
) -> pd.DataFrame:
    """ Generate cost curves given a Marketing Mix Model

    Args:
        model:
        spend_df:
        spend_start:
        spend_end:

    Returns:

    """
    if spend_df is None:
        spend_df = model.raw_df.copy()

    paid_channels = model.get_spend_cols()
    date_col = model.date_col
    max_adstock = model.get_max_adstock()

    if spend_start is None:
        spend_start = pd.to_datetime(spend_df[date_col].values[0])
    else:
        spend_start = pd.to_datetime(spend_start)

    if spend_end is None:
        spend_end = pd.to_datetime(spend_df[date_col].values[-1])
    else:
        spend_end = pd.to_datetime(spend_end)

    outcome_start = spend_start
    outcome_end = spend_end + pd.Timedelta(days=max_adstock)

    # output
    cost_curves_dict = {
        'ch': list(),
        'total_spend': list(),
        'total_outcome': list(),
        'multiplier': list(),
    }

    spend_mask = (spend_df[date_col] >= spend_start) & (spend_df[date_col] <= spend_end)
    outcome_mask = (spend_df[date_col] >= outcome_start) & (spend_df[date_col] <= outcome_end)

    # use overall spend range to determine simualtion scenarios
    spend_summmary = spend_df.loc[spend_mask, paid_channels].sum()
    spend_summmary.index = spend_summmary.index.set_names(['channel'])
    spend_summmary = spend_summmary.reset_index(name='total_spend')
    spend_summmary['max_multiplier'] = np.max(spend_summmary['total_spend']) / spend_summmary['total_spend'] * 1.2

    for ch in tqdm(paid_channels):
        # multiplier always contains 1.0
        # multiplier is derived based on ratio of the maximum
        temp_max_multiplier = spend_summmary.loc[spend_summmary['channel'] == ch, 'max_multiplier'].values
        temp_multipliers = np.sort(np.concatenate([
            np.linspace(0, temp_max_multiplier, 10).squeeze(-1),
            np.ones(1),
        ]))

        for multiplier in temp_multipliers:
            temp_df = spend_df.copy()
            temp_df.loc[spend_mask, ch] = spend_df.loc[spend_mask, ch] * multiplier
            total_spend = np.sum(temp_df.loc[spend_mask, ch].values)

            pred = model.predict(df=temp_df)
            total_outcome = np.sum(pred.loc[outcome_mask, 'prediction'].values)
            cost_curves_dict['ch'].append(ch)
            cost_curves_dict['total_spend'].append(total_spend)
            cost_curves_dict['total_outcome'].append(total_outcome)
            cost_curves_dict['multiplier'].append(multiplier)

    cost_curves = pd.DataFrame(cost_curves_dict)
    return cost_curves


def plot_cost_curves(
        cost_curves: pd.DataFrame,
        spend_scaler: float = 1e3,
        outcome_scaler: float = 1e3,
        optim_cost_curves: Optional[pd.DataFrame] = None,
) -> None:
    paid_channels = cost_curves['ch'].unique().tolist()
    n = len(paid_channels)
    nrows = math.ceil(n / 2)
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(18, nrows * 2.2))
    axes = axes.flatten()
    y_min = np.min(cost_curves['total_outcome']) / outcome_scaler * 0.75
    y_max = np.max(cost_curves['total_outcome']) / outcome_scaler * 1.5

    for idx, ch in enumerate(paid_channels):
        temp_cc = cost_curves[cost_curves['ch'] == ch].reset_index(drop=True)
        curr_spend_mask = temp_cc['multiplier'] == 1
        axes[idx].plot(
            temp_cc['total_spend'].values / spend_scaler,
            temp_cc['total_outcome'].values / outcome_scaler,
            label='current cost curve',
            color='red',
            alpha=0.8,
        )
        curr_spend = temp_cc.loc[curr_spend_mask, 'total_spend'].values / spend_scaler
        curr_outcome = temp_cc.loc[curr_spend_mask, 'total_outcome'].values / outcome_scaler

        axes[idx].scatter(
            curr_spend,
            curr_outcome,
            c='orange',
            s=48,
            label='current spend',
        )

        if optim_cost_curves is not None:
            temp_optim_cc = optim_cost_curves[cost_curves['ch'] == ch].reset_index(drop=True)
            optim_spend_mask = temp_cc['multiplier'] == 1
            axes[idx].plot(
                temp_optim_cc['total_spend'].values / spend_scaler,
                temp_optim_cc['total_outcome'].values / outcome_scaler,
                label='optim cost curve',
                color='forestgreen',
                alpha=0.8,
            )

            optim_spend = temp_optim_cc.loc[optim_spend_mask, 'total_spend'].values / spend_scaler
            optim_outcome = temp_optim_cc.loc[optim_spend_mask, 'total_outcome'].values / outcome_scaler

            axes[idx].scatter(
                optim_spend,
                optim_outcome,
                c='green',
                s=48,
                label='optim spend',
            )

        axes[idx].set_title(ch, fontdict={'fontsize': 18})
        axes[idx].grid(linestyle='--', linewidth=0.5, color='grey', alpha=0.8)
        axes[idx].set_xlabel('spend')
        axes[idx].set_ylabel('signups')
        axes[idx].xaxis.set_major_formatter('${x:1.0f}')
        axes[idx].set_ylim(y_min, y_max)

        # axes[idx].legend(loc='upper left')
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc=9, ncol=2, bbox_to_anchor=(0.5, 0), prop={'size': 18})
    fig.tight_layout()
