import scipy.optimize as optim
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from typing import Optional, List, Union
import math
import matplotlib.pyplot as plt
import logging
from .models import MMM

logger = logging.getLogger("karpiu-planning")


class Optimizer:
    """Perform optimization with a given Marketing Mix Model"""


def generate_cost_curves(
        model: MMM,
        n_steps: int = 10,
        channels: Union[Optional[List[str]], str] = None,
        max_spend: Optional[float] = None,
        spend_df: Optional[pd.DataFrame] = None,
        spend_start: Optional[str] = None,
        spend_end: Optional[str] = None,
) -> pd.DataFrame:
    """ Generate cost curves given a Marketing Mix Model

    Args:
        model: fitted MMM object
        n_steps: number of steps to estimate outcomes to generate cost curves
        channels: list of channel names to generate cost curves
        max_spend: single integer, the maximum spend of a channel budget used in the simulation
        spend_df: data frame to use in
        spend_start: date string indicate the start of date to collect spend for simulation inclusively
        spend_end: date string indicate the end of date to collect spend for simulation inclusively

    Returns:
        cost_curves: data frame storing all result from the simulation
    """
    if spend_df is None:
        spend_df = model.raw_df.copy()

    paid_channels = model.get_spend_cols()

    if channels is not None and channels != 'overall':
        paid_channels = set(paid_channels).union(set(channels))

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

    # create a case with all spend set to zero to estimate organic
    temp_df = spend_df.copy()
    temp_df.loc[:, paid_channels] = 0.
    pred = model.predict(df=temp_df)
    total_outcome = np.sum(pred.loc[outcome_mask, 'prediction'].values)
    cost_curves_dict['ch'].append('organic')
    cost_curves_dict['total_spend'].append(0.)
    cost_curves_dict['total_outcome'].append(total_outcome)
    cost_curves_dict['multiplier'].append(1.)

    # decide to compute overall cost curves or channel level
    if channels == 'overall':

        # (n_steps, n_channels)
        spend_matrix = spend_df.loc[spend_mask, paid_channels].values
        temp_multipliers = np.linspace(1e-3, 100.0, n_steps)

        for multiplier in temp_multipliers:
            temp_df = spend_df.copy()
            temp_df.loc[spend_mask, paid_channels] = spend_matrix * multiplier
            pred = model.predict(df=temp_df)
            total_spend = np.sum(spend_matrix * multiplier)
            total_outcome = np.sum(pred.loc[outcome_mask, 'prediction'].values)
            cost_curves_dict['ch'].append('overall')
            cost_curves_dict['total_spend'].append(total_spend)
            cost_curves_dict['total_outcome'].append(total_outcome)
            cost_curves_dict['multiplier'].append(multiplier)

    else:
        # (n_steps, n_channels)
        spend_matrix = spend_df.loc[spend_mask, paid_channels].values
        # (n_channels, )
        total_spend_arr = np.sum(spend_matrix, axis=0)
        zero_spend_flag = total_spend_arr < 1e-3
        if sum(zero_spend_flag) > 0:
            logger.info("Zero spend of a channel detected. Impute with value 1e-3.")
            spend_matrix[:, zero_spend_flag] = 1e-3
            spend_df.loc[spend_mask, paid_channels] = spend_matrix
            total_spend_arr = np.sum(spend_matrix, axis=0)
            # temp_df = spend_df.copy()
            # temp_df.loc[spend_mask, paid_channels] = spend_df.loc[spend_mask, paid_channels] * multiplier

        # use overall spend range to determine simulation scenarios
        spend_summary = pd.DataFrame(paid_channels, columns=['channel'])
        spend_summary['total_spend'] = total_spend_arr

        if max_spend is not None:
            spend_summary['max_multiplier'] = max_spend / spend_summary['total_spend']
        else:
            spend_summary['max_multiplier'] = np.max(spend_summary['total_spend']) / spend_summary['total_spend']

        for ch in tqdm(paid_channels):
            # multiplier always contains 1.0 to indicate current spend
            # multiplier is derived based on ratio of the maximum
            temp_max_multiplier = spend_summary.loc[spend_summary['channel'] == ch, 'max_multiplier'].values
            temp_multipliers = np.sort(np.concatenate([
                np.linspace(0, temp_max_multiplier, n_steps).squeeze(-1),
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
    paid_channels = list(set(paid_channels) - set('organic'))
    n = len(paid_channels)
    nrows = math.ceil(n / 2)

    y_min = 0.0
    y_max = np.max(cost_curves['total_outcome']) / outcome_scaler
    x_max = np.max(cost_curves['total_spend'].values) / spend_scaler

    if optim_cost_curves is not None:
        # y_min2 = np.min(optim_cost_curves['total_outcome']) / outcome_scaler
        y_max2 = np.max(optim_cost_curves['total_outcome']) / outcome_scaler
        # y_min = min(y_min, y_min2)
        y_max = max(y_max, y_max2)

    organic_outcome = cost_curves.loc[cost_curves['ch'] == 'organic', 'total_outcome'].values

    if len(paid_channels) > 2:
        fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(18, nrows * 2.2))
        axes = axes.flatten()

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

            axes[idx].axhline(y=organic_outcome / spend_scaler, linestyle='dashed', label='organic')

            if optim_cost_curves is not None:
                temp_optim_cc = optim_cost_curves[cost_curves['ch'] == ch].reset_index(drop=True)
                optim_spend_mask = temp_optim_cc['multiplier'] == 1
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
            axes[idx].grid(linestyle='dotted', linewidth=0.7, color='grey', alpha=0.8)
            axes[idx].set_xlabel('spend')
            axes[idx].set_ylabel('signups')
            axes[idx].xaxis.set_major_formatter('${x:1.0f}')
            axes[idx].set_ylim(y_min * 0.95, y_max * 1.025)
            axes[idx].set_xlim(left=0., right=x_max)

            # axes[idx].legend(loc='upper left')
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc=9, ncol=2, bbox_to_anchor=(0.5, 0), prop={'size': 18})
        fig.tight_layout()
    else:
        fig, ax = plt.subplots(1, 1, figsize=(18, 12))
        temp_cc = cost_curves[cost_curves['ch'] == 'overall'].reset_index(drop=True)
        curr_spend_mask = temp_cc['multiplier'] == 1
        ax.plot(
            temp_cc['total_spend'].values / spend_scaler,
            temp_cc['total_outcome'].values / outcome_scaler,
            label='current cost curve',
            color='red',
            alpha=0.8,
        )
        curr_spend = temp_cc.loc[curr_spend_mask, 'total_spend'].values / spend_scaler
        curr_outcome = temp_cc.loc[curr_spend_mask, 'total_outcome'].values / outcome_scaler

        ax.scatter(
            curr_spend,
            curr_outcome,
            c='orange',
            s=48,
            label='current spend',
        )

        ax.axhline(y=organic_outcome / spend_scaler, linestyle='dashed', label='organic')
