import math
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from typing import Optional, List, Union, Dict
from ..models import MMM
import logging

logger = logging.getLogger("karpiu-planning")

class CostCurves:
    def __init__(
        self,
        model: MMM,
        n_steps: int = 10,
        channels: Optional[Union[List[str], str]] = None,
        spend_df: Optional[pd.DataFrame] = None,
        spend_start: Optional[str] = None,
        spend_end: Optional[str] = None,
        multipliers: Optional[Union[np.array, Dict[str, np.array]]] = None,
        min_spend: float = 1., 
    ):

        if spend_df is None:
            self.spend_df = model.raw_df.copy()
        else:
            self.spend_df = spend_df

        all_channels = model.get_spend_cols()

        # generate the intersection of input and available channels
        if channels is not None and channels != 'overall':
            self.channels = list(set(all_channels).intersection(set(channels)))
        else:
            self.channels = all_channels

        self.date_col = model.date_col
        self.n_max_adstock = model.get_max_adstock()

        if spend_start is None:
            self.spend_start = pd.to_datetime(spend_df[self.date_col].values[0])
        else:
            self.spend_start = pd.to_datetime(spend_start)

        if spend_end is None:
            self.spend_end = pd.to_datetime(
                spend_df[self.date_col].values[-1]
            ) - pd.Timedelta(days=self.n_max_adstock)
        else:
            self.spend_end = pd.to_datetime(spend_end)

        self.outcome_start = self.spend_start
        self.outcome_end = self.spend_end + pd.Timedelta(days=self.n_max_adstock)
        
        if multipliers is not None:
            self.multipliers = multipliers
        else:
            self.multipliers = self.derive_multipliers()

        spend_mask = (
            (self.spend_df[self.date_col] >= spend_start) & 
            (self.spend_df[self.date_col] <= spend_end)
        )
        # outcome_mask = (spend_df[date_col] >= outcome_start) & (spend_df[date_col] <= outcome_end)

        # (n_steps, n_channels)
        spend_matrix = self.spend_df.loc[spend_mask, self.channels].values
        # it's buggy to estimate slope at zero or working with multiplier
        # hence, add a very small delta to make the analysis work
        zero_spend_flag = spend_matrix < min_spend
        if np.any(np.sum(zero_spend_flag)):
            logger.info(
                "Minimum spend threshold is hit in some channel(s). Update with value {}.".format(
                min_spend)
            )
            spend_matrix[zero_spend_flag] = min_spend
            self.spend_df.loc[spend_mask, self.channels] = spend_matrix    

    def derive_multipliers(
        self,
        max_multiplier=2.0,
    ):
        # compute flat multiplier if it is an overall cost curves
        # otherwise, compute cost curves based on max spend across all channels
        # with the max multiplier
        if self.channels == 'overall':
            self.multipliers = np.linspace(0.0, max_multiplier, self.n_steps)
        else:
            spend_mask = (
                (self.spend_df[self.date_col] >= self.spend_start) & 
                (self.spend_df[self.date_col] <= self.spend_end)
            )            
            # (n_steps, n_channels)
            spend_matrix = self.spend_df.loc[spend_mask, self.channels].values
            # (n_channels, )
            total_channel_spend = np.sum(spend_matrix, axis=0)
            multipliers_arr = np.max(total_channel_spend) / total_channel_spend
            self.multipliers = {
                k:v for (k, v) in zip(self.channels, multipliers_arr)
            }


# def generate_cost_curves(
#         model: MMM,
#         n_steps: int = 10,
#         channels: Optional[Union[List[str], str]] = None,
#         spend_df: Optional[pd.DataFrame] = None,
#         spend_start: Optional[str] = None,
#         spend_end: Optional[str] = None,
#         max_multiplier: Optional[float] = 2.0,
#         min_spend=1e-3,
# ) -> pd.DataFrame:
#     """ Generate cost curves given a Marketing Mix Model

#     Args:
#         model: fitted MMM object
#         n_steps: number of steps to estimate outcomes to generate cost curves
#         channels: list of channel names to generate cost curves
#         single numeric value, the maximum spend of a channel budget used in the simulation
#         spend_df: data frame to use in
#         spend_start: date string indicate the start of date to collect spend for simulation inclusively
#         spend_end: date string indicate the end of date to collect spend for simulation inclusively
#         max_multiplier: when generating overall cost curve only, it is the value to set the end point for simulating
#         cost curve; when generating channel specific cost curve, it only applies to the channel which has the
#         highest maximum spend across channels
#         min_spend: minimum spend of a channel per entry; otherwise, the number will be imputed by the min_spend
#     Returns:
#         cost_curves: data frame storing all result from the simulation
#     """



#     # output
#     cost_curves_dict = {
#         'ch': list(),
#         'total_spend': list(),
#         'total_outcome': list(),
#         'multiplier': list(),
#     }



#     # create a case with all spend in the spend range set to zero to estimate organic
#     # note that it doesn't include past spend as it already happens
#     temp_df = spend_df.copy()
#     temp_df.loc[spend_mask, paid_channels] = 0.
#     pred = model.predict(df=temp_df)
#     total_outcome = np.sum(pred.loc[outcome_mask, 'prediction'].values)
#     cost_curves_dict['ch'].append('organic')
#     cost_curves_dict['total_spend'].append(0.)
#     cost_curves_dict['total_outcome'].append(total_outcome)
#     cost_curves_dict['multiplier'].append(1.)

#     # decide to compute overall cost curves or channel level
#     if channels == 'overall':
#         temp_multipliers = np.linspace(0.0, max_multiplier, n_steps)

#         for multiplier in temp_multipliers:
#             temp_df = spend_df.copy()
#             temp_df.loc[spend_mask, paid_channels] = spend_matrix * multiplier
#             pred = model.predict(df=temp_df)
#             total_spend = np.sum(spend_matrix * multiplier)
#             total_outcome = np.sum(pred.loc[outcome_mask, 'prediction'].values)
#             cost_curves_dict['ch'].append('overall')
#             cost_curves_dict['total_spend'].append(total_spend)
#             cost_curves_dict['total_outcome'].append(total_outcome)
#             cost_curves_dict['multiplier'].append(multiplier)

#     else:
#         # (n_channels, )
#         total_spend_arr = np.sum(spend_matrix, axis=0)
#         # use overall spend range to determine simulation scenarios
#         spend_summary = pd.DataFrame(paid_channels, columns=['channel'])
#         spend_summary['total_spend'] = total_spend_arr

#         max_spend = np.max(spend_summary['total_spend']) / spend_summary['total_spend']
#         spend_summary['max_multiplier'] = max_spend * max_multiplier

#         for ch in tqdm(paid_channels):
#             # multiplier always contains 1.0 to indicate current spend
#             # multiplier is derived based on ratio of the maximum
#             temp_max_multiplier = spend_summary.loc[spend_summary['channel'] == ch, 'max_multiplier'].values
#             temp_multipliers = np.sort(np.concatenate([
#                 np.linspace(0, temp_max_multiplier, n_steps).squeeze(-1),
#                 np.ones(1),
#             ]))
#             # remove duplicates
#             temp_multipliers = np.unique(temp_multipliers)

#             for multiplier in temp_multipliers:
#                 temp_df = spend_df.copy()
#                 temp_df.loc[spend_mask, ch] = spend_df.loc[spend_mask, ch] * multiplier
#                 total_spend = np.sum(temp_df.loc[spend_mask, ch].values)

#                 pred = model.predict(df=temp_df)
#                 total_outcome = np.sum(pred.loc[outcome_mask, 'prediction'].values)
#                 cost_curves_dict['ch'].append(ch)
#                 cost_curves_dict['total_spend'].append(total_spend)
#                 cost_curves_dict['total_outcome'].append(total_outcome)
#                 cost_curves_dict['multiplier'].append(multiplier)

#     cost_curves = pd.DataFrame(cost_curves_dict)
#     return cost_curves


# def plot_cost_curves(
#         cost_curves: pd.DataFrame,
#         spend_scaler: float = 1e3,
#         outcome_scaler: float = 1e3,
#         optim_cost_curves: Optional[pd.DataFrame] = None,
#         plot_margin: float = 0.05,
# ) -> None:
#     paid_channels = cost_curves['ch'].unique().tolist()
#     paid_channels = list(set(paid_channels) - set('organic'))
#     n = len(paid_channels)
#     nrows = math.ceil(n / 2)

#     y_min = np.min(cost_curves['total_outcome'].values) / outcome_scaler
#     y_max = np.max(cost_curves['total_outcome'].values) / outcome_scaler
#     x_max = np.max(cost_curves['total_spend'].values) / spend_scaler

#     if optim_cost_curves is not None:
#         y_min2 = np.min(optim_cost_curves['total_outcome']) / outcome_scaler
#         y_max2 = np.max(optim_cost_curves['total_outcome']) / outcome_scaler
#         y_min = min(y_min, y_min2)
#         y_max = max(y_max, y_max2)

#     organic_outcome = cost_curves.loc[cost_curves['ch'] == 'organic', 'total_outcome'].values
#     if len(paid_channels) > 2:
#         # mulitple cost curves
#         fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(18, nrows * 2.2))
#         axes = axes.flatten()

#         for idx, ch in enumerate(paid_channels):
#             temp_cc = cost_curves[cost_curves['ch'] == ch].reset_index(drop=True)
#             curr_spend_mask = temp_cc['multiplier'] == 1
#             axes[idx].plot(
#                 temp_cc['total_spend'].values / spend_scaler,
#                 temp_cc['total_outcome'].values / outcome_scaler,
#                 label='current cost curve',
#                 color='red',
#                 alpha=0.8,
#             )
#             curr_spend = temp_cc.loc[curr_spend_mask, 'total_spend'].values / spend_scaler
#             curr_outcome = temp_cc.loc[curr_spend_mask, 'total_outcome'].values / outcome_scaler

#             axes[idx].scatter(
#                 curr_spend,
#                 curr_outcome,
#                 c='orange',
#                 s=48,
#                 label='current spend',
#             )

#             axes[idx].axhline(y=organic_outcome / spend_scaler, linestyle='dashed', label='organic')

#             if optim_cost_curves is not None:
#                 temp_optim_cc = optim_cost_curves[cost_curves['ch'] == ch].reset_index(drop=True)
#                 optim_spend_mask = temp_optim_cc['multiplier'] == 1
#                 axes[idx].plot(
#                     temp_optim_cc['total_spend'].values / spend_scaler,
#                     temp_optim_cc['total_outcome'].values / outcome_scaler,
#                     label='optim cost curve',
#                     color='forestgreen',
#                     alpha=0.8,
#                 )

#                 optim_spend = temp_optim_cc.loc[optim_spend_mask, 'total_spend'].values / spend_scaler
#                 optim_outcome = temp_optim_cc.loc[optim_spend_mask, 'total_outcome'].values / outcome_scaler

#                 axes[idx].scatter(
#                     optim_spend,
#                     optim_outcome,
#                     c='green',
#                     s=48,
#                     label='optim spend',
#                 )

#             axes[idx].set_title(ch, fontdict={'fontsize': 18})
#             axes[idx].grid(linestyle='dotted', linewidth=0.7, color='grey', alpha=0.8)
#             axes[idx].set_xlabel('spend')
#             axes[idx].set_ylabel('signups')
#             axes[idx].xaxis.set_major_formatter('${x:1.0f}')
#             axes[idx].set_ylim(y_min * (1 - plot_margin), y_max * (1 + plot_margin))
#             axes[idx].set_xlim(left=0., right=x_max)

#             # axes[idx].legend(loc='upper left')
#         handles, labels = axes[-1].get_legend_handles_labels()
#         fig.legend(handles, labels, loc=9, ncol=2, bbox_to_anchor=(0.5, 0), prop={'size': 18})
#         fig.tight_layout()
#     else:
#         # single cost curve
#         fig, ax = plt.subplots(1, 1, figsize=(18, 12))
#         temp_cc = cost_curves[cost_curves['ch'] == 'overall'].reset_index(drop=True)
#         curr_spend_mask = temp_cc['multiplier'] == 1
#         ax.plot(
#             temp_cc['total_spend'].values / spend_scaler,
#             temp_cc['total_outcome'].values / outcome_scaler,
#             label='current cost curve',
#             color='red',
#             alpha=0.8,
#         )
#         curr_spend = temp_cc.loc[curr_spend_mask, 'total_spend'].values / spend_scaler
#         curr_outcome = temp_cc.loc[curr_spend_mask, 'total_outcome'].values / outcome_scaler

#         ax.scatter(
#             curr_spend,
#             curr_outcome,
#             c='orange',
#             s=48,
#             label='current spend',
#         )

#         ax.axhline(y=organic_outcome / spend_scaler, linestyle='dashed', label='organic')
#         ax.set_xlim(left=0., right=x_max)


# def calculate_marginal_cost(
#         model: MMM,
#         channels: List[str],
#         spend_start: str,
#         spend_end: str,
#         spend_df: Optional[pd.DataFrame] = None,
#         delta: float = 1e-7,
# ) -> pd.DataFrame:
#     """ Generate overall marginal cost per channel with given period [spend_start, spend_end]
#     Args:
#         model:
#         channels:
#         spend_df:
#         spend_start:
#         spend_end:
#         delta:

#     Returns:

#     """
#     if spend_df is None:
#         df = model.raw_df.copy()
#     else:
#         df = spend_df.copy()

#     date_col = model.date_col
#     max_adstock = model.get_max_adstock()
#     full_regressors = model.get_regressors()
#     event_regressors = model.get_event_cols()
#     control_regressors = model.get_control_feat_cols()
#     sat_df = model.get_saturation()

#     spend_start = pd.to_datetime(spend_start)
#     spend_end = pd.to_datetime(spend_end)
#     mea_start = spend_start
#     mea_end = spend_end + pd.Timedelta(days=max_adstock)
#     calc_start = spend_start - pd.Timedelta(days=max_adstock)
#     calc_end = spend_end + pd.Timedelta(days=max_adstock)

#     spend_mask = (df[date_col] >= spend_start) & (df[date_col] <= spend_end)
#     mea_mask = (df[date_col] >= mea_start) & (df[date_col] <= mea_end)
#     calc_mask = (df[date_col] >= calc_start) & (df[date_col] <= calc_end)

#     dummy_pred_df = model.predict(df=df, decompose=True)
#     # log scale (mea_steps, )
#     trend = dummy_pred_df.loc[mea_mask, 'trend'].values
#     # seas = dummy_pred_df.loc[mea_mask, 'weekly seasonality'].values
#     # base_comp = trend + seas
#     base_comp = trend

#     # background regressors
#     bg_regressors = list(
#         set(full_regressors) - set(channels) - set(event_regressors) - set(control_regressors)
#     )

#     if len(bg_regressors) > 0:
#         # (n_regressors, )
#         bg_coef_array = model.get_coef_vector(regressors=bg_regressors)
#         # (n_regressors, )
#         bg_sat_array = sat_df.loc[bg_regressors, 'saturation'].values
#         # (calc_steps, n_regressors)
#         bg_regressor_matrix = df.loc[calc_mask, bg_regressors].values
#         bg_adstock_filter_matrix = model.get_adstock_matrix(bg_regressors)
#         # (mea_steps, n_regressors)
#         bg_adstock_regressor_matrix = adstock_process(
#             bg_regressor_matrix,
#             bg_adstock_filter_matrix,
#         )

#         base_comp += np.sum(
#             bg_coef_array * np.log1p(bg_adstock_regressor_matrix / bg_sat_array),
#             -1,
#         )

#     if len(event_regressors) > 0:
#         event_coef_array = model.get_coef_vector(regressors=event_regressors)
#         # (mea_steps, n_regressors)
#         event_regressor_matrix = df.loc[mea_mask, event_regressors].values
#         base_comp += np.sum(event_coef_array * event_regressor_matrix, -1)

#     if len(control_regressors) > 0:
#         control_coef_array = model.get_coef_vector(regressors=control_regressors)
#         # (mea_steps, n_regressors)
#         control_regressor_matrix = np.log1p(df.loc[mea_mask, control_regressors].values)
#         base_comp += np.sum(control_coef_array * control_regressor_matrix, -1)

#     # base_comp calculation finished above
#     # the varying comp is computed below
#     attr_regressor_matrix = df.loc[calc_mask, channels].values
#     attr_coef_array = model.get_coef_vector(regressors=channels)
#     attr_sat_array = sat_df.loc[channels, 'saturation'].values
#     attr_adstock_matrix = model.get_adstock_matrix(channels)
#     attr_adstock_regressor_matrix = adstock_process(
#         attr_regressor_matrix,
#         attr_adstock_matrix
#     )
#     # log scale
#     attr_comp = np.sum(
#         attr_coef_array * np.log1p(attr_adstock_regressor_matrix / attr_sat_array),
#         -1,
#     )
#     mcac = np.empty(len(channels))
#     for idx, ch in enumerate(channels):
#         # (calc_steps, n_regressors)
#         delta_matrix = np.zeros_like(attr_regressor_matrix)
#         delta_matrix[max_adstock:-max_adstock, idx] = delta
#         # (calc_steps, n_regressors)
#         new_attr_regressor_matrix = attr_regressor_matrix + delta_matrix
#         new_attr_adstock_regressor_matrix = adstock_process(
#             new_attr_regressor_matrix,
#             attr_adstock_matrix
#         )
#         new_attr_comp = np.sum(
#             attr_coef_array * np.log1p(new_attr_adstock_regressor_matrix / attr_sat_array),
#             -1,
#         )

#         m_acq = np.exp(base_comp) * (np.exp(new_attr_comp) - np.exp(attr_comp))
#         mcac[idx] = np.sum(delta_matrix) / np.sum(m_acq)

#     return pd.DataFrame({
#         "regressor": channels,
#         "mcac": mcac,
#     }).set_index("regressor")
