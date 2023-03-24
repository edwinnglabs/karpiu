# import numpy as np
# import pandas as pd
# from copy import deepcopy
# from typing import Optional, Tuple, List

# from ..utils import adstock_process, np_shift
# from ..models import MMM



# class Attributor:
#     """The class to make attribution on a state-space model in an object-oriented way; Algorithm assumes model is
#      with a Karpiu and Orbit structure

#     Attributes:
#         date_col : str
#         verbose : bool
#         attr_start : attribution start date string; if None, use minimum start date based on model provided
#         attr_end : attribution end date string; if None, use maximum end date based on model provided
#         calc_start : datetime
#             the starting datetime required from the input dataframe to support calculation; date is shifted backward
#         from attr_start according to the maximum adstock introduced
#         calc_end : datetime
#             the ending datetime required from the input dataframe to support calculation; date is shifted forward
#         from attr_end according to the maximum adstock introduced
#         max_adstock : int
#         full_regressors : list
#     """

#     def __init__(
#         self,
#         model: MMM,
#         attr_regressors: Optional[List[str]] = None,
#         start: Optional[str] = None,
#         end: Optional[str] = None,
#         df: Optional[pd.DataFrame] = None,
#         kpi_name: Optional[str] = None,
#         verbose: bool = False,
#     ):
#         """

#         Args:
#             model:
#             attr_regressors: the regressors required attribution; if None, use model.get_spend_cols()
#             start: attribution start date string; if None, use minimum start date based on model provided
#             end: attribution end date string; if None, use maximum end date based on model provided
#             df: data frame; if None, use original data based on model provided
#             kpi_name: kpi label; if None, use information from model provided
#             verbose: whether to print process information


#         Attributes:
#             date_col: date column of the core dataframe
#             verbose: control verbose
#             max_adstock: number of adstock delay i.e. max_adstock = adstock steps - 1 and max_adstock = 0
#             if no adstock exists in the model
#             full_regressors: all regressors except internal seasonal regressors
#             bkg_regressors: the background regressors not included in the required attribution regressors
#             , events and control features. it is used as the starting point to pre-calculate base component
#             for fast attribution process. Note that in general it includes non-attribution spend
#             and seasonal regressors.
#             event_regressors: event regressors
#             control_regressors: control regressors
#             attr_start: attribution start date see inputs.
#             attr_end: attribution end date. see inputs.
#             kpi_col: response column of the core data frame
#             attr_regressors: the regressors required attribution; if None, use model.get_spend_cols()
#             calc_start: the full expanded range of date range used for calculation i.e. extending with
#             max_adstock days upfront
#             calc_end: the full expanded range of date range used for calculation i.e. extending with
#             max_adstock days after attr_end
#             df_bau: the snapshot of the input data frame within the [calc_start, calc_end] range with
#             trimmed columns which are necessary for attribution process
#             dt_array: date array of df_bau
#             attr_adstock_matrix: the adstock matrix with same alignment of attr_regressors
#             attr_regressor_matrix: the regressor matrix extracted from df_bau
#             attr_sat_array: saturation array of attr_regressors
#             attr_adstock_regressor_matrix: adstock transformed matrix from attr_regressor_matrix with
#             zeros padded upfront to account lose of adstock(=max_adstock) steps
#             attr_coef_matrix: the attribution regressor coefficients extracted from the model
#             pred_zero: prediction when all attributing regressor are turned off
#         """

#         self.date_col = model.date_col
#         self.verbose = verbose

#         date_col = self.date_col
#         self.max_adstock = model.get_max_adstock()
#         # it excludes the fourier-series columns
#         self.full_regressors = model.get_regressors()
#         # FIXME: right now it DOES NOT work with including control features;
#         self.event_regressors = model.get_event_cols()
#         self.control_regressors = model.get_control_feat_cols()

#         if df is None:
#             df = model.raw_df.copy()
#         else:
#             df = df.copy()

#         if start is None:
#             self.attr_start = pd.to_datetime(
#                 df[self.date_col].values[0]
#             ) + pd.Timedelta(days=2 * self.max_adstock)
#         else:
#             self.attr_start = pd.to_datetime(start)

#         if end is None:
#             self.attr_end = pd.to_datetime(df[self.date_col].values[-1]) - pd.Timedelta(
#                 days=2 * self.max_adstock
#             )
#         else:
#             self.attr_end = pd.to_datetime(end)

#         df[date_col] = pd.to_datetime(df[date_col])

#         if kpi_name is None:
#             self.kpi_col = model.kpi_col
#         else:
#             self.kpi_col = kpi_name

#         if attr_regressors is None:
#             self.attr_regressors = model.get_spend_cols()
#         else:
#             self.attr_regressors = attr_regressors

#         # for debug
#         self.delta_matrix = None

#         # better date operations
#         # organize the dates. This pads the range with the carry over before it starts
#         self.calc_start = self.attr_start - 2 * pd.Timedelta(days=self.max_adstock)
#         self.calc_end = self.attr_end + 2 * pd.Timedelta(days=self.max_adstock)

#         if verbose:
#             print(
#                 "Full calculation start={} and end={}".format(
#                     self.calc_start.strftime("%Y-%m-%d"),
#                     self.calc_end.strftime("%Y-%m-%d"),
#                 )
#             )
#             print(
#                 "Attribution start={} and end={}".format(
#                     self.attr_start.strftime("%Y-%m-%d"),
#                     self.attr_end.strftime("%Y-%m-%d"),
#                 )
#             )

#         if self.calc_start < df[date_col].min():
#             raise Exception(
#                 "Dataframe provided starts at {} must be before {} due to max_adstock={}".format(
#                     df[date_col].iloc[0], self.calc_start, self.max_adstock
#                 )
#             )

#         if self.calc_end > df[date_col].max():
#             raise Exception(
#                 "Dataframe provided ends at {} must be after {} due to max_adstock={}".format(
#                     df[date_col].iloc[-1], self.calc_end, self.max_adstock
#                 )
#             )

#         self.calc_mask = (df[date_col] >= self.calc_start) & (
#             df[date_col] <= self.calc_end
#         )
#         self.attr_mask = (df[date_col] >= self.attr_start) & (
#             df[date_col] <= self.attr_end
#         )

#         # set a business-as-usual case data frame
#         self.df = df.copy()
#         self.dt_array = df.loc[self.calc_mask, date_col].values

#         # just make sure attr_regressors input by user align original all regressors order
#         attr_regressors_idx = [
#             idx
#             for idx in range(len(self.full_regressors))
#             if self.full_regressors[idx] in self.attr_regressors
#         ]
#         self.attr_regressors = [
#             self.full_regressors[idx] for idx in attr_regressors_idx
#         ]

#         # prepare a few matrices and arrays for rest of the calculations
#         # required matrices and arrays such as saturation, adstock, regressors matrix etc.
#         self.attr_adstock_matrix = model.get_adstock_matrix(self.attr_regressors)
#         # untransformed regressor matrix
#         # (n_steps, n_regressors)
#         self.attr_regressor_matrix = df.loc[self.calc_mask, self.attr_regressors].values
#         # (n_regressors, )
#         sat_df = model.get_saturation()
#         self.attr_sat_array = sat_df.loc[self.attr_regressors, "saturation"].values

#         # adstock_regressor_matrix dim: time x num of regressors
#         if self.max_adstock >= 1:
#             # adstock transformed regressors will be used to calculate pred_bau later
#             self.attr_adstock_regressor_matrix = adstock_process(
#                 self.attr_regressor_matrix, self.attr_adstock_matrix
#             )
#             # we lose first n(=max_adstock) observations; to maintain original dimension,
#             # we need to pad zeros n(=max_adstock) time
#             self.attr_adstock_regressor_matrix = np.concatenate(
#                 (
#                     np.zeros(
#                         (self.max_adstock, self.attr_adstock_regressor_matrix.shape[1])
#                     ),
#                     self.attr_adstock_regressor_matrix,
#                 ),
#                 0,
#             )
#         else:
#             self.attr_adstock_regressor_matrix = deepcopy(self.attr_regressor_matrix)
#         self.attr_coef_matrix = model.get_coef_matrix(
#             date_array=self.dt_array,
#             regressors=self.attr_regressors,
#         )

#         # organic, zero values baseline prediction
#         df_zero = df.copy()
#         df_zero.loc[:, self.attr_regressors] = 0.0

#         # prediction with all attr regressors turned to zero
#         # (n_steps, )
#         zero_pred_df = model.predict(df=df_zero, decompose=True)

#         # seas = zero_pred_df['weekly seasonality'].values
#         # original scale
#         # (n_steps, )
#         self.pred_zero = zero_pred_df.loc[self.calc_mask, "prediction"].values
#         # log scale
#         self.base_comp = np.log(self.pred_zero)
#         # dependent on the coefficients (can be specified by users in next step)
#         self.pred_bau = None

#         # store background target regressors spend before and after budget period due to adstock
#         # only background spend involved; turn off all spend during budget decision period
#         if self.max_adstock > 0:
#             bkg_attr_regressor_matrix = df.loc[
#                 self.calc_mask, self.attr_regressors
#             ].values
#             bkg_attr_regressor_matrix[self.max_adstock : -self.max_adstock, ...] = 0.0
#             self.bkg_attr_regressor_matrix = bkg_attr_regressor_matrix
#         else:
#             self.bkg_attr_regressor_matrix = np.zeros_like(
#                 df.loc[self.calc_mask, self.attr_regressors].values
#             )

#     def make_attribution(
#         self,
#         new_coef_name: Optional[str] = None,
#         new_coef: Optional[float] = None,
#         true_up: bool = True,
#         debug: bool = False,
#     ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
#         """
#         A general time-based attribution on stat-space model with regression. This is
#         much faster than the legacy method.


#         Parameters
#         ----------
#         new_coef_name :
#         new_coef:
#         true_up : bool
#         """
#         print("I'm here.")
#         # get the number of lags in adstock expressed as days
#         date_col = self.date_col
#         # base df for different report
#         # (n_steps, n_channels)
#         df_bau = self.df.loc[self.calc_mask,]
#         df_bau = df_bau.reset_index(drop=True)

#         # (n_steps, n_attr_regressors)
#         attr_coef_matrix = self.attr_coef_matrix.copy()
#         if new_coef is not None:
#             coef_search_found = False
#             for idx, x in enumerate(self.attr_regressors):
#                 if new_coef_name == x:
#                     coef_search_found = True
#                     attr_coef_matrix[:, idx] = new_coef
#                     break
#             if not coef_search_found:
#                 raise Exception(
#                     "New coefficient name does not match any regressors in the model."
#                 )
#         # TODO: consider update the prediction function such that the shape of the prediction
#         # is the same as the input
#         # modified coef matrix would impact this prediction vector
#         # prediction with full spend in business-as-usual case
#         # NOTE: this vector has less length (a `max_adstock` shift from the beginning)
#         varying_comp = np.sum(
#             attr_coef_matrix
#             * np.log1p(self.attr_adstock_regressor_matrix / self.attr_sat_array),
#             -1,
#         )
#         print(varying_comp.shape)
#         print(self.base_comp.shape)
#         # (n_steps, ); already tested value is the same as model.predict()
#         self.pred_bau = np.exp(self.base_comp + varying_comp)

#         if true_up:
#             true_up_array = df_bau[self.kpi_col].values
#         else:
#             true_up_array = self.pred_bau

#         (
#             activities_attr_matrix,
#             spend_attr_matrix,
#             delta_matrix,
#         ) = self.make_attribution_numpy(
#             coef_matrix=attr_coef_matrix,
#             regressor_matrix=self.attr_regressor_matrix,
#             adstock_regressor_matrix=self.attr_adstock_regressor_matrix,
#             pred_bau=self.pred_bau,
#             pred_zero=self.pred_zero,
#             adstock_matrix=self.attr_adstock_matrix,
#             saturation_array=self.attr_sat_array,
#             true_up_arr=true_up_array,
#         )

#         if debug:
#             self.delta_matrix = delta_matrix

#         # note that activities based attribution only makes sense when spend is fully settled after adstock process
#         # hence first n(=max_adstock) need to be discarded

#         # note that spend attribution only makes sense when all attributing metric fully observed in the entire
#         # adstock process
#         # also need to discard first n(=max_adstock) observation as you cannot observe the correct pred_bau
#         # hence last n(=max_adstock) need to be discarded

#         activities_attr_df = self.df.loc[self.attr_mask, [date_col]]
#         activities_attr_df[["organic"] + self.attr_regressors] = activities_attr_matrix
#         activities_attr_df = activities_attr_df.reset_index(drop=True)

#         spend_attr_df = self.df.loc[self.attr_mask, [date_col]]
#         spend_attr_df[["organic"] + self.attr_regressors] = spend_attr_matrix
#         spend_attr_df = spend_attr_df.reset_index(drop=True)

#         spend_df = self.df.loc[self.attr_mask, [date_col] + self.attr_regressors]
#         spend_df = spend_df.reset_index(drop=True)

#         cost_df = spend_df[[date_col]].copy()
#         cost_df[self.attr_regressors] = (
#             spend_df[self.attr_regressors] / spend_attr_df[self.attr_regressors]
#         )

#         return activities_attr_df, spend_attr_df, spend_df, cost_df

#     @staticmethod
#     def make_attribution_numpy(
#         coef_matrix: np.array,
#         regressor_matrix: np.array,
#         adstock_regressor_matrix: np.array,
#         pred_bau: np.array,
#         pred_zero: np.array,
#         adstock_matrix: np.array,
#         saturation_array: np.array,
#         true_up_arr: np.array,
#     ) -> Tuple[np.array, np.array]:
#         """A numpy version of making attribution

#         Notes
#         -----
#         Assuming n_steps = max_adstock + spend range + 4 * max_adstock;
#         where n_steps cover (attr_start - 2 * max_adstock, attr_end + 2 * max_adstock)

#         Parameters
#         ----------
#         coef_matrix: array in shape (n_steps, n_regressors); although probably only range(
#             max_adstock, spend range + max_adstock
#         ) is used; to preserve same index and easier readability, keep same shape as input regressor
#         regressor_matrix: array in shape (n_steps, n_regressors)
#         adstock_regressor_matrix: array in shape (n_steps, n_regressors) adstock transformed regressor
#         matrix
#         pred_bau:  (n_steps, )
#         pred_zero: (n_step, )
#         adstock_matrix: (n_regressors, max_adstock + 1)
#         saturation_array:  (n_regressors, )
#         true_up_arr: (n_steps, )
#         """

#         # a delta matrix with extra dimension (num of attr_regressor) and store the delta at calendar date view
#         # so that we do the normalization within each calendar date (not spend date)
#         # delta matrix stores the impact on calendar date by each channel spend due to each adstock impact
#         # the first channel dimension is added by an extra 1 to include organic
#         # note that this channel should not have any adstock effect simply because
#         # we don't care organic adstock; it doesn't impact paid on calculation which is the whole
#         # purpose of resolving adstock effect
#         # (n_steps = spend_range + 2 * max_adstock)
#         n_steps, n_regressors = regressor_matrix.shape
#         max_adstock = adstock_matrix.shape[1] - 1

#         # the first max_adstock steps are consumed to derive the first observed adstock transformed spend
#         # (n_steps - max_adstock, max_adstock + 1, n_regressors + 1)
#         delta_matrix = np.empty(
#             (n_steps - max_adstock, max_adstock + 1, n_regressors + 1)
#         )

#         # (n_steps - 3 * max_adstock, max_adstock + 1, n_regressors + 1)
#         paid_on_attr_matrix = np.empty(
#             (n_steps - 3 * max_adstock, max_adstock + 1, n_regressors + 1)
#         )

#         # active on can directly take organic as it does not care adstock;
#         # hence all impact contribute at first(current) time
#         if max_adstock > 0:
#             delta_matrix[:, 0, 0] = pred_zero[max_adstock:]
#         else:
#             delta_matrix[:, 0, 0] = pred_zero

#         # loop through the channels
#         for i in range(n_regressors):
#             # store the delta where row is the time spend is turned off,
#             # and column is the subsequent impact from time t (size=adstock + 1)
#             # (n_steps -max_adstock, max_adstock + 1)
#             temp_delta_matrix = np.zeros((n_steps - max_adstock, max_adstock + 1))

#             # (n_steps, )
#             temp_bau_regressor = deepcopy(regressor_matrix[:, i])
#             # (n_steps, )
#             temp_bau_regressor_adstock = deepcopy(adstock_regressor_matrix[:, i])
#             # (1, max_adstock)
#             temp_adstock_filter = np.expand_dims(deepcopy(adstock_matrix[i, :]), 0)

#             # loop over time to turn off spend; note that j here is not necessarily time prediction target at!
#             # j equals t only if adstock = 0
#             # attr end - attr start = 10
#             # max_adstock 55
#             # attr_regressor_matrix needs to be (55 * 2 + 10)
#             # add max_adstock period at the beginning for active-on
#             # add max_adstock period at the end for paid-on
#             for j in range(0, n_steps - 2 * max_adstock):
#                 # (n_steps, )
#                 temp_attr_regressor_zero = deepcopy(temp_bau_regressor)
#                 # (n_steps, 1); make it work for adstock process
#                 temp_attr_regressor_zero = np.expand_dims(temp_attr_regressor_zero, -1)
#                 # turn off spend at time j + max_adstock (cannot do j because we won't know the residuals)
#                 # we have to wait for max_adstock steps to know the residuals
#                 temp_attr_regressor_zero[j + max_adstock] = 0

#                 if max_adstock > 0:
#                     # (n_steps - max_adstock, )
#                     temp_attr_regressor_zero = np.squeeze(
#                         adstock_process(temp_attr_regressor_zero, temp_adstock_filter),
#                         -1,
#                     )
#                     # pad zeros; since both bau and set-zero condition yields the same number in the
#                     # zero-padding period, the delta ends up to be (constant x (1 - 1 / 1)) =  0
#                     # it yields the same result as legacy version.
#                     # (n_steps, )
#                     temp_attr_regressor_zero = np.concatenate(
#                         (np.zeros(max_adstock), temp_attr_regressor_zero)
#                     )
#                 else:
#                     temp_attr_regressor_zero = np.squeeze(temp_attr_regressor_zero, -1)

#                 # measure impact from j to j + max_adstock only
#                 # (max_adstock + 1, )
#                 # note that coef_matrix already skip the first max_adostock steps
#                 coef_array = coef_matrix[
#                     (j + max_adstock) : (j + 2 * max_adstock + 1), i
#                 ]

#                 # compute the delta who is the ratio between lift of bau spend and zero spend at time j
#                 # (max_adstock + 1, )
#                 numerator = (
#                     1
#                     + temp_attr_regressor_zero[
#                         (j + max_adstock) : (j + 2 * max_adstock + 1)
#                     ]
#                     / saturation_array[i]
#                 ) ** coef_array
#                 # (max_adstock + 1, )
#                 denominator = (
#                     1
#                     + temp_bau_regressor_adstock[
#                         (j + max_adstock) : (j + 2 * max_adstock + 1)
#                     ]
#                     / saturation_array[i]
#                 ) ** coef_array
#                 # delta calculation with analytical formula
#                 # (max_adstock + 1, )
#                 delta = pred_bau[(j + max_adstock) : (j + 2 * max_adstock + 1)] * (
#                     1 - numerator / denominator
#                 )
#                 # the last max_adstock is reserved for downward shifting
#                 # (n_steps - max_adstock, max_adstock + 1)
#                 temp_delta_matrix[j, 0 : len(delta)] = delta

#             # so far what does delta matrix tell us?
#             # it is the delta on each channel, each time spend turn off, its effect on each adstock
#             # however, we need to shift the adstock effect to the next day (downward) as adstock
#             # impact subsequent time, not the current time
#             # that's why we have the next step to shift the matrix down for the delta_matrix

#             # shift down the arrays by adstock effect size in j dimension; NAs / zeros padded at original place (elements get shifted)
#             # note that the first dimension is dedicated to organic in i dimension; so index from left need to be
#             # shifted one up
#             if max_adstock > 0:
#                 delta_matrix[:, :, i + 1] = np_shift(
#                     temp_delta_matrix, np.arange(max_adstock + 1)
#                 )
#             else:
#                 delta_matrix[:, :, i + 1] = temp_delta_matrix

#         # the first and last max_adstock_steps need to be discarded
#         # first max_adstock: due to adstock causing non-fully-observed
#         # last max_adstock: only the previous is the last one needed; but to keep easier tracking
#         # (n_steps - 3 * max_adstock, max_adstock + 1, n_regressors + 1)
#         if max_adstock > 0:
#             useful_delta_matrix = delta_matrix[max_adstock:-max_adstock, ...]
#         else:
#             useful_delta_matrix = deepcopy(delta_matrix)

#         # get the sum for all channels and adstock effect
#         # (n_steps - 3 * max_adstock, 1, 1)
#         total_delta = np.sum(useful_delta_matrix, axis=(-1, -2), keepdims=True)

#         # remove zeros to avoid divide-by-zero issue
#         index_zero = total_delta == 0
#         total_delta[index_zero] = 1

#         # get the normalized delta
#         # (n_steps - 3 * max_adstock, max_adstock + 1, n_regressors + 1)
#         norm_delta_matrix = useful_delta_matrix / total_delta

#         # (n_steps - 3 * max_adstock, max_adstock + 1, n_regressors + 1)
#         # the covering time is on (attr_start, attr_end + max_adstock)
#         # hence, true up array is cut by first 2 max_adstocks and last max_adostocks
#         true_up_arr_reshape = np.expand_dims(
#             true_up_arr[(2 * max_adstock) : len(true_up_arr) - max_adstock], (-1, -2)
#         )
#         full_attr_matrix = norm_delta_matrix * true_up_arr_reshape

#         # sum over lags (adstock);
#         # (n_steps - 3 * max_adstock, n_attr_regressors + 1)
#         activities_attr_matrix = np.sum(full_attr_matrix, axis=-2)
#         # for now cut last max_adstock to preserve shapes just like the attribution matrix
#         if max_adstock > 0:
#             # (n_steps - 4 * max_adstock, n_attr_regressors + 1)
#             activities_attr_matrix = activities_attr_matrix[:-max_adstock, ...]

#         ########################################################################################
#         # get the total from a channel in a day (sum over lag); this is for the paid on
#         ########################################################################################
#         # shift up arrays by lags; NAs / zeros padded at the end
#         if max_adstock > 0:
#             for idx in range(full_attr_matrix.shape[2]):
#                 paid_on_attr_matrix[:, :, idx] = np_shift(
#                     full_attr_matrix[:, :, idx],
#                     np.arange(0, -1 * (max_adstock + 1), -1),
#                 )
#             # first adstock steps are not fully observed anyway, so discard them
#             useful_paid_on_attr_matrix = paid_on_attr_matrix[max_adstock:, ...]
#         else:
#             useful_paid_on_attr_matrix = deepcopy(full_attr_matrix)

#         # sum over lags (adstock);
#         # (n_steps, n_attr_regressors + 1)
#         spend_attr_matrix = np.sum(useful_paid_on_attr_matrix, axis=-2)

#         # output delta_matrix mainly for debug
#         return activities_attr_matrix, spend_attr_matrix, useful_delta_matrix
