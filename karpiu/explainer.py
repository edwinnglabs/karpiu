import numpy as np
import pandas as pd
from copy import deepcopy
from .utils import adstock_process, np_shift
from typing import Optional, List, Dict

from .models import MMM


class Attributor(object):
    """The class to make attribution on a state-space model in an object-oriented way; Algorithm assumes model is
     with a Karpiu and Orbit structure

    Attributes:
        date_col : str
        verbose : bool
        attr_start : datetime
        attr_end : datetime
        calc_start : datetime
            the starting datetime required from the input dataframe to support calculation; date is shifted backward
        from attr_start according to the maximum adstock introduced
        calc_end : datetime
            the ending datetime required from the input dataframe to support calculation; date is shifted forward
        from attr_end according to the maximum adstock introduced
        max_adstock : int
        full_regressors : list
    """

    def __init__(
            self,
            model: MMM,
            start: Optional[str] = None,
            end: Optional[str] = None,
            df: Optional[pd.DataFrame] = None,
            kpi_name: Optional[str] = None,
            verbose: bool = False,
    ):
        """

        Args:
            model:
            start: attribution start date string; if None, use minimum start date based on model provided
            end: attribution end date string; if None, use maximum end date based on model provided
            df: data frame; if None, use original data based on model provided
            kpi_name: kpi label; if None, use information from model provided
            verbose: whether to print process information
        """

        self.date_col = model.date_col
        self.verbose = verbose

        date_col = self.date_col
        self.max_adstock = model.get_max_adstock()
        self.adstock_matrix = model.get_adstock_matrix()
        # excluding the fourier-series columns
        self.full_regressors = model.get_regressors()

        # FIXME: keep model object for debug only
        self.model = deepcopy(model)

        if df is None:
            df = model.raw_df.copy()
        else:
            df = df.copy()

        if start is None:
            self.attr_start = pd.to_datetime(df[self.date_col].values[0])
        else:
            self.attr_start = pd.to_datetime(start)

        if end is None:
            self.attr_end = pd.to_datetime(df[self.date_col].values[-1])
        else:
            self.attr_end = pd.to_datetime(end)

        df[date_col] = pd.to_datetime(df[date_col])

        if kpi_name is None:
            self.kpi_col = model.kpi_col
        else:
            self.kpi_col = kpi_col

        self.attr_regressors = model.get_spend_cols()

        # better date operations
        # organize the dates. This pads the range with the carry over before it starts
        self.calc_start = self.attr_start + pd.Timedelta(days=-self.max_adstock)
        self.calc_end = self.attr_end + pd.Timedelta(days=self.max_adstock)

        if verbose:
            print('Full calculation start={} and end={}'.format(
                self.calc_start.strftime('%Y-%m-%d'), self.calc_end.strftime('%Y-%m-%d')))
            print('Attribution start={} and end={}'.format(
                self.attr_start.strftime('%Y-%m-%d'), self.attr_end.strftime('%Y-%m-%d')))

        if self.calc_start < df[date_col].min():
            raise Exception('Dataframe provided starts at {} must be before {} due to max_adstock={}'.format(
                df[date_col].iloc[0], self.calc_start, self.max_adstock))

        if self.calc_end > df[date_col].max():
            raise Exception('Dataframe provided ends at {} must be after {} due to max_adstock={}'.format(
                df[date_col].iloc[-1], self.calc_end, self.max_adstock))
        # set a business-as-usual case data frame
        self.df_bau = df.loc[
            (df[date_col] >= self.calc_start) & (df[date_col] <= self.calc_end),
            [date_col, self.kpi_col] + self.full_regressors
        ].reset_index(drop=True)

        # prepare a few matrices and arrays for rest of the calculations
        # required matrices and arrays such as saturation, adstock, regressors matrix etc.
        # just make sure attr_regressors input by user align original all regresors order
        attr_regressors_idx = [
            idx for idx in range(len(self.full_regressors)) if self.full_regressors[idx] in self.attr_regressors
        ]
        self.attr_regressors = [self.full_regressors[idx] for idx in attr_regressors_idx]

        # untransformed regressor matrix
        # attr_regressor_matrix dim: time x num of regressor
        self.attr_regressor_matrix = self.df_bau[self.attr_regressors].values

        # saturation_array dim: 1 x num of regressor
        self.saturation_array = np.expand_dims(
            model.get_saturation().loc[self.attr_regressors, 'saturation'].values,
            0
        )

        # adstock_regressor_matrix dim: time x num of regressors
        if self.max_adstock >= 1:
            # adstock transformed regressors will be used to calculate pred_bau later
            self.adstock_regressor_matrix = adstock_process(
                self.attr_regressor_matrix, self.adstock_matrix
            )
            # we lose first n(=max_adstock) observations; to maintain original dimension, we need to pad zeros n(=max_adstock) time
            self.adstock_regressor_matrix = np.concatenate((
                np.zeros((self.max_adstock, self.adstock_regressor_matrix.shape[1])),
                self.adstock_regressor_matrix
            ), 0)
        else:
            self.adstock_regressor_matrix = deepcopy(self.attr_regressor_matrix)

        self.coef_matrix = model.get_coef_matrix(
            self.attr_regressors,
            date_array=self.df_bau[date_col],
        )

        # organic, zero values baseline prediction
        self.df_zero = self.df_bau.copy()
        self.df_zero.loc[:, self.attr_regressors] = 0.0
        # prediction with all predictors turned to zero
        pred_zero = model.predict(df=self.df_zero)['prediction'].values

        # NOTE: this vector has less length (a `max_adstock` shift from the beginning)
        if self.max_adstock > 0:
            # pad zeros due to adstock
            self.pred_zero = np.concatenate((
                np.zeros(self.max_adstock), pred_zero
            ))
        else:
            self.pred_zero = pred_zero

        # dependent on the coefficients (can be specified by users in next step)
        self.pred_bau = None

    def make_attribution(self, new_coef_name=None, new_coef=None, true_up=True):
        """
        A general time-based attribution on stat-space model with regression. This is 
        much faster than the legacy method.


        Parameters
        ----------
        new_coef : tuple
        true_up : bool
        """
        # get the number of lags in adstock expressed as days
        date_col = self.date_col
        df_bau = self.df_bau.copy()
        kpi_col = self.kpi_col

        coef_matrix = self.coef_matrix.copy()

        if new_coef is not None:
            coef_search_found = False
            for idx, x in enumerate(self.attr_regressors):
                if new_coef_name == x:
                    coef_search_found = True
                    coef_matrix[:, idx] = new_coef
                    break
            if not coef_search_found:
                raise Exception("New coefficient name does not match any regressors in the model.")

        # TODO: consider update the prediction function such that the shape of the prediction is the same as the input
        # modified coef matrix would impact this prediction vector
        # prediction with full spend in business-as-usual case
        # NOTE: this vector has less length (a `max_adstock` shift from the beginning)

        # adstock_regressor_matrix dim: time x num of regressors
        # saturation_array dim: 1 x num of regressor

        # FIXME: how come they look different?
        # FIXME: the difference is small; we can re-examine this later
        pred_bau = self.model.predict(df_bau)['prediction'].values
        if self.max_adstock > 0:
            # pad zeros due to adstock
            self.pred_bau = np.concatenate((
                np.zeros(self.max_adstock), pred_bau
            ))

        else:
            self.pred_bau = pred_bau

        # a delta matrix with extra dimension (num of attr_regressor) and store the delta at calendar date view
        # so that we do the normalization within each calendar date (not spend date)
        # delta matrix stores the impact on calendar date by each channel spend due to each adstock impact
        # the first channel dimension is added by an extra 1 to include organic
        # note that this channel should not have any adstock effect simply because
        # we don't care organic adstock; it doesn't impact paid on calculation which is the whole
        # purpose of resolving adstock effect
        delta_matrix = np.zeros((
            self.attr_regressor_matrix.shape[0],
            self.max_adstock + 1,
            len(self.attr_regressors) + 1
        ))

        # a same size of matrix declared to prepare paid date dimension
        paid_on_attr_matrix = np.zeros(delta_matrix.shape)

        # active on can directly take organic as it does not care adstock;
        # hence all impact contribute at first(current) time
        delta_matrix[:, 0, 0] = self.pred_zero

        # loop through the channels
        for i, c in enumerate(self.attr_regressors):
            # store the delta where row is the time spend is turned off,
            # and column is the subsequent impact from time t (size=adstock + 1)
            temp_delta_matrix = np.zeros((self.attr_regressor_matrix.shape[0], self.max_adstock + 1))

            # time x num of regressors
            temp_bau_regressor = deepcopy(self.attr_regressor_matrix[:, i])
            # time x num of regressors
            temp_bau_regressor_adstock = deepcopy(self.adstock_regressor_matrix[:, i])
            # 1 x num of adstock 
            temp_adstock_filter = np.expand_dims(deepcopy(self.adstock_matrix[i, :]), 0)

            # loop over time to turn off spend; note that j here is not necessarily time prediction target at!
            # j equals t only if adstock = 0 
            # attr end - attr start = 10
            # max_adstock 55
            # attr_regressor_matrix needs to be (55 * 2 + 10)
            # add max_adstock period at the beginning for active-on
            # add max_adstock period at the end for paid-on
            for j in range(0, self.attr_regressor_matrix.shape[0]):
                # time x num of regressors
                temp_attr_regressor_zero = deepcopy(temp_bau_regressor)
                # time x num of regressors x 1
                temp_attr_regressor_zero = np.expand_dims(temp_attr_regressor_zero, -1)
                # turn off spend at time j
                temp_attr_regressor_zero[j] = 0
                # time x num of regressors
                temp_attr_regressor_zero = np.squeeze(
                    adstock_process(temp_attr_regressor_zero, temp_adstock_filter),
                    -1
                )
                # pad zeros; since both bau and set-zero condition yields the same number in the
                # zero-padding period, the delta ends up to be (constant x (1 - 1 / 1)) =  0
                # it yields the same result as legacy version.
                if self.max_adstock > 0:
                    temp_attr_regressor_zero = np.concatenate((
                        np.zeros(self.max_adstock),
                        temp_attr_regressor_zero
                    ))

                # measure impact from j to j + max_adstock only
                coef_array = coef_matrix[j:(j + self.max_adstock + 1), i]
                # compute the delta who is the ratio between lift of bau spend and zero spend at time j
                numerator = (1 + temp_attr_regressor_zero[j:(j + self.max_adstock + 1)] /
                             self.saturation_array[:, i]) ** coef_array
                denominator = (1 + temp_bau_regressor_adstock[j:(j + self.max_adstock + 1)] /
                               self.saturation_array[:, i]) ** coef_array

                delta = self.pred_bau[j:(j + self.max_adstock + 1)] * (1 - numerator / denominator)
                temp_delta_matrix[j, 0:len(delta)] = delta

            # so far what does delta matrix tell us?
            # it is the delta on each channel, each time spend turn off, its effect on each adstock
            # however, we need to shift the adstock effect to the next day (downward) as adstock
            # impact  subsequent time, not the current time
            # that's why we have the next step to shift the matrix down for the delta_matrix

            # shift down the arrays by adstock effect size in j dimension; NAs / zeros padded at original place (elements get shifted)
            # note that the first dimension is dedicated to organic in i dimension; so index from left need to be 
            # shifted one up
            if self.max_adstock > 0:
                delta_matrix[:, :, i + 1] = np_shift(temp_delta_matrix, np.arange(self.max_adstock + 1))
            else:
                delta_matrix[:, :, i + 1] = temp_delta_matrix

        # get the sum for all channels and adstock effect
        total_delta = np.sum(delta_matrix, axis=(-1, -2), keepdims=True)

        # remove zeros to avoid divide-by-zero issue
        index_zero = total_delta == 0
        total_delta[index_zero] = 1

        # get the normalized delta
        norm_delta_matrix = delta_matrix / total_delta

        # FIXME: assuming true up now; applying on kpi provided
        # full_attr_matrix is on activities (not spend) date view
        # this is the most important matrix in time x adstock x num of regressors
        # which represents the most granular attribution
        full_attr_matrix = norm_delta_matrix * df_bau[kpi_col].values.reshape(-1, 1, 1)

        # sum over lags; reduce dimension to time x num of regressors
        activities_attr_matrix = np.sum(full_attr_matrix, axis=1)

        ########################################################################################
        # get the total from a channel in a day (summed over lag); this is for the paid on
        ########################################################################################
        # shift up arrays by lags; NAs / zeros padded at the end
        if self.max_adstock > 0:
            for idx in range(full_attr_matrix.shape[2]):
                paid_on_attr_matrix[:, :, idx] = np_shift(
                    full_attr_matrix[:, :, idx], np.arange(0, -1 * (self.max_adstock + 1), -1))
        else:
            paid_on_attr_matrix = deepcopy(full_attr_matrix)

        # remove first max_adstock rows
        # sum over lags; reduce dimension to time x attr_regressor
        spend_attr_matrix = np.sum(paid_on_attr_matrix, axis=1)
        # paid on and active on should be the same here for organic
        # assert np.allclose(activities_attr_matrix[:, 0], spend_attr_matrix[:, 0])

        # note that activities attribution only makes sense when spend is fully sentled after adstock process
        # hence first n(=max_adstock) need to be be disgard
        activities_attr_df = df_bau[self.max_adstock:].reset_index(drop=True)
        activities_attr_df = activities_attr_df[[date_col]]
        activities_attr_df[['organic'] + self.attr_regressors] = activities_attr_matrix[self.max_adstock:]

        # note that spend attribution only makes sense when all attributing metric fully observed in the entire
        # adstock process
        # also need to disgard first n(=max_adstock) observation as you cannot observe the correct pred_bau
        # hence last n(=max_adstock) need to be be disgard
        spend_attr_df = df_bau[self.max_adstock:-self.max_adstock].reset_index(drop=True)
        spend_attr_df = spend_attr_df[[date_col]]
        spend_attr_df[['organic'] + self.attr_regressors] = spend_attr_matrix[self.max_adstock:-self.max_adstock]

        spend_df = df_bau[self.max_adstock:-self.max_adstock].reset_index(drop=True)
        spend_df = spend_df[[date_col] + self.attr_regressors]

        cost_df = spend_df[[date_col]].copy()
        cost_df[self.attr_regressors] = (
                spend_df[self.attr_regressors] / spend_attr_df[self.attr_regressors]
        )

        return activities_attr_df, spend_attr_df, spend_df, cost_df

# def extract_adstock_matrix(adstock_df, regressor):
#    """return a adstock-filter matrix for machine computation ready 1D-convolution"""
#    # get intersected features
#    adstock_regressors = adstock_df['subchannel'].unique()
#    # problematic
#    # adstock_regressors1 = list(set(regressor) & set(adstock_regressors))
#    adstock_regressors = [x for x in regressor if x in adstock_regressors]

#    adstock_index_df = adstock_df.set_index('subchannel')
#    adstock_matrix = np.array(
#        adstock_index_df.loc[adstock_regressors,
#                             adstock_index_df.columns.str.startswith('lag')].values, dtype=np.float64
#    )

#    if len(adstock_regressors) == len(regressor):
#        return adstock_matrix
#    else:
#        adstock_matrix_expand = np.zeros((len(regressor), adstock_matrix.shape[1]))
#        k = 0
#        for idx, x in enumerate(regressor):
#            if x in adstock_regressors:
#                adstock_matrix_expand[idx, :] = adstock_matrix[k, :]
#                k += 1
#            else:
#                dummy_filter = np.zeros(adstock_matrix.shape[1])
#                dummy_filter[0] = 1.0
#                adstock_matrix_expand[idx, :] = dummy_filter
#        return adstock_matrix_expand


# deprecated
# def make_attr(model, df=None, regressors=None, match_actual=True, return_df=False):
#     """ A MVP version that provides decomposition on activities due to spend. However, this approach
#     only provides calendar date view but not spend date view when there is an adstock in the model.
#
#     Notes
#     -----
#     1. Propose regressors set $x1, x2, \cdots$ to be explained
#     2. Make baseline prediction with all regressors original values as $y_\text{full}$
#     3. Iterate $x_i$ from $i$ in $1$ to $I$
#     - store the prediction as $y_i$
#     - derive the "one-off" delta as $\delta_i = y_\text{full} - y_i$
#     4. finally, make prediction when all regressors values set to zero and set it as $y_\text{zero}$
#     5. calculate the "all-off" delta as $\delta_0 = y_\text{full} - y_\text{zero} $
#     6. derive the decomposed components including the baseline and regressors $x_i$ by
#     - derive the normalized weight $w_i = \delta_i / \sum_{i=0}^{i=I}\delta_i$
#     - then the final decomposed components can be derived by $\text{comp}_i =  w_i \cdot y_\text{actual}$
#
#     Parameters
#     ----------
#     model : object
#     df : pd.DataFrame
#     regressors : list
#
#     """
#     if not regressors:
#         regressors = model.spend_cols
#
#     if df is None:
#         df = model.raw_df.copy()
#
#     # organice plus all regressors
#     pred_array = np.empty((df.shape[0], len(regressors)))
#     delta_array = np.empty((df.shape[0], len(regressors) + 1))
#     full_pred = model.predict(df)['prediction'].values.reshape(-1, 1)
#
#     for idx, x in enumerate(regressors):
#         temp_df = df.copy()
#         temp_df[x] = 0.0
#         pred_array[:, idx] = model.predict(temp_df)['prediction'].values
#
#     temp_df = df.copy()
#     temp_df[regressors] = 0.0
#     # zero prediction used for organic
#     delta_array[:, 0] = model.predict(temp_df)['prediction'].values
#     delta_array[:, 1:] = full_pred - pred_array
#     total_delta = np.sum(delta_array, axis=1, keepdims=True)
#     assert total_delta.shape[0] == df.shape[0]
#     weight_array = delta_array / total_delta
#
#     if match_actual:
#         target_array = df[model.kpi_col].values.reshape(-1, 1)
#     else:
#         target_array = pred_array.reshape(-1, 1)
#
#     if return_df:
#         res = pd.DataFrame(weight_array * target_array, columns=['organic'] + regressors)
#         res[model.date_col] = df[model.date_col]
#         # re-arrange columns
#         res = res[[model.date_col, 'organic'] + regressors].reset_index(drop=True)
#     else:
#         res = weight_array * target_array
#     return res
