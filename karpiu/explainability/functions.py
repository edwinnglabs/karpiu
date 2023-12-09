import numpy as np
from copy import deepcopy

from typing import Tuple

from ..utils import np_shift, adstock_process


def make_attribution_numpy_beta(
    attr_coef_array: np.ndarray,
    attr_regressor_matrix: np.ndarray,
    attr_transformed_regressor_matrix: np.ndarray,
    pred_bau: np.ndarray,
    pred_zero: np.ndarray,
    adstock_matrix: np.ndarray,
    attr_saturation_array: np.ndarray,
    true_up_arr: np.ndarray,
    fixed_intercept: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A numpy version of making attribution
    Notes
    -----
    Assuming n_steps = spend range + 2 * max_adstock
    Parameters
    ----------
    coef_matrix: array in shape (n_steps, n_regressors)
    regressor_matrix: array in shape (n_steps, n_regressors)
    adstock_regressor_matrix: array in shape (n_steps, n_regressors)
    pred_bau:  (n_steps, )
    pred_zero: (n_step, )
    adstock_matrix: (n_regressors, max_adstock + 1)
    saturation_array:  (n_regressors, )
    true_up_arr: (n_steps, )
    Returns
    -------
    activities_attr_matrix: (n_steps, )
    spend_attr_matrix: (n_steps, )
    delta_matrix: (n_steps, )
    """
    n_calc_steps, n_attr_reg = attr_regressor_matrix.shape
    max_adstock = adstock_matrix.shape[1] - 1

    # (n_calc_steps, max_adstock + 1, n_attr_reg + 1)
    delta_matrix = np.zeros((n_calc_steps, max_adstock + 1, n_attr_reg + 1))

    # a same size of matrix declared to prepare paid date dimension
    # (n_calc_steps, max_adstock + 1, n_attr_reg + 1)
    paid_on_attr_matrix = np.zeros(delta_matrix.shape)

    # active on can directly take organic as it does not care adstock;
    # hence all impact contribute at first(current) time
    # delta_matrix[:, 0, 0] = self.base_comp_calc
    delta_matrix[:, 0, 0] = pred_zero

    # TODO: the design matrix and access matrix can be pre-computed
    design_matrix = np.ones((n_calc_steps, n_calc_steps))
    np.fill_diagonal(design_matrix, 0.0)
    # (n_calc_steps, n_calc_steps, 1)
    design_matrix = np.expand_dims(design_matrix, axis=-1)

    # single row each step
    # (n_calc_steps, 1)
    access_row_matrix = (np.expand_dims(np.arange(n_calc_steps), -1),)
    # (n_calc_steps, max_adstock + 1)
    access_col_matrix = np.stack(
        [np.arange(x, x + max_adstock + 1) for x in range(n_calc_steps)]
    )

    # expand for delta calculation later
    pred_bau = np.concatenate(
        [
            np.expand_dims(pred_bau, -1),
            np.zeros((max_adstock, 1)),
        ],
        axis=0,
    )
    # (n_calc_steps, max_adstock + 1)
    pred_bau = np.squeeze(pred_bau[access_col_matrix].copy(), -1)

    for idx in range(n_attr_reg):
        # derive the bau regressor matrix
        # (n_calc_steps + max_adstock, 1)
        temp_bau_transformed_regressor = np.concatenate(
            [
                deepcopy(attr_transformed_regressor_matrix[:, [idx]]),
                np.zeros((max_adstock, 1)),
            ],
            axis=0,
        )
        # (n_calc_steps, max_adstock + 1)
        temp_bau_transformed_regressor = np.squeeze(
            temp_bau_transformed_regressor[access_col_matrix],
            -1,
        )
        # (n_calc_steps, 1)
        temp_bau_regressor = deepcopy(attr_regressor_matrix[:, [idx]])
        # scenario with spend-off step by step
        # (n_scenarios, n_calc_steps, 1)
        temp_full_regressor_zero = design_matrix * temp_bau_regressor

        if max_adstock > 0:
            temp_adstock_filter = deepcopy(adstock_matrix[[idx], :])
            temp_full_regressor_zero = np.squeeze(
                adstock_process(temp_full_regressor_zero, temp_adstock_filter)
            )
            # (n_scenarios, n_calc_steps + max_adstock, )
            temp_full_regressor_zero = np.concatenate(
                [
                    np.zeros((n_calc_steps, max_adstock)),
                    temp_full_regressor_zero,
                    # append max_adstock of zeros for access purpose later
                    np.zeros((n_calc_steps, max_adstock)),
                ],
                axis=-1,
            )

        else:
            # (n_scenarios, n_calc_steps, 1)
            temp_full_regressor_zero = np.squeeze(temp_full_regressor_zero, -1)

        # (n_calc_steps, max_adstock + 1)
        temp_full_regressor_zero_reduced = np.squeeze(
            temp_full_regressor_zero[
                access_row_matrix,
                access_col_matrix,
            ],
            0,
        )

        # (n_calc_steps, max_adstock + 1)
        numerator = (
            1 + temp_full_regressor_zero_reduced / attr_saturation_array[idx]
        ) ** attr_coef_array[idx]

        # (n_calc_steps, max_adstock + 1)
        denominator = (
            1 + temp_bau_transformed_regressor / attr_saturation_array[idx]
        ) ** attr_coef_array[idx]

        # (n_calc_steps, max_adstock + 1)
        temp_delta_matrix = pred_bau * (1 - numerator / denominator)

        # temp delta is the view anchored with spend date for convenient delta
        # calculation; however, they need to be shifted to be anchored with activities
        # date in order to perform normalization; hence, the step below shift
        # down the derived delta to make them aligned at activities date

        if max_adstock > 0:
            delta_matrix[:, :, idx + 1] = np_shift(
                temp_delta_matrix, np.arange(max_adstock + 1)
            )
        else:
            delta_matrix[:, :, idx + 1] = temp_delta_matrix

    # fix numeric problem
    # force invalid number to be zero
    # force small number to be zero
    delta_matrix[np.logical_not(np.isfinite(delta_matrix))] = 0.0
    delta_matrix[delta_matrix <= 1e-7] = 0.0

    if fixed_intercept:
        # intercept has no adstock
        true_up_arr_sub = true_up_arr - delta_matrix[:, 0, 0]
        delta_matrix_sub = delta_matrix[:, :, 1:]
        # (n_steps, 1, 1)
        total_delta = np.sum(delta_matrix_sub, axis=(-1, -2), keepdims=True)
        # remove zeros to avoid divide-by-zero issue
        index_zero = total_delta == 0
        total_delta[index_zero] = 1
        # get the normalized delta
        # (n_steps, max_adstock + 1, n_regressors + 1)
        norm_delta_matrix = delta_matrix_sub / total_delta

        # (n_steps, max_adstock + 1, n_regressors + 1)
        full_attr_matrix = np.concatenate(
            [
                delta_matrix[:, :, :1],
                norm_delta_matrix * true_up_arr_sub.reshape(-1, 1, 1),
            ],
            axis=-1,
        )
    else:
        # (n_steps, 1, 1)
        total_delta = np.sum(delta_matrix, axis=(-1, -2), keepdims=True)

        # remove zeros to avoid divide-by-zero issue
        index_zero = total_delta == 0
        total_delta[index_zero] = 1

        # get the normalized delta
        # (n_steps, max_adstock + 1, n_regressors + 1)
        norm_delta_matrix = delta_matrix / total_delta

        # (n_steps, max_adstock + 1, n_regressors + 1)
        full_attr_matrix = norm_delta_matrix * true_up_arr.reshape(-1, 1, 1)

    # sum over lags (adstock);
    # (n_steps, n_attr_regressors + 1)
    activities_attr_matrix = np.sum(full_attr_matrix, axis=-2)
    # remove the first and last max_adstock periods as they are not fully observed
    if max_adstock > 0:
        activities_attr_matrix = activities_attr_matrix[max_adstock:-max_adstock]

    ########################################################################################
    # get the total from a channel in a day (sum over lag); this is for the paid on
    ########################################################################################
    # shift up arrays by lags; NAs / zeros padded at the end
    if max_adstock > 0:
        for idx in range(full_attr_matrix.shape[2]):
            paid_on_attr_matrix[:, :, idx] = np_shift(
                full_attr_matrix[:, :, idx], np.arange(0, -1 * (max_adstock + 1), -1)
            )
    else:
        paid_on_attr_matrix = deepcopy(full_attr_matrix)

    # sum over lags (adstock);
    # (n_steps, n_attr_regressors + 1)
    spend_attr_matrix = np.sum(paid_on_attr_matrix, axis=-2)
    if max_adstock > 0:
        spend_attr_matrix = spend_attr_matrix[max_adstock:-max_adstock]

    # output norm_delta_matrix mainly for debug
    return activities_attr_matrix, spend_attr_matrix, delta_matrix


def make_attribution_numpy_alpha(
    coef_matrix: np.ndarray,
    regressor_matrix: np.ndarray,
    adstock_regressor_matrix: np.ndarray,
    pred_bau: np.ndarray,
    pred_zero: np.ndarray,
    adstock_matrix: np.ndarray,
    saturation_array: np.ndarray,
    true_up_arr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A numpy version of making attribution
    Notes
    -----
    Assuming n_steps = spend range + 2 * max_adstock
    Parameters
    ----------
    coef_matrix: array in shape (n_steps, n_regressors)
    regressor_matrix: array in shape (n_steps, n_regressors)
    adstock_regressor_matrix: array in shape (n_steps, n_regressors)
    pred_bau:  (n_steps, )
    pred_zero: (n_step, )
    adstock_matrix: (n_regressors, max_adstock + 1)
    saturation_array:  (n_regressors, )
    true_up_arr: (n_steps, )
    """

    # a delta matrix with extra dimension (num of attr_regressor) and store the delta at calendar date view
    # so that we do the normalization within each calendar date (not spend date)
    # delta matrix stores the impact on calendar date by each channel spend due to each adstock impact
    # the first channel dimension is added by an extra 1 to include organic
    # note that this channel should not have any adstock effect simply because
    # we don't care organic adstock; it doesn't impact paid on calculation which is the whole
    # purpose of resolving adstock effect
    # (n_steps = spend_range + 2 * max_adstock)
    n_steps, n_regressors = regressor_matrix.shape
    max_adstock = adstock_matrix.shape[1] - 1

    # (n_steps, max_adstock + 1, n_regressors + 1)
    delta_matrix = np.zeros((n_steps, max_adstock + 1, n_regressors + 1))

    # a same size of matrix declared to prepare paid date dimension
    # (n_steps, max_adstock + 1, n_regressors + 1)
    paid_on_attr_matrix = np.zeros(delta_matrix.shape)

    # active on can directly take organic as it does not care adstock;
    # hence all impact contribute at first(current) time
    delta_matrix[:, 0, 0] = pred_zero

    # loop through the channels
    for i in range(n_regressors):
        # store the delta where row is the time spend is turned off,
        # and column is the subsequent impact from time t (size=adstock + 1)
        # (n_steps, max_adstock + 1)
        temp_delta_matrix = np.zeros((n_steps, max_adstock + 1))

        # (n_steps, )
        temp_bau_regressor = deepcopy(regressor_matrix[:, i])
        # (n_steps, )
        temp_bau_regressor_adstock = deepcopy(adstock_regressor_matrix[:, i])
        # (1, max_adstock)
        temp_adstock_filter = np.expand_dims(deepcopy(adstock_matrix[i, :]), 0)

        # loop over time to turn off spend; note that j here is not necessarily time prediction target at!
        # j equals t only if adstock = 0
        # attr end - attr start = 10
        # max_adstock 55
        # attr_regressor_matrix needs to be (55 * 2 + 10)
        # add max_adstock period at the beginning for active-on
        # add max_adstock period at the end for paid-on
        for j in range(0, n_steps):
            # (n_steps, )
            temp_attr_regressor_zero = deepcopy(temp_bau_regressor)
            # (n_steps, 1); make it work for adstock process
            temp_attr_regressor_zero = np.expand_dims(temp_attr_regressor_zero, -1)
            # turn off spend at time j
            temp_attr_regressor_zero[j] = 0

            if max_adstock > 0:
                # (n_steps - max_adstock, )
                temp_attr_regressor_zero = np.squeeze(
                    adstock_process(temp_attr_regressor_zero, temp_adstock_filter), -1
                )
                # pad zeros; since both bau and set-zero condition yields the same number in the
                # zero-padding period, the delta ends up to be (constant x (1 - 1 / 1)) =  0
                # it yields the same result as legacy version.

                # (n_steps, )
                temp_attr_regressor_zero = np.concatenate(
                    (np.zeros(max_adstock), temp_attr_regressor_zero)
                )
            else:
                temp_attr_regressor_zero = np.squeeze(temp_attr_regressor_zero, -1)

            # measure impact from j to j + max_adstock only
            # (max_adstock + 1, )
            coef_array = coef_matrix[j : (j + max_adstock + 1), i]

            # compute the delta who is the ratio between lift of bau spend and zero spend at time j
            # (max_adstock + 1, )
            numerator = (
                1
                + temp_attr_regressor_zero[j : (j + max_adstock + 1)]
                / saturation_array[i]
            ) ** coef_array
            # (max_adstock + 1, )
            denominator = (
                1
                + temp_bau_regressor_adstock[j : (j + max_adstock + 1)]
                / saturation_array[i]
            ) ** coef_array

            delta = pred_bau[j : (j + max_adstock + 1)] * (1 - numerator / denominator)
            # (n_steps, max_adstock + 1)
            temp_delta_matrix[j, 0 : len(delta)] = delta

        # so far what does delta matrix tell us?
        # it is the delta on each channel, each time spend turn off, its effect on each adstock
        # however, we need to shift the adstock effect to the next day (downward) as adstock
        # impact subsequent time, not the current time
        # that's why we have the next step to shift the matrix down for the delta_matrix

        # shift down the arrays by adstock effect size in j dimension; NAs / zeros padded at original place (elements get shifted)
        # note that the first dimension is dedicated to organic in i dimension; so index from left need to be
        # shifted one up
        if max_adstock > 0:
            delta_matrix[:, :, i + 1] = np_shift(
                temp_delta_matrix, np.arange(max_adstock + 1)
            )
        else:
            delta_matrix[:, :, i + 1] = temp_delta_matrix

    # get the sum for all channels and adstock effect
    # (n_steps, 1, 1)
    total_delta = np.sum(delta_matrix, axis=(-1, -2), keepdims=True)

    # remove zeros to avoid divide-by-zero issue
    index_zero = total_delta == 0
    total_delta[index_zero] = 1

    # get the normalized delta
    # (n_steps, max_adstock + 1, n_regressors + 1)
    norm_delta_matrix = delta_matrix / total_delta

    # (n_steps, max_adstock + 1, n_regressors + 1)
    full_attr_matrix = norm_delta_matrix * true_up_arr.reshape(-1, 1, 1)

    # sum over lags (adstock);
    # (n_steps, n_attr_regressors + 1)
    activities_attr_matrix = np.sum(full_attr_matrix, axis=-2)
    # remove the first and last max_adstock periods as they are not fully observed
    if max_adstock > 0:
        activities_attr_matrix = activities_attr_matrix[max_adstock:-max_adstock]

    ########################################################################################
    # get the total from a channel in a day (sum over lag); this is for the paid on
    ########################################################################################
    # shift up arrays by lags; NAs / zeros padded at the end
    if max_adstock > 0:
        for idx in range(full_attr_matrix.shape[2]):
            paid_on_attr_matrix[:, :, idx] = np_shift(
                full_attr_matrix[:, :, idx], np.arange(0, -1 * (max_adstock + 1), -1)
            )
    else:
        paid_on_attr_matrix = deepcopy(full_attr_matrix)

    # sum over lags (adstock);
    # (n_steps, n_attr_regressors + 1)
    spend_attr_matrix = np.sum(paid_on_attr_matrix, axis=-2)
    if max_adstock > 0:
        spend_attr_matrix = spend_attr_matrix[max_adstock:-max_adstock]

    # output norm_delta_matrix mainly for debug
    return activities_attr_matrix, spend_attr_matrix, delta_matrix


def make_attribution_numpy_gamma(
    attr_coef_array: np.ndarray,
    attr_regressor_matrix: np.ndarray,
    attr_transformed_regressor_matrix: np.ndarray,
    organic_arr: np.ndarray,
    pred_zero: np.ndarray,
    adstock_matrix: np.ndarray,
    attr_saturation_array: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A numpy version of making attribution
    Notes
    -----
    Assuming n_steps = spend range + 2 * max_adstock
    Parameters
    ----------
    coef_matrix: array in shape (n_steps, n_regressors)
    regressor_matrix: array in shape (n_steps, n_regressors)
    adstock_regressor_matrix: array in shape (n_steps, n_regressors)
    pred_bau:  (n_steps, )
    pred_zero: (n_step, )
    adstock_matrix: (n_regressors, max_adstock + 1)
    saturation_array:  (n_regressors, )
    true_up_arr: (n_steps, )
    Returns
    -------
    activities_attr_matrix: (n_steps, )
    spend_attr_matrix: (n_steps, )
    delta_matrix: (n_steps, )
    """
    n_calc_steps, n_attr_reg = attr_regressor_matrix.shape
    max_adstock = adstock_matrix.shape[1] - 1

    # TODO: n_calc_steps mean n_attr_steps (result_steps?) + 2 * max_adstock
    # delta matrix store the shares from each step, each lag effect and each channel 
    # (n_calc_steps, max_adstock + 1, n_attr_reg)
    delta_matrix = np.zeros((n_calc_steps, max_adstock + 1, n_attr_reg))
    # a same size of delta matrix to prepare paid date dimension
    paid_on_attr_matrix = np.zeros(delta_matrix.shape)
    # TODO: the design matrix and access matrix can be pre-computed
    design_matrix = np.ones((n_calc_steps, n_calc_steps))
    np.fill_diagonal(design_matrix, 0.0)
    # (n_calc_steps, n_calc_steps, 1)
    design_matrix = np.expand_dims(design_matrix, axis=-1)

    # single row each step
    # TODO: not sure what it is but looks it is used to control what to access on each step
    # (n_calc_steps, 1)
    access_row_matrix = (np.expand_dims(np.arange(n_calc_steps), -1),)
    # (n_calc_steps, max_adstock + 1)
    access_col_matrix = np.stack(
        [np.arange(x, x + max_adstock + 1) for x in range(n_calc_steps)]
    )

    # expand for delta calculation later
    pred_bau = np.concatenate(
        [
            np.expand_dims(pred_bau, -1),
            np.zeros((max_adstock, 1)),
        ],
        axis=0,
    )
    # (n_calc_steps, max_adstock + 1)
    pred_bau = np.squeeze(pred_bau[access_col_matrix].copy(), -1)

    for idx in range(n_attr_reg):
        # derive the bau regressor matrix
        # (n_calc_steps + max_adstock, 1)
        temp_bau_transformed_regressor = np.concatenate(
            [
                deepcopy(attr_transformed_regressor_matrix[:, [idx]]),
                np.zeros((max_adstock, 1)),
            ],
            axis=0,
        )
        # (n_calc_steps, max_adstock + 1)
        temp_bau_transformed_regressor = np.squeeze(
            temp_bau_transformed_regressor[access_col_matrix],
            -1,
        )
        # (n_calc_steps, 1)
        temp_bau_regressor = deepcopy(attr_regressor_matrix[:, [idx]])
        # scenario with spend-off step by step
        # (n_scenarios, n_calc_steps, 1)
        temp_full_regressor_zero = design_matrix * temp_bau_regressor

        if max_adstock > 0:
            temp_adstock_filter = deepcopy(adstock_matrix[[idx], :])
            temp_full_regressor_zero = np.squeeze(
                adstock_process(temp_full_regressor_zero, temp_adstock_filter)
            )
            # (n_scenarios, n_calc_steps + max_adstock, )
            temp_full_regressor_zero = np.concatenate(
                [
                    np.zeros((n_calc_steps, max_adstock)),
                    temp_full_regressor_zero,
                    # append max_adstock of zeros for access purpose later
                    np.zeros((n_calc_steps, max_adstock)),
                ],
                axis=-1,
            )

        else:
            # (n_scenarios, n_calc_steps, 1)
            temp_full_regressor_zero = np.squeeze(temp_full_regressor_zero, -1)

        # (n_calc_steps, max_adstock + 1)
        temp_full_regressor_zero_reduced = np.squeeze(
            temp_full_regressor_zero[
                access_row_matrix,
                access_col_matrix,
            ],
            0,
        )

        # (n_calc_steps, max_adstock + 1)
        numerator = (
            1 + temp_full_regressor_zero_reduced / attr_saturation_array[idx]
        ) ** attr_coef_array[idx]

        # (n_calc_steps, max_adstock + 1)
        denominator = (
            1 + temp_bau_transformed_regressor / attr_saturation_array[idx]
        ) ** attr_coef_array[idx]

        # (n_calc_steps, max_adstock + 1)
        temp_delta_matrix = pred_bau * (1 - numerator / denominator)

        # temp delta is the view anchored with spend date for convenient delta
        # calculation; however, they need to be shifted to be anchored with activities
        # date in order to perform normalization; hence, the step below shift
        # down the derived delta to make them aligned at activities date

        if max_adstock > 0:
            delta_matrix[:, :, idx] = np_shift(
                temp_delta_matrix, np.arange(max_adstock + 1)
            )
        else:
            delta_matrix[:, :, idx] = temp_delta_matrix

    # fix numeric problem
    # force invalid number to be zero
    # force small number to be zero
    delta_matrix[np.logical_not(np.isfinite(delta_matrix))] = 0.0
    delta_matrix[delta_matrix <= 1e-7] = 0.0

    if fixed_intercept:
        # intercept has no adstock
        true_up_arr_sub = true_up_arr - delta_matrix[:, 0, 0]
        delta_matrix_sub = delta_matrix[:, :, 1:]
        # (n_steps, 1, 1)
        total_delta = np.sum(delta_matrix_sub, axis=(-1, -2), keepdims=True)
        # remove zeros to avoid divide-by-zero issue
        index_zero = total_delta == 0
        total_delta[index_zero] = 1
        # get the normalized delta
        # (n_steps, max_adstock + 1, n_regressors + 1)
        norm_delta_matrix = delta_matrix_sub / total_delta

        # (n_steps, max_adstock + 1, n_regressors + 1)
        full_attr_matrix = np.concatenate(
            [
                delta_matrix[:, :, :1],
                norm_delta_matrix * true_up_arr_sub.reshape(-1, 1, 1),
            ],
            axis=-1,
        )
    else:
        # (n_steps, 1, 1)
        total_delta = np.sum(delta_matrix, axis=(-1, -2), keepdims=True)

        # remove zeros to avoid divide-by-zero issue
        index_zero = total_delta == 0
        total_delta[index_zero] = 1

        # get the normalized delta
        # (n_steps, max_adstock + 1, n_regressors + 1)
        norm_delta_matrix = delta_matrix / total_delta

        # (n_steps, max_adstock + 1, n_regressors + 1)
        full_attr_matrix = norm_delta_matrix * true_up_arr.reshape(-1, 1, 1)

    # sum over lags (adstock);
    # (n_steps, n_attr_regressors + 1)
    activities_attr_matrix = np.sum(full_attr_matrix, axis=-2)
    # remove the first and last max_adstock periods as they are not fully observed
    if max_adstock > 0:
        activities_attr_matrix = activities_attr_matrix[max_adstock:-max_adstock]

    ########################################################################################
    # get the total from a channel in a day (sum over lag); this is for the paid on
    ########################################################################################
    # shift up arrays by lags; NAs / zeros padded at the end
    if max_adstock > 0:
        for idx in range(full_attr_matrix.shape[2]):
            paid_on_attr_matrix[:, :, idx] = np_shift(
                full_attr_matrix[:, :, idx], np.arange(0, -1 * (max_adstock + 1), -1)
            )
    else:
        paid_on_attr_matrix = deepcopy(full_attr_matrix)

    # sum over lags (adstock);
    # (n_steps, n_attr_regressors + 1)
    spend_attr_matrix = np.sum(paid_on_attr_matrix, axis=-2)
    if max_adstock > 0:
        spend_attr_matrix = spend_attr_matrix[max_adstock:-max_adstock]

    # output norm_delta_matrix mainly for debug
    return activities_attr_matrix, spend_attr_matrix, delta_matrix


# deprecated
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
#     # organic plus all regressors
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
