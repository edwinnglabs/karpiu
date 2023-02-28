import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Optional, Tuple, List

from .utils import adstock_process, np_shift
from .models import MMM
from .model_shell import MMMShell


class FastAttributor(MMMShell):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def make_attribution(self, true_up: bool = True):
        n_regressors = len(self.target_regressors)
        zero_paddings = np.zeros((self.max_adstock, n_regressors))
        # (n_calc_steps, n_regressors)
        simulated_regressors_matrix = np.concatenate(
            [zero_paddings, self.target_regressors_matrix, zero_paddings], axis=0
        )

        # (n_regressors + 1,, n_calc_steps, n_regressors)
        simulated_regressors_matrix = (
            np.tile(
                np.expand_dims(simulated_regressors_matrix, 0),
                reps=(n_regressors + 1, 1, 1),
            )
            * self.design_matrix
        )
        # (n_regressors + 1, n_calc_steps, n_regressors)
        simulated_regressors_matrix += self.target_regressor_bkg_matrix

        # take resulting regressors into adstock process
        # (n_regressor + 1, n_result_steps, n_regressor)
        if self.max_adstock > 0:
            simulated_regressors_matrix = adstock_process(
                simulated_regressors_matrix,
                self.target_adstock_matrix,
            )

        # one-off regression comp
        # (n_regressors, n_result_steps)
        one_off_reg_comp = np.sum(
            self.target_coef_matrix
            * np.log1p(simulated_regressors_matrix[1:, ...] / self.target_sat_array),
            -1,
        )
        # (n_result_steps, )
        full_reg_comp = np.sum(
            self.target_coef_matrix
            * np.log1p(simulated_regressors_matrix[0] / self.target_sat_array),
            -1,
        )

        # un-normalized attribution
        # (n_regressors, n_result_steps)
        attr_matrix = self.pred_zero * (
            -np.exp(one_off_reg_comp) + np.exp(full_reg_comp)
        )
        # insert the first row for organic / non-target regressors attribution lump sum
        attr_matrix = np.concatenate(
            [
                np.expand_dims(self.pred_zero, 0),
                attr_matrix,
            ]
        )

        if true_up:
            true_up_arr = self.df.loc[self.result_mask, self.kpi_col].values
        else:
            true_up_arr = self.pred_zero * np.exp(full_reg_comp)

        # linearization to make decomp additive
        # (n_regressors, n_result_steps, )
        norm_attr_matrix = (
            attr_matrix / np.sum(attr_matrix, 0, keepdims=True) * true_up_arr
        )
        # since they are not decomposable in spend x time dimension
        # aggregate them into regressors level
        # agg_attr = np.sum(norm_attr_matrix, -1)
        return attr_matrix, norm_attr_matrix


class Attributor:
    """The class to make attribution on a state-space model in an object-oriented way; Algorithm assumes model is
     with a Karpiu and Orbit structure

    Attributes:
        date_col : str
        verbose : bool
        attr_start : attribution start date string; if None, use minimum start date based on model provided
        attr_end : attribution end date string; if None, use maximum end date based on model provided
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
        attr_regressors: Optional[List[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
        kpi_name: Optional[str] = None,
        verbose: bool = False,
    ):
        """

        Args:
            model:
            attr_regressors: the regressors required attribution; if None, use model.get_spend_cols()
            start: attribution start date string; if None, use minimum start date based on model provided
            end: attribution end date string; if None, use maximum end date based on model provided
            df: data frame; if None, use original data based on model provided
            kpi_name: kpi label; if None, use information from model provided
            verbose: whether to print process information


        Attributes:
            date_col: date column of the core dataframe
            verbose: control verbose
            max_adstock: number of adstock delay i.e. max_adstock = adstock steps - 1 and max_adstock = 0
            if no adstock exists in the model
            full_regressors: all regressors except internal seasonal regressors
            bkg_regressors: the background regressors not included in the required attribution regressors
            , events and control features. it is used as the starting point to pre-calculate base component
            for fast attribution process. Note that in general it includes non-attribution spend
            and seasonal regressors.
            event_regressors: event regressors
            control_regressors: control regressors
            attr_start: attribution start date see inputs.
            attr_end: attribution end date. see inputs.
            kpi_col: response column of the core data frame
            attr_regressors: the regressors required attribution; if None, use model.get_spend_cols()
            calc_start: the full expanded range of date range used for calculation i.e. extending with
            max_adstock days upfront
            calc_end: the full expanded range of date range used for calculation i.e. extending with
            max_adstock days after attr_end
            df_bau: the snapshot of the input data frame within the [calc_start, calc_end] range with
            trimmed columns which are necessary for attribution process
            dt_array: date array of df_bau
            attr_adstock_matrix: the adstock matrix with same alignment of attr_regressors
            attr_regressor_matrix: the regressor matrix extracted from df_bau
            attr_sat_array: saturation array of attr_regressors
            attr_adstock_regressor_matrix: adstock transformed matrix from attr_regressor_matrix with
            zeros padded upfront to account lose of adstock(=max_adstock) steps
            attr_coef_matrix: the attribution regressor coefficients extracted from the model
            pred_zero: prediction when all attributing regressor are turned off
        """

        self.date_col = model.date_col
        self.verbose = verbose

        date_col = self.date_col
        self.max_adstock = model.get_max_adstock()
        # it excludes the fourier-series columns
        self.full_regressors = model.get_regressors()
        # FIXME: right now it DOES NOT work with including control features;
        self.event_regressors = model.get_event_cols()
        self.control_regressors = model.get_control_feat_cols()

        if df is None:
            df = model.raw_df.copy()
        else:
            df = df.copy()

        if start is None:
            self.attr_start = pd.to_datetime(
                df[self.date_col].values[0]
            ) + pd.Timedelta(days=self.max_adstock)
        else:
            self.attr_start = pd.to_datetime(start)

        if end is None:
            self.attr_end = pd.to_datetime(df[self.date_col].values[-1]) - pd.Timedelta(
                days=self.max_adstock
            )
        else:
            self.attr_end = pd.to_datetime(end)

        df[date_col] = pd.to_datetime(df[date_col])

        if kpi_name is None:
            self.kpi_col = model.kpi_col
        else:
            self.kpi_col = kpi_name

        if attr_regressors is None:
            self.attr_regressors = model.get_spend_cols()
        else:
            self.attr_regressors = attr_regressors

        # for debug
        self.delta_matrix = None

        # better date operations
        # organize the dates. This pads the range with the carry over before it starts
        self.calc_start = self.attr_start - pd.Timedelta(days=self.max_adstock)
        self.calc_end = self.attr_end + pd.Timedelta(days=self.max_adstock)

        if verbose:
            print(
                "Full calculation start={} and end={}".format(
                    self.calc_start.strftime("%Y-%m-%d"),
                    self.calc_end.strftime("%Y-%m-%d"),
                )
            )
            print(
                "Attribution start={} and end={}".format(
                    self.attr_start.strftime("%Y-%m-%d"),
                    self.attr_end.strftime("%Y-%m-%d"),
                )
            )

        if self.calc_start < df[date_col].min():
            raise Exception(
                "Dataframe provided starts at {} must be before {} due to max_adstock={}".format(
                    df[date_col].iloc[0], self.calc_start, self.max_adstock
                )
            )

        if self.calc_end > df[date_col].max():
            raise Exception(
                "Dataframe provided ends at {} must be after {} due to max_adstock={}".format(
                    df[date_col].iloc[-1], self.calc_end, self.max_adstock
                )
            )

        # set a business-as-usual case data frame
        self.calc_mask = (df[date_col] >= self.calc_start) & (
            df[date_col] <= self.calc_end
        )
        # self.df_bau = df.loc[
        #     calc_mask, [date_col, self.kpi_col] + self.full_regressors
        # ].reset_index(drop=True)
        self.df = df.copy()
        self.dt_array = df.loc[self.calc_mask, date_col].values

        # just make sure attr_regressors input by user align original all regressors order
        attr_regressors_idx = [
            idx
            for idx in range(len(self.full_regressors))
            if self.full_regressors[idx] in self.attr_regressors
        ]
        self.attr_regressors = [
            self.full_regressors[idx] for idx in attr_regressors_idx
        ]

        # prepare a few matrices and arrays for rest of the calculations
        # required matrices and arrays such as saturation, adstock, regressors matrix etc.
        self.attr_adstock_matrix = model.get_adstock_matrix(self.attr_regressors)
        # untransformed regressor matrix
        # (n_steps, n_regressors)
        self.attr_regressor_matrix = df.loc[self.calc_mask, self.attr_regressors].values
        # (n_regressors, )
        sat_df = model.get_saturation()
        self.attr_sat_array = sat_df.loc[self.attr_regressors, "saturation"].values

        # adstock_regressor_matrix dim: time x num of regressors
        if self.max_adstock >= 1:
            # adstock transformed regressors will be used to calculate pred_bau later
            self.attr_adstock_regressor_matrix = adstock_process(
                self.attr_regressor_matrix, self.attr_adstock_matrix
            )
            # we lose first n(=max_adstock) observations; to maintain original dimension,
            # we need to pad zeros n(=max_adstock) time
            self.attr_adstock_regressor_matrix = np.concatenate(
                (
                    np.zeros(
                        (self.max_adstock, self.attr_adstock_regressor_matrix.shape[1])
                    ),
                    self.attr_adstock_regressor_matrix,
                ),
                0,
            )
        else:
            self.attr_adstock_regressor_matrix = deepcopy(self.attr_regressor_matrix)
        self.attr_coef_matrix = model.get_coef_matrix(
            date_array=self.dt_array,
            regressors=self.attr_regressors,
        )

        # organic, zero values baseline prediction
        df_zero = df.copy()
        df_zero.loc[:, self.attr_regressors] = 0.0

        # prediction with all attr regressors turned to zero
        # (n_steps, )
        zero_pred_df = model.predict(df=df_zero, decompose=True)

        # seas = zero_pred_df['weekly seasonality'].values
        # original scale
        self.pred_zero = zero_pred_df.loc[self.calc_mask, "prediction"].values
        # log scale
        self.base_comp = np.log(self.pred_zero)
        # dependent on the coefficients (can be specified by users in next step)
        self.pred_bau = None

        # store background target regressors spend before and after budget period due to adstock
        # only background spend involved; turn off all spend during budget decision period
        if self.max_adstock > 0:
            bkg_attr_regressor_matrix = df.loc[
                self.calc_mask, self.attr_regressors
            ].values
            bkg_attr_regressor_matrix[self.max_adstock : -self.max_adstock, ...] = 0.0
            self.bkg_attr_regressor_matrix = bkg_attr_regressor_matrix
        else:
            self.bkg_attr_regressor_matrix = np.zeros_like(
                df.loc[self.calc_mask, self.attr_regressors].values
            )

        # trend = zero_pred_df.loc[self.calc_mask, "trend"].values
        # # base_comp = trend + seas
        # base_comp = trend

        # self.non_attr_regressors = list(
        #     set(self.full_regressors)
        #     - set(self.attr_regressors)
        #     - set(self.event_regressors)
        #     - set(self.control_regressors)
        # )
        # if len(self.non_attr_regressors) > 0:
        #     non_attr_coef_matrix = model.get_coef_matrix(
        #         date_array=self.dt_array,
        #         regressors=self.non_attr_regressors,
        #     )
        #     non_attr_sat_array = sat_df.loc[self.non_attr_regressors, "saturation"].values
        #     non_attr_regressor_matrix = self.df.loc[self.calc_mask, self.non_attr_regressors].values
        #     non_attr_adstock_matrix = model.get_adstock_matrix(self.non_attr_regressors)
        #     non_attr_adstock_regressor_matrix = adstock_process(
        #         non_attr_regressor_matrix, non_attr_adstock_matrix
        #     )
        #     non_attr_adstock_regressor_matrix = np.concatenate(
        #         (
        #             np.zeros((self.max_adstock, non_attr_adstock_regressor_matrix.shape[1])),
        #             non_attr_adstock_regressor_matrix,
        #         ),
        #         0,
        #     )
        #     base_comp += np.sum(
        #         non_attr_coef_matrix
        #         * np.log1p(non_attr_adstock_regressor_matrix / non_attr_sat_array),
        #         -1,
        #     )

        # if len(self.event_regressors) > 0:
        #     event_coef_matrix = model.get_coef_matrix(
        #         date_array=self.dt_array,
        #         regressors=self.event_regressors,
        #     )

        #     event_regressor_matrix = self.df.loc[self.calc_mask, self.event_regressors].values
        #     base_comp += np.sum(event_coef_matrix * event_regressor_matrix, -1)

        # if len(self.control_regressors) > 0:
        #     control_coef_matrix = model.get_coef_matrix(
        #         date_array=self.dt_array,
        #         regressors=self.control_regressors,
        #     )
        #     control_regressor_matrix = np.log1p(
        #         self.df.loc[self.calc_mask, self.control_regressors].values
        #     )
        #     base_comp += np.sum(control_coef_matrix * control_regressor_matrix, -1)

        # self.base_comp = base_comp
        # self.pred_zero = np.exp(base_comp)

    def make_attribution(
        self,
        new_coef_name: Optional[str] = None,
        new_coef: Optional[float] = None,
        true_up: bool = True,
        debug: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        A general time-based attribution on stat-space model with regression. This is
        much faster than the legacy method.


        Parameters
        ----------
        new_coef_name :
        new_coef:
        true_up : bool
        """
        # get the number of lags in adstock expressed as days
        date_col = self.date_col
        # base df for different report
        # (n_steps, n_channels)
        df_bau = self.df.loc[self.calc_mask,]
        df_bau = df_bau.reset_index(drop=True)

        # (n_steps, n_attr_regressors)
        attr_coef_matrix = self.attr_coef_matrix.copy()
        if new_coef is not None:
            coef_search_found = False
            for idx, x in enumerate(self.attr_regressors):
                if new_coef_name == x:
                    coef_search_found = True
                    attr_coef_matrix[:, idx] = new_coef
                    break
            if not coef_search_found:
                raise Exception(
                    "New coefficient name does not match any regressors in the model."
                )
        # TODO: consider update the prediction function such that the shape of the prediction is the same as the input
        # modified coef matrix would impact this prediction vector
        # prediction with full spend in business-as-usual case
        # NOTE: this vector has less length (a `max_adstock` shift from the beginning)
        varying_comp = np.sum(
            attr_coef_matrix
            * np.log1p(self.attr_adstock_regressor_matrix / self.attr_sat_array),
            -1,
        )
        # (n_steps + 2 * max_adstock, ); already tested value is the same as model.predict()
        self.pred_bau = np.exp(self.base_comp + varying_comp)

        if true_up:
            true_up_array = df_bau[self.kpi_col].values
        else:
            true_up_array = self.pred_bau

        (
            activities_attr_matrix,
            spend_attr_matrix,
            delta_matrix,
        ) = make_attribution_numpy(
            coef_matrix=attr_coef_matrix,
            regressor_matrix=self.attr_regressor_matrix,
            adstock_regressor_matrix=self.attr_adstock_regressor_matrix,
            pred_bau=self.pred_bau,
            pred_zero=self.pred_zero,
            adstock_matrix=self.attr_adstock_matrix,
            saturation_array=self.attr_sat_array,
            true_up_arr=true_up_array,
        )
        # print(spend_attr_matrix.shape)
        # print(activities_attr_matrix.shape)
        # print(self.max_adstock)
        if debug:
            self.delta_matrix = delta_matrix

        # note that activities based attribution only makes sense when spend is fully settled after adstock process
        # hence first n(=max_adstock) need to be discarded

        # note that spend attribution only makes sense when all attributing metric fully observed in the entire
        # adstock process
        # also need to discard first n(=max_adstock) observation as you cannot observe the correct pred_bau
        # hence last n(=max_adstock) need to be discarded
        if self.max_adstock > 0:
            activities_attr_df = df_bau[self.max_adstock :][[date_col]]
            activities_attr_df[
                ["organic"] + self.attr_regressors
            ] = activities_attr_matrix[self.max_adstock :]
            activities_attr_df = activities_attr_df.reset_index(drop=True)
            spend_attr_df = df_bau[self.max_adstock : -self.max_adstock][[date_col]]
            spend_attr_df[["organic"] + self.attr_regressors] = spend_attr_matrix[
                self.max_adstock : -self.max_adstock
            ]
            spend_attr_df = spend_attr_df.reset_index(drop=True)
            spend_df = df_bau[self.max_adstock : -self.max_adstock][
                [date_col] + self.attr_regressors
            ]
            spend_df = spend_df.reset_index(drop=True)
        else:
            activities_attr_df = df_bau[[date_col]]
            activities_attr_df[
                ["organic"] + self.attr_regressors
            ] = activities_attr_matrix
            spend_attr_df = df_bau[[date_col]]
            spend_attr_df[["organic"] + self.attr_regressors] = spend_attr_matrix
            spend_df = df_bau[[date_col] + self.attr_regressors]

        cost_df = spend_df[[date_col]].copy()
        cost_df[self.attr_regressors] = (
            spend_df[self.attr_regressors] / spend_attr_df[self.attr_regressors]
        )

        return activities_attr_df, spend_attr_df, spend_df, cost_df


def make_attribution_numpy(
    coef_matrix: np.array,
    regressor_matrix: np.array,
    adstock_regressor_matrix: np.array,
    pred_bau: np.array,
    pred_zero: np.array,
    adstock_matrix: np.array,
    saturation_array: np.array,
    true_up_arr: np.array,
) -> Tuple[np.array, np.array]:
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
