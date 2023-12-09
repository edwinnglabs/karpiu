import numpy as np
import pandas as pd
from copy import deepcopy
import logging
from typing import Optional, Tuple, List

from ..utils import adstock_process
from ..model_shell import MMMShell
from ..models import MMM
from ..utils import np_shift, adstock_process, get_logger


class AttributorGamma(MMMShell):
    def __init__(
        self,
        model: MMM,
        # attr_regressors: Optional[List[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
        logger: Optional[logging.Logger] = None,
        debug: bool = False,
        **kwargs
    ):
        super().__init__(model=model, start=start, end=end, df=df, **kwargs)

        self.attr_start = self.start
        self.attr_end = self.end
        self.debug = debug

        if logger is None:
            self.logger = get_logger("karpiu-planning")
        else:
            self.logger = logger

        if self.debug:
            self.logger.setLevel(logging.DEBUG)

        # for debug
        self.norm_delta_matrix = None

        self.logger.info(
            "Full calculation start={} and end={}".format(
                self.calc_start.strftime("%Y-%m-%d"),
                self.calc_end.strftime("%Y-%m-%d"),
            )
        )
        self.logger.info(
            "Attribution start={} and end={}".format(
                self.attr_start.strftime("%Y-%m-%d"),
                self.attr_end.strftime("%Y-%m-%d"),
            )
        )

        if self.calc_start < self.df[self.date_col].min():
            raise Exception(
                "Dataframe provided starts at {} must be before {} due to max_adstock={}".format(
                    self.df[self.date_col].iloc[0], self.calc_start, self.max_adstock
                )
            )

        if self.calc_end > self.df[self.date_col].max():
            raise Exception(
                "Dataframe provided ends at {} must be after {} due to max_adstock={}".format(
                    self.df[self.date_col].iloc[-1], self.calc_end, self.max_adstock
                )
            )

        # this is equivalent to np.exp(trend + seas + control + marketing)
        # it is dynamic due to coefficient matrix and spend input which impact marketing
        self.design_matrix = None
        self.access_row_vector = None
        self.access_col_matrix = None
        self.n_calc_steps = np.sum(self.calc_mask)

        if self.max_adstock > 0:
            self._derive_design_and_access_matrix()

        self.attr_coef_matrix = model.get_coef_matrix(
            date_array=self.calc_dt_array,
            regressors=self.target_regressors,
        )

        self.delta_matrix = None
        self.norm_delta_matrix = None
        self.activities_attr_matrix = None
        self.pred_bau_array = None
        self.target_transformed_matrix = None
        self.attr_marketing = None

    def _derive_design_and_access_matrix(
        self,
    ) -> None:
        self.design_matrix = np.ones((self.n_calc_steps, self.n_calc_steps))
        np.fill_diagonal(self.design_matrix, 0.0)
        # (n_calc_steps, n_calc_steps, 1)
        self.design_matrix = np.expand_dims(self.design_matrix, axis=-1)

        # single row each step
        # (n_calc_steps, 1)
        self.access_row_vector = (np.expand_dims(np.arange(self.n_calc_steps), -1),)
        # (n_calc_steps, max_adstock + 1)
        self.access_col_matrix = np.stack(
            [np.arange(x, x + self.max_adstock + 1) for x in range(self.n_calc_steps)]
        )

    # override parent properties
    def _define_masks(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # better date operations
        # organize the dates. This pads the range with the carry over before it starts
        calc_start = self.start - pd.Timedelta(days=self.max_adstock)
        calc_end = self.end + pd.Timedelta(days=self.max_adstock)
        # result_end = self.end + pd.Timedelta(days=self.max_adstock)
        input_mask = (self.df[self.date_col] >= self.start) & (
            self.df[self.date_col] <= self.end
        )
        result_mask = (self.df[self.date_col] >= self.start) & (
            self.df[self.date_col] <= calc_end
        )
        calc_mask = (self.df[self.date_col] >= calc_start) & (
            self.df[self.date_col] <= calc_end
        )
        self.calc_start = calc_start
        self.calc_end = calc_end

        return input_mask, result_mask, calc_mask

    def _derive_target_transformed_matrix(
        self,
        # (n_calc_steps, n_regressors)
        target_calc_regressors_matrix: np.ndarray,
    ) -> np.ndarray:
        if self.max_adstock > 0:
            target_transformed_matrix = adstock_process(
                target_calc_regressors_matrix,
                self.target_adstock_matrix,
            )

            target_transformed_matrix = np.concatenate(
                (
                    np.zeros(
                        (
                            self.max_adstock,
                            target_transformed_matrix.shape[1],
                        )
                    ),
                    target_transformed_matrix,
                ),
                axis=0,
            )
            return target_transformed_matrix
        else:
            return deepcopy(target_calc_regressors_matrix)

    def _derive_attr_marketing(
        self,
        # (n_calc_steps, n_regressors)
        target_transformed_matrix: np.ndarray,
        # (n_regressors, )
        target_coef_array: np.ndarray,
    ) -> np.ndarray:
        attr_marketing = (
            np.exp(
                np.sum(
                    target_coef_array
                    * np.log(1 + target_transformed_matrix / self.target_sat_array),
                    -1,
                )
            )
            - 1
        ) * self.attr_organic[self.calc_mask]

        return attr_marketing

    def _derive_bau_array(
        self,
        # (n_calc_steps, )
        transformed_regressors_matrix: np.ndarray,
        # (n_regressors, )
        target_coef_array: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Derive a time-based array to be used in attribution
        Returns:
            np.ndarray: the array in shape (n_calc_steps, ) where it is equal to
            np.exp(trend + seas + control + marketing)
        """
        return (
            np.exp(
                np.sum(
                    target_coef_array
                    * np.log(1 + transformed_regressors_matrix / self.target_sat_array),
                    -1,
                )
            )
            * self.base_comp_array[self.calc_mask]
        )

    def _derive_market_shares_delta_matrix(
        self,
        # (n_calc_steps, )
        pred_bau: np.ndarray,
        # (n_calc_steps, n_regressors)
        target_calc_regressors_matrix: np.ndarray,
        # (n_calc_steps, n_regressors) but first max-adstock matrix is padded with zeros
        target_transformed_matrix: np.ndarray,
        # (n_regressors, )
        target_coef_array: np.ndarray,
    ) -> np.ndarray:
        """Derive a normalized-market shares of contribution of marketing on each period and time lag

        Returns:
            np.ndarray: _description_
        """
        # n_calc_steps, n_attr_reg = target_calc_regressors_matrix.shape
        # assert n_calc_steps == self.n_calc_steps
        _, n_attr_reg = target_calc_regressors_matrix.shape
        delta_matrix = np.zeros((self.n_calc_steps, self.max_adstock + 1, n_attr_reg))

        # expand in adstock dimension for pred_bau if max_adstock > 0
        if self.max_adstock > 0:
            # expand for delta calculation and pad zeros to preserve rectangular access (the tail is not used)
            pred_bau_expand = np.concatenate(
                [
                    np.expand_dims(pred_bau, -1),
                    np.zeros((self.max_adstock, 1)),
                ],
                axis=0,
            )

            # (n_calc_steps, max_adstock + 1)
            pred_bau_expand = np.squeeze(
                pred_bau_expand[self.access_col_matrix].copy(), -1
            )
        else:
            pred_bau_expand = np.expand_dims(pred_bau, -1)

        for idx in range(n_attr_reg):
            # delta matrix is basically the numerator / denominator
            # denominator: derive the bau regressor matrix which will be used to slice for each channel
            if self.max_adstock > 0:
                # (n_calc_steps + max_adstock, 1)
                temp_bau_transformed_regressor = np.concatenate(
                    [
                        deepcopy(target_transformed_matrix[:, [idx]]),
                        np.zeros((self.max_adstock, 1)),
                    ],
                    axis=0,
                )
                # (n_calc_steps, max_adstock + 1)
                temp_bau_transformed_regressor = np.squeeze(
                    temp_bau_transformed_regressor[self.access_col_matrix],
                    -1,
                )
            else:
                # (n_calc_steps, 1)
                temp_bau_transformed_regressor = deepcopy(
                    target_transformed_matrix[:, [idx]]
                )

            # (n_calc_steps, max_adstock + 1)
            denominator = (
                1 + temp_bau_transformed_regressor / self.target_sat_array[idx]
            ) ** target_coef_array[idx]

            # numerator: numerator is 1 in no-adstock case; otherwise, create scenario which each spend with lags turn
            # to zero
            if self.max_adstock > 0:
                # (n_calc_steps, 1)
                temp_bau_regressor = deepcopy(target_calc_regressors_matrix[:, [idx]])
                # scenario with spend-off step by step
                # (n_scenarios, n_calc_steps, 1)
                temp_full_regressor_zero = self.design_matrix * temp_bau_regressor

                temp_adstock_filter = deepcopy(self.target_adstock_matrix[[idx], :])
                temp_full_regressor_zero = np.squeeze(
                    adstock_process(temp_full_regressor_zero, temp_adstock_filter)
                )
                # (n_scenarios, n_calc_steps + max_adstock, )
                temp_full_regressor_zero = np.concatenate(
                    [
                        np.zeros((self.n_calc_steps, self.max_adstock)),
                        temp_full_regressor_zero,
                        # append max_adstock of zeros for access purpose later
                        np.zeros((self.n_calc_steps, self.max_adstock)),
                    ],
                    axis=-1,
                )

                # (n_calc_steps, max_adstock + 1)
                temp_full_regressor_zero_reduced = np.squeeze(
                    temp_full_regressor_zero[
                        self.access_row_vector,
                        self.access_col_matrix,
                    ],
                    0,
                )

                # (n_calc_steps, max_adstock + 1)
                numerator = (
                    1 + temp_full_regressor_zero_reduced / self.target_sat_array[idx]
                ) ** self.target_coef_array[idx]

            else:
                # in no-adstock case, the "reduced spend" to be the zero spend which
                # leads the numerator to be 1
                numerator = np.ones((self.n_calc_steps, 1))

            # (n_calc_steps, max_adstock + 1)
            temp_delta_matrix = pred_bau_expand * (1 - numerator / denominator)

            # temp delta is the view anchored with spend date for convenient delta
            # calculation; however, they need to be shifted to be anchored with activities
            # date in order to perform normalization; hence, the step below shift
            # down the derived delta to make them aligned at activities date
            if self.max_adstock > 0:
                delta_matrix[:, :, idx] = np_shift(
                    temp_delta_matrix, np.arange(self.max_adstock + 1)
                )
            else:
                delta_matrix[:, :, idx] = temp_delta_matrix

        # fix numeric problem
        # force invalid number to be zero
        # force small number to be zero
        delta_fix_flag = np.logical_and(
            np.logical_not(np.isfinite(delta_matrix)), delta_matrix <= 1e-7
        )
        if self.debug:
            self.logger.info(
                "Total delta fixed flag: {} out of {}".format(
                    np.sum(delta_fix_flag), len(delta_fix_flag.flatten())
                )
            )

        # (n_steps, 1, 1)
        total_delta = np.sum(delta_matrix, axis=(-1, -2), keepdims=True)
        # remove zeros to avoid divide-by-zero issue
        total_delta[total_delta == 0] = 1
        # (n_steps, max_adstock + 1, n_regressors)
        norm_delta_matrix = delta_matrix / total_delta
        return norm_delta_matrix, delta_matrix

    def _derive_attr_matrix(
        self,
        # (n_calc_steps, max_adstock + 1, n_regressors)
        norm_delta_matrix: np.ndarray,
        # (n_calc_steps, )
        attr_marketing: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # (n_steps, max_adstock + 1, n_regressors + 1)
        full_attr_matrix = norm_delta_matrix * attr_marketing.reshape(-1, 1, 1)

        # resolve activities view and spend view attribution matrix
        # activities_attr:
        # sum over lags (adstock) to derive activities_attr
        # (n_steps, n_attr_regressors + 1)
        activities_attr_matrix = np.sum(full_attr_matrix, axis=-2)
        # remove the first and last max_adstock periods as they are not fully observed
        if self.max_adstock > 0:
            activities_attr_matrix = activities_attr_matrix[
                self.max_adstock : -self.max_adstock
            ]

        # spend_attr:
        if self.max_adstock > 0:
            spend_attr_matrix = np.zeros(norm_delta_matrix.shape)
            for idx in range(full_attr_matrix.shape[2]):
                spend_attr_matrix[:, :, idx] = np_shift(
                    full_attr_matrix[:, :, idx],
                    np.arange(0, -1 * (self.max_adstock + 1), -1),
                )
            spend_attr_matrix = np.sum(spend_attr_matrix, axis=-2)
            spend_attr_matrix = spend_attr_matrix[self.max_adstock : -self.max_adstock]
        else:
            # TODO: can verify if both of them are equal later
            spend_attr_matrix = deepcopy(activities_attr_matrix)

        # prevent overflow
        activities_attr_matrix[activities_attr_matrix <= 1e-7] = 0.0
        spend_attr_matrix[spend_attr_matrix <= 1e-7] = 0.0
        return activities_attr_matrix, spend_attr_matrix

    def make_attribution(
        self,
        new_coef_name: Optional[str] = None,
        new_coef: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # update coefficients if new values are specified
        # (n_attr_regressors, )
        target_coef_array = self.target_coef_array
        if new_coef is not None:
            coef_search_found = False
            for idx, x in enumerate(self.target_regressors):
                if new_coef_name == x:
                    coef_search_found = True
                    target_coef_array[idx] = new_coef
                    break
            if not coef_search_found:
                raise Exception(
                    "New coefficient name does not match any regressors in the model."
                )
        target_transformed_matrix = self._derive_target_transformed_matrix(
            self.target_calc_regressors_matrix
        )
        attr_marketing = self._derive_attr_marketing(
            target_transformed_matrix,
            target_coef_array,
        )

        pred_bau_array = self._derive_bau_array(
            target_transformed_matrix,
            target_coef_array,
        )

        norm_delta_matrix, delta_matrix = self._derive_market_shares_delta_matrix(
            pred_bau=pred_bau_array,
            target_calc_regressors_matrix=self.target_calc_regressors_matrix,
            target_transformed_matrix=target_transformed_matrix,
            target_coef_array=target_coef_array,
        )

        activities_attr_matrix, spend_attr_matrix = self._derive_attr_matrix(
            norm_delta_matrix=norm_delta_matrix,
            attr_marketing=attr_marketing,
        )
        activities_attr_matrix = np.round(activities_attr_matrix, 5)
        spend_attr_matrix = np.round(spend_attr_matrix, 5)

        if self.debug:
            self.delta_matrix = delta_matrix
            self.norm_delta_matrix = norm_delta_matrix
            self.activities_attr_matrix = activities_attr_matrix
            self.pred_bau_array = pred_bau_array
            self.target_transformed_matrix = target_transformed_matrix
            self.attr_marketing = attr_marketing

        # note that activities based attribution only makes sense when spend is fully settled after adstock process
        # hence first n(=max_adstock) need to be discarded

        # note that spend attribution only makes sense when all attributing metric fully observed in the entire
        # adstock process
        # also need to discard first n(=max_adstock) observation as you cannot observe the correct pred_bau
        # hence last n(=max_adstock) need to be discarded
        activities_attr_df = pd.DataFrame(
            {self.date_col: self.df[self.input_mask][self.date_col].values}
        )
        # merge back with organic
        # input mask sounds weird although it matches the requirement that
        # the reporting range should be the input spend range
        activities_attr_df["organic"] = self.attr_organic[self.input_mask]
        activities_attr_df[self.target_regressors] = activities_attr_matrix

        spend_attr_df = pd.DataFrame(
            {self.date_col: self.df[self.input_mask][self.date_col].values}
        )
        spend_attr_df["organic"] = self.attr_organic[self.input_mask]
        spend_attr_df[self.target_regressors] = spend_attr_matrix

        spend_df = self.df.loc[
            self.input_mask, [self.date_col] + self.target_regressors
        ]
        spend_df = spend_df.reset_index(drop=True)

        cost_df = spend_df[[self.date_col]].copy()
        cost_df[self.target_regressors] = (
            spend_df[self.target_regressors] / spend_attr_df[self.target_regressors]
        )

        return activities_attr_df, spend_attr_df, spend_df, cost_df
