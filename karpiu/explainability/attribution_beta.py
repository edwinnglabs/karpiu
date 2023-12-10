import numpy as np
import pandas as pd
from copy import deepcopy
import logging
from typing import Optional, Tuple, List

from ..utils import adstock_process
from ..model_shell import MMMShellLegacy
from ..models import MMM
from .functions import make_attribution_numpy_beta


class AttributorBeta(MMMShellLegacy):
    def __init__(
        self,
        model: MMM,
        attr_regressors: Optional[List[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs
    ):
        logger.warning(
            "This is the Alpha version of attribution class. Be aware this may be deprecated in future version."
            "For future support, please use the AttributorBeta instead."
        )
        super().__init__(
            model=model,
            target_regressors=attr_regressors,
            start=start,
            end=end,
            df=df,
            **kwargs
        )

        self.attr_start = self.start
        self.attr_end = self.end

        if logger is None:
            self.logger = logging.getLogger("karpiu-planning")
        else:
            self.logger = logger

        # for debug
        self.delta_matrix = None

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

        # dynamic due to coefficient matrix input
        self.pred_bau = None

        # adstock_regressor_matrix dim: time x num of regressors
        if self.max_adstock >= 1:
            # adstock transformed regressors will be used to calculate pred_bau later
            self.attr_transformed_regressor_matrix = adstock_process(
                self.target_calc_regressors_matrix, self.target_adstock_matrix
            )
            # we lose first n(=max_adstock) observations; to maintain original dimension,
            # we need to pad zeros n(=max_adstock) time; note that they will not be
            # really used; it is for preserving shape
            self.attr_transformed_regressor_matrix = np.concatenate(
                (
                    np.zeros(
                        (
                            self.max_adstock,
                            self.attr_transformed_regressor_matrix.shape[1],
                        )
                    ),
                    self.attr_transformed_regressor_matrix,
                ),
                axis=0,
            )
        else:
            self.attr_transformed_regressor_matrix = deepcopy(
                self.target_input_regressors_matrix
            )

        self.attr_coef_matrix = model.get_coef_matrix(
            date_array=self.calc_dt_array,
            regressors=self.target_regressors,
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

    def make_attribution(
        self,
        new_coef_name: Optional[str] = None,
        new_coef: Optional[float] = None,
        true_up: bool = True,
        fixed_intercept: bool = True,
        debug: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """_summary_

        Args:
            new_coef_name (Optional[str], optional): Optional. If supplied, it also requires the new_coef to be
            supplied with the same length to replace to existing regression coefficients of the model in performing
            attribution
            new_coef (Optional[float], optional): _description_. Defaults to None. If supplied, it will be used to
            replace to existing regression coefficients of the model in performing attribution
            true_up (bool, optional): _description_. Defaults to True. If true, the final attribution total will be
            true-up with the original observed response.
            fixed_intercept (bool, optional): _description_. Defaults to True. If true, intercept estimate will not be
            involved in the linearization process with other channels raw-attribution. It will be forced to turn-off if
            it violates the condition of true-up response < intercept.
            debug (bool, optional): _description_. Defaults to False. Logic to turn on debug statements.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: attribution result
        """
        date_col = self.date_col

        # (n_attr_regressors, )
        attr_coef_array = self.target_coef_array
        if new_coef is not None:
            coef_search_found = False
            for idx, x in enumerate(self.target_regressors):
                if new_coef_name == x:
                    coef_search_found = True
                    attr_coef_array[idx] = new_coef
                    break
            if not coef_search_found:
                raise Exception(
                    "New coefficient name does not match any regressors in the model."
                )

        varying_comp = np.sum(
            attr_coef_array
            * np.log1p(self.attr_transformed_regressor_matrix / self.target_sat_array),
            -1,
        )

        # (n_calc_steps, )
        self.pred_bau = self.base_comp_calc * np.exp(varying_comp)

        if true_up:
            true_up_array = self.df.loc[self.calc_mask, self.kpi_col].values
            if fixed_intercept:
                if np.any(true_up_array < self.base_comp_calc):
                    raise Exception(
                        "Fixed intercept is not allowed due to true_up_array < intercept."
                    )
        else:
            true_up_array = self.pred_bau

        (
            activities_attr_matrix,
            spend_attr_matrix,
            delta_matrix,
        ) = make_attribution_numpy_beta(
            attr_coef_array=attr_coef_array,
            attr_regressor_matrix=self.target_calc_regressors_matrix,
            attr_transformed_regressor_matrix=self.attr_transformed_regressor_matrix,
            pred_bau=self.pred_bau,
            pred_zero=self.base_comp_calc,
            adstock_matrix=self.target_adstock_matrix,
            attr_saturation_array=self.target_sat_array,
            true_up_arr=true_up_array,
            fixed_intercept=fixed_intercept,
        )

        if debug:
            self.delta_matrix = delta_matrix

        # note that activities based attribution only makes sense when spend is fully settled after adstock process
        # hence first n(=max_adstock) need to be discarded

        # note that spend attribution only makes sense when all attributing metric fully observed in the entire
        # adstock process
        # also need to discard first n(=max_adstock) observation as you cannot observe the correct pred_bau
        # hence last n(=max_adstock) need to be discarded
        activities_attr_df = pd.DataFrame(
            {date_col: self.df[self.input_mask][date_col].values}
        )
        activities_attr_df[
            ["organic"] + self.target_regressors
        ] = activities_attr_matrix

        spend_attr_df = pd.DataFrame(
            {date_col: self.df[self.input_mask][date_col].values}
        )
        spend_attr_df[["organic"] + self.target_regressors] = spend_attr_matrix

        spend_df = self.df.loc[self.input_mask, [date_col] + self.target_regressors]
        spend_df = spend_df.reset_index(drop=True)

        cost_df = spend_df[[date_col]].copy()
        cost_df[self.target_regressors] = (
            spend_df[self.target_regressors] / spend_attr_df[self.target_regressors]
        )

        return activities_attr_df, spend_attr_df, spend_df, cost_df
