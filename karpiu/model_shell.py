import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Optional, Tuple, List

from .models import MMM


class MMMShell:
    """A Shell version of MMM freezing a snapshot and target regressors for fast computations"""

    def __init__(
        self,
        model: MMM,
        target_regressors: List[str],
        start: str,
        end: str,
    ):
        # when no adstock, this is zero
        self.max_adstock = model.get_max_adstock()
        # it excludes the fourier-series columns
        self.full_regressors = model.get_regressors()
        # FIXME: right now it DOES NOT work with including control features;
        self.event_regressors = model.get_event_cols()
        self.control_regressors = model.get_control_feat_cols()
        # business as usual dataframe
        self.df = model.raw_df.copy()
        self.start = pd.to_datetime(start)
        self.end = pd.to_datetime(end)
        self.kpi_col = model.kpi_col
        self.date_col = model.date_col

        # better date operations
        # organize the dates. This pads the range with the carry over before it starts
        self.calc_start = self.start - pd.Timedelta(days=self.max_adstock)
        self.calc_end = self.end + pd.Timedelta(days=self.max_adstock)
        self.input_mask = (self.df[self.date_col] >= self.start) & (
            self.df[self.date_col] <= self.end
        )
        self.result_mask = (self.df[self.date_col] >= self.start) & (
            self.df[self.date_col] <= self.calc_end
        )

        self.calc_mask = (self.df[self.date_col] >= self.calc_start) & (
            self.df[self.date_col] <= self.calc_end
        )

        # (n_result_steps, )
        self.dt_array = self.df.loc[self.result_mask, self.date_col].values

        # target related
        # make sure target_regressors input by user align original order from model
        self.target_regressors = [
            x for x in self.full_regressors if x in target_regressors
        ]
        # store background target regressors spend before and after budget period due to adstock
        target_regressor_bkg_matrix = self.df.loc[
            self.calc_mask, self.target_regressors
        ].values
        # only background spend involved; turn off all spend during budget decision period
        target_regressor_bkg_matrix[self.max_adstock : -self.max_adstock, ...] = 0.0
        # (n_calc_steps, n_regressors)
        self.target_regressor_bkg_matrix = target_regressor_bkg_matrix

        # create design matrix for fast computations
        # one off design-matrix with additional first row for full spend
        self.n_regressors = len(self.target_regressors)
        design_matrix_first_row = np.ones((1, 1, self.n_regressors))
        # Note that np.fill is mutating the numpy object; so DO NOT need the assign operator
        one_off_design_matrix = np.ones((self.n_regressors, self.n_regressors))
        np.fill_diagonal(one_off_design_matrix, 0.0)
        one_off_design_matrix = np.expand_dims(one_off_design_matrix, -2)
        # (n_regressors + 1, 1, n_regressors)
        self.design_matrix = np.concatenate(
            [
                design_matrix_first_row,
                one_off_design_matrix,
            ],
            axis=0,
        )
        # (n_input_steps, n_regressors)
        self.target_regressors_matrix = self.df.loc[self.input_mask, self.target_regressors].values
        self.target_adstock_matrix = model.get_adstock_matrix(self.target_regressors)
        sat_df = model.get_saturation()
        # (n_result_steps, n_regressors)
        self.target_coef_matrix = model.get_coef_matrix(
            date_array=self.dt_array,
            regressors=self.target_regressors,
        )
        self.target_sat_array = sat_df.loc[self.target_regressors, "saturation"].values

        # base comp related
        df_zero = self.df.copy()
        df_zero.loc[:, self.target_regressors] = 0.0
        # (n_steps, )
        zero_pred_df = model.predict(df=df_zero, decompose=True)
        # prediction when all target regressors are set to zero
        # (n_result_steps, )
        self.pred_zero = zero_pred_df.loc[self.result_mask, "prediction"].values
