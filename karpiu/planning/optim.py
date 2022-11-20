import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from typing import Optional
import logging
from ..models import MMM
import scipy.optimize as optim
from copy import deepcopy

from ..explainability import Attributor
from ..utils import adstock_process


class ResponseMaximizer:
    """Perform optimization with a given Marketing Mix Model"""

    def __init__(
        self,
        model: MMM,
        budget_start: str,
        budget_end: str,
        optim_channel: str,
        response_scaler: float=1e2,
        spend_scaler: float=1e4,
        logger: Optional[logging.Logger] = None,
    ):
        if logger is None:
            self.logger = logging.getLogger("karpiu-planning")
        else:
            self.logger = logger

        # self.model = model
        self.df = model.get_raw_df()
        df = self.df.copy()
        self.date_col = model.date_col
        self.budget_start = pd.to_datetime(budget_start)
        self.budget_end = pd.to_datetime(budget_end)
        self.optim_channel = optim_channel
        # TODO: check optim channel is in the mmm.spend_cols
        self.response_scaler = response_scaler
        self.spend_scaler = spend_scaler

        # derive basic attributes
        self.n_max_adstock = model.get_max_adstock()
        self.calc_start = self.budget_start - pd.Timedelta(days=self.n_max_adstock)  
        self.calc_end = self.budget_end + pd.Timedelta(days=self.n_max_adstock) 
        self.all_regressor = model.get_regressors() 
        self.event_regressor = model.get_event_cols()
        self.spend_regressor = model.get_spend_cols()
        self.non_optim_regressor = list(
            set(self.all_regressor) - set(self.optim_channel) - set(self.event_regressor)
        )

        # TODO: some log info here

        # derive masks and dt_array
        budget_mask = (df[self.date_col] >= self.budget_start) & (df[self.date_col] <= self.budget_end)
        self.budget_mask = budget_mask
        calc_mask =  (df[self.date_col] >= self.calc_start) & (df[self.date_col] <= self.calc_end)
        self.calc_mask = calc_mask
        calc_dt_array = df.loc[calc_mask, self.date_col].values

        # derive optimization input
        # derive init values
        # (n_budget_steps * n_optim_channels, )
        self.init_spend_matrix  = df.loc[budget_mask, self.optim_channel].values
        self.curr_spend_matrix = deepcopy(self.init_spend_matrix)
        self.n_optim_channels = len(self.optim_channel)
        n_budget_steps = np.sum(budget_mask)
        self.total_budget = np.sum(self.init_spend_matrix)
        self.n_budget_steps = n_budget_steps

        # store required matrices and arrays for transformation of channel which is being optimized 
        sat_df = model.get_saturation()
        # (n_optim_channels, )
        self.optim_sat_array = sat_df.loc[optim_channel, 'saturation'].values
        # (n_optim_channels, n_max_adstock + 1)
        self.optim_adstock_matrix = model.get_adstock_matrix(optim_channel)
        # (n_budget_steps + n_max_adstock, n_optim_channels)
        optim_coef_matrix = model.get_coef_matrix(date_array=calc_dt_array, regressors=optim_channel)
        self.optim_coef_matrix = optim_coef_matrix[self.n_max_adstock:]

        # derive budget constraints based on total sum of init values
        total_budget_constraint = optim.LinearConstraint(
            np.ones(n_budget_steps * self.n_optim_channels),
            np.zeros(1),
            np.ones(1) * self.total_budget / self.spend_scaler,
        )
        self.constraints = [total_budget_constraint]

        # derive budget bounds for each step and each channel
        self.budget_bounds = optim.Bounds(
            lb=np.zeros(n_budget_steps * self.n_optim_channels),
            ub=np.ones(n_budget_steps * self.n_optim_channels) * self.total_budget / self.spend_scaler,
        )

        # derive base comp
        pred_df = model.predict(df, decompose=True)
        pred_df = pred_df.loc[calc_mask].reset_index(drop=True)
        trend = pred_df['trend'].values
        base_comp = trend[self.n_max_adstock:]

        # TODO: so this code can be replaced by Attributor?
        # TODO: check how it did the calculation in RevenueMaximizer
        if len(self.non_optim_regressor) > 0:
            coef_matrix = model.get_coef_matrix(date_array=calc_dt_array, regressors=self.non_optim_regressor)
            coef_matrix = coef_matrix[self.n_max_adstock:, ...]
        
            sat_df = model.get_saturation()
            sat_array = sat_df.loc[self.non_optim_regressor, 'saturation'].values
            temp_df = df.copy()
            temp_df = temp_df.set_index('dt')
            regressor_matrix = temp_df.loc[calc_dt_array, self.non_optim_regressor].values
            adstock_matrix = model.get_adstock_matrix(self.non_optim_regressor)
            adj_regressor_matrix = adstock_process(regressor_matrix, adstock_matrix)
            reg_non_optim = np.sum(coef_matrix * np.log1p(adj_regressor_matrix / sat_array), -1)
            base_comp += reg_non_optim
            
        if len(self.event_regressor) > 0:
            coef_matrix = model.get_coef_matrix(date_array=calc_dt_array, regressors=self.event_regressor)
            coef_matrix = coef_matrix[self.n_max_adstock:, ...]
            temp_df = df.copy()
            temp_df = temp_df.set_index('dt')
            regressor_matrix = temp_df.loc[calc_dt_array, self.event_regressor].values
            regressor_matrix = regressor_matrix[self.n_max_adstock:, ...]
            reg_non_optim = np.sum(coef_matrix * regressor_matrix, -1)
            base_comp += reg_non_optim

        self.base_comp = base_comp

         # store background spend before and after budget period due to adstock
        bkg_spend_matrix = df.loc[calc_mask, self.optim_channel].values
        # only background spend involved; turn off all spend during budget decision period
        bkg_spend_matrix[self.n_max_adstock:-self.n_max_adstock, ...] = 0.0
        self.bkg_spend_matrix = bkg_spend_matrix

    def objective_func(self, spend):
        spend_matrix = spend.reshape(-1, self.n_optim_channels) * self.spend_scaler
        zero_paddings = np.zeros((self.n_max_adstock, self.n_optim_channels))
        spend_matrix = np.concatenate([zero_paddings, spend_matrix, zero_paddings], 0)
        spend_matrix += self.bkg_spend_matrix
        transformed_spend_matrix = adstock_process(spend_matrix, self.optim_adstock_matrix)
        # regression
        spend_comp = np.sum(self.optim_coef_matrix * np.log1p(transformed_spend_matrix / self.optim_sat_array), -1)
        pred_outcome = np.exp(self.base_comp + spend_comp)
        loss = -1 * np.sum(pred_outcome) / self.response_scaler
        return loss

    def optimize(
        self, 
        init: Optional[np.array] = None,
        maxiter: int = 2,
        eps: float = 1.,
        ftol: float = 1e-07,
    ) -> None:
        
        if init is None:
            x0 = self.init_spend_matrix.flatten() / self.spend_scaler
        else:
            x0 = init.flatten()

        sol = optim.minimize(
            self.objective_func,
            x0=x0,
            method='SLSQP',
            bounds=self.budget_bounds,
            constraints=self.constraints,
            options={
                'disp': True,
                'maxiter': maxiter,
                'eps': eps,
                'ftol': ftol,
            }
        )

        optim_spend_matrix = sol.x.reshape(-1, self.n_optim_channels) * self.spend_scaler
        optim_df = self.get_df()
        optim_df.loc[self.budget_mask, self.optim_channel] = optim_spend_matrix
        self.curr_spend_matrix = optim_spend_matrix
        return optim_df

    def get_df(self) -> pd.DataFrame:
        df = self.df.copy()
        return df
    
    def get_current_state(self) -> np.array:
        return deepcopy(self.curr_spend_matrix)

    def get_total_budget(self) -> float:
        return deepcopy(self.total_budget)
    
    def get_init_state(self) -> np.array:
        return deepcopy(self.init_spend_matrix)


class RevenueMaximizer:
    """Perform revenue optimization with a given Marketing Mix Model and 
    lift-time values (LTV) per channel 
    """

    def __init__(
        self,
        model: MMM,
        budget_start: str,
        budget_end: str,
        optim_channel: str,
        ltv_arr: np.array,
        response_scaler: float=1e2,
        spend_scaler: float=1e4,
        logger: Optional[logging.Logger] = None,
    ):
        if logger is None:
            self.logger = logging.getLogger("karpiu-planning")
        else:
            self.logger = logger

        # self.model = model
        self.df = model.get_raw_df()
        df = self.df.copy()
        self.date_col = model.date_col
        self.budget_start = pd.to_datetime(budget_start)
        self.budget_end = pd.to_datetime(budget_end)
        self.optim_channel = optim_channel
        # TODO: check optim channel is in the mmm.spend_cols
        self.response_scaler = response_scaler
        self.spend_scaler = spend_scaler
        self.ltv_arr = ltv_arr

        # derive basic attributes
        self.n_max_adstock = model.get_max_adstock()
        self.calc_start = self.budget_start - pd.Timedelta(days=self.n_max_adstock)  
        self.calc_end = self.budget_end + pd.Timedelta(days=self.n_max_adstock) 
        self.all_regressor = model.get_regressors() 
        self.event_regressor = model.get_event_cols()
        self.spend_regressor = model.get_spend_cols()
        self.non_optim_regressor = list(
            set(self.all_regressor) - set(self.optim_channel) - set(self.event_regressor)
        )

        # TODO: some log info here
        # derive masks and dt_array
        budget_mask = (df[self.date_col] >= self.budget_start) & (df[self.date_col] <= self.budget_end)
        self.budget_mask = budget_mask
        calc_mask =  (df[self.date_col] >= self.calc_start) & (df[self.date_col] <= self.calc_end)
        self.calc_mask = calc_mask
        calc_dt_array = df.loc[calc_mask, self.date_col].values
        n_calc_steps = np.sum(calc_mask)

        # derive optimization input
        # derive init values
        # (n_budget_steps * n_optim_channels, )
        self.init_spend_matrix  = df.loc[budget_mask, self.optim_channel].values
         # (n_budget_steps * n_optim_channels, )
        self.curr_spend_matrix = deepcopy(self.init_spend_matrix)
        self.n_optim_channels = len(self.optim_channel)
        n_budget_steps = np.sum(budget_mask)
        self.total_budget = np.sum(self.init_spend_matrix)
        self.n_budget_steps = n_budget_steps

        # leverage Attributor to get base comp and pred_zero (for 1-off approximation)
        attr_obj = Attributor(
            model, 
            attr_regressors=optim_channel, 
            start=budget_start, 
            end=budget_end,
        )
         # (n_budget_steps + n_max_adstock, )
        self.base_comp = attr_obj.base_comp[self.n_max_adstock:]
        self.optim_adstock_matrix = attr_obj.attr_adstock_matrix
        # (n_optim_channels, )
        self.optim_sat_array = attr_obj.attr_sat_array
        # (n_budget_steps + n_max_adstock, n_optim_channels)
        self.optim_coef_matrix = attr_obj.attr_coef_matrix[self.n_max_adstock:]

         # store background spend before and after budget period due to adstock
        bkg_spend_matrix = df.loc[calc_mask, self.optim_channel].values
        # only background spend involved; turn off all spend during budget decision period
        bkg_spend_matrix[self.n_max_adstock:-self.n_max_adstock, ...] = 0.0
        self.bkg_spend_matrix = bkg_spend_matrix
        transformed_bkg_matrix = adstock_process(bkg_spend_matrix, self.optim_adstock_matrix)
        self.bkg_attr_comp = self.optim_coef_matrix * np.log1p(transformed_bkg_matrix / self.optim_sat_array)
        
        # derive budget constraints based on total sum of init values
        total_budget_constraint = optim.LinearConstraint(
            np.ones(n_budget_steps * self.n_optim_channels),
            np.zeros(1),
            np.ones(1) * self.total_budget / self.spend_scaler,
        )
        self.constraints = [total_budget_constraint]

        # derive budget bounds for each step and each channel
        self.budget_bounds = optim.Bounds(
            lb=np.zeros(n_budget_steps * self.n_optim_channels),
            ub=np.ones(n_budget_steps * self.n_optim_channels) * self.total_budget / self.spend_scaler,
        )

    
    def objective_func(self, spend):
        spend_matrix = spend.reshape(-1, self.n_optim_channels) * self.spend_scaler
        zero_paddings = np.zeros((self.n_max_adstock, self.n_optim_channels))
        spend_matrix = np.concatenate([zero_paddings, spend_matrix, zero_paddings], 0)
        spend_matrix += self.bkg_spend_matrix
        transformed_spend_matrix = adstock_process(spend_matrix, self.optim_adstock_matrix)
        # regression
        # (n_budget_steps + n_max_adstock, n_optim_channels)
        varying_attr_comp = self.optim_coef_matrix  * np.log1p(transformed_spend_matrix / self.optim_sat_array)
        # one-off approximation
        # base comp does not have the channel dimension so we need expand on last dim
        # (n_budget_steps + n_max_adstock, n_optim_channels)
        attr_matrix = (
            np.exp(np.expand_dims(self.base_comp, -1)) * 
            (np.exp(varying_attr_comp) - np.exp(self.bkg_attr_comp))
        )
        revenue = self.ltv_arr * np.sum(attr_matrix, 0)
        # (n_optim_channels, )
        cost = np.sum(spend_matrix, 0)
        net_profit = np.sum(revenue - cost)
        loss = -1 * net_profit / self.response_scaler
        return loss

    def optimize(
            self, 
            init: Optional[np.array] = None,
            maxiter: int = 2,
            eps: float = 1e-3,
            ftol: float = 1e-07,
        ) -> None:
            
            if init is None:
                x0 = self.init_spend_matrix.flatten() / self.spend_scaler
            else:
                x0 = init.flatten()

            sol = optim.minimize(
                self.objective_func,
                x0=x0,
                method='SLSQP',
                bounds=self.budget_bounds,
                constraints=self.constraints,
                options={
                    'disp': True,
                    'maxiter': maxiter,
                    'eps': eps,
                    'ftol': ftol,
                }
            )

            optim_spend_matrix = sol.x.reshape(-1, self.n_optim_channels) * self.spend_scaler
            optim_df = self.get_df()
            optim_df.loc[self.budget_mask, self.optim_channel] = optim_spend_matrix
            self.curr_spend_matrix = optim_spend_matrix
            return optim_df
    
    def get_df(self) -> pd.DataFrame:
        df = self.df.copy()
        return df
    
    def get_current_state(self) -> np.array:
        return deepcopy(self.curr_spend_matrix)

    def get_total_budget(self) -> float:
        return deepcopy(self.total_budget)
    
    def get_init_state(self) -> np.array:
        return deepcopy(self.init_spend_matrix)