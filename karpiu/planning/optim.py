import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from typing import Optional, List
import logging
from ..models import MMM
import scipy.optimize as optim
from copy import deepcopy

from ..explainability import Attributor
from ..utils import adstock_process


# TODO:
# add more issues e.g.
# 1. add individual budget constraints [done]
# 2. add tutorials and demo [done]
# 3. add attribution tutorials
# 4. add better auto doc workflow [done]
# 5. add unit test of attribution
# 6. add unit test of optimization


class BudgetOptimizer:
    """Base class for optimization solution"""

    def __init__(
        self,
        model: MMM,
        budget_start: str,
        budget_end: str,
        optim_channel: str,
        response_scaler: float = 1e2,
        spend_scaler: float = 1e4,
        logger: Optional[logging.Logger] = None,
    ):
        if logger is None:
            self.logger = logging.getLogger("karpiu-planning")
        else:
            self.logger = logger

        self.df = model.get_raw_df()
        df = self.df.copy()
        self.date_col = model.date_col
        self.budget_start = pd.to_datetime(budget_start)
        self.budget_end = pd.to_datetime(budget_end)
        self.optim_channel = optim_channel
        self.optim_channel.sort()
        self.logger.info(
            "Optimizing channels is sorted. They are now : {}".format(
                self.optim_channel
            )
        )
        # TODO: check optim channel is in the model.spend_cols
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
            set(self.all_regressor)
            - set(self.optim_channel)
            - set(self.event_regressor)
        )
        self.constraints = list()

        # some masks derivation to extract data with specific periods effectively
        budget_mask = (df[self.date_col] >= self.budget_start) & (
            df[self.date_col] <= self.budget_end
        )
        self.budget_mask = budget_mask
        calc_mask = (df[self.date_col] >= self.calc_start) & (
            df[self.date_col] <= self.calc_end
        )
        self.calc_mask = calc_mask
        self.calc_dt_array = df.loc[calc_mask, self.date_col].values

        # derive optimization input
        # derive init values
        # (n_budget_steps * n_optim_channels, )
        self.init_spend_matrix = df.loc[budget_mask, self.optim_channel].values
        # (n_budget_steps * n_optim_channels, ); this stores current optimal spend
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
        self.base_comp = attr_obj.base_comp[self.n_max_adstock :]

        # store some numpy arrays for channels to be optimized so that it can be
        # used in objective function
        self.optim_adstock_matrix = attr_obj.attr_adstock_matrix
        # (n_optim_channels, )
        self.optim_sat_array = attr_obj.attr_sat_array
        # (n_budget_steps + n_max_adstock, n_optim_channels)
        self.optim_coef_matrix = attr_obj.attr_coef_matrix[self.n_max_adstock :]

        # store background spend before and after budget period due to adstock
        bkg_spend_matrix = df.loc[calc_mask, self.optim_channel].values
        # only background spend involved; turn off all spend during budget decision period
        bkg_spend_matrix[self.n_max_adstock : -self.n_max_adstock, ...] = 0.0
        self.bkg_spend_matrix = bkg_spend_matrix

        total_budget_constraint = self.generate_total_budget_constraint()
        self.add_constraints([total_budget_constraint])
        # ind_budget_constraints = self.generate_individual_channel_constraints(delta=0.1)
        # self.add_constraints(ind_budget_constraints)

        # derive budget bounds for each step and each channel
        self.budget_bounds = optim.Bounds(
            lb=np.zeros(n_budget_steps * self.n_optim_channels),
            ub=np.ones(n_budget_steps * self.n_optim_channels)
            * self.total_budget
            / self.spend_scaler,
        )

    def set_constraints(self, constraints: List[optim.LinearConstraint]):
        self.constraints = constraints

    def add_constraints(self, constraints: List[optim.LinearConstraint]):
        self.constraints += constraints

    def generate_total_budget_constraint(self) -> optim.LinearConstraint:
        # derive budget constraints based on total sum of init values
        # scipy.optimize.LinearConstraint notation: lb <= A.dot(x) <= ub
        total_budget_constraint = optim.LinearConstraint(
            A=np.ones(self.n_budget_steps * self.n_optim_channels),
            lb=np.zeros(1),
            ub=np.ones(1) * self.total_budget / self.spend_scaler,
        )
        return total_budget_constraint

    def generate_individual_channel_constraints(self, delta=0.1):
        constraints = list()
        init_spend_channel_total_arr = (
            np.sum(self.init_spend_matrix, 0) / self.spend_scaler
        )
        for idx in range(self.n_optim_channels):
            lb = (1 - delta) * init_spend_channel_total_arr[idx]
            ub = (1 + delta) * init_spend_channel_total_arr[idx]
            A = np.zeros((self.n_budget_steps, self.n_optim_channels))
            A[:, idx] = 1.0
            ind_constraint = optim.LinearConstraint(
                A=A.flatten(),
                lb=lb,
                ub=ub,
            )
            constraints.append(ind_constraint)
        return constraints

    def get_df(self) -> pd.DataFrame:
        df = self.df.copy()
        return df

    def get_current_state(self) -> np.array:
        return deepcopy(self.curr_spend_matrix)

    def get_total_budget(self) -> float:
        return deepcopy(self.total_budget)

    def get_init_state(self) -> np.array:
        return deepcopy(self.init_spend_matrix)

    def objective_func(self, spend):
        raise Exception(
            "Abstract objective function. Child class needs to override this method to have concrete result."
        )

    def optimize(
        self,
        init: Optional[np.array] = None,
        maxiter: int = 2,
        eps: float = 1.0,
        ftol: float = 1e-07,
    ) -> None:

        if init is None:
            x0 = self.init_spend_matrix.flatten() / self.spend_scaler
        else:
            x0 = init.flatten()

        sol = optim.minimize(
            self.objective_func,
            x0=x0,
            method="SLSQP",
            bounds=self.budget_bounds,
            constraints=self.constraints,
            options={
                "disp": True,
                "maxiter": maxiter,
                "eps": eps,
                "ftol": ftol,
            },
        )

        optim_spend_matrix = (
            sol.x.reshape(-1, self.n_optim_channels) * self.spend_scaler
        )
        optim_df = self.get_df()
        optim_df.loc[self.budget_mask, self.optim_channel] = optim_spend_matrix
        self.curr_spend_matrix = optim_spend_matrix
        return optim_df


class TargetMaximizer(BudgetOptimizer):
    """Perform optimization with a given Marketing Mix Model"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def objective_func(self, spend):
        spend_matrix = spend.reshape(-1, self.n_optim_channels) * self.spend_scaler
        zero_paddings = np.zeros((self.n_max_adstock, self.n_optim_channels))
        spend_matrix = np.concatenate([zero_paddings, spend_matrix, zero_paddings], 0)
        spend_matrix += self.bkg_spend_matrix
        transformed_spend_matrix = adstock_process(
            spend_matrix, self.optim_adstock_matrix
        )
        # regression
        spend_comp = np.sum(
            self.optim_coef_matrix
            * np.log1p(transformed_spend_matrix / self.optim_sat_array),
            -1,
        )
        pred_outcome = np.exp(self.base_comp + spend_comp)
        loss = -1 * np.sum(pred_outcome) / self.response_scaler
        return loss


class RevenueMaximizer(BudgetOptimizer):
    """Perform revenue optimization with a given Marketing Mix Model and
    lift-time values (LTV) per channel
    """

    def __init__(self, ltv_arr: np.array, **kwargs):
        super().__init__(**kwargs)
        transformed_bkg_matrix = adstock_process(
            self.bkg_spend_matrix, self.optim_adstock_matrix
        )
        self.bkg_attr_comp = self.optim_coef_matrix * np.log1p(
            transformed_bkg_matrix / self.optim_sat_array
        )
        self.ltv_arr = ltv_arr

    def objective_func(self, spend):
        spend_matrix = spend.reshape(-1, self.n_optim_channels) * self.spend_scaler
        zero_paddings = np.zeros((self.n_max_adstock, self.n_optim_channels))
        spend_matrix = np.concatenate([zero_paddings, spend_matrix, zero_paddings], 0)
        spend_matrix += self.bkg_spend_matrix
        transformed_spend_matrix = adstock_process(
            spend_matrix, self.optim_adstock_matrix
        )
        # regression
        # (n_budget_steps + n_max_adstock, n_optim_channels)
        varying_attr_comp = self.optim_coef_matrix * np.log1p(
            transformed_spend_matrix / self.optim_sat_array
        )
        # one-off approximation
        # base comp does not have the channel dimension so we need expand on last dim
        # (n_budget_steps + n_max_adstock, n_optim_channels)
        attr_matrix = np.exp(np.expand_dims(self.base_comp, -1)) * (
            np.exp(varying_attr_comp) - np.exp(self.bkg_attr_comp)
        )
        revenue = self.ltv_arr * np.sum(attr_matrix, 0)
        # (n_optim_channels, )
        cost = np.sum(spend_matrix, 0)
        net_profit = np.sum(revenue - cost)
        loss = -1 * net_profit / self.response_scaler
        return loss

    # def optimize(
    #     self,
    #     init: Optional[np.array] = None,
    #     maxiter: int = 2,
    #     eps: float = 1e-3,
    #     ftol: float = 1e-07,
    # ) -> None:

    #     if init is None:
    #         x0 = self.init_spend_matrix.flatten() / self.spend_scaler
    #     else:
    #         x0 = init.flatten()

    #     sol = optim.minimize(
    #         self.objective_func,
    #         x0=x0,
    #         method="SLSQP",
    #         bounds=self.budget_bounds,
    #         constraints=self.constraints,
    #         options={
    #             "disp": True,
    #             "maxiter": maxiter,
    #             "eps": eps,
    #             "ftol": ftol,
    #         },
    #     )

    #     optim_spend_matrix = (
    #         sol.x.reshape(-1, self.n_optim_channels) * self.spend_scaler
    #     )
    #     optim_df = self.get_df()
    #     optim_df.loc[self.budget_mask, self.optim_channel] = optim_spend_matrix
    #     self.curr_spend_matrix = optim_spend_matrix
    #     return optim_df
