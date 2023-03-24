import pandas as pd
import numpy as np
from typing import Optional, List
import logging
import scipy.optimize as optim
from copy import deepcopy

from karpiu.models import MMM
from karpiu.model_shell import MMMShell

# from karpiu.explainability import Attributor


class TotalBudgetOptimizer(MMMShell):
    """Base class for optimization solution"""

    def __init__(
        self,
        model: MMM,
        budget_start: str,
        budget_end: str,
        optim_channels: str,
        response_scaler: float = 1.0,
        spend_scaler: float = 1.0,
        logger: Optional[logging.Logger] = None,
        total_budget_override: Optional[float] = None,
        weight: Optional[np.array] = None,
    ):
        if logger is None:
            self.logger = logging.getLogger("karpiu-planning")
        else:
            self.logger = logger

        super().__init__(
            model=model,
            target_regressors=optim_channels,
            start=budget_start,
            end=budget_end,
        )

        self.budget_start = self.start
        self.budget_end = self.end
        self.optim_channels = optim_channels
        self.logger.info("Optimizing channels : {}".format(self.optim_channels))

        self.response_scaler = response_scaler
        self.spend_scaler = spend_scaler
        self.constraints = list()
        self.budget_mask = self.input_mask

        # derive optimization input
        # derive init values
        # (n_budget_steps * n_optim_channels, )
        self.init_spend_matrix = self.df.loc[
            self.budget_mask, self.optim_channels
        ].values
        # total spend per channel
        # (n_optim_channels, )
        self.init_spend_array = np.sum(self.init_spend_matrix, 0)
        # this stores current optimal spend
        self.curr_spend_matrix = deepcopy(self.init_spend_matrix)
        self.curr_spend_array = deepcopy(self.init_spend_array)

        self.n_optim_channels = len(self.optim_channels)
        n_budget_steps = np.sum(self.input_mask)
        if total_budget_override is not None and total_budget_override > 0:
            self.total_budget = total_budget_override
        else:
            self.total_budget = np.sum(self.init_spend_array)
        self.n_budget_steps = n_budget_steps

        total_budget_constraint = self.generate_total_budget_constraint(
            total_budget=self.total_budget
        )
        self.add_constraints([total_budget_constraint])
        # ind_budget_constraints = self.generate_individual_channel_constraints(delta=0.1)
        # self.add_constraints(ind_budget_constraints)

        # derive budget bounds for each step and each channel
        self.budget_bounds = optim.Bounds(
            lb=np.zeros(self.n_optim_channels),
            ub=np.ones(self.n_optim_channels) * self.total_budget / self.spend_scaler,
        )

        # spend allocation on time dimension
        # (n_budget_steps, )
        if weight is None:
            self.weight = self.base_comp_input / np.sum(self.base_comp_input)
        else:
            self.weight = weight

        if len(self.weight) != self.n_budget_steps:
            raise Exception("Input weight has different length from budget period.")


    def set_constraints(self, constraints: List[optim.LinearConstraint]):
        self.constraints = constraints

    def add_constraints(self, constraints: List[optim.LinearConstraint]):
        self.constraints += constraints

    def generate_total_budget_constraint(
        self, total_budget: float
    ) -> optim.LinearConstraint:
        # derive budget constraints based on total sum of init values
        # scipy.optimize.LinearConstraint notation: lb <= A.dot(x) <= ub
        total_budget_constraint = optim.LinearConstraint(
            A=np.ones(self.n_optim_channels),
            lb=np.zeros(1),
            ub=np.ones(1) * total_budget / self.spend_scaler,
        )
        return total_budget_constraint

    # def generate_individual_channel_constraints(self, delta=0.1):
    #     constraints = list()
    #     init_spend_channel_total_arr = (
    #         np.sum(self.init_spend_matrix, 0) / self.spend_scaler
    #     )
    #     for idx in range(self.n_optim_channels):
    #         lb = (1 - delta) * init_spend_channel_total_arr[idx]
    #         ub = (1 + delta) * init_spend_channel_total_arr[idx]
    #         A = np.zeros((self.n_budget_steps, self.n_optim_channels))
    #         A[:, idx] = 1.0
    #         ind_constraint = optim.LinearConstraint(
    #             A=A.flatten(),
    #             lb=lb,
    #             ub=ub,
    #         )
    #         constraints.append(ind_constraint)
    #     return constraints

    def get_df(self) -> pd.DataFrame:
        df = self.df.copy()
        return df

    def get_current_state(self) -> np.array:
        return deepcopy(self.curr_spend_matrix)

    def get_total_budget(self) -> float:
        return deepcopy(self.total_budget)

    def get_init_state(self) -> np.array:
        return deepcopy(self.init_spend_matrix)

    def objective_func(self, spend: np.array):
        raise Exception(
            "Abstract objective function. Child class needs to override this method to have concrete result."
        )

    def optimize(
        self,
        init: Optional[np.array] = None,
        maxiter: int = 2,
        eps: float = 1e-03,
        ftol: float = 1e-03,
    ) -> None:
        if init is None:
            x0 = self.init_spend_array / self.spend_scaler
        else:
            x0 = init.flatten() / self.spend_scaler

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

        optim_spend_array = sol.x * self.spend_scaler
        optim_spend_matrix = (
            optim_spend_array
            * np.ones((self.n_budget_steps, self.n_optim_channels))
            * np.expand_dims(self.weight, -1) 
        )

        optim_df = self.get_df()
        optim_df.loc[self.budget_mask, self.optim_channels] = optim_spend_matrix
        self.curr_spend_matrix = optim_spend_matrix
        self.curr_spend_array = optim_spend_array
        return optim_df
