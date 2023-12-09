import pandas as pd
import numpy as np
from typing import Optional, List, Dict
import logging
import scipy.optimize as optim
from copy import deepcopy

from karpiu.models import MMM
from karpiu.model_shell import MMMShellLegacy

class BudgetOptimizer(MMMShellLegacy):
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
    ):
        self.optim_channels = optim_channels
        self.optim_channels.sort()

        super().__init__(
            model=model,
            target_regressors=self.optim_channels,
            start=budget_start,
            end=budget_end,
        )

        if logger is None:
            self.logger = logging.getLogger("karpiu-planning")
        else:
            self.logger = logger

        self.logger.info(
            "Optimizing channels is sorted. They are now : {}".format(
                self.optim_channels
            )
        )

        self.budget_start = self.start
        self.budget_end = self.end

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
        # (n_budget_steps * n_optim_channels, ); this stores current optimal spend
        self.curr_spend_matrix = deepcopy(self.init_spend_matrix)
        self.n_optim_channels = len(self.optim_channels)
        n_budget_steps = np.sum(self.input_mask)
        if total_budget_override is not None and total_budget_override > 0:
            self.total_budget = total_budget_override
        else:
            self.total_budget = np.sum(self.init_spend_matrix)
        self.n_budget_steps = n_budget_steps

        total_budget_constraint = self.generate_total_budget_constraint(
            total_budget=self.total_budget
        )
        self.add_constraints([total_budget_constraint])
        # ind_budget_constraints = self.generate_individual_channel_constraints(delta=0.1)
        # self.add_constraints(ind_budget_constraints)

        # derive budget bounds for each step and each channel
        self.budget_bounds = optim.Bounds(
            lb=np.zeros(n_budget_steps * self.n_optim_channels),
            ub=np.ones(n_budget_steps * self.n_optim_channels) * np.inf,
        )

        # create a dict to store all return metrics from callback
        self.callback_metrics = dict()
        self._init_callback_metrics()
        self.bounds_and_constraints_df = None

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
            A=np.ones(self.n_budget_steps * self.optim_channels),
            lb=np.zeros(1),
            # lb=np.ones(1) * total_budget / self.spend_scaler,
            ub=np.ones(1) * total_budget / self.spend_scaler,
        )
        return total_budget_constraint

    def get_df(self) -> pd.DataFrame:
        df = self.df.copy()
        return df

    def get_current_state(self) -> np.ndarray:
        return deepcopy(self.curr_spend_matrix)

    def get_total_budget(self) -> float:
        return deepcopy(self.total_budget)

    def get_init_state(self) -> np.ndarray:
        return deepcopy(self.init_spend_matrix)

    def get_callback_metrics(self) -> Dict[str, np.ndarray]:
        return deepcopy(self.callback_metrics)

    def objective_func(self, spend: np.ndarray, extra_info: bool = False) -> np.ndarray:
        raise Exception(
            "Abstract objective function. Child class needs to override this method to have concrete result."
        )

    def optimize(
        self,
        init: Optional[np.ndarray] = None,
        maxiter: int = 2,
        eps: Optional[float] = None,
        ftol: float = 1e-7,
    ) -> None:
        if init is None:
            x0 = self.init_spend_matrix.flatten() / self.spend_scaler
        else:
            x0 = init.flatten() / self.spend_scaler

        # clear all solutions
        self._init_callback_metrics()

        options = {
            "disp": True,
            "maxiter": maxiter,
            "ftol": ftol,
        }
        if eps is not None:
            options["eps"] = eps

        sol = optim.minimize(
            self.objective_func,
            x0=x0,
            method="SLSQP",
            bounds=self.budget_bounds,
            constraints=self.constraints,
            options=options,
            callback=self.optim_callback,
        )

        optim_spend_matrix = (
            sol.x.reshape(-1, self.n_optim_channels) * self.spend_scaler
        )
        optim_spend_matrix = np.round(optim_spend_matrix, 5)
        optim_df = self.get_df()
        optim_df.loc[self.budget_mask, self.optim_channels] = optim_spend_matrix
        self.curr_spend_matrix = optim_spend_matrix
        return optim_df

    def _init_callback_metrics(self):
        self.callback_metrics = {"xs": list()}

    def optim_callback(self, xk: np.ndarray, *_):
        """the callback used for each iteration within optimization.
        See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html for details.
        """
        self.callback_metrics["xs"].append(xk)



