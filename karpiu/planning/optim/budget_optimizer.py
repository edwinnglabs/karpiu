import pandas as pd
import numpy as np
from typing import Optional, List, Dict
import logging
import scipy.optimize as optim
from copy import deepcopy

from karpiu.models import MMM
from karpiu.model_shell import MMMShell


class BudgetOptimizer(MMMShell):
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
            A=np.ones(self.n_budget_steps * self.n_optim_channels),
            lb=np.zeros(1),
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
        self.callback_metrics = {
            "xs": list()
        }
        
    def optim_callback(self, xk: np.ndarray, *_):
        """the callback used for each iteration within optimization.
        See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html for details.
        """
        self.callback_metrics["xs"].append(xk)

class ChannelBudgetOptimizer(MMMShell):
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
        weight: Optional[np.ndarray] = None,
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
            ub=np.ones(self.n_optim_channels) * np.inf,
        )

        # spend allocation on time dimension
        # (n_budget_steps, )
        if weight is None:
            self.weight = self.base_comp_input / np.sum(self.base_comp_input)
        else:
            self.weight = weight

        if len(self.weight) != self.n_budget_steps:
            raise Exception("Input weight has different length from budget period.")

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
            A=np.ones(self.n_optim_channels),
            lb=np.zeros(1),
            ub=np.ones(1) * total_budget / self.spend_scaler,
        )
        return total_budget_constraint

    def get_df(self) -> pd.DataFrame:
        df = self.df.copy()
        return df

    def get_current_state(self) -> np.ndarray:
        return deepcopy(self.curr_spend_array)

    def get_current_spend_matrix(self) -> np.ndarray:
        return deepcopy(self.curr_spend_matrix)

    def get_total_budget(self) -> float:
        return deepcopy(self.total_budget)

    def get_init_state(self) -> np.ndarray:
        return deepcopy(self.init_spend_array)

    def get_init_spend_matrix(self) -> np.ndarray:
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
        eps: float = 1e-3,
        ftol: float = 1e-7,
    ) -> None:
        if init is None:
            x0 = self.init_spend_array / self.spend_scaler
        else:
            x0 = init.flatten() / self.spend_scaler

        # clear all results stack from callback
        self._init_callback_metrics()

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
            callback=self.optim_callback,
        )

        optim_spend_array = sol.x * self.spend_scaler
        optim_spend_array = np.round(optim_spend_array, 5)
        optim_spend_matrix = (
            optim_spend_array
            * np.ones((self.n_budget_steps, self.n_optim_channels))
            * np.expand_dims(self.weight, -1)
        )
        optim_spend_matrix = np.round(optim_spend_matrix, 5)

        optim_df = self.get_df()
        optim_df.loc[self.budget_mask, self.optim_channels] = optim_spend_matrix
        self.curr_spend_matrix = optim_spend_matrix
        self.curr_spend_array = optim_spend_array
        return optim_df

    def _init_callback_metrics(self):
        self.callback_metrics = {
            "xs": list()
        } 

    def optim_callback(self, xk: np.ndarray, *_):
        """the callback used for each iteration within optimization.
        See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html for details.
        """
        self.callback_metrics["xs"].append(xk)

    def set_bounds_and_constraints(self, df: pd.DataFrame) -> None:
        """_summary_

        Args:
            df (pd.DataFrame): must contain column named as "channel" which can map with the channel index; a special
            channel can be specified once as "total" which will be used as budget constraints instead of bounds
        """
        self.bounds_and_constraints_df = df
        bounds_and_constraints_df = df.copy()
        bounds_and_constraints_df = bounds_and_constraints_df.set_index("channels")
        bounds_df = bounds_and_constraints_df.loc[self.optim_channels]

        self.logger.info("Set bounds.")
        self.budget_bounds = optim.Bounds(
            lb=bounds_df["lower"].values / self.spend_scaler,
            ub=bounds_df["upper"].values / self.spend_scaler,
        )

        if "total" in bounds_and_constraints_df.index.to_list():
            total_budget_upper = bounds_and_constraints_df.loc["total", "upper"]
            total_budget_lower = bounds_and_constraints_df.loc["total", "lower"]
            self.logger.info("Set total budget constraints.")

            total_budget_constraint = optim.LinearConstraint(
                A=np.ones(self.n_optim_channels),
                lb=np.ones(1) * total_budget_lower / self.spend_scaler,
                ub=np.ones(1) * total_budget_upper / self.spend_scaler,
            )
            self.set_constraints([total_budget_constraint])

class TimeBudgetOptimizer(MMMShell):
    """_summary_

    Args:
        MMMShell (_type_): _description_
    """

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
        weight: Optional[np.ndarray] = None,
        # additional optim config
        lb_ratio: float = 0.05,
        ub_ratio: float = 5.0,
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

        self.lb_ratio = lb_ratio
        self.ub_ratio = ub_ratio

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

        # total spend per time step
        # (n_optim_channels, )
        self.init_spend_array = np.sum(self.init_spend_matrix, -1)
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
            lb=np.zeros(self.n_budget_steps),
            ub=np.ones(self.n_budget_steps) * np.inf,
        )

        # spend allocation on time dimension
        # (n_budget_steps, )
        if weight is None:
            # sum of channels over total sum
            self.weight = np.sum(self.init_spend_matrix, 0) / np.sum(
                self.init_spend_matrix
            )
        else:
            self.weight = weight

        if len(self.weight) != self.n_optim_channels:
            raise Exception(
                "Input weight has different length from number of optimizing channels."
            )

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
            A=np.ones(self.n_budget_steps),
            lb=np.zeros(1),
            ub=np.ones(1) * total_budget / self.spend_scaler,
        )
        return total_budget_constraint

    def get_df(self) -> pd.DataFrame:
        df = self.df.copy()
        return df

    def get_current_state(self) -> np.ndarray:
        return deepcopy(self.curr_spend_array)

    def get_current_spend_matrix(self) -> np.ndarray:
        return deepcopy(self.curr_spend_matrix)

    def get_total_budget(self) -> float:
        return deepcopy(self.total_budget)

    def get_init_state(self) -> np.ndarray:
        return deepcopy(self.init_spend_array)

    def get_init_spend_matrix(self) -> np.ndarray:
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
        eps: float = 1e-3,
        ftol: float = 1e-7,
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
        optim_spend_array = np.round(optim_spend_array, 5)
        optim_spend_matrix = (
            np.expand_dims(optim_spend_array, -1)
            * np.ones((self.n_budget_steps, self.n_optim_channels))
            * self.weight
        )
        optim_spend_matrix = np.round(optim_spend_matrix, 5)

        optim_df = self.get_df()
        optim_df.loc[self.budget_mask, self.optim_channels] = optim_spend_matrix
        self.curr_spend_matrix = optim_spend_matrix
        self.curr_spend_array = optim_spend_array
        return optim_df

    def set_bounds_and_constraints(self, df: pd.DataFrame) -> None:
        """_summary_

        Args:
            df (pd.DataFrame): must contain column named as "channel" which can map with the channel index; a special
            channel can be specified once as "total" which will be used as budget constraints instead of bounds
        """
        # "date" is a reserved keyword
        self.bounds_and_constraints_df = df
        bounds_and_constraints_df = df.copy()
        bounds_df = bounds_and_constraints_df.loc[
            bounds_and_constraints_df["date"] != "total", :].reset_index(drop=True)
        assert bounds_df.shape[0] == self.n_budget_steps

        self.logger.info("Set bounds.")
        self.budget_bounds = optim.Bounds(
            lb=bounds_df["lower"].values / self.spend_scaler,
            ub=bounds_df["upper"].values / self.spend_scaler,
        )

        if "total" in bounds_and_constraints_df["date"].to_list():
            constraints_df = bounds_and_constraints_df.loc[
                bounds_and_constraints_df["date"] == "total", :].reset_index(drop=True)
            assert constraints_df.shape[0] == 1
            total_budget_upper = constraints_df["upper"]
            total_budget_lower = constraints_df["lower"]
            self.logger.info("Set total budget constraints.")

            total_budget_constraint = optim.LinearConstraint(
                A=np.ones(self.n_budget_steps),
                lb=np.ones(1) * total_budget_lower / self.spend_scaler,
                ub=np.ones(1) * total_budget_upper / self.spend_scaler,
            )
            self.set_constraints([total_budget_constraint])