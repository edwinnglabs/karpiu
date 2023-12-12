import pandas as pd
import numpy as np
from typing import Optional, List, Dict
import logging
import scipy.optimize as optim
from copy import deepcopy

from karpiu.models import MMM
from karpiu.model_shell import MMMShellLegacy
from ...explainability.attribution_gamma import AttributorGamma


class TimeBudgetOptimizer(MMMShellLegacy):
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

        self.optim_channels = optim_channels
        self.full_channels = model.get_spend_cols()
        self.optim_channels_idx = list()
        for ch in optim_channels:
            for idx in range(len(self.full_channels)):
                if ch == self.full_channels[idx]:
                    self.optim_channels_idx.append(idx)
        self.logger.info("Optimizing channels : {}".format(self.optim_channels))

        # this is more for calculating all attribution required math
        super().__init__(
            model=model,
            target_regressors=self.full_channels,
            start=budget_start,
            end=budget_end,
        )

        self.lb_ratio = lb_ratio
        self.ub_ratio = ub_ratio

        self.budget_start = self.start
        self.budget_end = self.end

        self.response_scaler = response_scaler
        self.spend_scaler = spend_scaler
        self.constraints = list()
        self.budget_mask = self.input_mask

        n_budget_steps = np.sum(self.input_mask)
        self.n_budget_steps = n_budget_steps
        self.n_optim_channels = len(self.optim_channels)

        # derive optimization input
        # derive init values
        # (n_budget_steps, n_optim_channels)
        self.init_spend_matrix = self.df.loc[
            self.budget_mask, self.optim_channels
        ].values

        # total spend per time step
        # (n_budget_steps, )
        self.init_spend_array = np.sum(self.init_spend_matrix, -1)
        # this stores current optimal spend
        self.curr_spend_matrix = deepcopy(self.init_spend_matrix)
        self.curr_spend_array = deepcopy(self.init_spend_array)

        if total_budget_override is not None and total_budget_override > 0:
            self.total_budget = total_budget_override
        else:
            self.total_budget = np.sum(self.init_spend_array)

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

        self.full_channels_spend_matrix = self.df.loc[
            self.budget_mask, self.full_channels
        ].values

        # spend allocation on time dimension
        # (1, n_channels)
        if weight is None:
            # equally split weight
            # self.weight = np.ones((1, len(optim_channels))) / len(optim_channels)
            # input volume weight based
            self.weight = np.sum(self.init_spend_matrix, 0) / np.sum(
                self.init_spend_matrix
            )
        else:
            self.weight = weight

        if self.weight.shape[-1] != self.n_optim_channels:
            raise Exception(
                "Input weight has different length from number of optimizing channels."
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
            A=np.ones(self.n_budget_steps),
            lb=np.zeros(1),
            # lb=np.ones(1) * total_budget / self.spend_scaler,
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
        disp: bool = True,
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
                "disp": disp,
                "maxiter": maxiter,
                "eps": eps,
                "ftol": ftol,
            },
            callback=self.optim_callback,
        )

        optim_spend_array = sol.x * self.spend_scaler
        optim_spend_matrix = (
            np.expand_dims(optim_spend_array, -1)
            * np.ones((self.n_budget_steps, self.n_optim_channels))
            * self.weight
        )
        optim_spend_matrix = np.round(optim_spend_matrix, 5)
        # after round-up, recalculate this to make it consistent with sum
        optim_spend_array = np.sum(optim_spend_matrix, -1)

        optim_df = self.get_df()
        optim_df.loc[self.budget_mask, self.optim_channels] = optim_spend_matrix
        self.curr_spend_matrix = optim_spend_matrix
        self.curr_spend_array = optim_spend_array
        return optim_df

    def set_bounds_and_constraints(self, df: pd.DataFrame) -> None:
        """_summary_

        Args:
            df (pd.DataFrame): df assumes each index sequentially represents the time step in optimization period;
            a special index can be specified once as "total" which will be used as budget constraints instead of bounds
        """
        # "date" is a reserved keyword
        self.bounds_and_constraints_df = df
        bounds_and_constraints_df = df.copy()
        bounds_df = bounds_and_constraints_df.loc[
            bounds_and_constraints_df["date"] != "total", :
        ].reset_index(drop=True)
        assert bounds_df.shape[0] == self.n_budget_steps

        self.logger.info("Set bounds.")
        self.budget_bounds = optim.Bounds(
            lb=bounds_df["lower"].values / self.spend_scaler,
            ub=bounds_df["upper"].values / self.spend_scaler,
        )

        if "total" in bounds_and_constraints_df["date"].to_list():
            constraints_df = bounds_and_constraints_df.loc[
                bounds_and_constraints_df["date"] == "total", :
            ].reset_index(drop=True)
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


class TimeNetProfitMaximizer(TimeBudgetOptimizer):
    """Perform revenue optimization with a given Marketing Mix Model and
    lift-time values (LTV) per channel; an extra variance penalty is introduced to resolve
    identifiability due to adstock
    """

    def __init__(
        self,
        attributor: AttributorGamma,
        ltv_arr: np.ndarray,
        variance_penalty: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        # (n_optim_channels, )
        self.ltv_arr = ltv_arr
        self.variance_penalty = variance_penalty
        self.attributor = attributor

    def objective_func(self, spend: np.ndarray, extra_info: bool = False):
        # spend(n_budget_steps, ) -> (expand) -> spend(n_budget_steps, 1)
        # (multiply channels weight(n_optim_channels)
        # -> (broadcast) -> input_channel_spend_matrix (n_budget_steps, n_optim_channels)
        input_channel_spend_matrix = np.expand_dims(spend, -1) * self.weight
        # the full spend matrix pass into attribution calculation
        spend_matrix = self.full_channels_spend_matrix.copy()
        spend_matrix[:, self.optim_channels_idx] = input_channel_spend_matrix

        if self.max_adstock > 0:
            zero_paddings = np.zeros((self.max_adstock, self.n_optim_channels))
            # (n_calc_steps, n_optim_channels)
            spend_matrix = np.concatenate(
                [zero_paddings.copy(), spend_matrix, zero_paddings.copy()], axis=0
            )
            spend_matrix += self.target_regressor_bkg_matrix

        target_coef_array = self.target_coef_array
        target_transformed_matrix = self.attributor._derive_target_transformed_matrix(
            target_calc_regressors_matrix=spend_matrix,
        )
        attr_marketing = self.attributor._derive_attr_marketing(
            target_transformed_matrix,
            target_coef_array,
        )

        pred_bau_array = self.attributor._derive_bau_array(
            target_transformed_matrix,
            target_coef_array,
        )

        norm_delta_matrix, _ = self.attributor._derive_market_shares_delta_matrix(
            pred_bau=pred_bau_array,
            target_calc_regressors_matrix=spend_matrix,
            target_transformed_matrix=target_transformed_matrix,
            target_coef_array=target_coef_array,
        )

        _, spend_attr_matrix = self.attributor._derive_attr_matrix(
            norm_delta_matrix=norm_delta_matrix,
            attr_marketing=attr_marketing,
        )

        # For attribution, revenue, and cost are calculated
        # with all channels spend (not just the two we are optimizing) as the shape
        # (n_optim_channels, )
        revenue = self.ltv_arr * np.sum(spend_attr_matrix, 0)
        # (n_optim_channels, )
        cost = np.sum(spend_matrix, 0)
        net_profit = np.sum(revenue - cost)
        loss = -1 * net_profit / self.response_scaler
        # add punishment of variance of spend; otherwise may risk of identifiability issue with adstock
        loss += self.variance_penalty * np.var(spend)
        if extra_info:
            return np.sum(revenue), np.sum(cost), input_channel_spend_matrix
        else:
            return loss

    def _init_callback_metrics(self):
        self.callback_metrics = {
            "xs": list(),
            "optim_revenues": list(),
            "optim_costs": list(),
            "spend_matrix": list(),
        }

    # override parent class
    def optim_callback(self, xk: np.ndarray, *_):
        """the callback used for each iteration within optimization.
        See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html for details.
        """
        self.callback_metrics["xs"].append(xk)
        revs, costs, spend_matrix = self.objective_func(xk, extra_info=True)
        self.callback_metrics["optim_revenues"].append(revs)
        self.callback_metrics["optim_costs"].append(costs)
        self.callback_metrics["spend_matrix"].append(spend_matrix)


# assert budget start and and end are only two steps
def time_based_net_profit_response_curve(
    t_npm: TimeNetProfitMaximizer, model: MMM, n_iters=10
):
    net_profits = np.empty((n_iters, n_iters))
    total_budget = t_npm.total_budget
    date_col = t_npm.date_col
    budget_start = t_npm.budget_start
    budget_end = t_npm.budget_end

    logger = logging.getLogger("karpiu-planning-test")
    logger.setLevel(30)

    def time_based_net_profit_response(
        x1, x2, attributor, channels_weight, base_spend_df, optim_channels, ltv_arr
    ) -> np.ndarray:
        # (n_steps, n_channels)
        input_spend_matrix = np.vstack([x1, x2]) * channels_weight
        temp_spend_df = base_spend_df.copy()
        temp_spend_df.loc[
            (temp_spend_df[date_col] >= budget_start)
            & (temp_spend_df[date_col] <= budget_end),
            optim_channels,
        ] = input_spend_matrix

        attributor = AttributorGamma(
            model=model,
            df=temp_spend_df,
            start=budget_start,
            end=budget_end,
            logger=logger,
        )
        _, spend_attr, _, _ = attributor.make_attribution()

        # For attribution, revenue, and cost are calculated with all channels spend (not just the two we are optimizing) as the input
        cost = np.sum(
            temp_spend_df.loc[
                (temp_spend_df[date_col] >= budget_start)
                & (temp_spend_df[date_col] <= budget_end),
                # always use full channels in time-based optimization
                model.get_spend_cols(),
            ].values
        )

        return np.sum(spend_attr.loc[:, model.get_spend_cols()].values * ltv_arr) - cost

    x1s = total_budget * np.linspace(0, 1, n_iters)
    x2s = total_budget * np.linspace(0, 1, n_iters)

    x1s, x2s = np.meshgrid(x1s, x2s)

    for i in range(n_iters):
        for j in range(n_iters):
            x1 = x1s[i, j]
            x2 = x2s[i, j]
            net_profits[i, j] = time_based_net_profit_response(
                x1,
                x2,
                attributor=t_npm.attributor,
                channels_weight=t_npm.weight,
                base_spend_df=t_npm.df,
                optim_channels=t_npm.optim_channels,
                ltv_arr=t_npm.ltv_arr,
            )

    return x1s, x2s, net_profits
