import numpy as np
from ...utils import adstock_process


from ...explainability.functions import make_attribution_numpy_beta
from ...utils import adstock_process
from .budget_optimizer import ChannelBudgetOptimizer, TimeBudgetOptimizer
from scipy.stats import skew


class ChannelNetProfitMaximizer(ChannelBudgetOptimizer):
    """Perform revenue optimization with a given Marketing Mix Model and
    lift-time values (LTV) per channel
    """

    def __init__(self, ltv_arr: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        # (n_optim_channels, )
        self.ltv_arr = ltv_arr

    def objective_func(self, spend: np.ndarray):
        # spend should be (n_optim_channels, )
        spend_matrix = (
            spend
            * np.ones((self.n_budget_steps, self.n_optim_channels))
            * np.expand_dims(self.weight, -1)
        )
        zero_paddings = np.zeros((self.max_adstock, self.n_optim_channels))
        # (n_calc_steps, n_optim_channels)
        spend_matrix = np.concatenate(
            [zero_paddings.copy(), spend_matrix, zero_paddings.copy()], axis=0
        )
        spend_matrix += self.target_regressor_bkg_matrix
        # (n_result_steps, n_optim_channels)
        transformed_spend_matrix = adstock_process(
            spend_matrix, self.target_adstock_matrix
        )
        # (n_calc_steps, n_optim_channels)
        transformed_spend_matrix = np.concatenate(
            [
                zero_paddings.copy(),
                transformed_spend_matrix,
            ],
            axis=0,
        )

        varying_comp = np.sum(
            self.target_coef_array
            * np.log1p(transformed_spend_matrix / self.target_sat_array),
            -1,
        )
        # (n_calc_steps, )
        pred_bau = self.base_comp_calc * np.exp(varying_comp)

        # (n_steps, n_optim_channels)
        (
            _,
            spend_attr_matrix,
            _,
        ) = make_attribution_numpy_beta(
            attr_coef_array=self.target_coef_array,
            attr_regressor_matrix=spend_matrix,
            attr_transformed_regressor_matrix=transformed_spend_matrix,
            pred_bau=pred_bau,
            pred_zero=self.base_comp_calc,
            adstock_matrix=self.target_adstock_matrix,
            attr_saturation_array=self.target_sat_array,
            true_up_arr=pred_bau,
            fixed_intercept=True,
        )

        # (n_optim_channels, )
        # ignore first column which is organic
        revenue = self.ltv_arr * np.sum(spend_attr_matrix[:, 1:], 0)
        # (n_optim_channels, )
        cost = np.sum(spend_matrix, 0)
        net_profit = np.sum(revenue - cost)
        loss = -1 * net_profit / self.response_scaler
        return loss


class TimeNetProfitMaximizer(TimeBudgetOptimizer):
    """Perform revenue optimization with a given Marketing Mix Model and
    lift-time values (LTV) per channel
    """

    def __init__(self, ltv_arr: np.ndarray, variance_penalty: float = 1e-3, **kwargs):
        super().__init__(**kwargs)
        # (n_optim_channels, )
        self.ltv_arr = ltv_arr
        self.variance_penalty = variance_penalty

    def objective_func(self, spend: np.ndarray):
        # spend should be (n_optim_channels, )
        spend_matrix = (
            np.expand_dims(spend, -1)
            * np.ones((self.n_budget_steps, self.n_optim_channels))
            * self.weight
        )
        zero_paddings = np.zeros((self.max_adstock, self.n_optim_channels))
        # (n_calc_steps, n_optim_channels)
        spend_matrix = np.concatenate(
            [zero_paddings.copy(), spend_matrix, zero_paddings.copy()], axis=0
        )
        spend_matrix += self.target_regressor_bkg_matrix
        # (n_result_steps, n_optim_channels)
        transformed_spend_matrix = adstock_process(
            spend_matrix, self.target_adstock_matrix
        )
        # (n_calc_steps, n_optim_channels)
        transformed_spend_matrix = np.concatenate(
            [
                zero_paddings.copy(),
                transformed_spend_matrix,
            ],
            axis=0,
        )

        varying_comp = np.sum(
            self.target_coef_array
            * np.log1p(transformed_spend_matrix / self.target_sat_array),
            -1,
        )
        # (n_calc_steps, )
        pred_bau = self.base_comp_calc * np.exp(varying_comp)

        # (n_steps, n_optim_channels)
        (
            _,
            spend_attr_matrix,
            _,
        ) = make_attribution_numpy_beta(
            attr_coef_array=self.target_coef_array,
            attr_regressor_matrix=spend_matrix,
            attr_transformed_regressor_matrix=transformed_spend_matrix,
            pred_bau=pred_bau,
            pred_zero=self.base_comp_calc,
            adstock_matrix=self.target_adstock_matrix,
            attr_saturation_array=self.target_sat_array,
            true_up_arr=pred_bau,
            fixed_intercept=True,
        )

        # (n_optim_channels, )
        # ignore first column which is organic
        revenue = self.ltv_arr * np.sum(spend_attr_matrix[:, 1:], 0)
        # (n_optim_channels, )
        cost = np.sum(spend_matrix, 0)
        net_profit = np.sum(revenue - cost)
        loss = -1 * net_profit / self.response_scaler
        # add punishment of variance of spend; otherwise may risk of identifiability issue with adstock
        loss += self.variance_penalty * np.var(spend)
        return loss
