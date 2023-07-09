import numpy as np

from .budget_optimizer import BudgetOptimizer
from ...utils import adstock_process


class TargetMaximizer(BudgetOptimizer):
    """Perform target maximization with a given Marketing Mix Model"""

    def __init__(self, variance_penalty: float = 1e-3, **kwargs):
        super().__init__(**kwargs)
        self.variance_penalty = variance_penalty

    def objective_func(self, spend):
        spend_matrix = spend.reshape(-1, self.n_optim_channels) * self.spend_scaler
        zero_paddings = np.zeros((self.max_adstock, self.n_optim_channels))
        spend_matrix = np.concatenate([zero_paddings, spend_matrix, zero_paddings], 0)
        spend_matrix += self.target_regressor_bkg_matrix
        transformed_spend_matrix = adstock_process(
            spend_matrix, self.target_adstock_matrix
        )
        # regression
        spend_comp = np.sum(
            self.target_coef_matrix
            * np.log1p(transformed_spend_matrix / self.target_sat_array),
            -1,
        )
        # (n_result_steps, )
        pred_outcome = self.base_comp_result * np.exp(spend_comp)
        loss = -1 * np.sum(pred_outcome) / self.response_scaler
        # add punishment of within channel variance of spend; otherwise may risk of identifiability issue with adstock
        loss += self.variance_penalty * np.sum(np.std(spend_matrix, 0))
        return loss
