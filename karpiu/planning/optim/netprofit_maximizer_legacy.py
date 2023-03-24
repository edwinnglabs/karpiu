# import numpy as np

# from ...explainability.functions import make_attribution_numpy_beta
# from ...utils import adstock_process
# from .budget_optimizer import BudgetOptimizer


# class NetProfitMaximizer(BudgetOptimizer):
#     """Perform revenue optimization with a given Marketing Mix Model and
#     lift-time values (LTV) per channel
#     """

#     def __init__(self, ltv_arr: np.array, **kwargs):
#         super().__init__(**kwargs)
#         transformed_bkg_matrix = adstock_process(
#             self.bkg_spend_matrix, self.optim_adstock_matrix
#         )
#         self.bkg_attr_comp = self.optim_coef_matrix * np.log1p(
#             transformed_bkg_matrix / self.optim_sat_array
#         )
#         self.ltv_arr = ltv_arr
#         self.design_broadcast_matrix = np.concatenate(
#             [
#                 np.ones((1, 1, self.n_optim_channels)),
#                 np.expand_dims(np.eye(self.n_optim_channels), -2),
#             ],
#             axis=0,
#         )

#     def objective_func(self, spend):
#         spend_matrix = spend.reshape(-1, self.n_optim_channels) * self.spend_scaler
#         zero_paddings = np.zeros((self.n_max_adstock, self.n_optim_channels))
#         # (n_steps + 2 * n_max_adstock, n_channels)
#         spend_matrix = np.concatenate([zero_paddings, spend_matrix, zero_paddings], 0)

#         # duplicate 1 + n_channels scenarios with full spend and one-off spend
#         full_sim_sp_matrix = np.tile(
#             np.expand_dims(spend_matrix, 0), reps=(self.n_optim_channels + 1, 1, 1)
#         )
#         full_sim_sp_matrix = full_sim_sp_matrix * self.design_broadcast_matrix
#         full_sim_sp_matrix += self.bkg_spend_matrix
#         # (n_regressor + 1, n_steps + n_adstock, n_regressor)
#         full_sim_tran_sp_matrix = adstock_process(
#             full_sim_sp_matrix,
#             self.optim_adstock_matrix,
#         )

# #         # (n_steps + n_adstock, n_regressor)
# #         full_tran_sp_matrix = full_sim_tran_sp_matrix[0]
# #         # (n_regressor, n_steps + n_adstock, n_regressor)
# #         one_off_sp_matrix = full_sim_tran_sp_matrix[1:, ...]

# #         # (n_steps + n_adstock, )
# #         full_comp = np.sum(
# #             self.optim_coef_matrix
# #             * np.log1p(full_tran_sp_matrix / self.optim_sat_array),
# #             -1,
# #         )
# #         # (n_regressor, n_steps + n_adstock, )
# #         one_off_comp = np.sum(
# #             self.optim_coef_matrix * np.log1p(one_off_sp_matrix / self.optim_sat_array),
# #             -1,
# #         )

# #         # (n_regressor, n_steps + n_adstock)
# #         attr_matrix = np.exp(self.base_comp) * (
# #             -np.exp(one_off_comp) + np.exp(full_comp)
# #         )

# #         # linearization to make decomp additive
# #         # (n_steps + n_adstock, )
# #         delta_t = np.exp(self.base_comp) * (np.exp(full_comp) - 1)
# #         # (n_regressor, n_steps + n_adstock)
# #         norm_attr_matrix = attr_matrix / np.sum(attr_matrix, 0, keepdims=True) * delta_t

# #         # (n_regressor, )
# #         revenue = self.ltv_arr * np.sum(norm_attr_matrix, -1)

# #         # (n_regressor, )
# #         cost = np.sum(spend_matrix, 0)
# #         net_profit = np.sum(revenue - cost)
# #         loss = -1 * net_profit / self.response_scaler

# #         return loss


# # class NetProfitMaximizer2(BudgetOptimizer):
# #     """Perform revenue optimization with a given Marketing Mix Model and
# #     lift-time values (LTV) per channel
# #     """

# #     def __init__(self, ltv_arr: np.array, **kwargs):
# #         super().__init__(**kwargs)
# #         transformed_bkg_matrix = adstock_process(
# #             self.bkg_spend_matrix, self.optim_adstock_matrix
# #         )
# #         self.bkg_attr_comp = self.optim_coef_matrix * np.log1p(
# #             transformed_bkg_matrix / self.optim_sat_array
# #         )
# #         self.ltv_arr = ltv_arr

# #     def objective_func(self, spend):
# #         spend_matrix = spend.reshape(-1, self.n_optim_channels) * self.spend_scaler
# #         zero_paddings = np.zeros((self.n_max_adstock, self.n_optim_channels))
# #         # (n_budget_steps + 2 * n_max_adstock, n_optim_channels)
# #         spend_matrix = np.concatenate([zero_paddings, spend_matrix, zero_paddings], 0)
# #         spend_matrix += self.bkg_spend_matrix
# #         # (n_budget_steps + n_max_adstock, n_optim_channels)
# #         transformed_spend_matrix = adstock_process(
# #             spend_matrix, self.optim_adstock_matrix
# #         )
# #         # (n_budget_steps + n_max_adstock, n_optim_channels)
# #         varying_attr_comp = self.optim_coef_matrix * np.log1p(
# #             transformed_spend_matrix / self.optim_sat_array
# #         )

# #         # one-on approximation
# #         # FIXME: one-on does not capture multiplicative effect; not enough
# #         # base comp does not have the channel dimension so we need expand on last dim
# #         # (n_budget_steps + n_max_adstock, n_optim_channels)
# #         attr_matrix = np.exp(np.expand_dims(self.base_comp, -1)) * (
# #             np.exp(varying_attr_comp) - np.exp(self.bkg_attr_comp)
# #         )
# #         # (n_optim_channels, )
# #         revenue = self.ltv_arr * np.sum(attr_matrix, 0)
# #         # (n_optim_channels, )
# #         cost = np.sum(spend_matrix, 0)
# #         net_profit = np.sum(revenue - cost)
# #         loss = -1 * net_profit / self.response_scaler
# #         return loss
