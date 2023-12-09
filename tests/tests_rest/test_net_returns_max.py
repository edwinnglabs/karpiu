# import numpy as np
# from copy import deepcopy
# import pickle

# from karpiu.planning.optim import ChannelNetProfitMaximizer, TimeNetProfitMaximizer
# from karpiu.planning.common import generate_cost_report
# from karpiu.explainability import AttributorBeta as Attributor
# from karpiu.utils import np_shuffle


# def test_net_profit_maximizer():
#     with open("./tests/resources/seasonal-model.pkl", "rb") as f:
#         mmm = pickle.load(f)

#     ltv_arr = [48.5, 52.5, 38.6, 35.8, 60.8]
#     df = mmm.get_raw_df()

#     budget_start = "2020-01-01"
#     budget_end = "2020-01-31"
#     optim_channels = mmm.get_spend_cols()
#     # to be safe in beta version, use sorted list of channels
#     optim_channels.sort()

#     ch_npm = ChannelNetProfitMaximizer(
#         ltv_arr=ltv_arr,
#         model=mmm,
#         budget_start=budget_start,
#         budget_end=budget_end,
#         optim_channels=optim_channels,
#     )
#     temp_optim_spend_df = ch_npm.optimize(maxiter=1000)
#     ch_npm_curr_state = ch_npm.get_current_state()
#     ch_npm_init_state = ch_npm.get_init_state()

#     # check: optimization result should be indifferent with initial values
#     # create different initial spend df and plug back into the model
#     new_raw_df = mmm.get_raw_df()
#     new_spend_matrix = np_shuffle(
#         new_raw_df.loc[
#             (new_raw_df["date"] >= budget_start) & (new_raw_df["date"] <= budget_end),
#             optim_channels,
#         ].values
#     )
#     new_raw_df.loc[
#         (new_raw_df["date"] >= budget_start) & (new_raw_df["date"] <= budget_end),
#         optim_channels,
#     ] = new_spend_matrix

#     new_mmm = deepcopy(mmm)
#     new_mmm.raw_df = new_raw_df
#     new_ch_npm = ChannelNetProfitMaximizer(
#         ltv_arr=ltv_arr,
#         model=new_mmm,
#         budget_start=budget_start,
#         budget_end=budget_end,
#         optim_channels=optim_channels,
#     )
#     _ = new_ch_npm.optimize(maxiter=1000)
#     new_ch_npm_curr_state = new_ch_npm.get_current_state()
#     new_ch_npm_init_state = new_ch_npm.get_init_state()

#     # the final result should be closed in either by 1e-1 or .1%
#     assert np.any(np.not_equal(new_ch_npm_init_state, ch_npm_init_state))
#     assert np.allclose(new_ch_npm_curr_state, ch_npm_curr_state, atol=1e-1, rtol=1e-3)

#     temp_mmm = deepcopy(mmm)
#     temp_mmm.raw_df = temp_optim_spend_df

#     # pass into time budget optimizer
#     t_npm = TimeNetProfitMaximizer(
#         ltv_arr=ltv_arr,
#         model=temp_mmm,
#         budget_start=budget_start,
#         budget_end=budget_end,
#         optim_channels=optim_channels,
#     )
#     optim_spend_df = t_npm.optimize(maxiter=1000)

#     cost_report = generate_cost_report(
#         model=mmm,
#         channels=optim_channels,
#         start=budget_start,
#         end=budget_end,
#         pre_spend_df=df,
#         post_spend_df=optim_spend_df,
#     )
#     cost_report["ltv"] = np.array(ltv_arr)

#     # check general cost report condition
#     pre_opt_spend = cost_report["pre-opt-spend"].values
#     pre_ac = cost_report["pre-opt-avg-cost"].values[pre_opt_spend > 0]
#     pre_mc = cost_report["pre-opt-marginal-cost"].values[pre_opt_spend > 0]

#     post_opt_spend = cost_report["post-opt-spend"].values
#     post_ac = cost_report["post-opt-avg-cost"].values[post_opt_spend > 0]
#     post_mc = cost_report["post-opt-marginal-cost"].values[post_opt_spend > 0]

#     assert np.all(pre_mc >= pre_ac)
#     assert np.all(post_mc >= post_ac)

#     # check 3: underspend and overspend condition can be checked by
#     # whether pre_mc > ltv (overspend) and vice versa
#     pre_mc = cost_report["pre-opt-marginal-cost"].values
#     post_avg_mc = cost_report["post-opt-avg-cost"].values
#     positive_spend = cost_report["post-opt-spend"]
#     overspend = pre_mc > (cost_report["ltv"].values * 1.2)
#     underspend = pre_mc < (cost_report["ltv"].values * 0.8)
#     spend_delta = (
#         cost_report["post-opt-spend"].values - cost_report["pre-opt-spend"].values
#     )
#     assert np.all(spend_delta[overspend] < 0)
#     assert np.all(spend_delta[underspend] > 0)

#     # check 4: post-mc should be close or under ltv; post-avg-cost should be
#     # lower than ltv when they have spend
#     # 1.2 is pretty high tolerance; but it may be okay for a unit test
#     post_mc = cost_report["post-opt-marginal-cost"].values
#     assert np.all(
#         post_mc[positive_spend > 0]
#         < cost_report["ltv"].values[positive_spend > 0] * 1.2
#     )
#     assert np.all(
#         post_avg_mc[positive_spend > 0]
#         < cost_report["ltv"].values[positive_spend > 0] * 1.1
#     )

#     # check 5: all the marginal net return should be negative if a small delta is added to the current budget plan
#     attr_obj = Attributor(
#         mmm,
#         attr_regressors=optim_channels,
#         start=budget_start,
#         end=budget_end,
#         df=optim_spend_df,
#     )
#     res = attr_obj.make_attribution()

#     _, spend_attr_df, spend_df, _ = res
#     base_spend_attr_matrix = np.sum(spend_attr_df[optim_channels].values, 0)
#     base_spend_matrix = np.sum(spend_df[optim_channels].values, 0)
#     base_rev = base_spend_attr_matrix * (ltv_arr)
#     base_net_arr = base_rev - base_spend_matrix
#     baseline_net_rev = base_net_arr.sum()
#     input_mask = attr_obj.input_mask
#     delta = 1e-1

#     new_net_revs = np.empty(len(optim_channels))
#     for idx, ch in enumerate(optim_channels):
#         new_spend_df = deepcopy(optim_spend_df)
#         delta_matrix = np.zeros_like(new_spend_df.loc[input_mask, optim_channels])
#         delta_matrix[:, idx] += delta
#         new_spend_df.loc[input_mask, optim_channels] += delta_matrix
#         attr_obj = Attributor(
#             mmm,
#             attr_regressors=optim_channels,
#             start=budget_start,
#             end=budget_end,
#             df=new_spend_df,
#         )
#         res = attr_obj.make_attribution()
#         _, spend_attr_df, spend_df, _ = res
#         new_spend_attr_matrix = np.sum(spend_attr_df[optim_channels].values, 0)
#         new_spend_matrix = np.sum(spend_df[optim_channels].values, 0)
#         new_rev = new_spend_attr_matrix * (ltv_arr)
#         new_net_arr = new_rev - new_spend_matrix
#         new_net_revs[idx] = new_net_arr.sum()

#     assert np.all(new_net_revs - baseline_net_rev < 0)
