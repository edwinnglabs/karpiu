import numpy as np
from scipy.optimize import fsolve
from .explainer import Attributor

import logging

logger = logging.getLogger("karpiu-mmm")


class PriorSolver:
    """Solving Regression Coefficient Prior from MMM by using Attribution logic"""

    def __init__(self, tests_df):
        self.tests_df = tests_df

    def derive_prior(self, model):
        input_df = model.raw_df.copy()
        output_df = self.tests_df.copy()
        date_col = model.date_col

        for idx, row in self.tests_df.iterrows():
            test_start = row['test_start']
            test_end = row['test_end']
            test_icac = row['test_icac']
            test_channel = row['test_channel']
            test_se = row['test_se']

            # derive test spend
            mask = (input_df[date_col] >= test_start) & (input_df[date_col] <= test_end)
            sub_input_df = input_df[mask].reset_index(drop=True)
            test_spend = sub_input_df[test_channel].sum()
            # derive lift from spend data from model to ensure consistency
            test_lift = test_spend / test_icac
            test_lift_upper = test_spend / (test_icac - test_se)

            logger.info("test channel:{}".format(test_channel))
            logger.info("test spend: {:.3f}, test lift: {:.3f}".format(test_spend, test_lift))

            # create a callback used for scipy.optimize.fsolve
            attr_obj = Attributor(model, start=test_start, end=test_end)

            def attr_call_back(x, target):
                attr_res = attr_obj.make_attribution(
                    new_coef_name=test_channel,
                    new_coef=x,
                    true_up=True,
                )
                _, spend_attr_df, _, _ = attr_res
                mask = (spend_attr_df[date_col] >= test_start) & (spend_attr_df[date_col] <= test_end)
                res = np.sum(spend_attr_df.loc[mask, test_channel].values)
                loss = np.fabs(res - target)
                return loss

            init_search_pt = model.get_coef_vector([test_channel])
            coef_prior = fsolve(attr_call_back, x0=init_search_pt, args=test_lift)[0]
            coef_prior_upper = fsolve(attr_call_back, x0=init_search_pt, args=test_lift_upper)[0]

            # store derived result
            output_df.loc[idx, 'coef_prior'] = coef_prior
            # since model can be over-confident on empirical result and the non-linear relationship, 
            # introduce a 0.3 haircut on the derive sigma here
            output_df.loc[idx, 'sigma_prior'] = (coef_prior_upper - coef_prior) * 0.3
            output_df.loc[idx, 'test_spend'] = test_spend
            output_df.loc[idx, 'test_lift'] = test_lift

        return output_df
