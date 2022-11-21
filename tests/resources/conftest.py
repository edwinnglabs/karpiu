# import numpy as np
# from copy import deepcopy
# import pytest

# from karpiu.simulation import make_mmm_daily_data

# # request is a special keyword
# # see details in https://docs.pytest.org/en/latest/example/parametrize.html#apply-indirect-on-particular-arguments

# # to create test dataset on the fly
# @pytest.fixture
# def mmm_daily_test_data(request):
#     default_args = {
#         "n_steps" :365 * 3,
#         "coefs" :[0.03, 0.05, 0.028, 0.01, 0.03],
#         "channels" :['tv', 'radio', 'social', 'promo', 'search'],
#         "features_loc" :np.array([10000, 5000, 3000, 2000, 850]),
#         "features_scale" :np.array([5000,3000, 1000, 550, 500]),
#         "scalability" : np.array([1.1, 0.75, 1.3, 1.5, 0.9]),
#         "seed" :2022,
#         "start_date" :'2019-01-01',
#     }
#     final_args= deepcopy(default_args)
#     final_args.update(request.param)

#     df, scalability_df = make_mmm_daily_data(**final_args)
#     return df, scalability_df