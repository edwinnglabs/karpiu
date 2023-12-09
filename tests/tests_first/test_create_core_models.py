import numpy as np
import pickle
from karpiu.models import MMM
from karpiu.simulation import make_mmm_daily_data
import os


# this is a test to be run to create core models for tests such as attribution, optimization, ... etc.
def test_create_core_models():
    print("Create and test simple model.")
    # the simplest proof-of-concept case without adstock, events and seasonality.
    seed = 2022
    n_steps = 365 * 3
    channels_coef = [0.053, 0.08, 0.19, 0.125, 0.1]
    channels = ["promo", "radio", "search", "social", "tv"]
    features_loc = np.array([2000, 5000, 3850, 3000, 7500])
    features_scale = np.array([550, 2500, 500, 1000, 3500])
    scalability = np.array([3.0, 1.25, 0.8, 1.3, 1.5])
    start_date = "2019-01-01"
    np.random.seed(seed)
    df, scalability_df, adstock_df, _ = make_mmm_daily_data(
        channels_coef=channels_coef,
        channels=channels,
        features_loc=features_loc,
        features_scale=features_scale,
        scalability=scalability,
        n_steps=n_steps,
        start_date=start_date,
        adstock_args=None,
        country=None,
    )

    mmm = MMM(
        kpi_col="sales",
        date_col="date",
        spend_cols=channels,
        seed=seed,
    )
    mmm.derive_saturation(df=df, scalability_df=scalability_df)
    best_params = {
        "damped_factor": 0.949,
        "level_sm_input": 0.00245,
    }
    mmm.set_hyper_params(best_params)
    mmm.fit(df, num_warmup=1000, num_sample=1000, chains=4)

    if not os.path.exists("./tests/resources"):
        print("Creating resources folder...")
        os.makedirs("./tests/resources")

    print("Dumping simple-model.pkl...")

    model_file = "simple-model.pkl"
    model_path = os.path.join(os.path.dirname(__file__), 
                   '..',
                   'resources/',
                   model_file
                )
    model_path = os.path.abspath(model_path)
    with open(model_path, "wb") as f:
        pickle.dump(mmm, f, protocol=pickle.HIGHEST_PROTOCOL)

    # non-seasonal case with adstock; no events and seasonality
    print("Create and non-seasonal adstock model.")
    seed = 2023
    np.random.seed(seed)
    adstock_args = {
        "n_steps": 28,
        "peak_step": np.array([10, 8, 5, 3, 2]),
        "left_growth": np.array([0.05, 0.08, 0.1, 0.5, 0.75]),
        "right_growth": np.array([-0.03, -0.6, -0.5, -0.1, -0.25]),
    }
    df, scalability_df, adstock_df, _ = make_mmm_daily_data(
        channels_coef=channels_coef,
        channels=channels,
        features_loc=features_loc,
        features_scale=features_scale,
        scalability=scalability,
        n_steps=n_steps,
        start_date=start_date,
        adstock_args=adstock_args,
        country=None,
    )

    mmm = MMM(
        kpi_col="sales",
        date_col="date",
        spend_cols=channels,
        event_cols=[],
        adstock_df=adstock_df,
        seed=seed,
    )
    mmm.derive_saturation(df=df, scalability_df=scalability_df)
    best_params = {
        "damped_factor": 0.949,
        "level_sm_input": 0.00245,
    }
    mmm.set_hyper_params(best_params)
    mmm.fit(df, num_warmup=1000, num_sample=1000, chains=4)

    print("Dumping non-seasonal-model.pkl...")
    model_file = "non-seasonal-model.pkl"
    model_path = os.path.join(os.path.dirname(__file__), 
                   '..',
                   'resources/',
                   model_file
                )
    model_path = os.path.abspath(model_path)
    with open(model_path, "wb") as f:
        pickle.dump(mmm, f, protocol=pickle.HIGHEST_PROTOCOL)

    # seasonal case with adstock, events, seasonality
    print("Create and seasonal adstock model.")
    seed = 2024
    np.random.seed(seed)
    adstock_args = {
        "n_steps": 28,
        "peak_step": np.array([10, 8, 5, 3, 2]),
        "left_growth": np.array([0.05, 0.08, 0.1, 0.5, 0.75]),
        "right_growth": np.array([-0.03, -0.6, -0.5, -0.1, -0.25]),
    }
    df, scalability_df, adstock_df, event_cols = make_mmm_daily_data(
        channels_coef=channels_coef,
        channels=channels,
        features_loc=features_loc,
        features_scale=features_scale,
        scalability=scalability,
        n_steps=n_steps,
        start_date=start_date,
        adstock_args=adstock_args,
        with_weekly_seasonality=True,
        with_yearly_seasonality=True,
        country="US",
    )

    mmm = MMM(
        kpi_col="sales",
        date_col="date",
        spend_cols=channels,
        event_cols=event_cols,
        adstock_df=adstock_df,
        seasonality=[7, 365.25],
        fs_orders=[2, 3],
        seed=seed,
    )
    mmm.derive_saturation(df=df, scalability_df=scalability_df)
    best_params = {
        "damped_factor": 0.949,
        "level_sm_input": 0.00245,
    }
    mmm.set_hyper_params(best_params)
    mmm.fit(df, num_warmup=1000, num_sample=1000, chains=4)

    print("Dumping seasonal-model.pkl...")
    model_file = "seasonal-model.pkl"
    model_path = os.path.join(os.path.dirname(__file__), 
                   '..',
                   'resources/',
                   model_file
                )
    model_path = os.path.abspath(model_path)
    with open(model_path, "wb") as f:
        pickle.dump(mmm, f, protocol=pickle.HIGHEST_PROTOCOL)
