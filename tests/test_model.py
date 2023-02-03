import pytest
import numpy as np

from karpiu.models import MMM
from karpiu.simulation import make_mmm_daily_data
from orbit.diagnostics.metrics import smape

# @pytest.mark.parametrize(
#     "mmm_daily_test_data",
#     [
#         ({"seed": 1}),
#         ({"seed": 2022}),
#     ],
#     indirect=True
# )


@pytest.mark.parametrize("seed", [1, 2022], ids=["seed_1", "seed_2022"])
@pytest.mark.parametrize(
    "adstock_args",
    [
        {
            "n_steps": 28,
            "peak_step": np.array([10, 8, 5, 3, 2]),
            "left_growth": np.array([0.05, 0.08, 0.1, 0.5, 0.75]),
            "right_growth": np.array([-0.03, -0.6, -0.5, -0.1, -0.25]),
        },
        None,
    ],
    ids=["with_adstock", "no_adstock"],
)
def test_mmm_fit_predict(seed, adstock_args):
    # data_args
    n_steps = 365 * 3
    coefs = [0.03, 0.05, 0.028, 0.01, 0.03]
    channels = ["tv", "radio", "social", "promo", "search"]
    features_loc = np.array([10000, 5000, 3000, 2000, 850])
    features_scale = np.array([5000, 3000, 1000, 550, 500])
    scalability = np.array([1.1, 0.75, 1.3, 1.5, 0.9])
    start_date = "2019-01-01"

    # adstock_args = {
    #     "n_steps":28,
    #     "peak_step" : np.array([10, 8, 5, 3, 2]),
    #     "left_growth" : np.array([.05, .08, .1, .5, .75]),
    #     "right_growth" : np.array([-.03, -0.6, -0.5, -.1, -.25]),
    # }

    df, scalability_df, adstock_df = make_mmm_daily_data(
        coefs=coefs,
        channels=channels,
        features_loc=features_loc,
        features_scale=features_scale,
        scalability=scalability,
        n_steps=n_steps,
        start_date=start_date,
        adstock_args=adstock_args,
        seed=seed,
    )

    mmm = MMM(
        kpi_col="sales",
        date_col="date",
        spend_cols=channels,
        seed=seed,
        adstock_df=adstock_df,
    )
    mmm.derive_saturation(df=df, scalability_df=scalability_df)
    mmm.fit(df, num_warmup=400, num_sample=100, chains=4)

    saturation_df = mmm.get_saturation()
    assert saturation_df.index.name == "regressor"
    assert all(saturation_df.index == mmm.get_spend_cols())

    pred_df = mmm.predict(df)
    # make sure it returns the same shape
    assert pred_df.shape[0] == n_steps
    prediction = pred_df["prediction"].values
    actaul = df["sales"]
    metrics = smape(actual=actaul, prediction=prediction)
    # make sure in-sample fitting is reasonable
    assert metrics <= 0.5
