import math
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from typing import Optional, List, Union, Dict, Literal
from copy import deepcopy
from ..explainability import AttributorBeta
import logging

logger = logging.getLogger("karpiu-planning")

from ..models import MMM


class CostCurves:
    def __init__(
        self,
        model: MMM,
        n_steps: int = 10,
        curve_type: Literal["overall", "individual"] = "overall",
        channels: Optional[Union[List[str], str]] = None,
        spend_df: Optional[pd.DataFrame] = None,
        spend_start: Optional[str] = None,
        spend_end: Optional[str] = None,
        max_spend: Optional[Union[np.array, float]] = None,
        multipliers: Optional[Union[np.array, Dict[str, np.array]]] = None,
        min_spend: float = 1.0,
    ):
        if spend_df is None:
            self.spend_df = model.raw_df.copy()
        else:
            self.spend_df = spend_df

        self.n_steps = n_steps
        self.model = deepcopy(model)

        if curve_type == "overall":
            self.channels = model.get_spend_cols()
        elif curve_type == "individual":
            all_channels = model.get_spend_cols()
            # generate the intersection of input and available channels
            if channels is not None:
                # self.channels = list(set(all_channels).intersection(set(channels)))
                self.channels = [x for x in all_channels if x in channels]
            else:
                self.channels = all_channels
        else:
            raise Exception("Invalid {} curve type input.".format(curve_type))

        self.curve_type = curve_type
        self.date_col = model.date_col
        self.n_max_adstock = model.get_max_adstock()

        if spend_start is None:
            self.spend_start = pd.to_datetime(spend_df[self.date_col].values[0])
        else:
            self.spend_start = pd.to_datetime(spend_start)

        if spend_end is None:
            self.spend_end = pd.to_datetime(
                spend_df[self.date_col].values[-1]
            ) - pd.Timedelta(days=self.n_max_adstock)
        else:
            self.spend_end = pd.to_datetime(spend_end)

        # spend_mask, outcome_mask, outcome_start & outcome_end
        self.derive_masks()

        # (n_steps, n_channels)
        spend_matrix = self.spend_df.loc[self.spend_mask, self.channels].values
        # it's buggy to estimate slope at zero or working with multiplier
        # hence, add a very small delta to make the analysis work
        zero_spend_flag = spend_matrix < min_spend
        if np.any(np.sum(zero_spend_flag)):
            logger.info(
                "Minimum spend threshold is hit in some channel(s). Update with value {}.".format(
                    min_spend
                )
            )
            spend_matrix[zero_spend_flag] = min_spend
            self.spend_df.loc[self.spend_mask, self.channels] = spend_matrix

        # before derive multipliers, calculate the max spend (last simulation point = max spend * extend_multiplier)
        # if it is not supplied, we will use the max spend of the highest volume channel
        # (n_channels, )
        self.total_spend_arr = np.sum(spend_matrix, axis=0)
        if max_spend is not None:
            self.max_spend = max_spend
        else:
            self.max_spend = np.max(self.total_spend_arr)

        if multipliers is not None:
            self.multipliers = multipliers
        else:
            if curve_type == "overall":
                self.multipliers = self.derive_multipliers(extend_multiplier=2.0)
            elif curve_type == "individual":
                # since all the small channels can compare with the largest spend channel
                # we don't need to extend by default
                self.multipliers = self.derive_multipliers(extend_multiplier=2.0)
        # will be generated under generate_cost_curves
        self.cost_curves = None

    def derive_masks(self):
        self.outcome_start = self.spend_start
        self.outcome_end = self.spend_end + pd.Timedelta(days=self.n_max_adstock)
        self.spend_mask = (self.spend_df[self.date_col] >= self.spend_start) & (
            self.spend_df[self.date_col] <= self.spend_end
        )
        self.outcome_mask = (self.spend_df[self.date_col] >= self.outcome_start) & (
            self.spend_df[self.date_col] <= self.outcome_end
        )

    def derive_multipliers(
        self,
        extend_multiplier=1.0,
    ):
        # compute flat multiplier if it is an overall cost curves
        # otherwise, compute cost curves based on max spend across all channels
        # with the max multiplier
        if self.curve_type == "overall":
            multipliers = np.sort(
                np.unique(
                    np.concatenate(
                        [np.ones(1), np.linspace(0.0, extend_multiplier, self.n_steps)]
                    )
                )
            )
        elif self.curve_type == "individual":
            multipliers_arr = self.max_spend * extend_multiplier / self.total_spend_arr
            multipliers = {
                # always make sure we have an additional "1" for current spend
                k: np.sort(
                    np.unique(
                        np.concatenate([np.ones(1), np.linspace(0.0, m, self.n_steps)])
                    )
                )
                for k, m in zip(self.channels, multipliers_arr)
            }
            logger.info("Derived channels multipliers based on input spend.")
        else:
            raise Exception("Invalid {} curve type input.".format(self.curve_type))
        return multipliers

    def generate_cost_curves(
        self, multipliers: Optional[Union[List[float], Dict[str, float]]] = None
    ) -> pd.DataFrame:
        if multipliers is None:
            multipliers = self.get_multipliers()

        if not (
            isinstance(multipliers, (dict, list, np.ndarray, pd.core.series.Series))
        ):
            raise Exception("Invalid multipliers type -- must be in Dict or List type.")

        # output
        cost_curves_dict = {
            "ch": list(),
            "total_spend": list(),
            "total_outcome": list(),
            "multiplier": list(),
        }

        # create a case with all spend in the spend range set to zero to estimate organic
        # note that it doesn't include past spend as it already happens
        temp_df = self.spend_df.copy()
        temp_df.loc[self.spend_mask, self.channels] = 0.0
        pred = self.model.predict(df=temp_df)
        total_outcome = np.sum(pred.loc[self.outcome_mask, "prediction"].values)
        cost_curves_dict["ch"].append("organic")
        cost_curves_dict["total_spend"].append(0.0)
        cost_curves_dict["total_outcome"].append(total_outcome)
        cost_curves_dict["multiplier"].append(1.0)

        # decide to whether compute overall or individual channel cost curves
        if self.curve_type == "overall":
            # multiply with all channels, hence we can reuse the spend_matrix
            spend_matrix = self.spend_df.loc[self.spend_mask, self.channels].values
            # self.multipliers is a list
            for m in self.multipliers:
                temp_df = self.spend_df.copy()
                temp_df.loc[self.spend_mask, self.channels] = spend_matrix * m

                attr_obj = AttributorBeta(
                    self.model,
                    df=temp_df,
                    attr_regressors=self.channels,
                    start=self.spend_start,
                    end=self.spend_end,
                )

                _, spend_attr_df, _, _ = attr_obj.make_attribution(true_up=False, fixed_intercept=True)
                total_spend = np.sum(spend_matrix * m)
                total_outcome = np.sum(spend_attr_df.loc[:, self.channels].values)

                cost_curves_dict["ch"].append("overall")
                cost_curves_dict["total_spend"].append(total_spend)
                cost_curves_dict["total_outcome"].append(total_outcome)
                cost_curves_dict["multiplier"].append(m)

        elif self.curve_type == "individual":
            # self.multipliers is a dict
            for ch, multipliers in tqdm(self.multipliers.items()):
                for m in multipliers:
                    temp_df = self.spend_df.copy()
                    temp_df.loc[self.spend_mask, ch] = (
                        self.spend_df.loc[self.spend_mask, ch] * m
                    )
                    total_spend = np.sum(temp_df.loc[self.spend_mask, ch].values)

                    attr_obj = AttributorBeta(
                        self.model,
                        df=temp_df,
                        attr_regressors=[ch],
                        start=self.spend_start,
                        end=self.spend_end,
                    )
                    _, spend_attr_df, _, _ = attr_obj.make_attribution(true_up=False, fixed_intercept=True)
                    # pred = self.model.predict(df=temp_df)

                    total_outcome = np.sum(spend_attr_df.loc[:, [ch]].values)
                    cost_curves_dict["ch"].append(ch)
                    cost_curves_dict["total_spend"].append(total_spend)
                    cost_curves_dict["total_outcome"].append(total_outcome)
                    cost_curves_dict["multiplier"].append(m)
        else:
            raise Exception("Invalid {} curve type input.".format(self.curve_type))

        self.cost_curves = pd.DataFrame(cost_curves_dict)

    def get_multipliers(self) -> Union[np.array, Dict[str, np.array]]:
        return deepcopy(self.multipliers)

    def get_cost_curves(self) -> pd.DataFrame:
        return deepcopy(self.cost_curves)

    def get_max_spend(self) -> np.array:
        return deepcopy(self.max_spend)

    def plot(
        self,
        spend_scaler: float = 1e3,
        outcome_scaler: float = 1e3,
        optim_cost_curves: Optional[pd.DataFrame] = None,
        plot_margin: float = 0.05,
        is_visible: bool = True,
    ) -> None:
        n_channels = len(self.channels)
        nrows = math.ceil(n_channels / 2)

        y_min = np.min(self.cost_curves["total_outcome"].values) / outcome_scaler
        y_max = np.max(self.cost_curves["total_outcome"].values) / outcome_scaler
        x_max = np.max(self.cost_curves["total_spend"].values) / spend_scaler

        if optim_cost_curves is not None:
            y_min2 = np.min(optim_cost_curves["total_outcome"]) / outcome_scaler
            y_max2 = np.max(optim_cost_curves["total_outcome"]) / outcome_scaler
            y_min = min(y_min, y_min2)
            y_max = max(y_max, y_max2)

        organic_outcome = self.cost_curves.loc[
            self.cost_curves["ch"] == "organic", "total_outcome"
        ].values

        if self.curve_type == "individual":
            # multiple cost curves
            fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(18, nrows * 4.5))
            axes = axes.flatten()

            for idx, ch in enumerate(self.channels):
                temp_cc = self.cost_curves[self.cost_curves["ch"] == ch].reset_index(
                    drop=True
                )
                curr_spend_mask = temp_cc["multiplier"] == 1

                axes[idx].plot(
                    temp_cc["total_spend"].values / spend_scaler,
                    temp_cc["total_outcome"].values / outcome_scaler,
                    label="current cost curve",
                    color="red",
                    alpha=0.8,
                )

                curr_spend = (
                    temp_cc.loc[curr_spend_mask, "total_spend"].values / spend_scaler
                )
                curr_outcome = (
                    temp_cc.loc[curr_spend_mask, "total_outcome"].values
                    / outcome_scaler
                )

                # plot a dot for current spend
                axes[idx].scatter(
                    curr_spend,
                    curr_outcome,
                    c="orange",
                    s=48,
                    label="current spend",
                )

                axes[idx].axhline(
                    y=organic_outcome / spend_scaler,
                    linestyle="dashed",
                    label="organic",
                )

                if optim_cost_curves is not None:
                    temp_optim_cc = optim_cost_curves[
                        optim_cost_curves["ch"] == ch
                    ].reset_index(drop=True)
                    optim_spend_mask = temp_optim_cc["multiplier"] == 1
                    axes[idx].plot(
                        temp_optim_cc["total_spend"].values / spend_scaler,
                        temp_optim_cc["total_outcome"].values / outcome_scaler,
                        label="optim cost curve",
                        color="forestgreen",
                        alpha=0.8,
                    )

                    optim_spend = (
                        temp_optim_cc.loc[optim_spend_mask, "total_spend"].values
                        / spend_scaler
                    )
                    optim_outcome = (
                        temp_optim_cc.loc[optim_spend_mask, "total_outcome"].values
                        / outcome_scaler
                    )

                    axes[idx].scatter(
                        optim_spend,
                        optim_outcome,
                        c="green",
                        s=48,
                        label="optim spend",
                    )

                axes[idx].set_title(ch, fontdict={"fontsize": 18})
                axes[idx].grid(
                    linestyle="dotted", linewidth=0.7, color="grey", alpha=0.8
                )
                axes[idx].set_xlabel("spend")
                axes[idx].set_ylabel("signups")
                axes[idx].xaxis.set_major_formatter("${x:1.0f}")
                axes[idx].set_ylim(y_min * (1 - plot_margin), y_max * (1 + plot_margin))
                axes[idx].set_xlim(left=0.0, right=x_max)

            handles, labels = axes[-1].get_legend_handles_labels()

        elif self.curve_type == "overall":
            # single cost curve
            fig, ax = plt.subplots(1, 1, figsize=(18, 12))
            temp_cc = self.cost_curves[self.cost_curves["ch"] == "overall"].reset_index(
                drop=True
            )
            curr_spend_mask = temp_cc["multiplier"] == 1
            ax.plot(
                temp_cc["total_spend"].values / spend_scaler,
                temp_cc["total_outcome"].values / outcome_scaler,
                label="current cost curve",
                color="red",
                alpha=0.8,
            )
            curr_spend = (
                temp_cc.loc[curr_spend_mask, "total_spend"].values / spend_scaler
            )
            curr_outcome = (
                temp_cc.loc[curr_spend_mask, "total_outcome"].values / outcome_scaler
            )

            ax.scatter(
                curr_spend,
                curr_outcome,
                c="orange",
                s=48,
                label="current spend",
            )

            ax.axhline(
                y=organic_outcome / spend_scaler, linestyle="dashed", label="organic"
            )
            ax.set_xlim(left=0.0, right=x_max)

            if optim_cost_curves is not None:
                temp_optim_cc = optim_cost_curves[
                    optim_cost_curves["ch"] == "overall"
                ].reset_index(drop=True)
                optim_spend_mask = temp_optim_cc["multiplier"] == 1
                ax.plot(
                    temp_optim_cc["total_spend"].values / spend_scaler,
                    temp_optim_cc["total_outcome"].values / outcome_scaler,
                    label="optim cost curve",
                    color="forestgreen",
                    alpha=0.8,
                )

                optim_spend = (
                    temp_optim_cc.loc[optim_spend_mask, "total_spend"].values
                    / spend_scaler
                )
                optim_outcome = (
                    temp_optim_cc.loc[optim_spend_mask, "total_outcome"].values
                    / outcome_scaler
                )

                ax.scatter(
                    optim_spend,
                    optim_outcome,
                    c="green",
                    s=48,
                    label="optim spend",
                )

            handles, labels = ax.get_legend_handles_labels()
        # exit from overall vs. individual cost curve

        fig.legend(
            handles, labels, loc=9, ncol=2, bbox_to_anchor=(0.5, 0), prop={"size": 18}
        )
        fig.tight_layout()

        if is_visible:
            plt.show()
        else:
            plt.close()

        if self.curve_type == "individual":
            return axes
        else:
            return ax
