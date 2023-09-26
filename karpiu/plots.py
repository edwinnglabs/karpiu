import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import numpy as np

from .models import MMM


ORGANIC_COL = "organic"


class ColorConstants:
    QUALITATIVE_FIVE = [
        "#264653",
        "#2A9D8F",
        "#E9C46A",
        "#F4A261",
        "#E76F51",
    ]
    RAINBOW_SIX = [
        "#e4c1f9",
        "#1982c4",
        "#8ac926",
        "#ffca3a",
        "#ff924c",
        "#ff595e",
    ]
    RAINBOW_EIGHT = [
        "#FFD6A5",
        "#FDFFB6",
        "#CAFFBF",
        "#9BF6FF",
        "#A0C4FF",
        # extra
        # red
        "#FFADAD",
        # violet
        "#BDB2FF",
        # purple
        "#FFC6FF",
    ]


def plot_attribution_waterfall(
    model: MMM,
    attr_df: pd.DataFrame,
    figsize=None,
    colors=None,
    show=True,
    include_organic=True,
    alpha=0.8,
    grid_on=True,
    frame_on=False,
):
    if figsize is None:
        figsize = (16, 8)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_frame_on(frame_on)

    spend_cols = model.get_spend_cols()
    if include_organic:
        plot_cols = spend_cols + [ORGANIC_COL]
    else:
        plot_cols = spend_cols
    plot_cols = np.array([x for x in attr_df.columns if x in plot_cols])
    attr_sum = np.sum(attr_df[plot_cols].values, 0)
    attr_perc = attr_sum / np.sum(attr_sum)

    inds = np.argsort(attr_perc)
    attr_perc = attr_perc[inds]
    # reverse order to make largest go first
    attr_perc_cumsum = np.concatenate([np.zeros(1), np.cumsum(attr_perc[::-1])])
    # reverse this back so that the largest go top
    attr_perc_cumsum = attr_perc_cumsum[::-1]

    # plot columns order preserved the original
    plot_cols = plot_cols[inds]
    colors = np.array(colors)
    colors = colors[inds]

    ax.barh(
        plot_cols,
        width=attr_perc,
        left=attr_perc_cumsum[1:],
        color=colors,
        alpha=alpha,
        align="center",
    )

    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_xlim(0, 1)

    for idx, (p, p_cs) in enumerate(zip(attr_perc, attr_perc_cumsum[:-1])):
        ax.annotate(
            "{:.2%}".format(p),
            (p_cs - 0.08, idx - 0.05),
        )

    if grid_on:
        ax.grid(color="grey", linestyle="--", alpha=0.3)

    fig.tight_layout()

    if show:
        plt.show()
    else:
        plt.close()

    return ax


def plot_attribution_with_time(
    model: MMM,
    attr_df: pd.DataFrame,
    figsize=None,
    colors=None,
    show=True,
    dt_col="date",
    include_organic=True,
    alpha=0.8,
    grid_on=True,
    frame_on=False,
):
    if figsize is None:
        figsize = (16, 8)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_frame_on(frame_on)
    spend_cols = model.get_spend_cols()
    if include_organic:
        plot_cols = spend_cols + [ORGANIC_COL]
    else:
        plot_cols = spend_cols
    plot_cols = [x for x in attr_df.columns if x in plot_cols]

    ax.set_xlim(attr_df[dt_col].min(), attr_df[dt_col].max())
    ax.stackplot(
        attr_df[dt_col].values,
        attr_df[plot_cols].values.transpose(),
        labels=plot_cols,
        colors=colors,
        alpha=alpha,
    )
    if grid_on:
        ax.grid(color="grey", linestyle="--", alpha=0.5)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        frameon=False,
        loc="upper left",
        ncol=2,
        # bbox_to_anchor=(1.0, -.9),
    )
    fig.tight_layout()

    if show:
        plt.show()
    else:
        plt.close()

    return ax
