import torch
from torch.nn import functional as F
import numpy as np
import pandas as pd
from datetime import datetime
import holidays
import re
import logging
from rich.logging import RichHandler

from itertools import product
from scipy.stats import mode as scipy_mode
from functools import reduce
from typing import List, Dict, Any, Optional


def non_zero_quantile(x: np.array, q=0.75):
    assert len(x.shape) == 1
    new_x = x[x > 0]
    return np.quantile(new_x, q=q)


def adstock_process(
    regressor_matrix: np.array,
    adstock_matrix: np.array,
):
    """Perform 1-D convolution given data matrix and adstock filter matrix
    regressor_matrix: 2-D array like with shape (n_steps, n_regressors)
    adstock_matrix: 2-D array like with shape (n_regressors, n_adstock_weights)
    """
    # flipping to align with time and convert to torch
    adstock_filters = torch.from_numpy(np.flip(adstock_matrix, 1).copy())
    # middle shape = channels/group = 1
    adstock_filters = adstock_filters.unsqueeze(1)

    # torch.conv requires 3-dimension
    x = torch.from_numpy(regressor_matrix).transpose(1, 0).unsqueeze(0)

    assert (
        adstock_filters.shape[0] == x.shape[1]
    ), "adstock channels does not match with input channels"

    adstocked_values = (
        F.conv1d(
            x,
            adstock_filters,
            groups=adstock_matrix.shape[0],
        )
        .squeeze(0)
        .transpose(1, 0)
    )
    return adstocked_values.detach().numpy()


def np_shift(x, k):
    """Perform shifting to a 2-D array given an array of steps to shift up or down

    x : 2-D array like
    dim:num of steps x num of features
    k : 1-D array like
    dim:num of features; each element represent the step to shift (negative indicates upward)

    Examples
    --------
    x = np.arange(1, 17).reshape(4, 4)
    # shift with 0 lag for col 0, 1 lag for col 1, etc.
    k = np.array([0,1,2,3])
    print(x)
    # triangle from the top (shifting array downward)
    print(np_shift(x, k))
    # triangle from the bottom (shifting array upward)
    print(np_shift(x, -1 * k))
    """

    assert len(x.shape) == 2
    assert len(k.shape) == 1
    assert x.shape[1] == len(k)
    out = np.zeros(x.shape)
    for idx, kk in enumerate(k):
        # shift down; zero padded at the beginning
        if kk > 0:
            entry = x[: (-1 * kk), idx]
            out[-len(entry) :, idx] = entry
        # shift up; zero padded at the end
        elif kk < 0:
            entry = x[(-1 * kk) :, idx]
            out[: len(entry), idx] = entry
        else:
            out[:, idx] = x[:, idx]
    return out


def insert_events(df, date_col, country):
    """
    Returns
    -------
    df : data frame with inserted events
    event_cols : list of columns describing the events has been inserted
    """
    df = df.copy()
    min_year = min(df[date_col].dt.year)
    max_year = max(df[date_col].dt.year)
    yrs = np.arange(min_year, max_year + 1).tolist()
    events_dict = {}

    for yr in yrs:
        # us_holidays = holidays.US(years=yr)
        us_holidays = holidays.country_holidays(country, years=yr)
        for dt, name in sorted(us_holidays.items()):
            # clean up the tag for machine-readable format
            name = re.sub(r"[^a-zA-Z0-9 ]", r"", name)
            name = re.sub("\W+", "-", name)
            name = name.lower()
            dt = pd.to_datetime(datetime.strftime(dt, "%Y-%m-%d"))
            if name in events_dict.keys():
                events_dict[name].append(dt)
            else:
                events_dict[name] = [dt]

    event_cols = list(events_dict.keys())

    df[event_cols] = 0.0

    for name, dts in events_dict.items():
        mask = df[date_col].isin(dts)
        df.loc[mask, name] = 1.0

    df[event_cols] = df[event_cols].astype(int)
    return df, event_cols


def extend_ts_features(df, n_periods, date_col, rolling_window=30):
    """Extending Features with rolling median

    Parameters
    ----------
    df : pd.DataFrame
    n_periods : int
    date_col : str
    rolling_window : int
    """
    infer_freq = pd.infer_freq(df[date_col])
    features = [x for x in df.columns.tolist() if x != date_col]
    fill_vals = np.median(df[features].values[-rolling_window:], 0)

    # assuming data frame is sorted
    last_dt = df[date_col].values[-1]
    extended_df = pd.DataFrame()
    extended_df[date_col] = pd.to_datetime(last_dt) + pd.to_timedelta(
        np.arange(1, n_periods + 1), unit=infer_freq
    )
    extended_df[features] = fill_vals

    res = pd.concat([df, extended_df]).reset_index(drop=True)
    return res


def expand_grid(dictionary: Dict[str, Any]):
    return pd.DataFrame(
        [row for row in product(*dictionary.values())], columns=dictionary.keys()
    )


def generate_posteriors_mode(posteriors: Dict[str, np.array], var_names: str):
    posteriors_mode = {}
    for k, v in posteriors.items():
        if k in var_names:
            posteriors_mode[k] = scipy_mode(v)[0]

    return posteriors_mode


def merge_dfs(
    dfs: List[pd.DataFrame],
    on: List[str],
    how="inner",
) -> pd.DataFrame:
    return reduce(lambda left, right: pd.merge(left, right, on=on, how=how), dfs)


def make_info_logger(name: str, path: Optional[str] = None) -> logging.Logger:
    """generate new logger in a standardized way for Karpiu

    Returns:
        logging.Logger: _description_
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%s",
    )

    if path is None:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        fh = logging.FileHandler(path, mode="w")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
