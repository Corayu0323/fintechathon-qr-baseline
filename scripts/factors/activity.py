"""Volume and amount relative-activity factors."""

from __future__ import annotations

import pandas as pd

from .common import current_bar_ready


def _relative_to_prior_5d(frame: pd.DataFrame, field: str) -> pd.Series:
    group = frame.groupby("ts_code", sort=False, observed=True)
    rolling_mean = (
        group[field].rolling(window=5, min_periods=5).mean()
        .reset_index(level=0, drop=True)
    )
    prior_mean = rolling_mean.groupby(frame["ts_code"], sort=False).shift(1)
    current_value = frame[field]
    return (current_value / prior_mean - 1).where(
        current_bar_ready(frame, 6) & prior_mean.gt(0) & current_value.ge(0)
    )


def volume_relative_prior_5d(frame: pd.DataFrame) -> pd.Series:
    return _relative_to_prior_5d(frame, "vol")


def amount_relative_prior_5d(frame: pd.DataFrame) -> pd.Series:
    return _relative_to_prior_5d(frame, "amount")
