"""Close-to-close return factors."""

from __future__ import annotations

import pandas as pd

from .common import current_bar_ready


def _close_return(frame: pd.DataFrame, horizon: int) -> pd.Series:
    close = frame["close"]
    previous = frame.groupby("ts_code", sort=False, observed=True)["close"].shift(horizon)
    return (close / previous - 1).where(current_bar_ready(frame, horizon + 1))


def close_return_1d(frame: pd.DataFrame) -> pd.Series:
    return _close_return(frame, 1)


def close_return_5d(frame: pd.DataFrame) -> pd.Series:
    return _close_return(frame, 5)


def close_return_20d(frame: pd.DataFrame) -> pd.Series:
    return _close_return(frame, 20)
