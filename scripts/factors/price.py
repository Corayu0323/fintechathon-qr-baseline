"""Single-bar price shape factors."""

from __future__ import annotations

import pandas as pd

from .common import current_bar_ready


def intraday_return(frame: pd.DataFrame) -> pd.Series:
    open_ = frame["open"]
    return (frame["close"] / open_ - 1).where(current_bar_ready(frame, 1) & open_.gt(0))


def range_over_open(frame: pd.DataFrame) -> pd.Series:
    open_ = frame["open"]
    high = frame["high"]
    low = frame["low"]
    return ((high - low) / open_).where(
        current_bar_ready(frame, 1) & open_.gt(0) & high.ge(low)
    )


def close_position_in_range(frame: pd.DataFrame) -> pd.Series:
    high = frame["high"]
    low = frame["low"]
    close = frame["close"]
    position = (close - low) / (high - low)
    position = position.where(high.gt(low), 0.5)
    return position.where(
        current_bar_ready(frame, 1) & high.ge(low) & close.ge(low) & close.le(high)
    )
