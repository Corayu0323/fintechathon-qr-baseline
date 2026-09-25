"""Shared eligibility masks for factor formulas."""

from __future__ import annotations

import pandas as pd


def current_bar_ready(frame: pd.DataFrame, required_valid_bars: int) -> pd.Series:
    """Whether the current row ends a sufficiently long valid-bar run."""
    return (
        frame["bar_valid"].fillna(False).astype(bool)
        & pd.to_numeric(frame["valid_bar_run"], errors="coerce").fillna(0).ge(required_valid_bars)
    )
