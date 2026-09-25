"""Reusable preprocessing for factor panels before model fitting and scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_TRANSFORMS = ["train_zscore", "cs_zscore", "cs_rank", "identity"]


def transform_inputs(
    features: pd.DataFrame,
    ready: pd.Series,
    fit_mask: pd.Series,
    method: str,
    fitted_state: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Transform one raw factor panel; return transformed panel and state.

    ``fit_mask`` determines the static training-period scaler fit population.
    Cross-sectional transforms use all ready rows on each date, independent of
    label availability. Missing/unready rows stay missing and never become
    eligible through preprocessing.
    """
    if method not in {*DEFAULT_TRANSFORMS}:
        raise ValueError(f"unknown transform: {method}")
    result = pd.DataFrame(np.nan, index=features.index, columns=features.columns, dtype=np.float32)
    valid_rows = ready & pd.Series(np.isfinite(features.to_numpy()).all(axis=1), index=features.index)
    state: dict = {"method": method, "fit_rows": int((fit_mask & valid_rows).sum())}

    if method == "identity":
        result.loc[valid_rows] = features.loc[valid_rows].astype(np.float32)
    elif method == "train_zscore":
        if fitted_state is None:
            selected = features.loc[fit_mask & valid_rows].astype(np.float64)
            if selected.empty:
                raise ValueError("train_zscore has no eligible fit rows")
            mean = selected.mean(axis=0)
            scale = selected.std(axis=0, ddof=0)
            constant = scale.eq(0)
            scale = scale.mask(constant, 1.0)
        else:
            mean = pd.Series(fitted_state["mean"], dtype=np.float64).reindex(features.columns)
            scale = pd.Series(fitted_state["scale"], dtype=np.float64).reindex(features.columns)
            constant = pd.Series(
                features.columns.isin(fitted_state["constant_columns"]), index=features.columns
            )
            if mean.isna().any() or scale.isna().any():
                raise ValueError("stored scaler columns do not match the factor matrix")
        transformed = (features.loc[valid_rows].astype(np.float64) - mean) / scale
        result.loc[valid_rows] = transformed.astype(np.float32)
        state.update({
            "mean": {key: float(value) for key, value in mean.items()},
            "scale": {key: float(value) for key, value in scale.items()},
            "constant_columns": list(constant[constant].index),
        })
    else:
        full_values = features.loc[valid_rows].astype(np.float64)
        for trade_date, positions in full_values.groupby(level="trade_date", sort=True).groups.items():
            cross_section = full_values.loc[positions]
            count = len(cross_section)
            if count == 1:
                converted = cross_section * 0.0
            elif method == "cs_zscore":
                mean = cross_section.mean(axis=0)
                scale = cross_section.std(axis=0, ddof=0).replace(0.0, 1.0)
                converted = (cross_section - mean) / scale
            else:
                ranks = cross_section.rank(axis=0, method="average", ascending=True)
                converted = (ranks - 1.0) / (count - 1.0) - 0.5
            result.loc[positions] = converted.astype(np.float32)
        state.update({
            "cross_section": "all model_ready rows for the date, before label filtering",
            "tie_rule": "average rank" if method == "cs_rank" else None,
        })

    if not np.isfinite(result.loc[valid_rows].to_numpy()).all():
        raise ValueError(f"{method} produced nonfinite values on ready rows")
    return result, state
