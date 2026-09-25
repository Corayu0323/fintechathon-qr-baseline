"""Windowed adapter from the competition's factor contract to Qlib.

Qlib never reads the competition CSVs or its bundled market data here.  Factor
files are produced independently; this module only aligns them with the
unaltered official labels and auxiliary masks in ``data/prepared``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


KEY = ["trade_date", "ts_code"]
AUXILIARY = [
    "flag_limit_up", "flag_limit_down", "bar_valid", "volume_zero",
    "label_present", "label_maturity_date", "valid_bar_run", "source_row",
]


@dataclass
class Window:
    features: pd.DataFrame
    labels: pd.DataFrame
    auxiliary: pd.DataFrame

    @property
    def feature_names(self) -> list[str]:
        return list(self.features.columns)


def _years(start_date: int, end_date: int) -> range:
    if start_date > end_date or start_date // 10000 < 2018:
        raise ValueError("invalid date window")
    return range(start_date // 10000, end_date // 10000 + 1)


def _read_years(directory: Path, prefix: str, columns: list[str],
                start_date: int, end_date: int) -> pd.DataFrame:
    parts = []
    for year in _years(start_date, end_date):
        path = directory / f"{prefix}_{year}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"missing partition: {path}")
        table = pq.read_table(
            path, columns=columns,
            filters=[("trade_date", ">=", start_date),
                     ("trade_date", "<=", end_date)],
        )
        parts.append(table.to_pandas())
    return pd.concat(parts, ignore_index=True)


def _index(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if frame[KEY].isna().any().any() or frame.duplicated(KEY).any():
        raise ValueError(f"{name}: missing or duplicate (trade_date, ts_code)")
    return frame.set_index(KEY, verify_integrity=True).sort_index()


def load_window(
    prepared_dir: Path,
    factor_dir: Path,
    split: str,
    start_date: int,
    end_date: int,
    *,
    min_valid_run: int,
    label_asof_date: int | None = None,
) -> Window:
    """Load one bounded window with exact key alignment.

    ``features_YYYY.parquet`` must contain every source key and one or more
    numeric ``feature_*`` columns.  Raw labels remain unchanged.  Eligibility
    is exposed separately and applied only when building Qlib's learn group.
    """
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")
    if split == "train" and label_asof_date is None:
        raise ValueError("training requires label_asof_date")
    if min_valid_run < 1:
        raise ValueError("min_valid_run must come from the feature catalog")

    source_columns = KEY + AUXILIARY
    if split == "train":
        source_columns += ["y_ret_1d"]
    source = _read_years(
        prepared_dir, split, source_columns, start_date, end_date,
    )
    source = _index(source, "prepared")

    factor_files = [factor_dir / f"features_{year}.parquet"
                    for year in _years(start_date, end_date)]
    feature_names = None
    for path in factor_files:
        if not path.is_file():
            raise FileNotFoundError(f"factor matrix not ready: {path}")
        names = pq.read_schema(path).names
        selected = [name for name in names if name.startswith("feature_")]
        if not selected or not set(KEY).issubset(names):
            raise ValueError(f"{path}: expected keys and feature_* columns")
        if feature_names is None:
            feature_names = selected
        elif feature_names != selected:
            raise ValueError(f"{path}: feature names/order differ across years")

    factor = _index(_read_years(
        factor_dir, "features", KEY + feature_names, start_date, end_date,
    ), "features")
    if not factor.index.equals(source.index):
        missing = source.index.difference(factor.index)
        extra = factor.index.difference(source.index)
        if len(missing) or len(extra) or len(factor.index) != len(source.index):
            raise ValueError(f"factor keys mismatch: missing={len(missing)}, extra={len(extra)}")
        # Equivalent keys can arrive with different integer widths or ordering.
        # Reindex only after proving a one-to-one exact key set match.
        factor = factor.reindex(source.index)
    for name in feature_names:
        if not pd.api.types.is_numeric_dtype(factor[name]):
            raise TypeError(f"{name} is not numeric")
        factor[name] = factor[name].astype("float32")

    if split == "train":
        labels = source[["y_ret_1d"]].copy()
    else:
        # Test labels are not loaded from disk, even if a source partition
        # happens to contain a populated label column.
        labels = pd.DataFrame({"y_ret_1d": np.nan}, index=source.index)
    auxiliary = source[AUXILIARY].copy()
    if split == "train":
        auxiliary["label_available"] = source["label_present"] & (
            source["label_maturity_date"] <= label_asof_date
        ) & (source["label_maturity_date"] > 0)
    else:
        auxiliary["label_available"] = False
    auxiliary["history_ready_L"] = source["valid_bar_run"].ge(min_valid_run)
    auxiliary["model_ready"] = (
        source["bar_valid"] & auxiliary["history_ready_L"]
        & pd.Series(np.isfinite(factor.to_numpy()).all(axis=1), index=factor.index)
    )

    return Window(factor, labels, auxiliary)


def to_qlib_handler(window: Window):
    """Build Qlib's standard feature/label groups from an already sliced window."""
    from qlib.data.dataset import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader

    def qlib_index(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.reset_index()
        result["datetime"] = pd.to_datetime(
            result["trade_date"].astype(str), format="%Y%m%d"
        )
        result = result.rename(columns={"ts_code": "instrument"})
        return result.drop(columns="trade_date").set_index(
            ["datetime", "instrument"], verify_integrity=True
        ).sort_index()

    learn_labels = window.labels.copy()
    train_eligible = window.auxiliary["label_available"] & window.auxiliary["model_ready"]
    learn_labels.loc[~train_eligible, "y_ret_1d"] = np.nan
    loader = StaticDataLoader({
        "feature": qlib_index(window.features),
        "label": qlib_index(learn_labels),
    })
    return DataHandlerLP(
        data_loader=loader,
        infer_processors=[], learn_processors=[], shared_processors=[],
    )


def complete_predictions(window: Window, model_pred: pd.Series) -> pd.DataFrame:
    """Preserve every key and put model-unready rows below that day's valid scores."""
    if not window.features.index.equals(window.auxiliary.index):
        raise ValueError("feature and auxiliary keys differ")
    if not model_pred.index.is_unique:
        raise ValueError("duplicate model prediction keys")

    # Support both Qlib model indices and estimators fit directly on Window keys.
    raw = model_pred.rename("pred_model").reset_index()
    if {"datetime", "instrument"}.issubset(raw.columns):
        raw["trade_date"] = pd.to_datetime(raw["datetime"]).dt.strftime("%Y%m%d").astype("int32")
        raw = raw.rename(columns={"instrument": "ts_code"}).drop(columns="datetime")
    elif not set(KEY).issubset(raw.columns):
        raise ValueError("model prediction index must be (datetime, instrument) or (trade_date, ts_code)")
    raw = raw.set_index(KEY, verify_integrity=True)
    if not raw.index.equals(window.features.index):
        if set(raw.index) != set(window.features.index):
            raise ValueError("model prediction keys differ from source")
        raw = raw.reindex(window.features.index)

    model_ready = window.auxiliary["model_ready"].astype(bool)
    finite = pd.Series(np.isfinite(raw["pred_model"].to_numpy()), index=raw.index)
    if (model_ready & ~finite).any():
        bad = int((model_ready & ~finite).sum())
        raise ValueError(f"model produced nonfinite prediction for {bad} ready rows")
    ready = model_ready & finite
    result = pd.DataFrame({"pred": raw["pred_model"].astype("float64").where(ready),
                           "model_ready": ready}, index=window.features.index)
    for trade_date, positions in result.groupby(level="trade_date").indices.items():
        day = result.iloc[positions]
        valid = day["pred"].dropna()
        if valid.empty:
            raise ValueError(f"{trade_date}: no valid model predictions")
        eligible_nonlimit = int(
            (window.auxiliary.iloc[positions]["flag_limit_up"] == 0).sum()
        )
        ready_nonlimit = int(
            (day["model_ready"] &
             (window.auxiliary.iloc[positions]["flag_limit_up"].to_numpy() == 0)).sum()
        )
        if ready_nonlimit < max(eligible_nonlimit // 10, 1):
            raise ValueError(
                f"{trade_date}: only {ready_nonlimit} normal non-limit predictions; "
                f"need at least {max(eligible_nonlimit // 10, 1)} to keep fallback rows outside Top 10%"
            )
        spread = max(float(valid.max() - valid.min()), 1e-6)
        result.iloc[positions, result.columns.get_loc("pred")] = (
            result.iloc[positions]["pred"].fillna(float(valid.min()) - spread)
        )
    if not np.isfinite(result["pred"]).all():
        raise ValueError("submission predictions must all be finite")
    return result.reset_index()[KEY + ["pred", "model_ready"]]
