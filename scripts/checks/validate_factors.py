#!/usr/bin/env python3
"""Check factor-set integrity, old/new numerical parity, and temporal behavior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]

from scripts.factors.build import calculate, load_spec
from scripts.factors.diagnostics import evaluate_factor_panel
from scripts.modeling.preprocessing import transform_inputs


KEYS = ["trade_date", "ts_code"]


def validate_real_matrix(reference_dir: Path, version_dir: Path, prepared_dir: Path, spec: dict) -> None:
    names = [feature["name"] for feature in spec["features"]]
    for feature in spec["features"]:
        if "y_ret_1d" in feature["source_fields"]:
            raise AssertionError(f"label leak declared in factor spec: {feature['name']}")

    old_catalog = reference_dir / "catalog.json"
    if not old_catalog.is_file():
        raise FileNotFoundError(f"reference factor catalog missing: {old_catalog}")
    old_names = [feature["name"] for feature in json.loads(
        old_catalog.read_text(encoding="utf-8")
    )["features"]]
    if old_names != names:
        raise AssertionError(f"old/new factor column order differs: {old_names} != {names}")

    new_catalog_path = version_dir / "catalog.json"
    if not new_catalog_path.is_file():
        raise FileNotFoundError(f"version catalog missing: {new_catalog_path}")
    for new_partition in json.loads(new_catalog_path.read_text(encoding="utf-8"))["partitions"]:
        split, year = new_partition["split"], new_partition["year"]
        old = pq.read_table(reference_dir / f"features_{year}.parquet").to_pandas()
        new = pq.read_table(version_dir / f"features_{year}.parquet").to_pandas()
        source = pq.read_table(prepared_dir / f"{split}_{year}.parquet", columns=KEYS).to_pandas()
        for frame in (source, old, new):
            frame["trade_date"] = pd.to_numeric(frame["trade_date"]).astype("int64")
            frame["ts_code"] = frame["ts_code"].astype(str)
        source = source.sort_values(KEYS, kind="mergesort").reset_index(drop=True)
        old = old.sort_values(KEYS, kind="mergesort").reset_index(drop=True)
        new = new.sort_values(KEYS, kind="mergesort").reset_index(drop=True)
        if list(old.columns) != KEYS + names or list(new.columns) != KEYS + names:
            raise AssertionError(f"{year}: output columns/order differ from catalog")
        if old[KEYS].duplicated().any() or new[KEYS].duplicated().any():
            raise AssertionError(f"{year}: duplicate output keys")
        if not old[KEYS].equals(new[KEYS]) or not new[KEYS].equals(source[KEYS]):
            raise AssertionError(f"{year}: source keys changed")
        for name in names:
            before = old[name].to_numpy(dtype=np.float32)
            after = new[name].to_numpy(dtype=np.float32)
            if not np.array_equal(np.isnan(before), np.isnan(after)):
                raise AssertionError(f"{year}/{name}: missing-value mask changed")
            if not np.isfinite(after[~np.isnan(after)]).all():
                raise AssertionError(f"{year}/{name}: infinite values present")
            np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-7, equal_nan=True)
        ready = np.isfinite(new[names].to_numpy()).all(axis=1)
        if int(ready.sum()) != int(new_partition["ready_rows"]):
            raise AssertionError(f"{year}: ready-row count differs from catalog")
        differences = []
        for name in names:
            diff = np.abs(old[name].to_numpy(dtype=np.float64) - new[name].to_numpy(dtype=np.float64))
            finite_diff = diff[np.isfinite(diff)]
            differences.append(float(finite_diff.max()) if len(finite_diff) else 0.0)
        max_abs = max(differences)
        print(f"{year}: keys={len(new):,} ready={int(ready.sum()):,} max_abs_diff={max_abs:.3g}")


def synthetic_panel() -> pd.DataFrame:
    rows = []
    for stock in ("000001.SZ", "000002.SZ"):
        run = 0
        for offset in range(32):
            valid = not (stock == "000002.SZ" and offset == 17)
            run = run + 1 if valid else 0
            close = 10.0 + offset * 0.07 + (0.4 if stock == "000002.SZ" else 0)
            rows.append({
                "trade_date": 20240101 + offset,
                "ts_code": stock,
                "open": close * 0.995,
                "high": close * 1.02,
                "low": close * 0.98,
                "close": close,
                "vol": 1000.0 + 7 * offset,
                "amount": 50000.0 + 300 * offset,
                "bar_valid": valid,
                "valid_bar_run": run,
            })
    return pd.DataFrame(rows)


def validate_temporal_invariants(spec: dict) -> None:
    panel = synthetic_panel()
    full = calculate(panel, None, spec)
    cutoff = 20240122
    perturbed = panel.copy()
    future = perturbed["trade_date"] > cutoff
    perturbed.loc[future, ["open", "high", "low", "close", "vol", "amount"]] *= 3.0
    future_changed = calculate(perturbed, None, spec)
    past = full["trade_date"] <= cutoff
    names = [feature["name"] for feature in spec["features"]]
    for name in names:
        np.testing.assert_allclose(
            full.loc[past, name].to_numpy(), future_changed.loc[past, name].to_numpy(),
            rtol=0, atol=0, equal_nan=True,
        )

    boundary = 20240115
    prior = panel.loc[panel["trade_date"] < boundary]
    current = panel.loc[panel["trade_date"] >= boundary]
    partitioned = calculate(current, prior, spec)
    reference = full.loc[full["trade_date"] >= boundary].reset_index(drop=True)
    if not partitioned[KEYS].equals(reference[KEYS]):
        raise AssertionError("partitioned computation changed source keys")
    for name in names:
        np.testing.assert_allclose(
            reference[name].to_numpy(), partitioned[name].to_numpy(),
            rtol=1e-6, atol=1e-7, equal_nan=True,
        )
    print("synthetic checks: no future dependence; partition warm-up matches full-series computation")


def validate_diagnostics(spec: dict) -> None:
    names = [feature["name"] for feature in spec["features"]]
    rows = []
    for date_offset in range(3):
        for stock in range(40):
            rows.append({
                "trade_date": 20240101 + date_offset,
                "ts_code": f"{stock:06d}.SZ",
                **{name: float(stock) + date_offset * 0.01 for name in names},
                "y_ret_1d": float(stock) / 1000 + (stock % 3) * 1e-5,
                "label_available": True,
            })
    data = pd.DataFrame(rows).set_index(KEYS).sort_index()
    window = SimpleNamespace(
        features=data[names],
        labels=data[["y_ret_1d"]],
        auxiliary=data[["label_available"]],
    )
    one_factor_catalog = {"features": [spec["features"][0]]}
    summary, daily, correlations = evaluate_factor_panel(
        window, one_factor_catalog, label_asof_date=20240102,
    )
    row = summary.iloc[0]
    if row["eligible_label_rows"] != 80 or row["factor_valid_rows"] != 80:
        raise AssertionError("factor diagnostic did not enforce the label as-of boundary")
    if row["rankic_dates"] != 2 or row["rankic_mean"] < 0.99:
        raise AssertionError("factor diagnostic rank IC failed a known monotonic fixture")
    if len(daily) != 3 or len(correlations):
        raise AssertionError("factor diagnostic date/pair output shape is incorrect")
    print("synthetic diagnostic check: mature-label cutoff, coverage, daily Rank IC passed")


def validate_preprocessing() -> None:
    index = pd.MultiIndex.from_product(
        [[20240101, 20240102], ["000001.SZ", "000002.SZ", "000003.SZ"]],
        names=KEYS,
    )
    features = pd.DataFrame({"a": [1, 2, 3, 2, 4, 6], "b": [5, 5, 5, 1, 2, 3]}, index=index)
    ready = pd.Series([True, True, True, True, True, False], index=index)
    fit_mask = pd.Series([True, True, False, True, True, True], index=index)
    for method in ("identity", "train_zscore", "cs_zscore", "cs_rank"):
        transformed, state = transform_inputs(features, ready, fit_mask, method)
        if transformed.iloc[-1].notna().any():
            raise AssertionError(f"{method}: preprocessing made an unready row usable")
        if not np.isfinite(transformed.loc[ready].to_numpy()).all():
            raise AssertionError(f"{method}: ready synthetic rows became nonfinite")
        if method == "train_zscore":
            reused, reused_state = transform_inputs(features, ready, fit_mask, method, state)
            pd.testing.assert_frame_equal(transformed, reused)
            if reused_state["mean"] != state["mean"]:
                raise AssertionError("stored train scaler state changed on reuse")
        if method == "cs_rank":
            cross_section = transformed.loc[(20240101, slice(None)), "a"]
            if not cross_section.is_monotonic_increasing:
                raise AssertionError("cs_rank ordering does not match raw values")
    print("synthetic preprocessing check: four transforms, ready-row mask, and scaler-state reuse passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=Path("data/features/legacy_unversioned_reference"))
    parser.add_argument("--version-dir", type=Path, default=Path("data/features/factor_v0.2"))
    parser.add_argument("--prepared-dir", type=Path, default=Path("data/prepared"))
    parser.add_argument("--spec", type=Path, default=ROOT / "specs/factor_set_v0.2.json")
    args = parser.parse_args()
    spec = load_spec(args.spec)
    validate_real_matrix(args.reference_dir, args.version_dir, args.prepared_dir, spec)
    validate_temporal_invariants(spec)
    validate_diagnostics(spec)
    validate_preprocessing()
    print(f"PASS: {spec['version']} formula/output contract and old-matrix parity")


if __name__ == "__main__":
    main()
