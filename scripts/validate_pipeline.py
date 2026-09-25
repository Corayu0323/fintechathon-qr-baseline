#!/usr/bin/env python3
"""Validate fixed D/H/S artifacts and parity with the supplied evaluator."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from official_score import score_frame
from evaluate import evaluate
from qlib_bridge import load_window


def validate_score_parity() -> None:
    rng = np.random.default_rng(20260925)
    dates = [20240102, 20240103, 20240104, 20240105]
    rows = []
    for date in dates:
        for i in range(150):
            rows.append({
                "trade_date": date,
                "ts_code": f"{i:06d}.SZ",
                "pred": float(rng.normal()),
                "y_ret_1d": float(rng.normal(scale=0.03)),
                "flag_limit_up": int(i % 37 == 0),
            })
    panel = pd.DataFrame(rows)
    panel.loc[panel.index[::113], "y_ret_1d"] = np.nan
    predictions = panel[["trade_date", "ts_code", "pred"]]
    labels = panel[["trade_date", "ts_code", "y_ret_1d"]]
    auxiliary = panel[["trade_date", "ts_code", "flag_limit_up"]]
    local = score_frame(predictions, labels, auxiliary)

    with tempfile.TemporaryDirectory(prefix="qr_score_parity_") as tmp:
        temp = Path(tmp)
        predictions[["ts_code", "trade_date", "pred"]].to_csv(temp / "submission.csv", index=False)
        labels.to_csv(temp / "测试集_Y.csv", index=False)
        auxiliary.to_csv(temp / "测试集_X.csv", index=False)
        official = evaluate(str(temp / "submission.csv"), str(temp))
    for metric in ("ic_mean", "ic_std", "icir", "ic_positive_ratio", "annual_excess",
                   "top1_annual_ret", "mean_turnover", "final_score"):
        if not np.isclose(local[metric], official[metric], rtol=0, atol=1e-12):
            raise AssertionError(f"scorer mismatch for {metric}: {local[metric]} != {official[metric]}")


def validate_d(root: Path) -> None:
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("phase") != "D" or summary.get("selected_run_id") != "lightgbm__identity":
        raise AssertionError("D summary does not identify the registered selected candidate")
    run_ids = {run["run_id"] for run in summary["runs"]}
    expected = {
        f"{model}__{transform}"
        for model in ("ridge", "lightgbm")
        for transform in ("train_zscore", "cs_zscore", "cs_rank", "identity")
    } | {"ret1_momentum__identity", "ret1_reversal__identity"}
    if run_ids != expected:
        raise AssertionError(f"D candidate set mismatch: {run_ids ^ expected}")
    if any(run["prediction_rows"] != 3_380_550 for run in summary["runs"]):
        raise AssertionError("D runs do not all preserve the complete prediction key set")


def validate_h(root: Path) -> None:
    summary = json.loads((root / "H" / "summary.json").read_text(encoding="utf-8"))
    if summary.get("phase") != "H" or summary.get("fit_period") != [20210101, 20231231]:
        raise AssertionError("H stage boundaries are incorrect")
    if summary.get("prediction_period") != [20240101, 20241231]:
        raise AssertionError("H prediction period is incorrect")
    if summary.get("selected_run_id") is not None:
        raise AssertionError("H must evaluate the frozen candidate, not select among runs")
    run = summary["runs"][0]
    if run["run_id"] != "lightgbm__identity" or not np.isfinite(run["final_score"]):
        raise AssertionError("H frozen candidate or score is missing")
    if run["prediction_rows"] != 1_125_300:
        raise AssertionError("H prediction key count is incorrect")


def validate_s(root: Path) -> None:
    phase_dir = root / "S"
    summary = json.loads((phase_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("phase") != "S" or summary.get("fit_period") != [20220101, 20241231]:
        raise AssertionError("S fit period is incorrect")
    if summary.get("prediction_period") != [20250101, 20261231]:
        raise AssertionError("S prediction period is incorrect")
    run = summary["runs"][0]
    if "final_score" in run or any(k in run for k in ("ic_mean", "annual_excess", "mean_turnover")):
        raise AssertionError("S must not read or score against test labels")
    submission_path = phase_dir / "submission.csv"
    submission = pd.read_csv(submission_path)
    if list(submission.columns) != ["ts_code", "trade_date", "pred"]:
        raise AssertionError("submission columns/order differ from the official interface")
    if submission.duplicated(["trade_date", "ts_code"]).any():
        raise AssertionError("S submission contains duplicate keys")
    if len(submission) != 1_599_600 or not np.isfinite(submission["pred"]).all():
        raise AssertionError("S submission row count or finite-score check failed")
    expected_parts = [
        pq.read_table(ROOT / "data" / "prepared" / f"test_{year}.parquet", columns=["trade_date", "ts_code"]).to_pandas()
        for year in (2025, 2026)
    ]
    expected_keys = pd.concat(expected_parts, ignore_index=True)
    expected_index = pd.MultiIndex.from_frame(expected_keys[["trade_date", "ts_code"]])
    submission_keys = submission[["trade_date", "ts_code"]].copy()
    submission_keys["trade_date"] = submission_keys["trade_date"].astype(expected_keys["trade_date"].dtype)
    submission_keys["ts_code"] = submission_keys["ts_code"].astype(expected_keys["ts_code"].dtype)
    actual_index = pd.MultiIndex.from_frame(submission_keys)
    if len(actual_index.difference(expected_index)) or len(expected_index.difference(actual_index)):
        raise AssertionError("S submission keys are not exactly the provided test keys")
    details = pq.read_table(run["prediction_path"]).to_pandas()
    if len(details) != len(submission) or details.duplicated(["trade_date", "ts_code"]).any():
        raise AssertionError("S prediction artifact and submission keys differ")
    if not np.isfinite(details["pred"]).all() or int((~details["model_ready"]).sum()) != run["fallback_rows"]:
        raise AssertionError("S prediction finite-score/fallback check failed")


def validate_test_label_exclusion() -> None:
    with tempfile.TemporaryDirectory(prefix="qr_test_label_exclusion_") as tmp:
        base = Path(tmp)
        prepared = base / "prepared"
        factors = base / "features"
        prepared.mkdir()
        factors.mkdir()
        source = pd.DataFrame({
            "trade_date": [20250102, 20250102], "ts_code": ["000001.SZ", "000002.SZ"],
            "y_ret_1d": [0.5, -0.5], "flag_limit_up": [0, 0], "flag_limit_down": [0, 0],
            "bar_valid": [1, 1], "volume_zero": [0, 0], "label_present": [0, 0],
            "label_maturity_date": [0, 0], "valid_bar_run": [5, 5], "source_row": [0, 1],
        })
        feature = source[["trade_date", "ts_code"]].copy()
        feature["feature_test"] = [0.1, 0.2]
        pq.write_table(pa.Table.from_pandas(source, preserve_index=False), prepared / "test_2025.parquet")
        pq.write_table(pa.Table.from_pandas(feature, preserve_index=False), factors / "features_2025.parquet")
        window = load_window(prepared, factors, "test", 20250102, 20250102, min_valid_run=1)
        if window.labels["y_ret_1d"].notna().any():
            raise AssertionError("test label values leaked through load_window")


def main() -> None:
    root = ROOT / "outputs" / "fixed_baseline"
    validate_score_parity()
    validate_d(root)
    validate_h(root)
    validate_s(root)
    validate_test_label_exclusion()
    print("PASS: official-score parity; D candidate coverage; frozen H protocol; S keys/format; test labels excluded")


if __name__ == "__main__":
    main()
