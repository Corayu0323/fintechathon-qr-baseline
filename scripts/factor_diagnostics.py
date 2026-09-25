#!/usr/bin/env python3
"""Run single-factor D-stage IC diagnostics and record the immutable result."""

from __future__ import annotations

import argparse
import json
import math
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from experiment_tracking import (  # noqa: E402
    EXPERIMENT_NAME,
    initialize_tracking,
    parameter_values,
)
from qlib_bridge import load_window  # noqa: E402


MIN_CROSS_SECTION = 30
D_START = 20210101
D_END = 20231231


def _daily_corr(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    if len(x) < MIN_CROSS_SECTION or x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
        return math.nan, math.nan
    pearson = float(np.corrcoef(x.to_numpy(dtype=np.float64), y.to_numpy(dtype=np.float64))[0, 1])
    rank_ic = float(spearmanr(x, y).statistic)
    return pearson, rank_ic


def evaluate_factor_panel(window, catalog: dict, *, label_asof_date: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute per-date factor ICs, summary statistics, and daily pair correlations."""
    feature_names = [item["name"] for item in catalog["features"]]
    dates = window.features.index.get_level_values("trade_date")
    base = pd.DataFrame({
        "trade_date": dates.to_numpy(),
        "ts_code": window.features.index.get_level_values("ts_code").to_numpy(),
        "y_ret_1d": window.labels["y_ret_1d"].to_numpy(dtype=np.float64),
        "label_available": window.auxiliary["label_available"].to_numpy(dtype=bool),
    })
    base["diagnostic_label_valid"] = (
        base["label_available"]
        & base["y_ret_1d"].notna()
        & (base["trade_date"] <= label_asof_date)
    )
    denominator = int(base["diagnostic_label_valid"].sum())

    daily_rows = []
    summary_rows = []
    for feature in feature_names:
        panel = base[["trade_date", "ts_code", "y_ret_1d", "diagnostic_label_valid"]].copy()
        values = window.features[feature].to_numpy(dtype=np.float64)
        panel["factor_value"] = values
        usable = panel["diagnostic_label_valid"] & np.isfinite(panel["factor_value"])
        valid_rows = int(usable.sum())
        for date, group in panel.groupby("trade_date", sort=True):
            group = group.loc[group["diagnostic_label_valid"] & np.isfinite(group["factor_value"])]
            ic, rank_ic = _daily_corr(group["factor_value"], group["y_ret_1d"])
            daily_rows.append({
                "factor": feature,
                "trade_date": int(date),
                "n": len(group),
                "ic": ic,
                "rank_ic": rank_ic,
            })

        daily = pd.DataFrame([row for row in daily_rows if row["factor"] == feature])
        valid_rank_ic = daily["rank_ic"].dropna().to_numpy(dtype=np.float64)
        valid_ic = daily["ic"].dropna().to_numpy(dtype=np.float64)
        rankic_std = float(valid_rank_ic.std(ddof=1)) if len(valid_rank_ic) >= 2 else math.nan
        rankic_mean = float(valid_rank_ic.mean()) if len(valid_rank_ic) else math.nan
        ic_std = float(valid_ic.std(ddof=1)) if len(valid_ic) >= 2 else math.nan
        ic_mean = float(valid_ic.mean()) if len(valid_ic) else math.nan
        summary_rows.append({
            "factor": feature,
            "formula_id": next(item["formula_id"] for item in catalog["features"] if item["name"] == feature),
            "label_asof_date": label_asof_date,
            "eligible_label_rows": denominator,
            "factor_valid_rows": valid_rows,
            "coverage": valid_rows / denominator if denominator else math.nan,
            "rankic_dates": len(valid_rank_ic),
            "rankic_mean": rankic_mean,
            "rankic_std": rankic_std,
            "rankic_ir": rankic_mean / rankic_std if np.isfinite(rankic_std) and rankic_std > 0 else math.nan,
            "rankic_positive_ratio": float((valid_rank_ic > 0).mean()) if len(valid_rank_ic) else math.nan,
            "ic_dates": len(valid_ic),
            "ic_mean": ic_mean,
            "ic_std": ic_std,
        })

    correlation_rows = []
    corr_base = pd.DataFrame({"trade_date": dates.to_numpy()})
    for feature in feature_names:
        corr_base[feature] = window.features[feature].to_numpy(dtype=np.float64)
    for left_idx, left in enumerate(feature_names):
        for right in feature_names[left_idx + 1:]:
            pair_values = []
            for _, group in corr_base[["trade_date", left, right]].groupby("trade_date", sort=True):
                pair = group[[left, right]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(pair) >= MIN_CROSS_SECTION and pair[left].nunique() > 1 and pair[right].nunique() > 1:
                    corr = float(spearmanr(pair[left], pair[right]).statistic)
                    if np.isfinite(corr):
                        pair_values.append(corr)
            correlation_rows.append({
                "factor_a": left,
                "factor_b": right,
                "mean_daily_rank_correlation": float(np.mean(pair_values)) if pair_values else math.nan,
                "valid_dates": len(pair_values),
            })

    summary = pd.DataFrame(summary_rows)
    daily_frame = pd.DataFrame(daily_rows)
    correlation_frame = pd.DataFrame(correlation_rows)
    return summary, daily_frame, correlation_frame


def _json_safe(rows: list[dict]) -> list[dict]:
    clean = []
    for row in rows:
        clean.append({key: (None if isinstance(value, float) and not math.isfinite(value) else value)
                      for key, value in row.items()})
    return clean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=Path("data/prepared"))
    parser.add_argument("--factor-dir", type=Path, default=Path("data/features/factor_v0.2"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/factor_diagnostics"))
    parser.add_argument("--tracking-dir", type=Path, default=Path("outputs/experiment_tracking"))
    parser.add_argument("--start-date", type=int, default=D_START)
    parser.add_argument("--end-date", type=int, default=D_END)
    parser.add_argument("--label-asof-date", type=int, default=D_END)
    args = parser.parse_args()

    if args.start_date < D_START or args.end_date > D_END:
        parser.error(f"factor diagnostics are restricted to D dates {D_START}–{D_END}")
    if args.start_date > args.end_date:
        parser.error("--start-date must be no later than --end-date")
    if args.label_asof_date > D_END:
        parser.error(f"--label-asof-date cannot exceed D label cutoff {D_END}")

    catalog = json.loads((args.factor_dir / "catalog.json").read_text(encoding="utf-8"))
    window = load_window(
        args.prepared_dir, args.factor_dir, "train", args.start_date, args.end_date,
        min_valid_run=1, label_asof_date=args.label_asof_date,
    )
    summary, daily, correlations = evaluate_factor_panel(
        window, catalog, label_asof_date=args.label_asof_date,
    )

    run_uuid = uuid.uuid4().hex
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / catalog["version"] / "D" / f"{run_stamp}_{run_uuid[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    summary_path = run_dir / "factor_summary.csv"
    daily_path = run_dir / "daily_ic.csv"
    correlation_path = run_dir / "factor_correlations.csv"
    summary.to_csv(summary_path, index=False)
    daily.to_csv(daily_path, index=False)
    correlations.to_csv(correlation_path, index=False)

    manifest = {
        "run_uuid": run_uuid,
        "factor_set_version": catalog["version"],
        "phase": "D",
        "date_range": [args.start_date, args.end_date],
        "label_asof_date": args.label_asof_date,
        "minimum_cross_section": MIN_CROSS_SECTION,
        "label_rule": "label_present and label_maturity_date <= label_asof_date; factor finite; labels do not enter factor construction",
        "interpretation": "Factor-level diagnostics only; they screen hypotheses and do not replace the official model score.",
        "summary": _json_safe(summary.to_dict("records")),
        "factor_summary_path": str(summary_path),
        "daily_ic_path": str(daily_path),
        "factor_correlations_path": str(correlation_path),
    }
    manifest_path = run_dir / "diagnostic_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    initialize_tracking(args.tracking_dir)
    from qlib.workflow import R

    recorder_name = f"D__factor_diagnostic__{catalog['version']}__{run_stamp}__{run_uuid[:8]}"
    with R.start(experiment_name=EXPERIMENT_NAME, recorder_name=recorder_name):
        R.log_params(**parameter_values({
            "phase": "D",
            "model": "factor_diagnostic",
            "input_transform": "raw_factor_values",
            "factor_set_version": catalog["version"],
            "start_date": args.start_date,
            "end_date": args.end_date,
            "label_asof_date": args.label_asof_date,
            "minimum_cross_section": MIN_CROSS_SECTION,
            "factor_names": [item["name"] for item in catalog["features"]],
        }))
        R.set_tags(phase="D", model="factor_diagnostic", input_transform="raw_factor_values",
                   protocol="factor_diagnostics_v1", run_uuid=run_uuid)
        metrics = {}
        for row in summary.to_dict("records"):
            factor = row["factor"]
            for metric in ("coverage", "rankic_mean", "rankic_ir", "rankic_positive_ratio", "ic_mean"):
                value = row[metric]
                if pd.notna(value) and np.isfinite(value):
                    metrics[f"{metric}__{factor}"] = float(value)
        if metrics:
            R.log_metrics(**metrics)
        manifest["recorder_id"] = R.get_recorder().id
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for path in (summary_path, daily_path, correlation_path, manifest_path):
            R.log_artifact(str(path), artifact_path="factor_diagnostics")

    print(f"D factor diagnostics written to {run_dir}; Recorder={recorder_name}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
