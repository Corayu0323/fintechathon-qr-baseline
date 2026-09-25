#!/usr/bin/env python3
"""Run the fixed one-day protocol on D, H, or S.

D compares development candidates. H evaluates the frozen selected recipe on
2024. S fits the registered recipe on 2022-2024 and writes 2025-2026
submission predictions without scoring against test labels.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import hashlib
import resource
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import joblib

from scripts.evaluation.official_score import score_frame
from scripts.modeling.preprocessing import DEFAULT_TRANSFORMS, transform_inputs
from scripts.experiments.tracking import (
    EXPERIMENT_NAME,
    initialize_tracking,
    metric_values,
    new_run_name,
    parameter_values,
)
from scripts.modeling.qlib_bridge import complete_predictions, load_window
from scripts.modeling.config import load_model_config, load_protocol_config
from scripts.modeling.models import lightgbm, ridge
from scripts.modeling.models.signals import predict_signal


MODEL_CONFIG = load_model_config()
MODEL_RECIPES = MODEL_CONFIG["models"]
SIGNAL_MODELS = [name for name, recipe in MODEL_RECIPES.items() if recipe["implementation"] == "signal"]
DEFAULT_MODELS = SIGNAL_MODELS + ["ridge", "lightgbm"]


@dataclass(frozen=True)
class Stage:
    phase: str
    fit_split: str
    fit_start: int
    fit_end: int
    valid_start: int
    predict_split: str
    predict_start: int
    predict_end: int


def label_mask(window, asof_date: int, before_date: int | None = None) -> pd.Series:
    maturity = pd.to_numeric(window.auxiliary["label_maturity_date"], errors="coerce").fillna(0)
    mask = (
        window.auxiliary["model_ready"]
        & window.auxiliary["label_present"]
        & window.labels["y_ret_1d"].notna()
        & maturity.gt(0)
        & maturity.le(asof_date)
    )
    dates = window.features.index.get_level_values("trade_date")
    mask &= dates <= asof_date
    if before_date is not None:
        mask &= dates < before_date
    return mask


def fit_predict(
    fit_window,
    predict_window,
    stage: Stage,
    transform: str,
    model_name: str,
    *,
    threads: int,
) -> tuple[pd.DataFrame, dict, object]:
    if model_name in SIGNAL_MODELS:
        recipe = MODEL_RECIPES[model_name]
        sign = recipe["sign"]
        ready = predict_window.auxiliary["model_ready"]
        raw_pred = predict_signal(predict_window, recipe)
        raw_pred.name = "pred_model"
        completed = complete_predictions(predict_window, raw_pred)
        info = {
            "model": model_name,
            "input_transform": "identity",
            "train_rows": 0,
            "internal_train_rows": 0,
            "internal_valid_rows": 0,
            "selected_lightgbm_rounds": None,
            "transform_state": {"method": recipe["input_transform"], "source_factor": recipe["source_factor"]},
            "early_stopping_transform_state": None,
        }
        return completed, info, {"source_factor": recipe["source_factor"], "sign": sign}

    dates = fit_window.features.index.get_level_values("trade_date")
    eligible_all = label_mask(fit_window, stage.fit_end)
    prior_dates = dates[dates < stage.valid_start]
    if len(prior_dates) == 0:
        raise ValueError("fit window has no dates before internal validation")
    # Training labels must already have matured at the last common trading date
    # before validation begins. Validation labels are available only as of the
    # actual fit cutoff and are used solely to choose boosting rounds.
    early_cutoff = int(prior_dates.max())
    early_fit = label_mask(fit_window, early_cutoff, before_date=stage.valid_start)
    valid_fit = eligible_all & (dates >= stage.valid_start)
    if not early_fit.any() or not valid_fit.any():
        raise ValueError("internal train/valid split has no eligible labels")

    if model_name == "lightgbm":
        recipe = MODEL_RECIPES[model_name]
        early_x, early_state = transform_inputs(
            fit_window.features, fit_window.auxiliary["model_ready"], early_fit, transform
        )
        y = fit_window.labels["y_ret_1d"]
        rounds = lightgbm.select_rounds(
            early_x.loc[early_fit], y.loc[early_fit],
            early_x.loc[valid_fit], y.loc[valid_fit], recipe, threads=threads,
        )
    else:
        rounds = None
        early_state = None

    final_x, transform_state = transform_inputs(
        fit_window.features, fit_window.auxiliary["model_ready"], eligible_all, transform
    )
    y_train = fit_window.labels.loc[eligible_all, "y_ret_1d"].astype(np.float64)
    x_train = final_x.loc[eligible_all].astype(np.float32)
    if model_name == "ridge":
        recipe = MODEL_RECIPES[model_name]
        model = ridge.fit(fit_window, final_x, recipe, stage.fit_start, stage.fit_end)
        rounds = None
    elif model_name == "lightgbm":
        recipe = MODEL_RECIPES[model_name]
        model = lightgbm.fit(x_train, y_train, recipe, threads=threads, rounds=rounds)
    else:
        raise ValueError(f"unknown model: {model_name}")

    predict_x, _ = transform_inputs(
        predict_window.features, predict_window.auxiliary["model_ready"],
        pd.Series(False, index=predict_window.features.index), transform,
        fitted_state=transform_state if transform == "train_zscore" else None,
    )
    ready = predict_window.auxiliary["model_ready"]
    if model_name == "ridge":
        raw_pred = ridge.predict(
            model, predict_window, predict_x, stage.predict_start, stage.predict_end
        )
    else:
        raw_pred = pd.Series(np.nan, index=predict_window.features.index, name="pred_model")
        raw_pred.loc[ready] = model.predict(predict_x.loc[ready].astype(np.float32))
    completed = complete_predictions(predict_window, raw_pred)
    info = {
        "model": model_name,
        "input_transform": transform,
        "train_rows": int(eligible_all.sum()),
        "internal_train_rows": int(early_fit.sum()),
        "internal_valid_rows": int(valid_fit.sum()),
        "selected_lightgbm_rounds": rounds,
        "transform_state": transform_state,
        "early_stopping_transform_state": early_state,
    }
    return completed, info, model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("D", "H", "S"), default="D")
    parser.add_argument("--prepared-dir", type=Path, default=Path("data/prepared"))
    parser.add_argument("--factor-dir", type=Path, default=Path("data/features/factor_v0.2"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/fixed_baseline"))
    parser.add_argument("--models", nargs="+", choices=DEFAULT_MODELS)
    parser.add_argument("--transforms", nargs="+", choices=DEFAULT_TRANSFORMS)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--model-config", type=Path, default=Path("configs/model_recipes.json"))
    parser.add_argument("--protocol-config", type=Path, default=Path("configs/experiment_protocol.json"))
    parser.add_argument(
        "--tracking-dir", type=Path, default=Path("outputs/experiment_tracking"),
        help="local SQLite/MLflow tracking store (kept out of Git)",
    )
    args = parser.parse_args()
    global MODEL_RECIPES, SIGNAL_MODELS
    model_config = load_model_config(args.model_config)
    MODEL_RECIPES = model_config["models"]
    protocol_config = load_protocol_config(args.protocol_config)
    min_valid_run = protocol_config["min_valid_run"]
    SIGNAL_MODELS = [name for name, recipe in MODEL_RECIPES.items() if recipe["implementation"] == "signal"]
    stages = {
        phase: Stage(
            phase=phase,
            fit_split=values["fit_split"], fit_start=values["fit_start"], fit_end=values["fit_end"],
            valid_start=values["valid_start"], predict_split=values["predict_split"],
            predict_start=values["predict_start"], predict_end=values["predict_end"],
        )
        for phase, values in protocol_config["stages"].items()
    }
    locked_model = protocol_config["stages"]["H"]["models"][0]
    locked_transform = protocol_config["stages"]["H"]["transforms"][0]
    stage = stages[args.phase]
    stage_config = protocol_config["stages"][args.phase]
    args.models = args.models or stage_config["models"]
    args.transforms = args.transforms or stage_config["transforms"]
    if args.phase != "D" and (args.models != [locked_model] or args.transforms != [locked_transform]):
        parser.error(f"phase {args.phase} is frozen to {locked_model} + {locked_transform}")
    if args.threads < 1:
        parser.error("--threads must be positive")

    model_config_hash = hashlib.sha256(args.model_config.read_bytes()).hexdigest()
    protocol_config_hash = hashlib.sha256(args.protocol_config.read_bytes()).hexdigest()

    phase_output = args.output_dir if args.phase == "D" else args.output_dir / args.phase
    phase_output.mkdir(parents=True, exist_ok=True)
    initialize_tracking(args.tracking_dir)
    started = time.monotonic()
    fit_window = load_window(
        args.prepared_dir, args.factor_dir, stage.fit_split, stage.fit_start, stage.fit_end,
        min_valid_run=min_valid_run, label_asof_date=stage.fit_end,
    )
    predict_window = load_window(
        args.prepared_dir, args.factor_dir, stage.predict_split,
        stage.predict_start, stage.predict_end, min_valid_run=min_valid_run,
        label_asof_date=stage.predict_end if stage.predict_split == "train" else None,
    )
    if stage.predict_split == "test" and predict_window.labels["y_ret_1d"].notna().any():
        raise ValueError("S phase must not contain accessible test labels")

    runs = []
    for model_name in args.models:
        transforms = ["identity"] if model_name in SIGNAL_MODELS else args.transforms
        for transform in transforms:
            run_uuid = uuid.uuid4().hex
            run_id = f"{model_name}__{transform}"
            run_dir = phase_output / "runs" / f"{run_uuid}__{run_id}"
            run_dir.mkdir(parents=True, exist_ok=False)
            config_snapshot_dir = run_dir / "config"
            config_snapshot_dir.mkdir()
            model_config_snapshot = config_snapshot_dir / "model_recipes.json"
            protocol_config_snapshot = config_snapshot_dir / "experiment_protocol.json"
            shutil.copy2(args.model_config, model_config_snapshot)
            shutil.copy2(args.protocol_config, protocol_config_snapshot)
            output = run_dir / "predictions.parquet"
            model_suffix = "json" if model_name in SIGNAL_MODELS else "joblib" if model_name == "ridge" else "txt"
            model_path = run_dir / f"model.{model_suffix}"
            run_name = new_run_name(args.phase, model_name, transform)
            print(f"running run={run_name}", flush=True)
            run_started = time.monotonic()
            from qlib.workflow import R

            with R.start(experiment_name=EXPERIMENT_NAME, recorder_name=run_name):
                R.log_params(**parameter_values({
                    "phase": args.phase,
                    "model": model_name,
                    "input_transform": transform,
                    "fit_start": stage.fit_start,
                    "fit_end": stage.fit_end,
                    "valid_start": stage.valid_start,
                    "predict_start": stage.predict_start,
                    "predict_end": stage.predict_end,
                    "threads": args.threads,
                    "factor_catalog_version": json.loads((args.factor_dir / "catalog.json").read_text(encoding="utf-8"))["version"],
                    "model_recipe": {
                        **MODEL_RECIPES[model_name],
                        **({"n_jobs": args.threads} if MODEL_RECIPES[model_name]["implementation"] == "lightgbm" else {}),
                    },
                    "model_config_sha256": model_config_hash,
                    "protocol_config_sha256": protocol_config_hash,
                }))
                R.set_tags(protocol="fixed_v1", run_uuid=run_uuid, result_status="running")
                predictions, info, model = fit_predict(
                    fit_window, predict_window, stage, transform, model_name, threads=args.threads
                )
                scores = {}
                if args.phase != "S":
                    raw_labels = predict_window.labels.reset_index()
                    raw_auxiliary = predict_window.auxiliary.reset_index()
                    scores = score_frame(predictions, raw_labels, raw_auxiliary)
                if model_name in SIGNAL_MODELS:
                    model_path.write_text(
                        json.dumps(model, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                    )
                elif model_name == "ridge":
                    joblib.dump(model, model_path, compress=3)
                else:
                    model.booster_.save_model(str(model_path))
                pq.write_table(
                    pa.Table.from_pandas(predictions, preserve_index=False),
                    output, compression="zstd", row_group_size=100_000,
                )
                submission_path = None
                if args.phase == "S":
                    submission_path = phase_output / "submission.csv"
                    predictions[["ts_code", "trade_date", "pred"]].to_csv(submission_path, index=False)
                result = {
                    **info, **scores,
                    "phase": args.phase,
                    "run_id": run_id,
                    "run_uuid": run_uuid,
                    "recorder_name": run_name,
                    "prediction_rows": len(predictions),
                    "fallback_rows": int((~predictions["model_ready"]).sum()),
                    "elapsed_seconds": round(time.monotonic() - run_started, 2),
                    "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1),
                    "prediction_path": str(output),
                    "model_path": str(model_path),
                }
                if submission_path is not None:
                    result["submission_path"] = str(submission_path)
                R.log_params(**parameter_values({
                    "selected_lightgbm_rounds": result.get("selected_lightgbm_rounds"),
                    "train_rows": result.get("train_rows"),
                    "internal_train_rows": result.get("internal_train_rows"),
                    "internal_valid_rows": result.get("internal_valid_rows"),
                    "prediction_path": output.resolve(),
                    "model_path": model_path.resolve(),
                    "model_config_sha256": model_config_hash,
                    "protocol_config_sha256": protocol_config_hash,
                }))
                R.log_metrics(**metric_values(result))
                R.set_tags(result_status="finished", selected_candidate=(
                    "true" if args.phase == "D" and model_name == locked_model and transform == locked_transform else "false"
                ))
                result["recorder_id"] = R.get_recorder().id
                manifest_path = run_dir / "run_manifest.json"
                manifest_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                R.log_artifact(str(model_path), artifact_path="model")
                R.log_artifact(str(manifest_path), artifact_path="metadata")
                R.log_artifact(str(model_config_snapshot), artifact_path="config")
                R.log_artifact(str(protocol_config_snapshot), artifact_path="config")
            runs.append(result)
            print(json.dumps({k: v for k, v in result.items() if k not in {"transform_state", "early_stopping_transform_state"}}, ensure_ascii=False), flush=True)

    summary_path = phase_output / "summary.json"
    previous_runs = []
    if summary_path.exists():
        previous_runs = json.loads(summary_path.read_text(encoding="utf-8")).get("runs", [])
    runs = previous_runs + runs
    selected = None
    if args.phase == "D":
        runs = sorted(runs, key=lambda row: row["final_score"], reverse=True)
        best_score = runs[0]["final_score"]
        tied = [row for row in runs if abs(row["final_score"] - best_score) <= 1e-12]
        selected = min(
            tied,
            key=lambda row: (row["elapsed_seconds"], row["peak_rss_mib"], row["run_id"]),
        )
    feature_catalog = json.loads((args.factor_dir / "catalog.json").read_text(encoding="utf-8"))
    prepared_manifest = json.loads((args.prepared_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = {
        "protocol": f"{args.phase} fixed evaluation",
        "phase": args.phase,
        "fit_period": [stage.fit_start, stage.fit_end],
        "internal_valid_start": stage.valid_start,
        "prediction_period": [stage.predict_start, stage.predict_end],
        "label": "raw y_ret_1d; predictions are one-day returns",
        "selection_rule": "D: official composite score; H: frozen recipe evaluation; S: no test-label scoring",
        "models": sorted({row["model"] for row in runs}),
        "transforms": sorted({row["input_transform"] for row in runs}),
        "factor_catalog_version": feature_catalog["version"],
        "prepared_schema_version": prepared_manifest["schema_version"],
        "model_config_path": str(args.model_config),
        "model_config_sha256": model_config_hash,
        "protocol_config_path": str(args.protocol_config),
        "protocol_config_sha256": protocol_config_hash,
        "model_recipes": {
            name: MODEL_RECIPES[name]
            for name in sorted({row["model"] for row in runs})
            if name in MODEL_RECIPES
        },
        "elapsed_seconds_this_invocation": round(time.monotonic() - started, 2),
        "runs": runs,
        "selected_run_id": selected["run_id"] if selected else None,
        "selected_run_uuid": selected.get("run_uuid") if selected else None,
        "selected_recorder_id": selected.get("recorder_id") if selected else None,
        "registered_candidate": (
            {"model": selected["model"], "input_transform": selected["input_transform"]}
            if selected else {"model": locked_model, "input_transform": locked_transform}
        ),
        "interpretation": {
            "D": "Development interval; comparisons on D are not an untouched final test.",
            "H": "Frozen recipe; H results must not be used for post-hoc tuning and still called independent validation.",
            "S": "Submission predictions only; test labels are not used or scored.",
        }[args.phase],
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"phase={args.phase} selected={summary['selected_run_id']} summary={summary_path}")


if __name__ == "__main__":
    main()
