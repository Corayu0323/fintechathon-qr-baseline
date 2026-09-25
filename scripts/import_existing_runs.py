#!/usr/bin/env python3
"""Import existing D/H/S summaries into the local Qlib Recorder once."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from experiment_tracking import (  # noqa: E402
    EXPERIMENT_NAME,
    initialize_tracking,
    metric_values,
    parameter_values,
    safe_artifact_name,
)


def main() -> None:
    tracking_dir = ROOT / "outputs" / "experiment_tracking"
    uri = initialize_tracking(tracking_dir)
    from mlflow.tracking import MlflowClient
    from qlib.workflow import R

    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    existing = client.search_runs(
        [experiment.experiment_id], max_results=1000, order_by=["attributes.start_time ASC"]
    )
    imported_by_key = {
        row.data.tags.get("legacy_import_key"): row for row in existing
        if row.data.tags.get("legacy_import_key")
    }
    manifest_dir = tracking_dir / "import_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    created = 0
    skipped = 0
    for phase, summary_path in (
        ("D", ROOT / "outputs/fixed_baseline/summary.json"),
        ("H", ROOT / "outputs/fixed_baseline/H/summary.json"),
        ("S", ROOT / "outputs/fixed_baseline/S/summary.json"),
    ):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for result in summary["runs"]:
            key = f"{phase}/{result['run_id']}"
            if key in imported_by_key:
                old_run = imported_by_key[key]
                recipe = summary.get("model_recipes", {}).get(result["model"], {})
                if recipe and "model_recipe" not in old_run.data.params:
                    client.log_param(
                        old_run.info.run_id,
                        "model_recipe",
                        parameter_values({"model_recipe": recipe})["model_recipe"],
                    )
                skipped += 1
                continue
            recorder_name = f"legacy__{phase}__{safe_artifact_name(result['run_id'])}"
            with R.start(experiment_name=EXPERIMENT_NAME, recorder_name=recorder_name):
                R.log_params(**parameter_values({
                    "phase": phase,
                    "model": result["model"],
                    "input_transform": result["input_transform"],
                    "model_recipe": summary.get("model_recipes", {}).get(result["model"], {}),
                    "fit_period": summary["fit_period"],
                    "prediction_period": summary["prediction_period"],
                    "prediction_path": result.get("prediction_path", ""),
                    "model_path": result.get("model_path", ""),
                    "source": "existing fixed-baseline output before experiment tracking",
                }))
                R.set_tags(
                    protocol="fixed_v1",
                    phase=phase,
                    model=result["model"],
                    input_transform=result["input_transform"],
                    legacy_import_key=key,
                    result_status="legacy_import",
                )
                R.log_metrics(**metric_values(result))
                model_path = Path(result.get("model_path", ""))
                if model_path.is_file():
                    R.log_artifact(str(model_path), artifact_path="model")
                manifest_path = manifest_dir / f"{phase}__{safe_artifact_name(result['run_id'])}.json"
                manifest_path.write_text(json.dumps({
                    "protocol": summary["protocol"],
                    "fit_period": summary["fit_period"],
                    "prediction_period": summary["prediction_period"],
                    "factor_catalog_version": summary.get("factor_catalog_version"),
                    "run_result": result,
                    "legacy_import": True,
                }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                R.log_artifact(str(manifest_path), artifact_path="metadata")
            created += 1
    print(f"Imported {created} historical runs; skipped {skipped} already registered runs.")


if __name__ == "__main__":
    main()
