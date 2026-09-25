"""Local, append-only experiment tracking for the QR project."""

from __future__ import annotations

import json
import math
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mlflow.tracking import MlflowClient


EXPERIMENT_NAME = "FinTechathon-QR"


def initialize_tracking(tracking_dir: Path):
    """Initialize Qlib Recorder with a SQLite registry and local artifacts."""
    tracking_dir = tracking_dir.resolve()
    tracking_dir.mkdir(parents=True, exist_ok=True)
    artifact_root = tracking_dir / "artifacts" / "FinTechathon-QR"
    artifact_root.mkdir(parents=True, exist_ok=True)
    db_path = tracking_dir / "mlflow.db"
    uri = f"sqlite:///{db_path}"
    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        client.create_experiment(EXPERIMENT_NAME, artifact_location=artifact_root.as_uri())

    import qlib

    qlib.init(
        provider_uri=str(Path.cwd()),
        exp_manager={
            "class": "MLflowExpManager",
            "module_path": "qlib.workflow.expm",
            "kwargs": {"uri": uri, "default_exp_name": EXPERIMENT_NAME},
        },
    )
    return uri


def new_run_name(phase: str, model: str, transform: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{phase}__{model}__{transform}__{stamp}__{uuid.uuid4().hex[:8]}"


def metric_values(result: dict[str, Any]) -> dict[str, float]:
    values = {}
    for key, value in result.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if math.isfinite(float(value)):
            values[key] = float(value)
    return values


def parameter_values(values: dict[str, Any]) -> dict[str, str]:
    result = {}
    for key, value in values.items():
        if value is None or isinstance(value, (dict, list, tuple)):
            value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        result[key] = str(value)
    return result


def safe_artifact_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)

