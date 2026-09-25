"""Load and validate model recipes and fixed evaluation stages."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG_PATH = ROOT / "configs/model_recipes.json"
PROTOCOL_CONFIG_PATH = ROOT / "configs/experiment_protocol.json"
SUPPORTED_TRANSFORMS = {"identity", "train_zscore", "cs_zscore", "cs_rank"}


def load_model_config(path: Path = MODEL_CONFIG_PATH) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "model_recipes_v1":
        raise ValueError("unsupported model recipe schema_version")
    models = config.get("models", {})
    if set(models) != {"ret1_momentum", "ret1_reversal", "ridge", "lightgbm"}:
        raise ValueError("model recipes must define exactly the four registered baseline models")
    for name, recipe in models.items():
        implementation = recipe.get("implementation")
        if implementation not in {"signal", "qlib_ridge", "lightgbm"}:
            raise ValueError(f"{name}: unsupported implementation {implementation!r}")
        if implementation == "signal" and recipe.get("input_transform") != "identity":
            raise ValueError(f"{name}: signal baselines must use identity input")
        if implementation == "lightgbm":
            for key in ("max_estimators", "early_stopping_rounds", "num_leaves", "min_child_samples"):
                if int(recipe.get(key, 0)) <= 0:
                    raise ValueError(f"lightgbm: {key} must be positive")
            if not 0 < float(recipe.get("learning_rate", 0)) <= 1:
                raise ValueError("lightgbm: learning_rate must be in (0, 1]")
    return config


def load_protocol_config(path: Path = PROTOCOL_CONFIG_PATH) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("protocol_version") != "fixed_v1":
        raise ValueError("unsupported experiment protocol version")
    stages = config.get("stages", {})
    if set(stages) != {"D", "H", "S"}:
        raise ValueError("protocol must define D, H, and S")
    for phase, stage in stages.items():
        if stage["fit_start"] > stage["fit_end"] or stage["predict_start"] > stage["predict_end"]:
            raise ValueError(f"{phase}: invalid date range")
        if not stage["models"] or not stage["transforms"]:
            raise ValueError(f"{phase}: candidate models/transforms cannot be empty")
        unknown_transforms = set(stage["transforms"]) - SUPPORTED_TRANSFORMS
        if unknown_transforms:
            raise ValueError(f"{phase}: unsupported transforms {sorted(unknown_transforms)}")
    h, s = stages["H"], stages["S"]
    if h["models"] != s["models"] or h["transforms"] != s["transforms"]:
        raise ValueError("H and S must use the same frozen model/transform recipe")
    return config
