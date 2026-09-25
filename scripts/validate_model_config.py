#!/usr/bin/env python3
"""Validate frozen model/protocol defaults and exercise model adapters cheaply."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from models.config import load_model_config, load_protocol_config  # noqa: E402
from models.lightgbm_model import _estimator_params, fit as fit_lightgbm  # noqa: E402
from models.signals import predict_signal  # noqa: E402


def main() -> None:
    models = load_model_config()["models"]
    protocol = load_protocol_config()
    assert models["ridge"]["alpha"] == 1.0 and models["ridge"]["fit_intercept"] is True
    lgb = models["lightgbm"]
    expected = {
        "implementation": "lightgbm",
        "objective": "regression", "max_estimators": 500,
        "early_stopping_rounds": 50, "early_stopping_metric": "l2",
        "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 200,
        "reg_lambda": 1.0, "colsample_bytree": 0.9, "subsample": 1.0,
        "random_state": 42, "verbosity": -1,
    }
    assert lgb == expected, f"LightGBM baseline recipe changed unexpectedly: {lgb}"
    stages = protocol["stages"]
    assert stages["D"]["fit_start"] == 20180101 and stages["D"]["predict_end"] == 20231231
    assert stages["H"]["predict_start"] == 20240101 and stages["H"]["predict_end"] == 20241231
    assert stages["S"]["predict_split"] == "test" and stages["S"]["predict_end"] == 20261231
    assert stages["H"]["models"] == stages["S"]["models"] == ["lightgbm"]
    assert stages["H"]["transforms"] == stages["S"]["transforms"] == ["identity"]

    estimator_params = _estimator_params(lgb, threads=2)
    assert estimator_params["n_jobs"] == 2 and estimator_params["num_leaves"] == 31
    x = pd.DataFrame({"x": np.linspace(-1, 1, 120), "z": np.cos(np.arange(120))})
    y = pd.Series(x["x"].to_numpy() * 0.4 + x["z"].to_numpy() * 0.1)
    fitted = fit_lightgbm(x, y, lgb, threads=2, rounds=3)
    assert np.isfinite(fitted.predict(x)).all()

    index = pd.MultiIndex.from_tuples(
        [(20240102, "000001.SZ"), (20240102, "000002.SZ")],
        names=["trade_date", "ts_code"],
    )
    window = type("WindowFixture", (), {
        "features": pd.DataFrame({"feature_ret_1d": [0.1, -0.2]}, index=index),
        "auxiliary": pd.DataFrame({"model_ready": [True, False]}, index=index),
    })()
    signal = predict_signal(window, models["ret1_momentum"])
    assert signal.iloc[0] == 0.1 and pd.isna(signal.iloc[1])
    print("PASS: model recipes, D/H/S protocol defaults, LightGBM smoke fit, and signal adapter")


if __name__ == "__main__":
    main()
