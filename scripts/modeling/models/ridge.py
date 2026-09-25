"""Qlib Ridge adapter preserving the project's panel index and date segments."""

from __future__ import annotations

import pandas as pd
from dataclasses import replace
from qlib.contrib.model.linear import LinearModel
from qlib.data.dataset import DatasetH

from scripts.modeling.qlib_bridge import to_qlib_handler


def _qlib_date(date: int) -> str:
    return pd.to_datetime(str(date), format="%Y%m%d").strftime("%Y-%m-%d")


def fit(fit_window, transformed_features: pd.DataFrame, recipe: dict, fit_start: int, fit_end: int):
    window = replace(fit_window, features=transformed_features)
    dataset = DatasetH(
        handler=to_qlib_handler(window),
        segments={"train": (_qlib_date(fit_start), _qlib_date(fit_end))},
    )
    model = LinearModel(
        estimator=recipe["estimator"],
        alpha=recipe["alpha"],
        fit_intercept=recipe["fit_intercept"],
    )
    model.fit(dataset)
    return model


def predict(model, predict_window, transformed_features: pd.DataFrame, predict_start: int, predict_end: int):
    window = replace(predict_window, features=transformed_features)
    dataset = DatasetH(
        handler=to_qlib_handler(window),
        segments={"predict": (_qlib_date(predict_start), _qlib_date(predict_end))},
    )
    result = model.predict(dataset, segment="predict").rename("pred_model")
    result.index = result.index.set_names(["datetime", "instrument"])
    return result
