"""LightGBM construction, internal validation, and final fitting."""

from __future__ import annotations

import lightgbm as lgb
import pandas as pd


def _estimator_params(recipe: dict, threads: int) -> dict:
    return {
        "objective": recipe["objective"],
        "learning_rate": recipe["learning_rate"],
        "num_leaves": recipe["num_leaves"],
        "min_child_samples": recipe["min_child_samples"],
        "reg_lambda": recipe["reg_lambda"],
        "colsample_bytree": recipe["colsample_bytree"],
        "subsample": recipe["subsample"],
        "n_jobs": threads,
        "random_state": recipe["random_state"],
        "verbosity": recipe["verbosity"],
    }


def select_rounds(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    recipe: dict,
    *,
    threads: int,
) -> int:
    estimator = lgb.LGBMRegressor(
        **_estimator_params(recipe, threads), n_estimators=recipe["max_estimators"]
    )
    estimator.fit(
        x_train,
        y_train,
        eval_X=x_valid,
        eval_y=y_valid,
        eval_metric=recipe["early_stopping_metric"],
        callbacks=[
            lgb.early_stopping(recipe["early_stopping_rounds"], verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    return max(1, int(estimator.best_iteration_ or estimator.n_estimators))


def fit(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    recipe: dict,
    *,
    threads: int,
    rounds: int,
) -> lgb.LGBMRegressor:
    estimator = lgb.LGBMRegressor(
        **_estimator_params(recipe, threads), n_estimators=rounds
    )
    estimator.fit(x_train, y_train)
    return estimator
