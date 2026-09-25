"""Apply the three supplied contest metrics to a historical prediction panel.

This mirrors ``evaluate.py`` on an already aligned frame.  It does not claim
that the Jaccard term is real trading turnover or that the scores are PnL.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


KEY = ["trade_date", "ts_code"]


def score_frame(predictions: pd.DataFrame, labels: pd.DataFrame,
                auxiliary: pd.DataFrame) -> dict[str, float]:
    for name, frame, required in [
        ("predictions", predictions, KEY + ["pred"]),
        ("labels", labels, KEY + ["y_ret_1d"]),
        ("auxiliary", auxiliary, KEY + ["flag_limit_up"]),
    ]:
        if not set(required).issubset(frame.columns):
            raise ValueError(f"{name} lacks required columns")
        if frame.duplicated(KEY).any() or frame[KEY].isna().any().any():
            raise ValueError(f"{name} has duplicate or missing keys")
    if not np.isfinite(predictions["pred"]).all():
        raise ValueError("predictions contain nonfinite scores")

    data = predictions[KEY + ["pred"]].merge(
        labels[KEY + ["y_ret_1d"]], on=KEY, how="inner", validate="one_to_one"
    ).merge(
        auxiliary[KEY + ["flag_limit_up"]], on=KEY, how="inner",
        validate="one_to_one",
    )
    if len(data) != len(predictions) or len(data) != len(labels) or len(data) != len(auxiliary):
        raise ValueError("scoring panels must have exactly the same keys")

    ic_list = []
    excess_list = []
    top_return_list = []
    turnover_list = []
    previous_top = None
    for _, group in data.groupby("trade_date", sort=True):
        rank_valid = group.dropna(subset=["y_ret_1d"])
        if len(rank_valid) >= 30:
            ic, _ = spearmanr(rank_valid["pred"], rank_valid["y_ret_1d"])
            ic_list.append(ic)

        top_valid = group[(group["flag_limit_up"] == 0) & group["y_ret_1d"].notna()]
        if len(top_valid) >= 100:
            top_valid = top_valid.sort_values("pred", ascending=False)
            n_top = max(len(top_valid) // 10, 1)
            top_return = top_valid["y_ret_1d"].iloc[:n_top].mean()
            top_return_list.append(top_return)
            excess_list.append(top_return - top_valid["y_ret_1d"].mean())

        turnover_valid = group[group["flag_limit_up"] == 0]
        if len(turnover_valid) < 100:
            previous_top = None
            continue
        turnover_valid = turnover_valid.sort_values("pred", ascending=False)
        n_top = max(len(turnover_valid) // 10, 1)
        current_top = set(turnover_valid["ts_code"].iloc[:n_top])
        if previous_top:
            turnover_list.append(1 - len(current_top & previous_top) / len(current_top | previous_top))
        previous_top = current_top

    ic_mean = float(np.mean(ic_list))
    ic_std = float(np.std(ic_list, ddof=1))
    mean_turnover = float(np.mean(turnover_list))
    annual_excess = float(np.mean(excess_list) * 252)
    return {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "icir": ic_mean / ic_std if ic_std > 0 else 0.0,
        "ic_positive_ratio": float(np.mean(np.array(ic_list) > 0)),
        "annual_excess": annual_excess,
        "top1_annual_ret": float(np.mean(top_return_list) * 252),
        "mean_turnover": mean_turnover,
        "final_score": ic_mean * 0.4 + annual_excess * 0.3 + (1 - mean_turnover) * 0.3,
    }
