"""Fixed transparent score baselines built directly from configured factors."""

import pandas as pd


def predict_signal(window, recipe: dict) -> pd.Series:
    if recipe["source_factor"] not in window.features:
        raise KeyError(f"signal source factor is absent: {recipe['source_factor']}")
    prediction = window.features[recipe["source_factor"]] * float(recipe["sign"])
    return prediction.where(window.auxiliary["model_ready"]).rename("pred_model")
