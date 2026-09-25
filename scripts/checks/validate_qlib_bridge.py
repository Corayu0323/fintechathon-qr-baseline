#!/usr/bin/env python3
"""Exercise the Qlib bridge with temporary, deliberately meaningless factors.

This is an interface test, not a research result.  No factor files or model
artifacts are retained after the process exits.
"""

from __future__ import annotations

import json
import resource
import tempfile
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from qlib.contrib.model.linear import LinearModel
from qlib.data.dataset import DatasetH

from scripts.modeling.qlib_bridge import KEY, _read_years, complete_predictions, load_window, to_qlib_handler
from scripts.evaluation.official_score import score_frame


def temporary_factor(prepared: Path, split: str, year: int,
                     start: int, end: int, output: Path) -> None:
    source = _read_years(prepared, split, KEY + ["bar_valid"], start, end)
    # Fixed stock-code transform checks key alignment only; it has no alpha claim.
    source["feature_interface_code"] = (
        source["ts_code"].str[:6].astype("float32") / 1_000_000
    ).where(source["bar_valid"])
    output.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(source[KEY + ["feature_interface_code"]], preserve_index=False),
        output / f"features_{year}.parquet",
    )


def main() -> None:
    started = time.monotonic()
    prepared = Path("data/prepared")
    with tempfile.TemporaryDirectory(prefix="qlib-interface-check-") as scratch:
        factors = Path(scratch)
        temporary_factor(prepared, "train", 2024, 20241220, 20241231, factors)
        temporary_factor(prepared, "test", 2025, 20250102, 20250103, factors)

        train = load_window(prepared, factors, "train", 20241220, 20241231,
                            min_valid_run=1, label_asof_date=20241231)
        test = load_window(prepared, factors, "test", 20250102, 20250103,
                           min_valid_run=1)
        assert train.feature_names == test.feature_names == ["feature_interface_code"]
        assert not train.auxiliary.loc[20241231, "label_available"].any()
        assert train.labels.loc[20241231, "y_ret_1d"].notna().sum() == 4525
        assert not test.labels["y_ret_1d"].notna().any()
        later = load_window(prepared, factors, "train", 20241220, 20241231,
                            min_valid_run=1, label_asof_date=20250102)
        assert later.auxiliary.loc[20241231, "label_available"].sum() == 4525

        train_dataset = DatasetH(
            handler=to_qlib_handler(train),
            segments={"train": ("2024-12-20", "2024-12-31")},
        )
        test_dataset = DatasetH(
            handler=to_qlib_handler(test),
            segments={"test": ("2025-01-02", "2025-01-03")},
        )
        model = LinearModel(estimator="ridge", alpha=1.0, fit_intercept=True)
        model.fit(train_dataset)
        predicted = model.predict(test_dataset)
        complete = complete_predictions(test, predicted)
        expected = test.features.reset_index()[KEY]
        assert complete[KEY].equals(expected)
        assert complete["pred"].notna().all()
        assert (~complete["model_ready"]).sum() == (~test.auxiliary["bar_valid"]).sum()
        for _, group in complete.groupby("trade_date"):
            assert group.loc[~group["model_ready"], "pred"].max() < group.loc[
                group["model_ready"], "pred"
            ].min()

        train_pred = complete_predictions(
            train, model.predict(train_dataset, segment="train")
        )
        raw_labels = _read_years(
            prepared, "train", KEY + ["y_ret_1d"], 20241220, 20241231
        )
        score = score_frame(
            train_pred, raw_labels, train.auxiliary.reset_index()
        )
        assert all(pd.notna(value) for value in score.values())
        print(json.dumps({
            "status": "pass",
            "train_rows": len(train.features),
            "mature_train_labels": int(train.auxiliary["label_available"].sum()),
            "test_rows": len(test.features),
            "fallback_rows": int((~complete["model_ready"]).sum()),
            "official_scoring_path": "pass (in-sample mechanics only)",
            "factor": "temporary synthetic interface-check column; no alpha interpretation",
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1),
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
