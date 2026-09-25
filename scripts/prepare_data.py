#!/usr/bin/env python3
"""Preserve every competition row and add causal data-quality metadata.

This prepares a compact Parquet source for later factor calculation.  It does
not impute prices, construct factors, filter rows, or assign prediction scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PRICE = ["open", "high", "low", "close"]
BAR = PRICE + ["vol", "amount"]
INPUT = ["ts_code", "trade_date"] + BAR + ["flag_limit_up", "flag_limit_down"]
OUTPUT = INPUT + [
    "y_ret_1d", "source_row", "bar_valid", "volume_zero", "label_present",
    "label_maturity_date", "valid_bar_run",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def trading_calendar(paths: list[Path], chunk_size: int) -> list[int]:
    dates: set[int] = set()
    for path in paths:
        for part in pd.read_csv(path, usecols=["trade_date"], chunksize=chunk_size):
            dates.update(part["trade_date"].astype(int).unique())
    return sorted(dates)


def valid_run_lengths(frame: pd.DataFrame, valid: np.ndarray, state: dict[str, int]) -> np.ndarray:
    """Count consecutive valid daily rows per stock, carrying state across files."""
    result = np.zeros(len(frame), dtype=np.int16)
    for code, positions in frame.groupby("ts_code", sort=False).indices.items():
        run = state.get(str(code), 0)
        for position in positions:
            run = run + 1 if valid[position] else 0
            if run > np.iinfo(np.int16).max:
                raise OverflowError(f"valid_bar_run exceeds int16 for {code}")
            result[position] = run
        state[str(code)] = run
    return result


def prepare(
    paths: list[tuple[str, Path]],
    output_dir: Path,
    chunk_size: int,
) -> dict:
    calendar = trading_calendar([path for _, path in paths], chunk_size)
    next_date = {date: calendar[i + 1] for i, date in enumerate(calendar[:-1])}
    next_date[calendar[-1]] = 0

    output_dir.mkdir(parents=True, exist_ok=True)
    writers: dict[tuple[str, int], pq.ParquetWriter] = {}
    part_stats: dict[str, Counter] = {}
    source_stats: dict[str, Counter] = {}
    history_state: dict[str, int] = {}
    expected_schema: pa.Schema | None = None

    try:
        for split, path in paths:
            source_stats[split] = Counter()
            row_start = 0
            usecols = INPUT + (["y_ret_1d"] if split == "train" else [])
            for frame in pd.read_csv(
                path, usecols=usecols, chunksize=chunk_size, dtype={"ts_code": "string"}
            ):
                frame = frame.reset_index(drop=True)
                if split == "test":
                    frame["y_ret_1d"] = np.nan
                frame = frame[INPUT + ["y_ret_1d"]]
                frame["trade_date"] = frame["trade_date"].astype("int32")
                for flag in ["flag_limit_up", "flag_limit_down"]:
                    frame[flag] = frame[flag].astype("int8")
                frame["source_row"] = np.arange(
                    row_start, row_start + len(frame), dtype=np.int64
                )
                row_start += len(frame)

                bar_valid = frame[BAR].notna().all(axis=1).to_numpy(dtype=bool)
                frame["bar_valid"] = bar_valid
                frame["volume_zero"] = frame["vol"].eq(0).to_numpy(dtype=bool)
                frame["label_present"] = (
                    frame["y_ret_1d"].notna().to_numpy(dtype=bool)
                    if split == "train"
                    else np.zeros(len(frame), dtype=bool)
                )
                frame["label_maturity_date"] = (
                    frame["trade_date"].map(next_date).astype("int32")
                    if split == "train"
                    else np.zeros(len(frame), dtype=np.int32)
                )
                frame["valid_bar_run"] = valid_run_lengths(frame, bar_valid, history_state)
                frame = frame[OUTPUT]

                source_stats[split]["rows"] += len(frame)
                source_stats[split]["invalid_bar_rows"] += int((~bar_valid).sum())
                source_stats[split]["zero_volume_rows"] += int(frame["volume_zero"].sum())
                source_stats[split]["label_present_rows"] += int(frame["label_present"].sum())
                source_stats[split]["trainable_before_history_mask"] += int(
                    (frame["bar_valid"] & frame["label_present"]).sum()
                )

                for year, partition in frame.groupby(frame["trade_date"] // 10_000, sort=False):
                    key = (split, int(year))
                    table = pa.Table.from_pandas(partition.reset_index(drop=True), preserve_index=False)
                    if expected_schema is None:
                        expected_schema = table.schema
                    if table.schema != expected_schema:
                        raise ValueError(f"schema drift in {split} {year}: {table.schema}")
                    if key not in writers:
                        target = output_dir / f"{split}_{year}.parquet.partial"
                        writers[key] = pq.ParquetWriter(
                            target, expected_schema, compression="zstd"
                        )
                    writers[key].write_table(table)
                    name = f"{split}_{year}.parquet"
                    stats = part_stats.setdefault(name, Counter())
                    stats["rows"] += len(partition)
                    stats["invalid_bar_rows"] += int((~partition["bar_valid"]).sum())
                    stats["label_present_rows"] += int(partition["label_present"].sum())
        for writer in writers.values():
            writer.close()
        writers.clear()

        for name in part_stats:
            staged = output_dir / f"{name}.partial"
            os.replace(staged, output_dir / name)

        manifest = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "schema_version": "prepared-v1",
            "semantics": {
                "bar_valid": "All six current-day OHLCV/amount fields are non-missing.",
                "volume_zero": "Observed volume is exactly zero, not missing.",
                "label_present": "Provided y_ret_1d is non-missing; training still needs a maturity cutoff.",
                "label_maturity_date": "Next global trading date; 0 means unavailable/not applicable.",
                "valid_bar_run": "Consecutive valid daily rows through current date, reset to zero on invalid bar.",
                "trainable_before_history_mask": "bar_valid and label_present; factor-specific history not yet applied.",
            },
            "calendar_days": len(calendar),
            "source_files": {
                path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for _, path in paths
            },
            "source_stats": {key: dict(value) for key, value in source_stats.items()},
            "partitions": {key: dict(value) for key, value in sorted(part_stats.items())},
            "columns": OUTPUT,
        }
        metadata_path = output_dir / "manifest.json"
        staged_metadata = output_dir / "manifest.json.partial"
        staged_metadata.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(staged_metadata, metadata_path)
        return manifest
    finally:
        for writer in writers.values():
            writer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("赛题五数据"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/prepared"))
    parser.add_argument("--chunk-size", type=int, default=200_000)
    args = parser.parse_args()
    inputs = [
        ("train", args.data_dir / "训练集.csv"),
        ("test", args.data_dir / "测试集_X.csv"),
    ]
    manifest = prepare(inputs, args.output_dir, args.chunk_size)
    print(json.dumps(manifest["source_stats"], ensure_ascii=False, indent=2))
    print(f"Wrote {len(manifest['partitions'])} Parquet partitions to {args.output_dir}")


if __name__ == "__main__":
    main()
