#!/usr/bin/env python3
"""Build one immutable, versioned daily price-volume factor matrix."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.factors.formulas import FORMULAS


BAR_COLUMNS = [
    "trade_date", "ts_code", "open", "high", "low", "close", "vol",
    "amount", "bar_valid", "valid_bar_run",
]
SPEC_PATH = Path(__file__).resolve().parents[2] / "specs" / "factor_set_v0.2.json"
ALLOWED_SOURCE_FIELDS = {"open", "high", "low", "close", "vol", "amount"}


def load_spec(path: Path) -> dict:
    spec = json.loads(path.read_text(encoding="utf-8"))
    if not spec.get("version") or not spec.get("features"):
        raise ValueError(f"invalid factor set spec: {path}")
    names = [feature["name"] for feature in spec["features"]]
    if len(names) != len(set(names)):
        raise ValueError("factor names must be unique")
    for feature in spec["features"]:
        formula_id = feature["formula_id"]
        if formula_id not in FORMULAS:
            raise ValueError(f"unregistered formula_id: {formula_id}")
        if "y_ret_1d" in feature.get("source_fields", []):
            raise ValueError(f"labels cannot be factor inputs: {feature['name']}")
        unknown_fields = set(feature.get("source_fields", [])) - ALLOWED_SOURCE_FIELDS
        if unknown_fields:
            raise ValueError(f"unsupported source fields for {feature['name']}: {sorted(unknown_fields)}")
        if int(feature["required_valid_bars"]) < 1:
            raise ValueError(f"required_valid_bars must be positive: {feature['name']}")
    return spec


def source_part(prepared_dir: Path, split: str, year: int) -> pd.DataFrame | None:
    path = prepared_dir / f"{split}_{year}.parquet"
    if not path.exists():
        return None
    return pq.read_table(path, columns=BAR_COLUMNS).to_pandas()


def source_for_year(prepared_dir: Path, split: str, year: int) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    current = source_part(prepared_dir, split, year)
    if current is None:
        raise FileNotFoundError(prepared_dir / f"{split}_{year}.parquet")
    if split == "train":
        prior_split, prior_year = "train", year - 1
    elif year == 2025:
        prior_split, prior_year = "train", 2024
    else:
        prior_split, prior_year = "test", year - 1
    return current, source_part(prepared_dir, prior_split, prior_year)


def calculate(current: pd.DataFrame, prior: pd.DataFrame | None, spec: dict) -> pd.DataFrame:
    """Apply registered pure formulas; retain exact source-key coverage."""
    work = current.copy() if prior is None else pd.concat([prior, current], ignore_index=True)
    work = work.sort_values(["ts_code", "trade_date"], kind="mergesort").reset_index(drop=True)
    columns = {
        feature["name"]: FORMULAS[feature["formula_id"]](work)
        for feature in spec["features"]
    }
    factor_frame = pd.DataFrame(columns, index=work.index)
    values = factor_frame.to_numpy(dtype=np.float64)
    values[~np.isfinite(values)] = np.nan
    names = [feature["name"] for feature in spec["features"]]
    factor_frame = pd.DataFrame(values.astype(np.float32), columns=names, index=work.index)
    factor_frame.insert(0, "ts_code", work["ts_code"].to_numpy())
    factor_frame.insert(0, "trade_date", work["trade_date"].to_numpy())

    current_keys = current[["trade_date", "ts_code"]]
    result = current_keys.merge(
        factor_frame, on=["trade_date", "ts_code"], how="left", validate="one_to_one", sort=False,
    )
    result = result.sort_values(["trade_date", "ts_code"], kind="mergesort").reset_index(drop=True)
    if len(result) != len(current) or result.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError(f"factor key integrity failed for {len(current)} source rows")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=Path("data/prepared"))
    parser.add_argument("--output-root", type=Path, default=Path("data/features"))
    parser.add_argument("--spec", type=Path, default=SPEC_PATH)
    args = parser.parse_args()

    spec = load_spec(args.spec)
    partitions = []
    for split in ("train", "test"):
        for path in sorted(args.prepared_dir.glob(f"{split}_*.parquet")):
            partitions.append((split, int(path.stem.rsplit("_", 1)[1])))
    if not partitions:
        raise FileNotFoundError(f"no prepared annual parquet files in {args.prepared_dir}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    version_dir = args.output_root / spec["version"]
    if version_dir.exists():
        raise FileExistsError(
            f"immutable factor version already exists: {version_dir}; "
            "edit the spec version before building changed formulas"
        )

    staging = Path(tempfile.mkdtemp(prefix=f".{spec['version']}.staging-", dir=args.output_root))
    try:
        manifest_rows = []
        names = [feature["name"] for feature in spec["features"]]
        for split, year in partitions:
            current, prior = source_for_year(args.prepared_dir, split, year)
            factors = calculate(current, prior, spec)
            output = staging / f"features_{year}.parquet"
            pq.write_table(
                pa.Table.from_pandas(factors, preserve_index=False),
                output, compression="zstd", row_group_size=100_000,
            )
            manifest_rows.append({
                "split": split, "year": year, "rows": len(factors),
                "columns": names, "path": str(version_dir / output.name),
                "ready_rows": int(np.isfinite(factors[names].to_numpy()).all(axis=1).sum()),
            })
            print(f"{split} {year}: rows={len(factors):,} ready={manifest_rows[-1]['ready_rows']:,}")

        catalog = {**spec, "partitions": manifest_rows}
        (staging / "catalog.json").write_text(
            json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.chmod(staging, 0o755)
        staging.rename(version_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"Created immutable factor set {spec['version']}: {version_dir}")


if __name__ == "__main__":
    main()
