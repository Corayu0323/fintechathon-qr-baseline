#!/usr/bin/env python3
"""Chunked audit for the competition's raw daily-bar CSV files.

The script reads the source files without modifying them and writes only
metadata into ``reports/`` and ``data/metadata/``.  It deliberately keeps the
official label unchanged: recomputed labels are a consistency diagnostic, not
a replacement target.
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["ts_code", "trade_date"]
PRICE_COLUMNS = ["open", "high", "low", "close"]
BASE_COLUMNS = KEYS + PRICE_COLUMNS + [
    "vol",
    "amount",
    "flag_limit_up",
    "flag_limit_down",
]


def json_value(value):
    """Convert NumPy and pandas scalar values into JSON-safe Python values."""
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


@dataclass
class AuditState:
    path: Path
    has_label: bool
    rows: int = 0
    missing: Counter = field(default_factory=Counter)
    codes: set[str] = field(default_factory=set)
    dates: set[int] = field(default_factory=set)
    years: Counter = field(default_factory=Counter)
    zero_volume: int = 0
    nonpositive_price: int = 0
    ohlc_invalid: int = 0
    partial_bar_missing: int = 0
    invalid_flag: Counter = field(default_factory=Counter)
    sort_violations: int = 0
    duplicate_adjacent: int = 0
    first_key: tuple[str, int] | None = None
    last_key: tuple[str, int] | None = None
    label_checked: int = 0
    label_matched: int = 0
    label_abs_error_sum: float = 0.0
    label_abs_error_max: float = 0.0
    label_mismatch_over_tolerance: int = 0
    label_missing_with_next_close: int = 0
    label_present_without_next_close: int = 0
    label_missing: int = 0
    prev_row: pd.Series | None = None

    def as_dict(self) -> dict:
        output = {
            "file": self.path.name,
            "rows": self.rows,
            "stocks": len(self.codes),
            "trading_days": len(self.dates),
            "date_range": [min(self.dates), max(self.dates)] if self.dates else None,
            "first_key": self.first_key,
            "last_key": self.last_key,
            "rows_by_year": dict(sorted(self.years.items())),
            "missing_by_column": dict(sorted(self.missing.items())),
            "zero_volume": self.zero_volume,
            "nonpositive_price_cells": self.nonpositive_price,
            "partial_ohlc_missing_rows": self.partial_bar_missing,
            "ohlc_logic_invalid_rows": self.ohlc_invalid,
            "invalid_flags": dict(sorted(self.invalid_flag.items())),
            "key_order_violations": self.sort_violations,
            "adjacent_duplicate_keys": self.duplicate_adjacent,
        }
        if self.has_label:
            output["label_consistency"] = {
                "label_missing": self.label_missing,
                "checked_rows": self.label_checked,
                "matched_within_1e-10": self.label_matched,
                "mismatched_over_1e-10": self.label_mismatch_over_tolerance,
                "mean_absolute_error": (
                    self.label_abs_error_sum / self.label_checked if self.label_checked else None
                ),
                "max_absolute_error": self.label_abs_error_max if self.label_checked else None,
                "label_missing_with_next_observed_close": self.label_missing_with_next_close,
                "label_present_without_next_observed_close": self.label_present_without_next_close,
            }
        return output


def audit_file(path: Path, chunk_size: int) -> dict:
    columns = BASE_COLUMNS + (["y_ret_1d"] if path.name.startswith("训练") else [])
    state = AuditState(path=path, has_label="y_ret_1d" in columns)

    for chunk in pd.read_csv(path, usecols=columns, chunksize=chunk_size):
        state.rows += len(chunk)
        for column, count in chunk.isna().sum().items():
            state.missing[column] += int(count)
        state.codes.update(chunk["ts_code"].dropna().astype(str))
        state.dates.update(chunk["trade_date"].dropna().astype(int))
        state.years.update((chunk["trade_date"].dropna().astype(int) // 10_000).tolist())
        state.zero_volume += int(chunk["vol"].eq(0).sum())
        state.nonpositive_price += int((chunk[PRICE_COLUMNS] <= 0).sum().sum())

        ohlc_count = chunk[PRICE_COLUMNS].notna().sum(axis=1)
        state.partial_bar_missing += int(((ohlc_count > 0) & (ohlc_count < 4)).sum())
        complete = chunk[PRICE_COLUMNS].notna().all(axis=1)
        invalid_ohlc = complete & (
            (chunk["low"] > chunk[["open", "close"]].min(axis=1))
            | (chunk["high"] < chunk[["open", "close"]].max(axis=1))
            | (chunk["low"] > chunk["high"])
        )
        state.ohlc_invalid += int(invalid_ohlc.sum())

        for flag in ["flag_limit_up", "flag_limit_down"]:
            state.invalid_flag[flag] += int((~chunk[flag].isin([0, 1])).sum())

        codes = chunk["ts_code"].astype(str)
        dates = chunk["trade_date"].astype(int)
        if state.first_key is None:
            state.first_key = (codes.iat[0], int(dates.iat[0]))
        prior_code = codes.shift(1)
        prior_date = dates.shift(1)
        if state.last_key is not None:
            prior_code.iat[0], prior_date.iat[0] = state.last_key
        out_of_order = (codes < prior_code) | ((codes == prior_code) & (dates < prior_date))
        duplicates = (codes == prior_code) & (dates == prior_date)
        state.sort_violations += int(out_of_order.sum())
        state.duplicate_adjacent += int(duplicates.sum())
        state.last_key = (codes.iat[-1], int(dates.iat[-1]))

        if state.has_label:
            if state.prev_row is not None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=FutureWarning)
                    combined = pd.concat([state.prev_row.to_frame().T, chunk], ignore_index=True)
            else:
                combined = chunk.reset_index(drop=True)
            current = combined.iloc[:-1]
            following = combined.iloc[1:]
            same_code = current["ts_code"].to_numpy() == following["ts_code"].to_numpy()
            label = current["y_ret_1d"].to_numpy(dtype=float)
            current_close = current["close"].to_numpy(dtype=float)
            next_close = following["close"].to_numpy(dtype=float)
            label_is_missing = np.isnan(label)
            state.label_missing += int(label_is_missing.sum())
            both_prices = np.isfinite(current_close) & np.isfinite(next_close)
            eligible = same_code & both_prices
            state.label_missing_with_next_close += int((eligible & label_is_missing).sum())
            checked = eligible & ~label_is_missing
            expected = np.empty_like(label)
            expected[checked] = next_close[checked] / current_close[checked] - 1.0
            errors = np.abs(label[checked] - expected[checked])
            state.label_checked += int(checked.sum())
            state.label_matched += int((errors <= 1e-10).sum())
            state.label_mismatch_over_tolerance += int((errors > 1e-10).sum())
            state.label_abs_error_sum += float(errors.sum())
            if len(errors):
                state.label_abs_error_max = max(state.label_abs_error_max, float(errors.max()))
            missing_next = same_code & ~both_prices & ~label_is_missing
            cross_code = ~same_code & ~label_is_missing
            state.label_present_without_next_close += int(missing_next.sum() + cross_code.sum())
            state.prev_row = combined.iloc[-1].copy()

    if state.has_label and state.prev_row is not None:
        if pd.isna(state.prev_row["y_ret_1d"]):
            state.label_missing += 1
        else:
            state.label_present_without_next_close += 1
    return state.as_dict()


def audit_train_test_boundary(train_path: Path, test_path: Path, chunk_size: int) -> dict:
    """Check endpoint labels against each stock's first test-period close."""
    last_train: dict[str, tuple[int, float, float]] = {}
    first_test: dict[str, tuple[int, float]] = {}
    for chunk in pd.read_csv(
        train_path,
        usecols=["ts_code", "trade_date", "close", "y_ret_1d"],
        chunksize=chunk_size,
    ):
        for row in chunk.groupby("ts_code", sort=False).tail(1).itertuples(index=False):
            last_train[str(row.ts_code)] = (int(row.trade_date), row.close, row.y_ret_1d)
    for chunk in pd.read_csv(
        test_path, usecols=["ts_code", "trade_date", "close"], chunksize=chunk_size
    ):
        for row in chunk.groupby("ts_code", sort=False).head(1).itertuples(index=False):
            first_test.setdefault(str(row.ts_code), (int(row.trade_date), row.close))

    counts: Counter = Counter()
    errors: list[float] = []
    for code, (_, train_close, label) in last_train.items():
        test_row = first_test.get(code)
        if test_row is None:
            continue
        counts["stocks_with_boundary"] += 1
        _, test_close = test_row
        if pd.isna(label):
            counts["boundary_label_missing"] += 1
        elif pd.notna(train_close) and pd.notna(test_close):
            error = abs(float(label) - (test_close / train_close - 1.0))
            errors.append(error)
            counts["checked"] += 1
            counts["matched_within_1e-10" if error <= 1e-10 else "mismatched_over_1e-10"] += 1
        else:
            counts["label_present_but_close_missing"] += 1
    return {
        **dict(counts),
        "mean_absolute_error": float(np.mean(errors)) if errors else None,
        "max_absolute_error": float(np.max(errors)) if errors else None,
    }


def render_report(result: dict) -> str:
    train = result["training"]
    test = result["test"]
    label = train["label_consistency"]
    boundary = result["train_test_boundary_label_consistency"]
    lines = [
        "# 赛题五原始数据审计", "",
        f"生成时间：{result['generated_at']}", "",
        "## 结论摘要", "",
        f"- 训练集 {train['rows']:,} 行、{train['stocks']:,} 只股票、{train['trading_days']:,} 个交易日；测试集 {test['rows']:,} 行、{test['stocks']:,} 只股票、{test['trading_days']:,} 个交易日。",
        f"- 训练集标签缺失 {label['label_missing']:,} 行；它们应保留在历史特征计算中，但不进入监督训练。",
        f"- 可核对标签 {label['checked_rows']:,} 行，其中误差不超过 1e-10 的有 {label['matched_within_1e-10']:,} 行；不一致行 {label['mismatched_over_1e-10']:,} 行。",
        f"- 跨训练—测试边界可核对标签 {boundary.get('checked', 0):,} 行，其中匹配 {boundary.get('matched_within_1e-10', 0):,} 行。",
        "- 标签一致性只用于核对，不替换题给 y_ret_1d。",
        "",
        "## 结构与完整性", "",
        "| 数据集 | 行数 | 股票数 | 交易日数 | 日期范围 | 键顺序违例 | 相邻重复键 | OHLC 逻辑异常 |", "|---|---:|---:|---:|---|---:|---:|---:|",
        f"| 训练集 | {train['rows']:,} | {train['stocks']:,} | {train['trading_days']:,} | {train['date_range'][0]}—{train['date_range'][1]} | {train['key_order_violations']:,} | {train['adjacent_duplicate_keys']:,} | {train['ohlc_logic_invalid_rows']:,} |",
        f"| 测试集 | {test['rows']:,} | {test['stocks']:,} | {test['trading_days']:,} | {test['date_range'][0]}—{test['date_range'][1]} | {test['key_order_violations']:,} | {test['adjacent_duplicate_keys']:,} | {test['ohlc_logic_invalid_rows']:,} |",
        "",
        "## 缺失与异常", "",
        "| 数据集 | 收盘缺失 | 成交量缺失 | 标签缺失 | 零成交量 | 部分 OHLC 缺失 | 非正价格单元 | 非 0/1 涨跌停标志 |", "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| 训练集 | {train['missing_by_column'].get('close', 0):,} | {train['missing_by_column'].get('vol', 0):,} | {label['label_missing']:,} | {train['zero_volume']:,} | {train['partial_ohlc_missing_rows']:,} | {train['nonpositive_price_cells']:,} | {sum(train['invalid_flags'].values()):,} |",
        f"| 测试集 | {test['missing_by_column'].get('close', 0):,} | {test['missing_by_column'].get('vol', 0):,} | — | {test['zero_volume']:,} | {test['partial_ohlc_missing_rows']:,} | {test['nonpositive_price_cells']:,} | {sum(test['invalid_flags'].values()):,} |",
        "",
        "## 建议的数据口径", "",
        "1. 原始 CSV 作为只读快照；后续因子表另行生成，不覆盖原始字段。",
        "2. 日期 t 的因子仅使用截至 t 的日线；滚动指标跨训练边界时保留必要的先验历史。",
        "3. 标签缺失行不参与监督训练；测试集全部键均保留用于最终预测。",
        "4. 日期 t 的标签在下一条可用日线收盘后才成熟；训练集末端跨入测试集的标签不可在收益实现前用于滚动训练。",
        "5. 不将后复权价格与 amount / vol 直接构造成价格偏离指标，除非已证明二者统一复权口径。",
        "6. 训练、Top 组收益和 Jaccard 换手须分别按官方脚本过滤样本，不能用一套统一过滤替代。",
        "",
        "完整机器可读统计见 `data/metadata/raw_audit.json`。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("赛题五数据"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--metadata-dir", type=Path, default=Path("data/metadata"))
    parser.add_argument("--chunk-size", type=int, default=200_000)
    args = parser.parse_args()

    train_path = args.data_dir / "训练集.csv"
    test_path = args.data_dir / "测试集_X.csv"
    result = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source_files": {
            train_path.name: {"bytes": train_path.stat().st_size},
            test_path.name: {"bytes": test_path.stat().st_size},
        },
        "training": audit_file(train_path, args.chunk_size),
        "test": audit_file(test_path, args.chunk_size),
        "train_test_boundary_label_consistency": audit_train_test_boundary(
            train_path, test_path, args.chunk_size
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.metadata_dir.mkdir(parents=True, exist_ok=True)
    (args.metadata_dir / "raw_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=json_value) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "数据审计.md").write_text(render_report(result), encoding="utf-8")


if __name__ == "__main__":
    main()
