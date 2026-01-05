"""
Offline scan utilities for per-item `prices_*.csv` exports.

The snapshot under `data/<snapshot>/` contains many files like:
  prices_<item_id>_<snapshot>.csv

Each file is an exported time series from the `prices` table with columns:
  fetched_at, buy_unit_price, sell_unit_price, ...

This module scans those per-item CSV files to compute:
- volatility over a lookback window ending at a chosen timestamp
- profit potential (after fee) at the chosen timestamp (snapshot row)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PricesFileScanConfig:
    prices_dir: Path
    as_of: datetime
    lookback: timedelta = timedelta(hours=24)
    fee_rate: float = 0.15
    min_points: int = 30
    max_files: int | None = None

    def normalized(self) -> "PricesFileScanConfig":
        prices_dir = Path(self.prices_dir)
        as_of = self.as_of
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=UTC)
        else:
            as_of = as_of.astimezone(UTC)

        lookback = self.lookback
        if lookback.total_seconds() <= 0:
            lookback = timedelta(hours=24)

        fee_rate = float(self.fee_rate)
        if fee_rate < 0:
            fee_rate = 0.0
        if fee_rate > 1:
            fee_rate = 1.0

        min_points = int(self.min_points)
        if min_points < 2:
            min_points = 2

        max_files = None if self.max_files is None else max(int(self.max_files), 1)
        return PricesFileScanConfig(
            prices_dir=prices_dir,
            as_of=as_of,
            lookback=lookback,
            fee_rate=fee_rate,
            min_points=min_points,
            max_files=max_files,
        )


def _ceil_fee_copper(*, sell_price_copper: int, fee_rate: float) -> int:
    """
    Compute Trading Post fee in copper, rounded UP (as in-game).

    Default/common case is a 15% fee.
    """

    sell = int(sell_price_copper)
    if sell <= 0:
        return 0

    rate = float(fee_rate)
    if rate <= 0:
        return 0
    if rate >= 1:
        return sell

    # Exact fast-path for 15%: ceil(sell * 15 / 100)
    if abs(rate - 0.15) < 1e-12:
        return (sell * 15 + 99) // 100

    return int(math.ceil(sell * rate))


def _net_sell_after_fee_copper(*, sell_price_copper: int, fee_rate: float) -> int:
    sell = int(sell_price_copper)
    fee = _ceil_fee_copper(sell_price_copper=sell, fee_rate=fee_rate)
    return max(sell - fee, 0)


def iter_prices_csv_files(prices_dir: Path) -> Iterable[Path]:
    """
    Iterate `prices_*.csv` files under `prices_dir`.
    """

    base = Path(prices_dir)
    for entry in base.iterdir():
        if not entry.is_file():
            continue
        name = entry.name
        if not (name.startswith("prices_") and name.endswith(".csv")):
            continue
        yield entry


def _scan_single_prices_file(
    path: Path,
    *,
    as_of: datetime,
    start_time: datetime,
    fee_rate: float,
    min_points: int,
) -> dict[str, object] | None:
    try:
        df = pd.read_csv(
            path,
            usecols=["item_id", "buy_unit_price", "sell_unit_price", "fetched_at"],
            parse_dates=["fetched_at"],
        )
    except Exception:
        return None

    if df.empty:
        return None

    df["fetched_at"] = pd.to_datetime(df["fetched_at"], errors="coerce", utc=True)
    df = df[df["fetched_at"].notna()]
    if df.empty:
        return None

    df["buy_unit_price"] = pd.to_numeric(df["buy_unit_price"], errors="coerce")
    df["sell_unit_price"] = pd.to_numeric(df["sell_unit_price"], errors="coerce")
    df = df[(df["buy_unit_price"] > 0) & (df["sell_unit_price"] > 0)]
    if df.empty:
        return None

    # Snapshot row: last data point at or before as_of.
    snap = df[df["fetched_at"] <= as_of]
    if snap.empty:
        return None
    snap_row = snap.loc[snap["fetched_at"].idxmax()]

    buy_price = int(snap_row["buy_unit_price"])
    sell_price = int(snap_row["sell_unit_price"])
    item_id = int(snap_row["item_id"])
    snapshot_time = pd.Timestamp(snap_row["fetched_at"]).to_pydatetime()

    net_sell = _net_sell_after_fee_copper(sell_price_copper=sell_price, fee_rate=fee_rate)
    profit_after_fee = net_sell - buy_price
    roi_after_fee = (profit_after_fee / buy_price) if buy_price else float("nan")

    # Volatility window.
    window = df[(df["fetched_at"] >= start_time) & (df["fetched_at"] <= as_of)].copy()
    if window.empty:
        points = 0
        mid_avg = float("nan")
        mid_std = float("nan")
        mid_cv = float("nan")
        return_std = float("nan")
    else:
        window = window.sort_values("fetched_at")
        mid = (window["buy_unit_price"] + window["sell_unit_price"]) / 2.0
        points = int(mid.shape[0])
        mid_avg = float(mid.mean())
        mid_std = float(mid.std(ddof=1)) if points >= 2 else float("nan")
        mid_cv = float(mid_std / mid_avg) if mid_avg else float("nan")
        returns = mid.pct_change().dropna()
        return_std = float(returns.std(ddof=1)) if returns.shape[0] >= min_points else float("nan")

    return {
        "item_id": item_id,
        "snapshot_time": snapshot_time,
        "buy_price": int(buy_price),
        "sell_price": int(sell_price),
        "fee_rate": float(fee_rate),
        "profit_after_fee": int(profit_after_fee),
        "roi_after_fee": float(roi_after_fee),
        "is_profitable_after_fee": bool(profit_after_fee > 0),
        "points": int(points),
        "mid_avg": float(mid_avg),
        "mid_std": float(mid_std),
        "mid_cv": float(mid_cv),
        "return_std": float(return_std),
        "source_file": str(path),
    }


def build_opportunity_table_from_prices_dir(
    *,
    config: PricesFileScanConfig,
    workers: int = 8,
    log_every: int = 1000,
) -> pd.DataFrame:
    """
    Scan a directory of per-item `prices_*.csv` exports and build an opportunity table.

    - Uses thread workers because work is mostly IO + parsing small CSVs.
    - Returns one row per item id.
    """

    normalized = config.normalized()
    start_time = normalized.as_of - normalized.lookback

    files = list(iter_prices_csv_files(normalized.prices_dir))
    if normalized.max_files is not None:
        files = files[: normalized.max_files]

    if not files:
        return pd.DataFrame(
            columns=[
                "item_id",
                "snapshot_time",
                "buy_price",
                "sell_price",
                "profit_after_fee",
                "roi_after_fee",
                "is_profitable_after_fee",
                "return_std",
                "mid_cv",
                "points",
                "source_file",
            ]
        )

    # Local import to keep module import side-effects minimal.
    from concurrent.futures import ThreadPoolExecutor, as_completed

    rows: list[dict[str, object]] = []
    workers = max(int(workers), 1)
    log_every = max(int(log_every), 1)
    started_at = time.time()
    total = len(files)
    logger.info("Scanning %s prices CSV files with %s workers", total, workers)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _scan_single_prices_file,
                path,
                as_of=normalized.as_of,
                start_time=start_time,
                fee_rate=normalized.fee_rate,
                min_points=normalized.min_points,
            )
            for path in files
        ]
        processed = 0
        for future in as_completed(futures):
            result = future.result()
            processed += 1
            if result is not None:
                rows.append(result)
            if processed % log_every == 0 or processed == total:
                elapsed = max(time.time() - started_at, 1e-6)
                rate = processed / elapsed
                remaining = total - processed
                eta_s = remaining / rate if rate > 0 else float("inf")
                logger.info(
                    "Progress: %s/%s (%.1f%%) | %.1f files/s | ETA %.1fs",
                    processed,
                    total,
                    (processed / total) * 100.0,
                    rate,
                    eta_s,
                )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["points"] = pd.to_numeric(df.get("points"), errors="coerce").fillna(0).astype(int)
    for col in ["buy_price", "sell_price", "profit_after_fee", "roi_after_fee", "return_std", "mid_cv", "mid_avg", "mid_std"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


__all__ = [
    "PricesFileScanConfig",
    "iter_prices_csv_files",
    "build_opportunity_table_from_prices_dir",
]


