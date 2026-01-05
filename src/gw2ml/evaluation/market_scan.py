"""
Market scanning utilities for Guild Wars 2 Trading Post data.

This module is meant to support "fixed timestamp" analyses:
- compute per-item volatility over a lookback window ending at a chosen timestamp
- compute per-item profit potential at the chosen timestamp
- check whether the 15% Trading Post fee still yields positive profit

It operates directly against the `gw2tp_historical_prices` table and joins `items`
to restrict results to tradeable items and enrich output with item metadata.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pandas as pd
from sqlalchemy import text  # type: ignore[import]

from gw2ml.data.database_client import ConnectionInput, _ensure_client


@dataclass(frozen=True, slots=True)
class TpMarketScanConfig:
    """
    Configuration for a TP market scan.

    - `as_of`: "analysis timestamp" (inclusive) used for snapshot and lookback end.
    - `lookback`: historical window size used for volatility calculation.
    - `fee_rate`: Trading Post fee (default 15% = 0.15).
    - `min_points`: minimum number of points for volatility computation.
    """

    as_of: datetime
    lookback: timedelta = timedelta(hours=24)
    fee_rate: float = 0.15
    min_points: int = 30

    def normalized(self) -> "TpMarketScanConfig":
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

        return TpMarketScanConfig(as_of=as_of, lookback=lookback, fee_rate=fee_rate, min_points=min_points)


def _ceil_fee_copper_series(sell_price: pd.Series, *, fee_rate: float) -> pd.Series:
    """
    Compute Trading Post fee for each row, rounded UP (as in-game).
    """

    rate = float(fee_rate)
    sell = pd.to_numeric(sell_price, errors="coerce").fillna(0).astype(int)

    if rate <= 0:
        return sell * 0
    if rate >= 1:
        return sell

    # Exact fast-path for 15%: ceil(sell * 15 / 100)
    if abs(rate - 0.15) < 1e-12:
        return (sell * 15 + 99) // 100

    # Fallback (float): ceil(sell * rate)
    return sell.apply(lambda v: int(math.ceil(int(v) * rate)))


def _datetime_to_epoch_ms(value: datetime) -> int:
    utc_value = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return int(utc_value.astimezone(UTC).timestamp() * 1000)


def fetch_tp_snapshot_as_of(connection: ConnectionInput, *, as_of: datetime) -> pd.DataFrame:
    """
    Fetch a per-item snapshot (latest row per item) up to and including `as_of`.

    Returns a dataframe containing:
    - item metadata (id/name/rarity/level/vendor_value)
    - snapshot buy/sell prices + supply/demand
    - snapshot timestamp as both ms epoch and UTC timestamp
    """

    client = _ensure_client(connection)
    as_of_utc = as_of if as_of.tzinfo is not None else as_of.replace(tzinfo=UTC)
    end_ms = _datetime_to_epoch_ms(as_of_utc)

    query = text(
        """
        SELECT DISTINCT ON (h.item_id)
            h.item_id,
            i.name,
            i.rarity,
            i.level,
            i.vendor_value,
            h.timestamp AS timestamp_ms,
            to_timestamp(h.timestamp / 1000.0) AT TIME ZONE 'UTC' AS timestamp_utc,
            h.buy_price,
            h.sell_price,
            h.supply,
            h.demand
        FROM gw2tp_historical_prices h
        JOIN items i ON i.id = h.item_id
        WHERE i.is_tradeable = TRUE
          AND h.timestamp <= :end_ms
          AND h.buy_price IS NOT NULL
          AND h.sell_price IS NOT NULL
          AND h.buy_price > 0
          AND h.sell_price > 0
        ORDER BY h.item_id, h.timestamp DESC
        """
    )

    with client.engine.begin() as conn:
        df = pd.read_sql(query, conn, params={"end_ms": end_ms})

    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
    return df.reset_index(drop=True)


def fetch_tp_volatility_stats(
    connection: ConnectionInput,
    *,
    start_time: datetime,
    end_time: datetime,
    min_points: int = 30,
) -> pd.DataFrame:
    """
    Compute per-item volatility statistics over a time window.

    Volatility definition here is based on the standard deviation of mid-price
    returns, where mid = (buy_price + sell_price)/2.
    """

    client = _ensure_client(connection)
    start_utc = start_time if start_time.tzinfo is not None else start_time.replace(tzinfo=UTC)
    end_utc = end_time if end_time.tzinfo is not None else end_time.replace(tzinfo=UTC)
    start_ms = _datetime_to_epoch_ms(start_utc)
    end_ms = _datetime_to_epoch_ms(end_utc)
    min_points = max(int(min_points), 2)

    query = text(
        """
        WITH ordered AS (
            SELECT
                h.item_id,
                h.timestamp,
                ((h.buy_price + h.sell_price) / 2.0) AS mid_price,
                LAG(((h.buy_price + h.sell_price) / 2.0))
                    OVER (PARTITION BY h.item_id ORDER BY h.timestamp) AS mid_prev
            FROM gw2tp_historical_prices h
            JOIN items i ON i.id = h.item_id
            WHERE i.is_tradeable = TRUE
              AND h.timestamp BETWEEN :start_ms AND :end_ms
              AND h.buy_price IS NOT NULL
              AND h.sell_price IS NOT NULL
              AND h.buy_price > 0
              AND h.sell_price > 0
        )
        SELECT
            item_id,
            COUNT(*)::int AS points,
            AVG(mid_price) AS mid_avg,
            STDDEV_SAMP(mid_price) AS mid_std,
            STDDEV_SAMP((mid_price - mid_prev) / NULLIF(mid_prev, 0)) AS return_std
        FROM ordered
        WHERE mid_prev IS NOT NULL
        GROUP BY item_id
        HAVING COUNT(*) >= :min_points
        """
    )

    with client.engine.begin() as conn:
        df = pd.read_sql(
            query,
            conn,
            params={"start_ms": start_ms, "end_ms": end_ms, "min_points": min_points},
        )

    return df.reset_index(drop=True)


def build_tp_opportunity_table(connection: ConnectionInput, *, config: TpMarketScanConfig) -> pd.DataFrame:
    """
    Build a single per-item table combining:
    - snapshot prices at `config.as_of`
    - volatility stats over `[as_of - lookback, as_of]`
    - profit-after-fee metrics at the snapshot
    """

    normalized = config.normalized()
    start_time = normalized.as_of - normalized.lookback

    snapshot_df = fetch_tp_snapshot_as_of(connection, as_of=normalized.as_of)
    if snapshot_df.empty:
        return snapshot_df

    volatility_df = fetch_tp_volatility_stats(
        connection,
        start_time=start_time,
        end_time=normalized.as_of,
        min_points=normalized.min_points,
    )

    merged = snapshot_df.merge(volatility_df, how="left", on="item_id")

    merged["fee_rate"] = normalized.fee_rate
    merged["fee_copper"] = _ceil_fee_copper_series(merged["sell_price"], fee_rate=normalized.fee_rate)
    merged["net_sell_after_fee"] = (merged["sell_price"] - merged["fee_copper"]).clip(lower=0)
    merged["profit_after_fee"] = merged["net_sell_after_fee"] - merged["buy_price"]
    merged["roi_after_fee"] = merged["profit_after_fee"] / merged["buy_price"].where(merged["buy_price"] != 0, pd.NA)
    merged["is_profitable_after_fee"] = merged["profit_after_fee"] > 0
    merged["spread"] = merged["sell_price"] - merged["buy_price"]
    merged["mid_cv"] = merged["mid_std"] / merged["mid_avg"].where(merged["mid_avg"] != 0, pd.NA)

    numeric_cols = [
        "buy_price",
        "sell_price",
        "supply",
        "demand",
        "points",
        "mid_avg",
        "mid_std",
        "return_std",
        "fee_copper",
        "net_sell_after_fee",
        "profit_after_fee",
        "roi_after_fee",
        "spread",
        "mid_cv",
    ]
    for col in numeric_cols:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    return merged.reset_index(drop=True)


__all__ = [
    "TpMarketScanConfig",
    "fetch_tp_snapshot_as_of",
    "fetch_tp_volatility_stats",
    "build_tp_opportunity_table",
]


