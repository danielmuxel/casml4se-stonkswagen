"""Historical data helpers with consistent timestamp handling."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from .database import fetch_generic_df

__all__ = ["normalize_time_column", "fetch_historical_df"]


def normalize_time_column(
    df: pd.DataFrame,
    table_name: Optional[str],
    time_col: str,
) -> tuple[pd.DataFrame, str]:
    """Normalize a time column to timezone-aware datetime values."""
    if time_col not in df.columns:
        return df, time_col

    visible_col = time_col
    if table_name == "gw2bltc_historical_prices" and time_col == "timestamp":
        df["__ts"] = pd.to_datetime(df[time_col], unit="s", utc=True)
        visible_col = "__ts"
    elif table_name == "gw2tp_historical_prices" and time_col == "timestamp":
        df["__ts"] = pd.to_datetime(df[time_col], unit="ms", utc=True)
        visible_col = "__ts"
    else:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True)

    df = df.sort_values(visible_col)
    df = df[df[visible_col].notna()]
    return df, visible_col


def fetch_historical_df(
    connection_url: str,
    table_name: str,
    item_id: int,
    time_col: str = "timestamp",
    mode: str = "Days back",
    days_back: int = 7,
    limit: int = 1000,
) -> tuple[pd.DataFrame, str]:
    """Fetch historical rows with consistent timestamp handling."""
    where_sql = "item_id = :item_id"
    params: dict[str, object] = {"item_id": int(item_id)}
    eff_limit = int(limit) if mode == "Max rows" else 10000

    if mode == "Days back" and days_back and days_back > 0:
        start_time = datetime.utcnow() - timedelta(days=int(days_back))
        if table_name == "gw2bltc_historical_prices" and time_col == "timestamp":
            where_sql += " AND to_timestamp(timestamp) >= :start_time"
        elif table_name == "gw2tp_historical_prices" and time_col == "timestamp":
            where_sql += " AND to_timestamp(timestamp/1000.0) >= :start_time"
        else:
            where_sql += f" AND {time_col} >= :start_time"
        params["start_time"] = start_time

    order = f"{time_col} DESC"
    df = fetch_generic_df(connection_url, table_name, where_sql, params, order, eff_limit)
    df, visible_time_col = normalize_time_column(df, table_name, time_col)
    return df, visible_time_col

