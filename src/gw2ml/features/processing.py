"""Historical data helpers with consistent timestamp handling."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Optional

import pandas as pd

from .database import ConnectionInput, get_generic_rows

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
    connection: ConnectionInput,
    table_name: str,
    item_id: int,
    time_col: str = "timestamp",
    mode: str = "Days back",
    days_back: int = 7,
    limit: int = 1000,
) -> tuple[pd.DataFrame, str]:
    """Fetch historical rows with consistent timestamp handling."""

    eff_limit = int(limit) if mode == "Max rows" else 10000

    start_time: datetime | None = None
    if mode == "Days back" and days_back and days_back > 0:
        start_time = datetime.now(UTC) - timedelta(days=int(days_back))

    time_column_unit = "native"
    if table_name == "gw2bltc_historical_prices" and time_col == "timestamp":
        time_column_unit = "seconds"
    elif table_name == "gw2tp_historical_prices" and time_col == "timestamp":
        time_column_unit = "milliseconds"

    df = get_generic_rows(
        connection,
        table_name,
        item_id=int(item_id),
        start_time=start_time,
        time_column=time_col,
        limit=eff_limit,
        order="DESC",
        time_column_unit=time_column_unit,
    )

    df, visible_time_col = normalize_time_column(df, table_name, time_col)
    return df, visible_time_col

