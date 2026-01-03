"""
Database access utilities shared across ML components.

This module exposes a lightweight `DatabaseClient` wrapper along with helper
functions stored in sibling modules for convenient access patterns.

High-level loading utilities:
- load_gw2_series: Load price data as TimeSeries
- load_gw2_series_batch: Load multiple items at once
- load_and_split: Load and split by ratio
- load_and_split_days: Load and split by days
"""

from .database_client import DatabaseClient, ConnectionInput, resolve_connection_url
from .database_queries import (
    get_bltc_history,
    get_generic_rows,
    get_item_count,
    get_items,
    get_prices,
    get_prices_snapshot,
    get_tp_history,
    list_columns,
)
from .loaders import (
    GW2Series,
    load_gw2_series,
    load_gw2_series_batch,
    load_and_split,
    load_and_split_days,
    clear_cache,
    get_cache_info,
)

__all__ = [
    # Database client
    "DatabaseClient",
    "ConnectionInput",
    "resolve_connection_url",
    # Low-level queries
    "get_prices",
    "get_bltc_history",
    "get_prices_snapshot",
    "get_tp_history",
    "get_items",
    "get_item_count",
    "get_generic_rows",
    "list_columns",
    # High-level loaders
    "GW2Series",
    "load_gw2_series",
    "load_gw2_series_batch",
    "load_and_split",
    "load_and_split_days",
    "clear_cache",
    "get_cache_info",
]


