"""
Database access utilities shared across ML components.

This module exposes a lightweight `DatabaseClient` wrapper along with helper
functions stored in sibling modules for convenient access patterns.
"""

from .client import DatabaseClient, ConnectionInput, resolve_connection_url
from .queries import (
    get_bltc_history,
    get_generic_rows,
    get_item_count,
    get_items,
    get_prices,
    get_tp_history,
    list_columns,
)

__all__ = [
    "DatabaseClient",
    "ConnectionInput",
    "resolve_connection_url",
    "get_prices",
    "get_bltc_history",
    "get_tp_history",
    "get_items",
    "get_item_count",
    "get_generic_rows",
    "list_columns",
]


