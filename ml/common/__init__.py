"""Shared ML utilities grouped by concern for the Stonkswagen project."""

from . import database
from .database import DatabaseClient, ConnectionInput, resolve_connection_url
from .historical import normalize_time_column, fetch_historical_df
from .plotting import plot_prices, plot_quantity
from .query_params import (
    get_query_params,
    get_query_param,
    update_query_params,
    switch_page_with_params,
    remove_query_params,
    apply_pending_query_params,
)

__all__ = [
    "database",
    "DatabaseClient",
    "ConnectionInput",
    "resolve_connection_url",
    "normalize_time_column",
    "fetch_historical_df",
    "plot_prices",
    "plot_quantity",
    "get_query_params",
    "get_query_param",
    "update_query_params",
    "switch_page_with_params",
    "remove_query_params",
    "apply_pending_query_params",
]


