"""
Integration-style example for fetching live data and exporting CSV snapshots.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
import os

import pandas as pd
import pytest

from ml.common import DatabaseClient, database


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "tests"


def _resolve_client() -> DatabaseClient:
    try:
        return DatabaseClient.from_env()
    except ValueError as exc:
        pytest.skip(str(exc))


def _resolve_item_id() -> int:
    env_value = os.getenv("TEST_ITEM_ID")
    if env_value and env_value.isdigit():
        return int(env_value)
    return 19702


def _ensure_output_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def _write_csv(dataframe: pd.DataFrame, output_path: Path) -> None:
    dataframe.to_csv(output_path, index=False)
    assert output_path.exists(), f"Expected CSV to be written at {output_path}"
    if output_path.stat().st_size <= 0:
        pytest.skip(f"No data exported to {output_path.name}.")


def test_fetch_and_export_real_tables() -> None:
    client = _resolve_client()
    target_item_id = _resolve_item_id()
    output_dir = _ensure_output_dir()

    end_time = datetime.now(UTC)
    start_time = end_time - timedelta(days=int(os.getenv("TEST_DAYS_BACK", "7")))

    prices_df = database.get_prices(
        client,
        item_id=target_item_id,
        start_time=start_time,
        end_time=end_time,
        limit=int(os.getenv("TEST_LIMIT", "500")),
    )
    if prices_df.empty:
        pytest.skip("No price data returned; adjust TEST_ITEM_ID/TEST_DAYS_BACK.")
    _write_csv(prices_df, output_dir / "prices.csv")

    bltc_df = database.get_generic_rows(
        client,
        "gw2bltc_historical_prices",
        item_id=target_item_id,
        start_time=start_time,
        end_time=end_time,
        time_column="timestamp",
        time_column_unit="seconds",
        limit=int(os.getenv("TEST_LIMIT", "5000")),
        order="DESC",
    )
    if bltc_df.empty:
        pytest.skip("No gw2bltc historical data; adjust parameters.")
    _write_csv(bltc_df, output_dir / "gw2bltc_historical_prices.csv")

    tp_df = database.get_generic_rows(
        client,
        "gw2tp_historical_prices",
        item_id=target_item_id,
        start_time=start_time,
        end_time=end_time,
        time_column="timestamp",
        time_column_unit="milliseconds",
        limit=int(os.getenv("TEST_LIMIT", "5000")),
        order="DESC",
    )
    if tp_df.empty:
        pytest.skip("No gw2tp historical data; adjust parameters.")
    _write_csv(tp_df, output_dir / "gw2tp_historical_prices.csv")


