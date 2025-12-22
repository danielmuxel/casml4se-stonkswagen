"""
Integration-style example for fetching live data and exporting CSV snapshots.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
import os

import pandas as pd
import pytest

from gw2ml import DatabaseClient, database


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "tests"


def _resolve_client() -> DatabaseClient:
    return DatabaseClient.from_env()


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

    days_back = int(os.getenv("TEST_DAYS_BACK", "7"))
    limit = int(os.getenv("TEST_LIMIT", "500"))
    history_limit = int(os.getenv("TEST_HISTORY_LIMIT", "5000"))
    hours_back = int(os.getenv("TEST_HOURS_BACK", "24"))
    now = datetime.now(UTC)
    range_start = now - timedelta(days=days_back)

    prices_df = database.get_prices(
        client,
        item_id=target_item_id,
        last_days=days_back,
        limit=limit,
        order="DESC",
    )
    if prices_df.empty:
        pytest.skip("No price data returned; adjust TEST_ITEM_ID/TEST_DAYS_BACK.")
    _write_csv(prices_df, output_dir / "prices.csv")

    prices_range_df = database.get_prices(
        client,
        item_id=target_item_id,
        start_time=range_start,
        end_time=now,
        limit=limit,
        order="DESC",
    )
    if not prices_range_df.empty:
        _write_csv(prices_range_df, output_dir / "prices_range.csv")

    prices_hours_df = database.get_prices(
        client,
        item_id=target_item_id,
        last_hours=hours_back,
        limit=limit,
        order="DESC",
    )
    if not prices_hours_df.empty:
        _write_csv(prices_hours_df, output_dir / "prices_last_hours.csv")

    bltc_df = database.get_bltc_history(
        client,
        item_id=target_item_id,
        last_days=days_back,
        limit=history_limit,
        order="DESC",
    )
    if bltc_df.empty:
        pytest.skip("No gw2bltc historical data; adjust parameters.")
    _write_csv(bltc_df, output_dir / "gw2bltc_historical_prices.csv")

    bltc_range_df = database.get_bltc_history(
        client,
        item_id=target_item_id,
        start_time=range_start,
        end_time=now,
        limit=history_limit,
        order="DESC",
    )
    if not bltc_range_df.empty:
        _write_csv(bltc_range_df, output_dir / "gw2bltc_historical_prices_range.csv")

    bltc_hours_df = database.get_bltc_history(
        client,
        item_id=target_item_id,
        last_hours=hours_back,
        limit=history_limit,
        order="DESC",
    )
    if not bltc_hours_df.empty:
        _write_csv(bltc_hours_df, output_dir / "gw2bltc_historical_prices_last_hours.csv")

    tp_df = database.get_tp_history(
        client,
        item_id=target_item_id,
        last_days=days_back,
        limit=history_limit,
        order="DESC",
    )
    if tp_df.empty:
        pytest.skip("No gw2tp historical data; adjust parameters.")
    _write_csv(tp_df, output_dir / "gw2tp_historical_prices.csv")

    tp_range_df = database.get_tp_history(
        client,
        item_id=target_item_id,
        start_time=range_start,
        end_time=now,
        limit=history_limit,
        order="DESC",
    )
    if not tp_range_df.empty:
        _write_csv(tp_range_df, output_dir / "gw2tp_historical_prices_range.csv")

    tp_hours_df = database.get_tp_history(
        client,
        item_id=target_item_id,
        last_hours=hours_back,
        limit=history_limit,
        order="DESC",
    )
    if not tp_hours_df.empty:
        _write_csv(tp_hours_df, output_dir / "gw2tp_historical_prices_last_hours.csv")


def test_fetch_prices_snapshot_and_export() -> None:
    client = _resolve_client()
    target_item_id = _resolve_item_id()

    recent_df = database.get_prices(
        client,
        item_id=target_item_id,
        last_hours=int(os.getenv("TEST_SNAPSHOT_HOURS_BACK", "48")),
        limit=1,
        order="DESC",
    )
    if recent_df.empty:
        pytest.skip("No recent prices available for snapshot test.")

    latest_row = recent_df.iloc[0]
    target_timestamp = latest_row.get("fetched_at") or latest_row.get("created_at")
    if pd.isna(target_timestamp):
        pytest.skip("Recent price row missing timestamp data.")

    target_dt = pd.to_datetime(target_timestamp).to_pydatetime()

    snapshot_df = database.get_prices_snapshot(
        client,
        target_time=target_dt,
        grace_minutes=int(os.getenv("TEST_SNAPSHOT_GRACE_MINUTES", "4")),
        item_ids=[target_item_id],
    )
    if snapshot_df.empty:
        pytest.skip("Snapshot query returned no rows; adjust grace window or item id.")

    assert snapshot_df["item_id"].nunique() == 1
    assert snapshot_df.iloc[0]["item_id"] == target_item_id

    bound = target_dt + timedelta(minutes=int(os.getenv("TEST_SNAPSHOT_GRACE_MINUTES", "4")))
    candidate_times = []
    if "fetched_at" in snapshot_df and not snapshot_df["fetched_at"].isna().all():
        candidate_times.append(snapshot_df["fetched_at"].max())
    if "created_at" in snapshot_df and not snapshot_df["created_at"].isna().all():
        candidate_times.append(snapshot_df["created_at"].max())
    if candidate_times:
        latest_time = max(pd.to_datetime(candidate_times).tolist())
        bound_ts = pd.Timestamp(bound)
        if bound_ts.tzinfo is None or bound_ts.tz is None:
            bound_ts = bound_ts.tz_localize(UTC)
        else:
            bound_ts = bound_ts.tz_convert(UTC)
        assert latest_time <= bound_ts

    snapshot_dir = _ensure_output_dir() / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    timestamp_slug = target_dt.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    snapshot_path = snapshot_dir / f"prices_snapshot_{timestamp_slug}.csv"
    snapshot_df.to_csv(snapshot_path, index=False)
    assert snapshot_path.exists()
    if snapshot_path.stat().st_size <= 0:
        pytest.skip("Snapshot export file empty.")


