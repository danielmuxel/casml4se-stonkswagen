from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from gw2ml.data import DatabaseClient, get_items, get_prices
from gw2ml.paths import PROJECT_ROOT


# Populate this list with item ids you want to fetch.
# Leave it empty (or comment out entries) to export every tradeable item.
TARGET_ITEM_IDS: list[int] = [
    # 19702,
    # 19721,
]

# Adjust the number of parallel workers. Increase cautiously to avoid overloading the DB.
MAX_WORKERS = 8

TRAIN_DIR = PROJECT_ROOT / "data" / "train_arima"


def _ensure_output_dir() -> Path:
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    return TRAIN_DIR


def _build_output_path(item_id: int, timestamp: datetime) -> Path:
    slug = timestamp.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    filename = f"prices_{item_id}_{slug}.csv"
    return _ensure_output_dir() / filename


def _fetch_full_price_history(client: DatabaseClient, item_id: int) -> pd.DataFrame:
    df = get_prices(
        client,
        item_id=item_id,
        order="ASC",
    )
    if df.empty:
        message = f"No price records found for item_id={item_id}."
        raise RuntimeError(message)
    return df


def _fetch_all_tradeable_item_ids(client: DatabaseClient, page_size: int = 1000) -> list[int]:
    collected: list[int] = []
    offset = 0
    while True:
        items_df = get_items(client, limit=page_size, offset=offset)
        if items_df.empty:
            break
        collected.extend(items_df["id"].astype(int).tolist())
        offset += page_size
    return sorted(set(collected))


def _resolve_item_ids(client: DatabaseClient, overrides: Iterable[int] | None) -> list[int]:
    if overrides:
        return sorted({int(item_id) for item_id in overrides})
    return _fetch_all_tradeable_item_ids(client)


def _dispose_client(client: DatabaseClient) -> None:
    engine = getattr(client, "_engine", None)
    if engine is not None:
        engine.dispose()


def _export_single_item(connection_url: str, item_id: int, snapshot_time: datetime) -> Path:
    client = DatabaseClient(connection_url=connection_url)
    try:
        df = _fetch_full_price_history(client, item_id)
        output_path = _build_output_path(item_id, snapshot_time)
        df.to_csv(output_path, index=False)
        return output_path
    finally:
        _dispose_client(client)


def run(item_ids: Iterable[int] | None = None, max_workers: int | None = None) -> list[Path]:
    primary_client = DatabaseClient.from_env()
    snapshot_time = datetime.now(UTC)
    ids = _resolve_item_ids(primary_client, item_ids or TARGET_ITEM_IDS)
    if not ids:
        raise RuntimeError("No item ids resolved; populate TARGET_ITEM_IDS or ensure the database has tradeable items.")

    connection_url = primary_client.connection_url
    worker_count = max(1, max_workers or MAX_WORKERS)

    outputs: list[Path] = []
    executor = ThreadPoolExecutor(max_workers=worker_count)
    futures = {executor.submit(_export_single_item, connection_url, item_id, snapshot_time): item_id for item_id in ids}
    try:
        for future in as_completed(futures):
            item_id = futures[future]
            try:
                path = future.result()
                outputs.append(path)
            except Exception as exc:
                message = f"Failed to export item_id={item_id}: {exc}"
                raise RuntimeError(message) from exc
    except KeyboardInterrupt:
        for future in futures:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        _dispose_client(primary_client)

    return sorted(outputs, key=lambda p: p.name)


if __name__ == "__main__":
    export_paths = run()
    for path in export_paths:
        print(f"Exported ARIMA training snapshot to {path}")

