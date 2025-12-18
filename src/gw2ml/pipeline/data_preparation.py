from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Iterable, Sequence

import pandas as pd

from gw2ml.data import (
    DatabaseClient,
    get_generic_rows,
    get_prices,
    get_prices_snapshot,
    resolve_connection_url,
)

DatasetMeta = dict[str, object]

PRICE_COLUMNS: Sequence[str] = (
    "id",
    "item_id",
    "whitelisted",
    "buy_quantity",
    "buy_unit_price",
    "sell_quantity",
    "sell_unit_price",
    "fetched_at",
    "created_at",
)


def _ensure_utc(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _empty_prices_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=PRICE_COLUMNS)


@dataclass(slots=True)
class PipelineContext:
    """Shared metadata for training and validation data pulls."""

    model_key: str
    run_date: datetime
    snapshot_id: str | None = None
    lookback_days: int = 30
    connection_url: str | None = None

    def __post_init__(self) -> None:
        self.run_date = _ensure_utc(self.run_date)


def build_pipeline_context(
    model_key: str,
    *,
    run_date: datetime | None = None,
    snapshot_id: str | None = None,
    lookback_days: int = 30,
    connection_url: str | None = None,
) -> PipelineContext:
    """Factory helper with reasonable defaults for pipeline callers."""
    context = PipelineContext(
        model_key=model_key,
        run_date=_ensure_utc(run_date),
        snapshot_id=snapshot_id,
        lookback_days=max(1, int(lookback_days)),
        connection_url=connection_url,
    )
    if context.connection_url is None:
        resolved = resolve_connection_url()
        if resolved is None:
            message = "Unable to resolve database connection URL for pipeline."
            raise ValueError(message)
        context.connection_url = resolved
    return context


def _init_client(context: PipelineContext) -> DatabaseClient:
    if not context.connection_url:
        message = "PipelineContext is missing a resolved connection URL."
        raise ValueError(message)
    return DatabaseClient(connection_url=context.connection_url)


def _dispose_client(client: DatabaseClient | None) -> None:
    if client is None:
        return
    engine = getattr(client, "_engine", None)
    if engine is not None:
        engine.dispose()


def _normalize_item_ids(item_ids: Iterable[int] | None) -> list[int]:
    if not item_ids:
        return []
    return sorted({int(item_id) for item_id in item_ids})


def load_training_snapshot(
    context: PipelineContext,
    *,
    item_ids: Iterable[int] | None = None,
    grace_minutes: int = 4,
) -> tuple[pd.DataFrame, DatasetMeta]:
    """Return the latest snapshot per item up to ``run_date``."""
    client = _init_client(context)
    try:
        frame = get_prices_snapshot(
            client,
            target_time=context.run_date,
            grace_minutes=int(grace_minutes),
            item_ids=_normalize_item_ids(item_ids) or None,
        )
    finally:
        _dispose_client(client)

    if frame.empty:
        message = "Training snapshot query returned zero rows."
        raise RuntimeError(message)

    metadata: DatasetMeta = {
        "snapshot_time": context.run_date.isoformat(),
        "grace_minutes": int(grace_minutes),
        "row_count": int(len(frame)),
        "item_count": int(frame["item_id"].nunique()),
    }
    return frame, metadata


def load_latest_validation(
    context: PipelineContext,
    *,
    item_ids: Iterable[int] | None = None,
    lookback_days: int | None = None,
    rows_per_item: int = 2000,
    max_rows: int | None = 10000,
) -> tuple[pd.DataFrame, DatasetMeta]:
    """Fetch a rolling validation slice ending at ``run_date``."""
    effective_days = lookback_days or context.lookback_days
    window_end = context.run_date
    window_start = window_end - timedelta(days=max(1, int(effective_days)))
    client = _init_client(context)
    try:
        normalized_ids = _normalize_item_ids(item_ids)
        if normalized_ids:
            frames: list[pd.DataFrame] = []
            for item_id in normalized_ids:
                subset = get_prices(
                    client,
                    item_id=item_id,
                    start_time=window_start,
                    end_time=window_end,
                    limit=int(rows_per_item),
                    order="ASC",
                )
                if not subset.empty:
                    frames.append(subset)
            frame = pd.concat(frames, ignore_index=True) if frames else _empty_prices_frame()
        else:
            frame = get_generic_rows(
                client,
                "prices",
                start_time=window_start,
                end_time=window_end,
                limit=max_rows,
                order="ASC",
                time_column="fetched_at",
            )
    finally:
        _dispose_client(client)

    if frame.empty:
        message = "Validation window query returned zero rows."
        raise RuntimeError(message)

    metadata: DatasetMeta = {
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "row_count": int(len(frame)),
        "item_count": int(frame["item_id"].nunique()) if "item_id" in frame.columns else 0,
    }
    return frame, metadata


__all__ = [
    "DatasetMeta",
    "PipelineContext",
    "build_pipeline_context",
    "load_latest_validation",
    "load_training_snapshot",
]



