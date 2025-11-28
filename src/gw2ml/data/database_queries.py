"""
High-level query helpers that operate on top of `DatabaseClient`.

The helpers accept either a ready-to-use `DatabaseClient` instance or a raw
connection URL. That makes them convenient in sequential scripts while still
allowing dependency injection in larger applications.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Iterable, Literal, Sequence

import pandas as pd
from sqlalchemy import bindparam, text  # type: ignore[import]

from .database_client import ConnectionInput, _ensure_client


ITEM_COLUMNS: list[str] = [
    "id",
    "chat_link",
    "name",
    "icon",
    "description",
    "type",
    "rarity",
    "level",
    "vendor_value",
    "default_skin",
    "flags",
    "game_types",
    "restrictions",
    "is_tradeable",
    "upgrades_into",
    "upgrades_from",
    "details",
    "created_at",
    "updated_at",
]


def _normalize_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _build_time_filters(
    params: dict[str, object],
    column: str,
    start_time: datetime | None,
    end_time: datetime | None,
    unit: Literal["native", "seconds", "milliseconds"],
) -> list[str]:
    clauses: list[str] = []
    column_expr = column
    if unit == "seconds":
        column_expr = f"to_timestamp({column})"
    elif unit == "milliseconds":
        column_expr = f"to_timestamp({column}/1000.0)"

    if start_time is not None:
        params["start_time"] = _normalize_datetime(start_time)
        clauses.append(f"{column_expr} >= :start_time")
    if end_time is not None:
        params["end_time"] = _normalize_datetime(end_time)
        clauses.append(f"{column_expr} <= :end_time")
    return clauses


def _ensure_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _resolve_time_bounds(
    *,
    start_time: datetime | None,
    end_time: datetime | None,
    last_days: int | None,
    last_hours: int | None,
    last_minutes: int | None,
) -> tuple[datetime | None, datetime | None]:
    explicit_start = _ensure_utc(start_time)
    explicit_end = _ensure_utc(end_time)

    if explicit_start or explicit_end:
        return explicit_start, explicit_end

    params = {
        "days": last_days or 0,
        "hours": last_hours or 0,
        "minutes": last_minutes or 0,
    }
    if sum(params.values()) <= 0:
        return None, None

    end_dt = datetime.now(UTC)
    delta = timedelta(days=params["days"], hours=params["hours"], minutes=params["minutes"])
    return end_dt - delta, end_dt


def get_prices(
    connection: ConnectionInput,
    item_id: int,
    *,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    last_days: int | None = None,
    last_hours: int | None = None,
    last_minutes: int | None = None,
    limit: int | None = None,
    order: Literal["ASC", "DESC"] = "ASC",
) -> pd.DataFrame:
    """
    Retrieve price rows for a single item id within an optional time range.
    """

    client = _ensure_client(connection)

    params: dict[str, object] = {"item_id": int(item_id)}
    clauses = ["item_id = :item_id"]
    resolved_start, resolved_end = _resolve_time_bounds(
        start_time=start_time,
        end_time=end_time,
        last_days=last_days,
        last_hours=last_hours,
        last_minutes=last_minutes,
    )
    clauses.extend(_build_time_filters(params, "fetched_at", resolved_start, resolved_end, "native"))

    sql_lines = [
        "SELECT id, item_id, whitelisted, buy_quantity, buy_unit_price,",
        "       sell_quantity, sell_unit_price, fetched_at, created_at",
        "FROM prices",
        "WHERE " + " AND ".join(clauses),
        f"ORDER BY fetched_at {order}",
    ]
    if limit is not None:
        sql_lines.append("LIMIT :limit")
        params["limit"] = int(limit)

    with client.engine.begin() as conn:
        df = pd.read_sql(text("\n".join(sql_lines)), conn, params=params)

    if "fetched_at" in df.columns:
        df["fetched_at"] = pd.to_datetime(df["fetched_at"], errors="coerce", utc=True)
        df = df[df["fetched_at"].notna()]
    return df.reset_index(drop=True)


def get_items(
    connection: ConnectionInput,
    search: str | None = None,
    item_ids: Iterable[int] | None = None,
    *,
    limit: int = 100,
    offset: int = 0,
) -> pd.DataFrame:
    """
    Retrieve item records either by explicit id list or fuzzy name search.
    """

    client = _ensure_client(connection)

    with client.engine.begin() as conn:
        if item_ids is not None:
            ids = [int(item_id) for item_id in item_ids]
            if not ids:
                return pd.DataFrame(columns=ITEM_COLUMNS)

            query = (
                text(
                    """
                    SELECT {columns}
                    FROM items
                    WHERE id IN :ids
                    ORDER BY name ASC
                    LIMIT :limit OFFSET :offset
                    """.format(
                        columns=", ".join(ITEM_COLUMNS),
                    )
                ).bindparams(bindparam("ids", expanding=True))
            )

            params = {"ids": ids, "limit": int(limit), "offset": int(offset)}
            return pd.read_sql(query, conn, params=params)

        like = f"%{search.strip()}%" if search else "%"
        query = text(
            """
            SELECT {columns}
            FROM items
            WHERE is_tradeable = TRUE
              AND (name ILIKE :like OR CAST(id AS TEXT) ILIKE :like)
            ORDER BY name ASC
            LIMIT :limit OFFSET :offset
            """.format(
                columns=", ".join(ITEM_COLUMNS),
            )
        )
        params = {"like": like, "limit": int(limit), "offset": int(offset)}
        return pd.read_sql(query, conn, params=params)


def get_item_count(
    connection: ConnectionInput,
    search: str | None = None,
    item_ids: Iterable[int] | None = None,
) -> int:
    """
    Count items matching the provided filters.
    """

    client = _ensure_client(connection)

    with client.engine.begin() as conn:
        if item_ids is not None:
            ids = [int(item_id) for item_id in item_ids]
            if not ids:
                return 0
            query = text(
                """
                SELECT COUNT(*)::int AS count
                FROM items
                WHERE id IN :ids
                """
            ).bindparams(bindparam("ids", expanding=True))
            df = pd.read_sql(query, conn, params={"ids": ids})
            return int(df.iloc[0]["count"]) if not df.empty else 0

        like = f"%{search.strip()}%" if search else "%"
        query = text(
            """
            SELECT COUNT(*)::int AS count
            FROM items
            WHERE is_tradeable = TRUE
              AND (name ILIKE :like OR CAST(id AS TEXT) ILIKE :like)
            """
        )
        df = pd.read_sql(query, conn, params={"like": like})
        return int(df.iloc[0]["count"]) if not df.empty else 0


def get_generic_rows(
    connection: ConnectionInput,
    table: str,
    *,
    columns: Sequence[str] | None = None,
    item_id: int | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    last_days: int | None = None,
    last_hours: int | None = None,
    last_minutes: int | None = None,
    time_column: str = "timestamp",
    limit: int | None = None,
    order: str = "ASC",
    time_column_unit: Literal["native", "seconds", "milliseconds"] = "native",
) -> pd.DataFrame:
    """
    Fetch arbitrary table rows with optional item and time filters.
    """

    client = _ensure_client(connection)

    select_columns = ", ".join(columns) if columns else "*"
    sql_lines = [f"SELECT {select_columns} FROM {table}"]
    params: dict[str, object] = {}

    clauses: list[str] = []
    if item_id is not None:
        clauses.append("item_id = :item_id")
        params["item_id"] = int(item_id)

    resolved_start, resolved_end = _resolve_time_bounds(
        start_time=start_time,
        end_time=end_time,
        last_days=last_days,
        last_hours=last_hours,
        last_minutes=last_minutes,
    )
    clauses.extend(_build_time_filters(params, time_column, resolved_start, resolved_end, time_column_unit))
    if clauses:
        sql_lines.append("WHERE " + " AND ".join(clauses))

    order_direction = order.upper()
    if time_column and order_direction in {"ASC", "DESC"}:
        sql_lines.append(f"ORDER BY {time_column} {order_direction}")

    if limit is not None:
        sql_lines.append("LIMIT :limit")
        params["limit"] = int(limit)

    with client.engine.begin() as conn:
        df = pd.read_sql(text("\n".join(sql_lines)), conn, params=params or None)

    if time_column and time_column in df.columns:
        if time_column_unit == "native":
            df[time_column] = pd.to_datetime(df[time_column], errors="coerce", utc=True)
        elif time_column_unit == "seconds":
            df[time_column] = pd.to_datetime(df[time_column], unit="s", errors="coerce", utc=True)
        elif time_column_unit == "milliseconds":
            df[time_column] = pd.to_datetime(df[time_column], unit="ms", errors="coerce", utc=True)
        df = df[df[time_column].notna()]

    return df.reset_index(drop=True)


def list_columns(connection: ConnectionInput, table: str) -> list[str]:
    """
    Return column names for the requested table.
    """

    client = _ensure_client(connection)

    with client.engine.begin() as conn:
        result = conn.execute(text(f"SELECT * FROM {table} LIMIT 0"))
        return list(result.keys())


def get_bltc_history(
    connection: ConnectionInput,
    item_id: int,
    *,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    last_days: int | None = None,
    last_hours: int | None = None,
    last_minutes: int | None = None,
    limit: int | None = None,
    order: Literal["ASC", "DESC"] = "DESC",
) -> pd.DataFrame:
    """
    Convenience wrapper for `gw2bltc_historical_prices`.
    """

    return get_generic_rows(
        connection,
        "gw2bltc_historical_prices",
        item_id=item_id,
        start_time=start_time,
        end_time=end_time,
        last_days=last_days,
        last_hours=last_hours,
        last_minutes=last_minutes,
        limit=limit,
        order=order,
        time_column="timestamp",
        time_column_unit="seconds",
    )


def get_tp_history(
    connection: ConnectionInput,
    item_id: int,
    *,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    last_days: int | None = None,
    last_hours: int | None = None,
    last_minutes: int | None = None,
    limit: int | None = None,
    order: Literal["ASC", "DESC"] = "DESC",
) -> pd.DataFrame:
    """
    Convenience wrapper for `gw2tp_historical_prices`.
    """

    return get_generic_rows(
        connection,
        "gw2tp_historical_prices",
        item_id=item_id,
        start_time=start_time,
        end_time=end_time,
        last_days=last_days,
        last_hours=last_hours,
        last_minutes=last_minutes,
        limit=limit,
        order=order,
        time_column="timestamp",
        time_column_unit="milliseconds",
    )


__all__ = [
    "get_prices",
    "get_items",
    "get_item_count",
    "get_generic_rows",
    "list_columns",
    "get_bltc_history",
    "get_tp_history",
]


