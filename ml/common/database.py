"""Database-related helpers for accessing GW2 trading post data."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, UTC
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import bindparam, create_engine, text


__all__ = [
    "resolve_connection_url",
    "fetch_prices_df",
    "fetch_items_df",
    "count_items",
    "fetch_generic_df",
    "list_columns",
]


def resolve_connection_url(db_url_override: Optional[str] = None) -> Optional[str]:
    """Resolve DB URL from override, Streamlit secrets (if available), or env."""
    load_dotenv()

    candidates: list[str] = []
    if db_url_override and db_url_override.strip():
        candidates.append(db_url_override.strip())
    try:
        import streamlit as st  # type: ignore

        if "DB_URL" in st.secrets and st.secrets["DB_URL"]:
            candidates.append(str(st.secrets["DB_URL"]))
    except Exception:
        pass

    env_url = os.getenv("DB_URL")
    if env_url:
        candidates.append(env_url)

    for url in candidates:
        if url.startswith("postgres://"):
            return "postgresql+psycopg://" + url[len("postgres://") :]
        if url.startswith("postgresql://"):
            return "postgresql+psycopg://" + url[len("postgresql://") :]
        return url
    return None


def fetch_prices_df(
    connection_url: str,
    item_id: int,
    limit: int,
    days_back: Optional[int],
) -> pd.DataFrame:
    """Fetch recent price rows for a single item."""
    engine = create_engine(connection_url)
    base_query = """
        SELECT id, item_id, whitelisted, buy_quantity, buy_unit_price,
               sell_quantity, sell_unit_price, fetched_at, created_at
        FROM prices
        WHERE item_id = :item_id
    """
    params: dict[str, object] = {"item_id": item_id}
    if days_back is not None and days_back > 0:
        base_query += " AND fetched_at >= :start_time"
        params["start_time"] = datetime.now(UTC) - timedelta(days=days_back)
    base_query += " ORDER BY fetched_at DESC LIMIT :limit"
    params["limit"] = int(limit)
    with engine.begin() as conn:
        df = pd.read_sql(text(base_query), conn, params=params)
    if "fetched_at" in df.columns:
        df["fetched_at"] = pd.to_datetime(df["fetched_at"], errors="coerce")
        df = df.sort_values("fetched_at")
        df = df[df["fetched_at"].notna()]
    return df


def fetch_items_df(
    connection_url: str,
    search: str,
    limit: int,
    offset: int = 0,
    only_favorites_ids: Optional[list[int]] = None,
) -> pd.DataFrame:
    """Fetch tradeable item rows filtered by search or favorites."""
    engine = create_engine(connection_url)
    with engine.begin() as conn:
        if only_favorites_ids is not None:
            if not only_favorites_ids:
                return pd.DataFrame(
                    columns=[
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
                )
            query = text(
                """
                SELECT id, chat_link, name, icon, description, type, rarity, level,
                       vendor_value, default_skin, flags, game_types, restrictions,
                       is_tradeable, upgrades_into, upgrades_from, details,
                       created_at, updated_at
                FROM items
                WHERE id IN :ids
                ORDER BY name ASC
                LIMIT :limit OFFSET :offset
                """
            ).bindparams(bindparam("ids", expanding=True))
            df = pd.read_sql(
                query,
                conn,
                params={
                    "ids": list(only_favorites_ids),
                    "limit": int(limit),
                    "offset": int(offset),
                },
            )
            return df

        like = f"%{search.strip()}%" if search else "%"
        query = text(
            """
            SELECT id, chat_link, name, icon, description, type, rarity, level,
                   vendor_value, default_skin, flags, game_types, restrictions,
                   is_tradeable, upgrades_into, upgrades_from, details,
                   created_at, updated_at
            FROM items
            WHERE is_tradeable = TRUE
              AND (name ILIKE :like OR CAST(id AS TEXT) ILIKE :like)
            ORDER BY name ASC
            LIMIT :limit OFFSET :offset
            """
        )
        df = pd.read_sql(
            query,
            conn,
            params={"like": like, "limit": int(limit), "offset": int(offset)},
        )
        return df


def count_items(
    connection_url: str,
    search: str,
    only_favorites_ids: Optional[list[int]] = None,
) -> int:
    """Count tradeable items that match the search filter."""
    engine = create_engine(connection_url)
    with engine.begin() as conn:
        if only_favorites_ids is not None:
            if not only_favorites_ids:
                return 0
            query = text(
                """
                SELECT COUNT(*)::int AS count
                FROM items
                WHERE id IN :ids
                """
            ).bindparams(bindparam("ids", expanding=True))
            df = pd.read_sql(query, conn, params={"ids": list(only_favorites_ids)})
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


def list_columns(connection_url: str, table_name: str) -> list[str]:
    """List column names for the given table."""
    engine = create_engine(connection_url)
    with engine.begin() as conn:
        query = text("SELECT * FROM " + table_name + " LIMIT 0")
        result = conn.execute(query)
        return list(result.keys())


def fetch_generic_df(
    connection_url: str,
    table_name: str,
    where_sql: str | None,
    params: dict[str, object] | None,
    order_by: str | None,
    limit: int,
) -> pd.DataFrame:
    """Fetch arbitrary table rows with the provided filters."""
    engine = create_engine(connection_url)
    sql = f"SELECT * FROM {table_name}"
    if where_sql:
        sql += f" WHERE {where_sql}"
    if order_by:
        sql += f" ORDER BY {order_by}"
    if limit:
        sql += " LIMIT :limit"
        params = {**(params or {}), "limit": int(limit)}
    with engine.begin() as conn:
        df = pd.read_sql(text(sql), conn, params=params)
    return df

