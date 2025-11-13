"""
Database client utilities built on top of SQLAlchemy engines.

The helpers here keep engine management and connection resolution in one
place so the rest of the codebase can focus on query logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Optional

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


try:
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover
    st = None  # type: ignore


def resolve_connection_url(db_url_override: Optional[str] = None) -> Optional[str]:
    """
    Resolve a SQLAlchemy connection URL.

    The order of precedence is:
    1. Explicit override supplied via argument.
    2. Streamlit secrets (if running inside Streamlit).
    3. Environment variable `DB_URL` (dotenv supported).

    Returns
    -------
    Optional[str]
        A connection string compatible with SQLAlchemy or `None` if nothing can
        be resolved.
    """

    load_dotenv()

    candidates: list[str] = []

    if db_url_override and db_url_override.strip():
        candidates.append(db_url_override.strip())

    if st is not None:
        try:
            secret_url = st.secrets.get("DB_URL")  # type: ignore[attr-defined]
            if secret_url:
                candidates.append(str(secret_url))
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


@dataclass(slots=True)
class DatabaseClient:
    """
    Lightweight wrapper that manages a lazily created SQLAlchemy engine.
    """

    connection_url: str
    _engine: Engine | None = None

    @classmethod
    def from_env(cls, db_url_override: Optional[str] = None) -> "DatabaseClient":
        """
        Instantiate a client by pulling a connection string from environment
        sources.
        """

        connection_url = resolve_connection_url(db_url_override=db_url_override)
        if connection_url is None:
            message = "Unable to resolve database connection URL."
            raise ValueError(message)
        return cls(connection_url=connection_url)

    @property
    def engine(self) -> Engine:
        """
        Lazily construct and cache a SQLAlchemy engine.
        """

        if self._engine is None:
            self._engine = create_engine(self.connection_url)
        return self._engine


def _ensure_client(connection: "ConnectionInput") -> DatabaseClient:
    """
    Normalize inputs into a `DatabaseClient`.
    """

    if isinstance(connection, DatabaseClient):
        return connection

    if not isinstance(connection, str) or not connection.strip():
        message = "Connection input must be a connection string or DatabaseClient."
        raise ValueError(message)

    return DatabaseClient(connection_url=connection.strip())


ConnectionInput = str | DatabaseClient


__all__ = [
    "DatabaseClient",
    "ConnectionInput",
    "resolve_connection_url",
    "_ensure_client",
]


