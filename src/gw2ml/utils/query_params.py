"""Streamlit query-parameter helpers for multi-page navigation."""

from __future__ import annotations

__all__ = [
    "get_query_params",
    "get_query_param",
    "update_query_params",
    "switch_page_with_params",
    "remove_query_params",
    "apply_pending_query_params",
]


def get_query_params() -> dict[str, list[str]]:
    """Return current query params as dict[str, list[str]]."""
    try:
        import streamlit as st  # type: ignore

        qp = st.query_params  # type: ignore[attr-defined]
        norm: dict[str, list[str]] = {}
        for key, value in qp.items():
            if isinstance(value, list):
                norm[key] = [str(item) for item in value]
            else:
                norm[key] = [str(value)]
        return norm
    except Exception:
        try:
            import streamlit as st  # type: ignore

            return {key: [str(item) for item in value] for key, value in st.experimental_get_query_params().items()}
        except Exception:
            return {}


def get_query_param(key: str, default: str | None = None) -> str | None:
    """Return the first value for a query param key or the provided default."""
    params = get_query_params()
    return (params.get(key, [default]) or [default])[0]


def update_query_params(params: dict[str, str]) -> None:
    """Update query params in the URL without clearing others."""
    try:
        import streamlit as st  # type: ignore

        qp = st.query_params  # type: ignore[attr-defined]
        for key, value in params.items():
            qp[key] = value
    except Exception:
        try:
            import streamlit as st  # type: ignore

            current = st.experimental_get_query_params()
            current.update({key: [value] for key, value in params.items()})
            st.experimental_set_query_params(**current)
        except Exception:
            return


def switch_page_with_params(page_path: str, params: dict[str, str]) -> None:
    """Carry query params across page switches reliably."""
    try:
        import streamlit as st  # type: ignore

        st.session_state["__pending_qp"] = {key: str(value) for key, value in params.items()}
    except Exception:
        pass
    update_query_params(params)
    try:
        import streamlit as st  # type: ignore

        st.switch_page(page_path)
    except Exception:
        try:
            import streamlit as st  # type: ignore

            st.rerun()
        except Exception:
            return


def remove_query_params(keys: list[str]) -> None:
    """Remove specific keys from query params if present."""
    try:
        import streamlit as st  # type: ignore

        qp = st.query_params  # type: ignore[attr-defined]
        for key in keys:
            try:
                del qp[key]
            except Exception:
                pass
    except Exception:
        try:
            import streamlit as st  # type: ignore

            current = st.experimental_get_query_params()
            for key in keys:
                current.pop(key, None)
            st.experimental_set_query_params(**current)
        except Exception:
            return


def apply_pending_query_params() -> None:
    """If previous page left pending query params in session, apply and clear them."""
    try:
        import streamlit as st  # type: ignore

        pending = st.session_state.pop("__pending_qp", None)
        if isinstance(pending, dict) and pending:
            update_query_params({key: str(value) for key, value in pending.items()})
    except Exception:
        return

