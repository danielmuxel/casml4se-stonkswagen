from __future__ import annotations

from html import escape

import streamlit as st

_BRAND_TEXT = "🎮 Stonkswagen: Guild Wars 2 Trading Post Forecasting"
_STYLE_STATE_KEY = "_app_header_styles_injected"


def _inject_header_styles() -> None:
    if st.session_state.get(_STYLE_STATE_KEY):
        return

    st.markdown(
        """
<style>
.app-header {
  margin: 0 0 0.75rem 0;
}
.app-header .app-brand {
  font-size: 1.05rem;
  font-weight: 600;
  line-height: 1.2;
  margin: 0;
}
.app-header .app-page {
  font-size: 1.6rem;
  font-weight: 700;
  line-height: 1.2;
  margin: 0.15rem 0 0 0;
}
</style>
        """,
        unsafe_allow_html=True,
    )
    st.session_state[_STYLE_STATE_KEY] = True


def render_page_header(page_title: str) -> None:
    _inject_header_styles()
    safe_title = escape(page_title)
    st.markdown(
        f"""
<div class="app-header">
  <div class="app-brand">{_BRAND_TEXT}</div>
  <div class="app-page">{safe_title}</div>
</div>
        """,
        unsafe_allow_html=True,
    )
