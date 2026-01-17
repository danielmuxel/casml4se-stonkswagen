from __future__ import annotations

import streamlit as st

from app_header import render_page_header
from evaluation_app import render_evaluation_tab
from forecast_app import render_forecast_tab
from item_analysis_app import render_item_analysis_tab
from item_finder_app import render_item_finder_tab
from gw2ml.data.loaders import list_items
from gw2ml.utils import setup_logging, get_logger

logger = get_logger("app")

CACHE_TIMEOUT = 60 * 60 * 24 * 7  # 7 days


@st.cache_data(ttl=CACHE_TIMEOUT, persist="disk", show_spinner=True)
def cached_list_items(limit: int = 30000):
    return list_items(limit=limit)


def main() -> None:
    """Main entry point for the GW2ML Streamlit application with tabs."""
    setup_logging()
    logger.info("Starting GW2ML Streamlit application")
    
    st.set_page_config(page_title="Stonkswagen", layout="wide")
    
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["📈 Forecast", "🧪 Evaluation", "🔍 Item Analysis", "🔎 Item Finder"])

    page_title = "GW2ML"
    if selection == "📈 Forecast":
        page_title = "GW2ML: Forecast"
    elif selection == "🧪 Evaluation":
        page_title = "GW2ML: Evaluation & Backtest"
    elif selection == "🔍 Item Analysis":
        page_title = "GW2ML: Item Analysis"
    elif selection == "🔎 Item Finder":
        page_title = "GW2ML: Item Finder"
    render_page_header(page_title)

    items = cached_list_items()
    stored_ids = st.session_state.get("selected_item_ids", [])

    # Apply a programmatic selection (from Item Finder buttons).
    if st.session_state.get("apply_item_selection"):
        st.session_state["apply_item_selection"] = False
        if stored_ids:
            id_set = set(stored_ids)
            st.session_state["global_item_select"] = [
                item for item in items if str(item.get("item_id")) in id_set
            ]
        else:
            st.session_state["global_item_select"] = []

    # Build default items from stored IDs (only when widget state doesn't exist)
    default_items = []
    if stored_ids and "global_item_select" not in st.session_state:
        default_items = [item for item in items if str(item.get("item_id")) in stored_ids]

    selected_items = st.multiselect(
        "Search for items",
        options=items,
        default=default_items,
        format_func=lambda x: f"[{x['item_id']}] {x['item_name']}",
        key="global_item_select",
        help="This selection is shared across all tabs.",
    )
    st.session_state["selected_item_ids"] = [str(item["item_id"]) for item in selected_items]

    logger.info(f"Navigating to {selection}")
    if selection == "📈 Forecast":
        render_forecast_tab(selected_items=selected_items, show_header=False)
    elif selection == "🧪 Evaluation":
        render_evaluation_tab(selected_items=selected_items, show_header=False)
    elif selection == "🔍 Item Analysis":
        render_item_analysis_tab(selected_items=selected_items, show_header=False)
    elif selection == "🔎 Item Finder":
        render_item_finder_tab(selected_items=selected_items, show_header=False)


if __name__ == "__main__":
    main()
