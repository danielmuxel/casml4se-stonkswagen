from __future__ import annotations

import streamlit as st

from evaluation_app import render_evaluation_tab
from forecast_app import render_forecast_tab
from item_analysis_app import render_item_analysis_tab
from gw2ml.utils import setup_logging, get_logger

logger = get_logger("app")


def main() -> None:
    """Main entry point for the GW2ML Streamlit application with tabs."""
    setup_logging()
    logger.info("Starting GW2ML Streamlit application")
    
    st.set_page_config(page_title="GW2ML Application", layout="wide")
    
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["📈 Forecast", "🧪 Evaluation", "🔍 Item Analysis"])

    st.header("🎮 GW2ML: Guild Wars 2 Trading Post Analysis")
    
    logger.info(f"Navigating to {selection}")
    if selection == "📈 Forecast":
        render_forecast_tab()
    elif selection == "🧪 Evaluation":
        render_evaluation_tab()
    elif selection == "🔍 Item Analysis":
        render_item_analysis_tab()


if __name__ == "__main__":
    main()
