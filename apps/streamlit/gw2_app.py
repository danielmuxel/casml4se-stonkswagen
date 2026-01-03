from __future__ import annotations

import streamlit as st

from forecast_app import render_forecast_tab
from item_analysis_app import render_item_analysis_tab
from pretrained_models_app import render_pretrained_models_tab


def main() -> None:
    """Main entry point for the GW2ML Streamlit application with tabs."""
    st.set_page_config(page_title="GW2ML Application", layout="wide")
    
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["📈 Forecast", "🔍 Item Analysis", "🤖 Pretrained Models"])

    st.header("🎮 GW2ML: Guild Wars 2 Trading Post Analysis")
    
    if selection == "📈 Forecast":
        render_forecast_tab()
    elif selection == "🔍 Item Analysis":
        render_item_analysis_tab()
    elif selection == "🤖 Pretrained Models":
        render_pretrained_models_tab()


if __name__ == "__main__":
    main()
