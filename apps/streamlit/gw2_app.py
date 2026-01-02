from __future__ import annotations

import streamlit as st

from forecast_app import render_forecast_tab
from item_analysis_app import render_item_analysis_tab


def main() -> None:
    """Main entry point for the GW2ML Streamlit application with tabs."""
    st.set_page_config(page_title="GW2ML Application", layout="wide")
    
    st.header("🎮 GW2ML: Guild Wars 2 Trading Post Analysis")
    
    # Create tabs
    tab1, tab2 = st.tabs(["📈 Forecast", "🔍 Item Analysis"])
    
    with tab1:
        render_forecast_tab()
    
    with tab2:
        render_item_analysis_tab()


if __name__ == "__main__":
    main()
