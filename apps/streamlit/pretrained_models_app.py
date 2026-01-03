from __future__ import annotations

import streamlit as st
import pandas as pd
from gw2ml.data.loaders import list_items

CACHE_TIMEOUT = 60 * 60 * 24 * 7  # 7 days in seconds, TODO: if using disk, its possible this is ignored?

@st.cache_data(ttl=CACHE_TIMEOUT, persist="disk", show_spinner=True, show_time=True)
def cached_list_items(limit: int = 30000):
    return list_items(limit=limit)

def render_pretrained_models_tab() -> None:
    """Render the pretrained models tab content."""
    st.title("GW2ML: Pretrained Models")
    
    st.info("💡 These models are pretrained and do not require recalculation. They are suitable for heavier architectures.")
    
    st.subheader("Model Configuration")
    model_type = st.selectbox(
        "Select Model",
        options=["Facebook Prophet", "NeuralProphet", "TFT (Temporal Fusion Transformer)", "DeepAR"],
        help="Select a pretrained model architecture"
    )
    
    st.info(f"Currently selected: {model_type}")
    
    st.divider()
    
    st.subheader("Item Selection")
    
    # Use a cached version to avoid database hammering
    items = cached_list_items()

    default_ids = {"19698", "19699", "19697"}
    default_items = [
        item for item in items
        if str(item.get("item_id")) in default_ids
    ]

    selected_items = st.multiselect(
        "Search for items",
        options=items,
        default=default_items,
        format_func=lambda x: f"[{x['item_id']}] {x['item_name']}",
        help="Type to search and select one or more items"
    )

    if selected_items:
        st.session_state['pretrained_selected_item_ids'] = [str(item["item_id"]) for item in selected_items]
        
    st.divider()
    
    st.subheader("Upcoming Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("📅 **Scheduled Updates**: These models will be updated on a weekly basis using historical data.")
        st.info("🚀 **Performance**: Pretrained models provide faster inference for complex architectures.")
        
    with col2:
        st.info("📊 **Global Models**: Future support for models trained on multiple items simultaneously.")
        st.info("🔍 **Explainability**: Integration with SHAP/LIME for model prediction insights.")

    if st.button("Run Inference", type="primary"):
        if not selected_items:
            st.warning("Please select at least one item.")
        else:
            st.write(f"Running inference using **{model_type}** for items: {', '.join([item['item_name'] for item in selected_items])}...")
            st.warning("Inference logic is not yet implemented for these models.")

if __name__ == "__main__":
    st.set_page_config(page_title="GW2ML Pretrained Models", layout="wide")
    render_pretrained_models_tab()
