from __future__ import annotations

from typing import List

import streamlit as st
import pandas as pd
from gw2ml.data.loaders import list_items
from gw2ml.modeling.registry import list_models
from gw2ml.data.loaders import list_items
from gw2ml.modeling.registry import list_models
from gw2ml.pipelines.config import DEFAULT_CONFIG, merge_config
from gw2ml.pipelines.forecast import forecast_item
from gw2ml.pipelines.oneshot import oneshot_item_from_pretrained

CACHE_TIMEOUT = 60 * 60 * 24 * 7  # 7 days in seconds, TODO: if using disk, its possible this is ignored?

@st.cache_data(ttl=CACHE_TIMEOUT, persist="disk", show_spinner=True, show_time=True)
def cached_list_items(limit: int = 30000):
    return list_items(limit=limit)

def get_pretrained_models() -> List[str]:

    all_models = list_models()

    # TODO: this should be done in the model registry:
    allowed_models = ["XGBoost", "Chronos2"]

    # return only models in allowed_models
    return [model for model in all_models if model in allowed_models]

def render_pretrained_models_tab() -> None:
    """Render the pretrained models tab content."""
    st.title("GW2ML: Pretrained Models")
    
    st.info("💡 These models are pretrained and do not require recalculation. They are suitable for heavier architectures.")
    
    st.subheader("Model Configuration")
    models = get_pretrained_models()
    model_type = st.selectbox(
        "Select Model",
        options=models,
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
        st.toast("Not implemented yet!", icon="🚧")
        st.warning("Inference logic is not yet implemented for these models.")

    #if not selected_items:
        #    st.warning("Please select at least one item.")
        #else:
        #    st.write(f"Running inference using **{model_type}** for items: {', '.join([item['item_name'] for item in selected_items])}...")
        #    st.warning("Inference logic is not yet implemented for these models.")

def render_forecast_tab() -> None:
    """Render the forecast tab content."""

    st.subheader("Forecast")
    target_item = st.text_input("Forecast item_id", "19697")
    retrain_before = st.checkbox("Retrain before forecast (full search)", value=False)
    if st.button("Run forecast"):
        try:
            item_id = int(target_item)
            result = forecast_item(item_id, override_config=payload["config"], retrain=retrain_before)
            st.success("Forecast generated.")

            models_payload = result.get("models", [])
            missing_models = result.get("missing_models", [])
            if missing_models:
                st.warning(f"Missing artifacts for: {', '.join(missing_models)}. Retrain to generate them.")

            # Future forecasts (all models)
            future_rows = []
            for m in models_payload:
                f_ts = m.get("future", {}).get("timestamps", [])
                f_vals = m.get("future", {}).get("values", [])
                for ts, val in zip(f_ts, f_vals):
                    future_rows.append({"timestamp": ts, "value": val, "model": m.get("model_name")})
            if future_rows:
                df_future = pd.DataFrame(future_rows)
                palette = [
                    "#ff7f0e",
                    "#2ca02c",
                    "#d62728",
                    "#9467bd",
                    "#8c564b",
                    "#e377c2",
                    "#7f7f7f",
                    "#bcbd22",
                    "#17becf",
                ]
                models_unique = df_future["model"].unique().tolist()
                color_map = {m: palette[i % len(palette)] for i, m in enumerate(models_unique)}
                fig_future = px.line(
                    df_future,
                    x="timestamp",
                    y="value",
                    color="model",
                    title=f"Future forecasts for item {item_id}",
                    color_discrete_map=color_map,
                )
                st.plotly_chart(fig_future, use_container_width=True)

            # Historical backtests (forecasts per model + shared actual)
            hist_rows = []
            actual_rows = None
            for m in models_payload:
                h_ts = m.get("history", {}).get("timestamps", [])
                h_fc = m.get("history", {}).get("forecast", [])
                h_act = m.get("history", {}).get("actual", [])
                if h_ts and h_act and actual_rows is None:
                    actual_rows = [{"timestamp": ts, "actual": val} for ts, val in zip(h_ts, h_act) if val is not None]
                for ts, val in zip(h_ts, h_fc):
                    hist_rows.append({"timestamp": ts, "forecast": val, "model": m.get("model_name")})
            if hist_rows:
                df_hist_fc = pd.DataFrame(hist_rows)
                models_unique = df_hist_fc["model"].unique().tolist()
                palette = [
                    "#ff7f0e",
                    "#2ca02c",
                    "#d62728",
                    "#9467bd",
                    "#8c564b",
                    "#e377c2",
                    "#7f7f7f",
                    "#bcbd22",
                    "#17becf",
                ]
                color_map = {m: palette[i % len(palette)] for i, m in enumerate(models_unique)}

                fig_h = px.line(
                    df_hist_fc,
                    x="timestamp",
                    y="forecast",
                    color="model",
                    title="Historical backtest (forecasts)",
                    color_discrete_map=color_map,
                )
                if actual_rows:
                    df_act = pd.DataFrame(actual_rows)
                    fig_act = px.line(df_act, x="timestamp", y="actual")
                    for d in fig_act.data:
                        d.name = "actual"
                        d.showlegend = True
                        d.line.color = "#1f77b4"
                        fig_h.add_trace(d)
                st.plotly_chart(fig_h, use_container_width=True)

            # Metrics per model
            for m in models_payload:
                hist_metric = m.get("history", {}).get("metric")
                if hist_metric is not None:
                    st.metric(label=f"{m.get('model_name')} {primary_metric.upper()}", value=round(hist_metric, 4))

            st.json(result)
        except Exception as exc:  # pragma: no cover
            st.error(f"Forecast failed: {exc}")
            st.exception(exc)


if __name__ == "__main__":
    st.set_page_config(page_title="GW2ML Pretrained Models", layout="wide")
    render_pretrained_models_tab()
