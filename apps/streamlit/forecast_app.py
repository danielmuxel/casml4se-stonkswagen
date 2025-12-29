from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st

from gw2ml.modeling.registry import list_models
from gw2ml.pipelines.config import DEFAULT_CONFIG, merge_config
from gw2ml.pipelines.forecast import forecast_item
from gw2ml.pipelines.train import train_items


ALLOWED_CONFIG_KEYS = {"data", "split", "forecast", "metric", "models"}


def _parse_item_ids(text: str) -> List[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _sanitize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in cfg.items() if key in ALLOWED_CONFIG_KEYS}


def _build_override_config(
    item_ids_text: str,
    days_back: int,
    value_column: str,
    horizon: int,
    selected_models: List[str],
    primary_metric: str,
    metrics: List[str],
) -> Dict[str, Any]:
    item_ids = _parse_item_ids(item_ids_text)
    override: Dict[str, Any] = {
        "data": {
            "days_back": days_back,
            "value_column": value_column,
        },
        "forecast": {"horizon": horizon},
        "metric": {"primary": primary_metric, "metrics": metrics},
        "models": [{"name": name, "grid": None} for name in selected_models],
    }
    return {"item_ids": item_ids, "config": _sanitize_config(override)}


def main() -> None:
    st.set_page_config(page_title="GW2ML Train & Forecast", layout="wide")
    st.title("GW2ML: Train & Forecast")

    with st.sidebar:
        st.header("Configuration")
        days_back = st.slider("Days back", min_value=7, max_value=120, value=30, step=1)
        value_column = st.selectbox("Value column", ["buy_unit_price", "sell_unit_price"])
        horizon = st.slider("Forecast horizon (steps)", min_value=1, max_value=288, value=12, step=1)

        all_models = list_models()
        selected_models = st.multiselect("Models", options=all_models, default=all_models)

        available_metrics = ["mape"]
        primary_metric = st.selectbox("Primary metric", options=available_metrics, index=0)
        metrics = st.multiselect("Metrics to compute", options=available_metrics, default=available_metrics)

    st.subheader("Forecast")
    target_item = st.text_input("Forecast item_id", "19697")
    retrain_before = st.checkbox("Retrain before forecast (full search)", value=False)
    if st.button("Run forecast"):
        try:
            payload = _build_override_config(
                target_item, days_back, value_column, horizon, selected_models, primary_metric, metrics
            )
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
    main()

