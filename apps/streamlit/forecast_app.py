from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st

from gw2ml.data.loaders import list_items
from gw2ml.metrics.registry import list_metrics
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
    backtest_stride: Any = "auto",
) -> Dict[str, Any]:
    item_ids = _parse_item_ids(item_ids_text)

    # For very small datasets (≤3 days), use ratio-based splitting (80/20)
    # For larger datasets, use day-based splitting
    if days_back <= 3:
        split_config = {
            "train": 0.8,
            "val": 0.0,
            "test_days": None,
            "val_days": None,
        }
    else:
        # Dynamically calculate test_days to ensure enough training data
        # Use at most 1/4 of days_back for testing, with a minimum of 1 and maximum of 7
        test_days = min(7, max(1, days_back // 4))
        split_config = {
            "test_days": test_days,
            "val_days": None,
        }

    override: Dict[str, Any] = {
        "data": {
            "days_back": days_back,
            "value_column": value_column,
        },
        "split": split_config,
        "forecast": {
            "horizon": horizon,
            "backtest_stride": backtest_stride,
        },
        "metric": {"primary": primary_metric, "metrics": metrics},
        "models": [{"name": name, "grid": None} for name in selected_models],
    }
    return {"item_ids": item_ids, "config": _sanitize_config(override)}


def render_forecast_tab() -> None:
    """Render the forecast tab content."""
    st.title("GW2ML: Train & Forecast")

    with st.sidebar:
        st.header("Configuration")
        days_back = st.slider("Days back", min_value=3, max_value=120, value=7, step=1)
        value_column = st.selectbox("Value column", ["buy_unit_price", "sell_unit_price"])
        horizon = st.slider("Forecast horizon (steps)", min_value=1, max_value=288, value=24, step=1)

        st.subheader("Backtest Settings")
        use_auto_stride = st.checkbox("Auto stride (recommended)", value=True, help="Automatically sets stride = max(horizon//4, 6)")
        if use_auto_stride:
            backtest_stride = "auto"
            estimated_stride = max(horizon // 4, 6)
            st.caption(f"Auto stride will be: {estimated_stride} ({estimated_stride * 5} min)")
        else:
            backtest_stride = st.slider(
                "Backtest stride",
                min_value=1,
                max_value=horizon,
                value=max(horizon // 4, 6),
                step=1,
                help="stride=1: dense but can react to data | stride=horizon: sparse but realistic",
            )
            st.caption(f"Model retrains every {backtest_stride} steps ({backtest_stride * 5} min)")

        st.subheader("Models & Metrics")
        all_models = list_models()
        selected_models = st.multiselect("Models", options=all_models, default=all_models)

        available_metrics = list_metrics()
        primary_metric = st.selectbox("Primary metric", options=available_metrics, index=0)
        metrics = st.multiselect("Metrics to compute", options=available_metrics, default=available_metrics)

    st.subheader("Forecast")
    target_item = st.text_input("Forecast item_id", "19702")
    retrain_before = st.checkbox("Retrain before forecast (full search)", value=False)
    if st.button("Run forecast"):
        try:
            payload = _build_override_config(
                target_item, days_back, value_column, horizon, selected_models, primary_metric, metrics, backtest_stride
            )
            item_id = int(target_item)

            with st.status(f"Processing forecast for item {item_id}...", expanded=True) as status:
                if retrain_before:
                    st.write(f"🔄 Retraining models: {', '.join(selected_models)}")

                st.write(f"📊 Loading {days_back} days of data...")
                st.write(f"🔮 Generating {horizon}-step forecast...")
                st.write(f"📈 Running backtest validation...")

                result = forecast_item(item_id, override_config=payload["config"], retrain=retrain_before)

                status.update(label=f"✓ Forecast complete for item {item_id}", state="complete")

            st.success("Forecast generated.")

            # Show backtest configuration
            backtest_info = result.get("backtest_config", {})
            if backtest_info:
                st.info(f"🔍 Backtest: stride={backtest_info.get('stride', 'unknown')}, horizon={horizon}")
            else:
                # Calculate what stride should be used
                estimated_stride = max(horizon // 4, 6)
                st.warning(f"⚠️ Backtest config not in result. Expected stride would be: {estimated_stride} (horizon={horizon})")

            models_payload = result.get("models", [])
            missing_models = result.get("missing_models", [])

            if models_payload:
                successful_models = [m.get('model_name') for m in models_payload]
                st.info(f"✓ Successfully generated forecasts for {len(models_payload)} model(s): {', '.join(successful_models)}")

                # Check for models with failed backtests (NaN metrics)
                failed_backtests = [m.get('model_name') for m in models_payload if m.get('history', {}).get('metric') != m.get('history', {}).get('metric')]
                if failed_backtests:
                    st.warning(f"⚠️ Backtest failed for: {', '.join(failed_backtests)} (possible hardware incompatibility). Future forecast still available.")

            if missing_models:
                st.warning(f"⚠️ Missing artifacts for: {', '.join(missing_models)}. Retrain to generate them.")

            # If no models succeeded at all
            if not models_payload and not missing_models:
                st.error("❌ All models failed to generate forecasts. Check logs for details (possible MPS/GPU incompatibility).")

            # Future forecasts (all models) with historical context
            future_rows = []
            last_actual_point = None  # Will store the last actual data point for continuity

            for m in models_payload:
                f_ts = m.get("future", {}).get("timestamps", [])
                f_vals = m.get("future", {}).get("values", [])
                for ts, val in zip(f_ts, f_vals):
                    future_rows.append({"timestamp": ts, "value": val, "model": m.get("model_name"), "type": "forecast"})

            if future_rows:
                # Load historical data for context (5x the forecast horizon)
                from gw2ml.data.loaders import load_gw2_series

                context_points = 0
                hist_df = None
                try:
                    series_meta = load_gw2_series(
                        item_id=item_id,
                        days_back=days_back,
                        value_column=value_column,
                        fill_missing_dates=True,
                        resample_freq=None,
                    )
                    # Get last N points where N = 5 * horizon
                    context_points = min(5 * horizon, len(series_meta.series))
                    if context_points > 0:
                        historical_series = series_meta.series[-context_points:]

                        # Convert to dataframe - use the same approach as forecast.py
                        try:
                            hist_df = historical_series.pd_dataframe()
                        except AttributeError:
                            # Fallback: build from values and time_index
                            values = historical_series.values(copy=False)
                            time_idx = historical_series.time_index
                            hist_df = pd.DataFrame(values, index=time_idx, columns=["value"] if values.shape[1] == 1 else None)
                            if "value" not in hist_df.columns and hist_df.shape[1] == 1:
                                hist_df.columns = ["value"]

                        hist_df = hist_df.reset_index()
                        hist_df.columns = ["timestamp", "value"]
                        hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"]).dt.tz_localize(None)
                        hist_df["model"] = "Actual"
                        hist_df["type"] = "actual"

                        # Store the last actual point for continuity with forecasts
                        last_actual_point = {
                            "timestamp": hist_df.iloc[-1]["timestamp"],
                            "value": hist_df.iloc[-1]["value"],
                        }
                except Exception as e:
                    st.warning(f"Could not load historical data: {e}")
                    hist_df = None
                    context_points = 0

                # Combine historical and forecast data
                df_future = pd.DataFrame(future_rows)

                # Convert forecast timestamps to datetime objects
                if not df_future.empty:
                    df_future["timestamp"] = pd.to_datetime(df_future["timestamp"]).dt.tz_localize(None)

                # Add last actual point to each forecast model for visual continuity
                if last_actual_point is not None and not df_future.empty:
                    continuity_rows = []
                    for model_name in df_future["model"].unique():
                        continuity_rows.append({
                            "timestamp": last_actual_point["timestamp"],
                            "value": last_actual_point["value"],
                            "model": model_name,
                            "type": "continuity",
                        })
                    df_continuity = pd.DataFrame(continuity_rows)
                    df_future = pd.concat([df_continuity, df_future], ignore_index=True)

                if hist_df is not None and not hist_df.empty:
                    df_combined = pd.concat([hist_df, df_future], ignore_index=True)
                else:
                    df_combined = df_future
                    df_combined["type"] = "forecast"

                palette = [
                    "#1f77b4",  # Blue for actual
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
                models_unique = df_combined["model"].unique().tolist()
                # Ensure "Actual" is first and gets the blue color
                if "Actual" in models_unique:
                    models_unique.remove("Actual")
                    models_unique.insert(0, "Actual")
                color_map = {m: palette[i % len(palette)] for i, m in enumerate(models_unique)}

                fig_future = px.line(
                    df_combined,
                    x="timestamp",
                    y="value",
                    color="model",
                    title=f"Forecast for item {item_id} | Horizon: {horizon} steps | Historical context: {context_points} points",
                    color_discrete_map=color_map,
                )

                # Style the traces: thicker solid for actual, solid but thinner for forecasts
                for trace in fig_future.data:
                    if trace.name == "Actual":
                        trace.line.width = 3
                        trace.opacity = 1.0
                    else:
                        trace.line.width = 2.5
                        trace.opacity = 0.85

                # Add a vertical line to mark the transition from actual to forecast
                if hist_df is not None and not hist_df.empty:
                    try:
                        last_actual_time = hist_df.iloc[-1]["timestamp"]
                        # Convert pandas Timestamp to Python datetime for Plotly compatibility
                        if hasattr(last_actual_time, 'to_pydatetime'):
                            last_actual_time = last_actual_time.to_pydatetime()
                        fig_future.add_vline(
                            x=last_actual_time,
                            line_dash="dot",
                            line_color="gray",
                            opacity=0.5,
                            annotation_text="Forecast Start",
                            annotation_position="top",
                        )
                    except Exception:
                        # Skip vline if it causes compatibility issues
                        pass

                fig_future.update_layout(
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )

                st.plotly_chart(fig_future, width="stretch")

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
                df_hist_fc["timestamp"] = pd.to_datetime(df_hist_fc["timestamp"])

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
                    title=f"Historical backtest | Horizon: {horizon} steps | Stride: auto",
                    color_discrete_map=color_map,
                )
                if actual_rows:
                    df_act = pd.DataFrame(actual_rows)
                    df_act["timestamp"] = pd.to_datetime(df_act["timestamp"])
                    fig_act = px.line(df_act, x="timestamp", y="actual")
                    for d in fig_act.data:
                        d.name = "actual"
                        d.showlegend = True
                        d.line.color = "#1f77b4"
                        fig_h.add_trace(d)

                fig_h.update_layout(
                    hovermode="x unified",
                    showlegend=True,
                )
                st.plotly_chart(fig_h, width="stretch")

            # Metrics per model
            for m in models_payload:
                hist_metric = m.get("history", {}).get("metric")
                if hist_metric is not None:
                    st.metric(label=f"{m.get('model_name')} {primary_metric.upper()}", value=round(hist_metric, 4))

            st.json(result)
        except Exception as exc:  # pragma: no cover
            st.error(f"Forecast failed: {exc}")
            st.exception(exc)


def main() -> None:
    """Main entry point when running forecast_app.py standalone."""
    st.set_page_config(page_title="GW2ML Train & Forecast", layout="wide")
    render_forecast_tab()


if __name__ == "__main__":
    main()
