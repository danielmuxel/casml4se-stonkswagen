from __future__ import annotations

import time
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st

from gw2ml.data.loaders import list_items
from gw2ml.metrics.registry import get_metric, list_metrics
from gw2ml.modeling.registry import list_models
from gw2ml.pipelines.config import DEFAULT_CONFIG, merge_config
from gw2ml.pipelines.forecast import forecast_item
from gw2ml.pipelines.train import train_items
from gw2ml.utils import get_logger

logger = get_logger("forecast_app")


ALLOWED_CONFIG_KEYS = {"data", "split", "forecast", "metric", "models"}


def _validate_forecast_inputs(
    item_id_text: str, selected_models: List[str], days_back: int
) -> tuple[bool, str | None]:
    """
    Validate forecast inputs before running the forecast.

    Returns:
        tuple: (is_valid, error_message)
    """
    # Validate item ID is a valid integer
    try:
        item_id = int(item_id_text.strip())
        if item_id <= 0:
            return False, f"Item ID must be a positive integer, got: {item_id}"
    except ValueError:
        return False, f"Invalid item ID format: '{item_id_text}'. Must be a valid integer."

    # Validate at least one model is selected
    if not selected_models:
        return False, "No models selected. Please select at least one model."

    # Validate all selected models exist in registry
    available_models = list_models()
    invalid_models = [m for m in selected_models if m not in available_models]
    if invalid_models:
        return False, f"Invalid model(s): {', '.join(invalid_models)}. Available: {', '.join(available_models)}"

    # Validate minimum data requirements for train/val/test split
    if days_back < 3:
        return False, "Need at least 3 days of data for train/val/test split."

    # Validate required dependencies are importable
    try:
        from darts import TimeSeries  # noqa: F401
    except ImportError as e:
        return False, f"Required dependency 'darts' not available: {e}"

    return True, None


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

    # Determine split strategy based on data size
    # For small datasets, use proportional splits instead of fixed day counts
    test_days = DEFAULT_CONFIG["split"]["test_days"]
    val_days = DEFAULT_CONFIG["split"]["val_days"] or 0
    min_days_for_fixed_split = test_days + val_days + 3

    if days_back >= min_days_for_fixed_split:
        # Use fixed day splits (e.g., last 7 days for test)
        split_config = {
            "test_days": test_days,
            "val_days": val_days if val_days > 0 else None,
        }
    else:
        # Use proportional splits for small datasets
        split_config = {
            "train": 0.6,  # 60% training
            "val": 0.2,    # 20% validation
            "test_days": None,  # Disable fixed day splits
            "val_days": None,
        }

    override: Dict[str, Any] = {
        "data": {
            "days_back": days_back,
            "value_column": value_column,
        },
        "split": split_config,
        "forecast": {"horizon": horizon},
        "metric": {"primary": primary_metric, "metrics": metrics},
        "models": [{"name": name, "grid": None} for name in selected_models],
    }
    return {"item_ids": item_ids, "config": _sanitize_config(override)}


def render_forecast_tab() -> None:
    """Render the forecast tab content."""
    st.title("GW2ML: Train & Forecast")

    with st.sidebar:
        st.header("Configuration")

        # Calculate threshold for split strategy
        test_days = DEFAULT_CONFIG["split"]["test_days"]
        val_days = DEFAULT_CONFIG["split"]["val_days"] or 0
        fixed_split_threshold = test_days + val_days + 3

        days_back = st.slider(
            "Days back",
            min_value=3,
            max_value=120,
            value=30,
            step=1,
            help=(
                f"≥{fixed_split_threshold} days: fixed split (test={test_days}d, val={val_days}d). "
                f"<{fixed_split_threshold} days: proportional split (60/20/20%)."
            )
        )

        # Show which split mode is active
        if days_back >= fixed_split_threshold:
            split_info = f"📊 Using fixed split: {test_days}d test, {val_days}d validation"
        else:
            split_info = f"📊 Using proportional split: 60% train, 20% val, 20% test"
        st.caption(split_info)

        value_column = st.selectbox("Value column", ["buy_unit_price", "sell_unit_price"])
        horizon = st.slider("Forecast horizon (steps)", min_value=1, max_value=288, value=12, step=1)

        all_models = list_models()
        selected_models = st.multiselect("Models", options=all_models, default=all_models)

        available_metrics = list_metrics()
        primary_metric = st.selectbox("Primary metric", options=available_metrics, index=0)
        metrics = st.multiselect("Metrics to compute", options=available_metrics, default=available_metrics)

    st.subheader("Forecast")
    target_item = st.text_input("Forecast item_id", "19697")
    retrain_before = st.checkbox("Retrain before forecast (full search)", value=False)
    if st.button("Run forecast"):
        start_time = time.time()
        try:
            logger.info(f"Running forecast for item {target_item} (retrain={retrain_before})")
            payload = _build_override_config(
                target_item, days_back, value_column, horizon, selected_models, primary_metric, metrics
            )
            item_id = int(target_item)

            # Validate inputs before running forecast
            is_valid, error_msg = _validate_forecast_inputs(
                target_item, selected_models, days_back
            )
            if not is_valid:
                st.error(f"❌ Validation failed: {error_msg}")
                return

            with st.status(f"Processing forecast for item {item_id}...", expanded=True) as status:
                if retrain_before:
                    st.write(f"🔄 Retraining models: {', '.join(selected_models)}")

                st.write(f"📊 Loading {days_back} days of data...")
                st.write(f"🔮 Generating {horizon}-step forecast...")
                st.write(f"📈 Running backtest validation...")

                result = forecast_item(item_id, override_config=payload["config"], retrain=retrain_before)

                status.update(label=f"✓ Forecast complete for item {item_id}", state="complete")

            elapsed_time = time.time() - start_time
            st.success(f"Forecast generated in {elapsed_time:.2f} seconds.")

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

            # Metrics table: rows = models, columns = metrics
            if models_payload:
                metrics_data = []
                for m in models_payload:
                    h_fc = m.get("history", {}).get("forecast", [])
                    h_act = m.get("history", {}).get("actual", [])

                    row = {"Model": m.get("model_name")}

                    # Calculate all selected metrics if we have data
                    if h_fc and h_act and len(h_fc) == len(h_act):
                        # Convert to pandas Series for darts metrics
                        from darts import TimeSeries

                        try:
                            # Create simple TimeSeries for metric calculation
                            ts_actual = TimeSeries.from_values(h_act)
                            ts_forecast = TimeSeries.from_values(h_fc)

                            for metric_name in metrics:
                                try:
                                    metric_fn = get_metric(metric_name)
                                    value = metric_fn(ts_actual, ts_forecast)
                                    row[metric_name.upper()] = round(float(value), 4) if not pd.isna(value) else "NaN"
                                except Exception as e:
                                    row[metric_name.upper()] = "Error"
                        except Exception as e:
                            # If TimeSeries creation fails, mark all metrics as error
                            for metric_name in metrics:
                                row[metric_name.upper()] = "Error"
                    else:
                        # No backtest data available
                        for metric_name in metrics:
                            row[metric_name.upper()] = "N/A"

                    metrics_data.append(row)

                if metrics_data:
                    df_metrics = pd.DataFrame(metrics_data)
                    st.subheader("Model Performance Metrics")
                    st.dataframe(df_metrics, use_container_width=True, hide_index=True)

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
