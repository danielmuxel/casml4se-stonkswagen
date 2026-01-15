from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gw2ml.data.loaders import load_gw2_series
from gw2ml.modeling.registry import list_models
from gw2ml.pipelines.config import DEFAULT_CONFIG
from gw2ml.pipelines.forecast import forecast_item
from gw2ml.utils import get_logger

logger = get_logger("forecast_app")

ALLOWED_CONFIG_KEYS = {"data", "forecast", "models", "metric", "split", "train"}
VALUE_COLUMNS = [
    "buy_unit_price",
    "sell_unit_price",
    "buy_quantity",
    "sell_quantity",
]
RESAMPLE_OPTIONS = ["5min", "15min", "30min", "1h", "4h", "12h", "1d"]
MODEL_COLOR_MAP = {
    "ARIMA": "#ff7f0e",
    "ExponentialSmoothing": "#2ca02c",
    "XGBoost": "#d62728",
    "Chronos2": "#9467bd",
    "Chronos": "#9467bd",
}
FALLBACK_MODEL_COLORS = ["#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def _model_color(model_name: str) -> str:
    if model_name in MODEL_COLOR_MAP:
        return MODEL_COLOR_MAP[model_name]
    if "Chronos" in model_name:
        return MODEL_COLOR_MAP["Chronos"]
    import hashlib

    digest = hashlib.md5(model_name.encode("utf-8")).hexdigest()
    idx = int(digest, 16) % len(FALLBACK_MODEL_COLORS)
    return FALLBACK_MODEL_COLORS[idx]


def _seconds_to_next_five_minute(now: datetime) -> int:
    minute_block = (now.minute // 5 + 1) * 5
    next_hour = now.hour
    next_day = now.date()
    if minute_block >= 60:
        minute_block = 0
        next_hour += 1
        if next_hour >= 24:
            next_hour = 0
            next_day = next_day + timedelta(days=1)
    next_dt = now.replace(
        year=next_day.year,
        month=next_day.month,
        day=next_day.day,
        hour=next_hour,
        minute=minute_block,
        second=0,
        microsecond=0,
    )
    delta = next_dt - now
    return max(0, int(delta.total_seconds()))


def _load_actual_context(
    item_id: int,
    days_back: int,
    value_column: str,
    resample_freq: str,
) -> Dict[str, Any]:
    series_meta = load_gw2_series(
        item_id=item_id,
        days_back=days_back,
        value_column=value_column,
        fill_missing_dates=DEFAULT_CONFIG["data"]["fill_missing_dates"],
        resample_freq=resample_freq,
        use_cache=False,
    )
    df = series_meta.series.to_dataframe()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return {
        "timestamps": df.index.astype(str).tolist(),
        "values": df.iloc[:, 0].tolist(),
    }


def _validate_forecast_inputs(
    item_id_text: str, selected_models: List[str], horizon: int
) -> tuple[bool, str | None]:
    try:
        item_id = int(item_id_text.strip())
        if item_id <= 0:
            return False, f"Item ID must be a positive integer, got: {item_id}"
    except ValueError:
        return False, f"Invalid item ID format: '{item_id_text}'. Must be a valid integer."

    if not selected_models:
        return False, "No models selected. Please select at least one model."

    available_models = list_models()
    invalid_models = [m for m in selected_models if m not in available_models]
    if invalid_models:
        return False, f"Invalid model(s): {', '.join(invalid_models)}. Available: {', '.join(available_models)}"

    if horizon < 1:
        return False, "Forecast horizon must be at least 1 step."

    try:
        from darts import TimeSeries  # noqa: F401
    except ImportError as e:
        return False, f"Required dependency 'darts' not available: {e}"

    return True, None


def _sanitize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in cfg.items() if key in ALLOWED_CONFIG_KEYS}


def _normalize_resample_options(labels: List[str]) -> List[tuple[str, str]]:
    choices: List[tuple[str, str]] = []
    for label in labels:
        choices.append((label, label))
    return choices


def _build_override_config(
    days_back: int,
    value_column: str,
    horizon: int,
    selected_models: List[str],
    resample_freq: Optional[str],
    force_grid_search: bool,
) -> Dict[str, Any]:
    test_days = DEFAULT_CONFIG["split"]["test_days"]
    val_days = DEFAULT_CONFIG["split"]["val_days"] or 0
    min_days_for_fixed_split = test_days + val_days + 3

    if days_back >= min_days_for_fixed_split:
        split_config = {
            "test_days": test_days,
            "val_days": val_days if val_days > 0 else None,
        }
    else:
        split_config = {
            "train": 0.6,
            "val": 0.2,
            "test_days": None,
            "val_days": None,
        }

    override: Dict[str, Any] = {
        "data": {
            "days_back": days_back,
            "value_column": value_column,
            "resample_freq": resample_freq,
        },
        "split": split_config,
        "forecast": {"horizon": horizon},
        "train": {"force_grid_search": force_grid_search},
        "models": [{"name": name, "grid": None} for name in selected_models],
    }
    return _sanitize_config(override)


def _plot_history_and_forecast(
    result: Dict[str, Any],
    title: str,
    actual_points: int,
    actual_override: Dict[str, Any] | None = None,
) -> go.Figure:
    context = actual_override or result.get("context") or {}
    history_ts = pd.to_datetime(context.get("timestamps", []))
    history_vals = context.get("values", [])
    if actual_points > 0 and len(history_ts) > actual_points:
        history_ts = history_ts[-actual_points:]
        history_vals = history_vals[-actual_points:]

    last_actual = None
    if len(history_ts) and len(history_vals):
        for ts, val in zip(reversed(history_ts), reversed(history_vals)):
            if val is not None and not pd.isna(val):
                last_actual = (ts, val)
                break

    fig = go.Figure()
    if len(history_ts) > 0:
        fig.add_trace(
            go.Scatter(
                x=history_ts,
                y=history_vals,
                mode="lines",
                name="actual",
                line=dict(color="#1f77b4"),
            )
        )

    for model in result.get("models", []):
        future = model.get("future", {})
        future_ts = pd.to_datetime(future.get("timestamps", []))
        future_vals = future.get("values", [])
        if len(future_ts) == 0:
            continue
        if last_actual is not None:
            last_ts, last_val = last_actual
            if pd.to_datetime(future_ts[0]) != pd.to_datetime(last_ts):
                future_ts = [last_ts] + list(future_ts)
                future_vals = [last_val] + list(future_vals)
        fig.add_trace(
            go.Scatter(
                x=future_ts,
                y=future_vals,
                mode="lines",
                name=f"{model.get('model_name')} forecast",
                line=dict(color=_model_color(model.get("model_name", ""))),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="timestamp",
        yaxis_title="value",
        legend_title="series",
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        margin=dict(b=120),
    )
    return fig


def _build_profitability_plots(
    buy_result: Dict[str, Any],
    sell_result: Dict[str, Any],
    model_name: str,
    actual_points: int,
    buy_override: Dict[str, Any] | None = None,
    sell_override: Dict[str, Any] | None = None,
) -> tuple[go.Figure, go.Figure] | None:
    buy_context = buy_override or buy_result.get("context") or {}
    sell_context = sell_override or sell_result.get("context") or {}

    buy_hist = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(buy_context.get("timestamps", [])),
            "buy": buy_context.get("values", []),
        }
    )
    sell_hist = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(sell_context.get("timestamps", [])),
            "sell": sell_context.get("values", []),
        }
    )
    if actual_points > 0:
        buy_hist = buy_hist.tail(actual_points)
        sell_hist = sell_hist.tail(actual_points)

    buy_model = next((m for m in buy_result.get("models", []) if m.get("model_name") == model_name), None)
    sell_model = next((m for m in sell_result.get("models", []) if m.get("model_name") == model_name), None)

    if buy_model is None or sell_model is None:
        return None

    buy_future = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(buy_model.get("future", {}).get("timestamps", [])),
            "buy": buy_model.get("future", {}).get("values", []),
        }
    )
    sell_future = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(sell_model.get("future", {}).get("timestamps", [])),
            "sell": sell_model.get("future", {}).get("values", []),
        }
    )

    history = buy_hist.merge(sell_hist, on="timestamp", how="outer").sort_values("timestamp")
    future = buy_future.merge(sell_future, on="timestamp", how="outer").sort_values("timestamp")

    combined = pd.concat([history, future], ignore_index=True).sort_values("timestamp")
    combined["breakeven_sell"] = combined["buy"] / 0.85
    combined["max_buy"] = combined["sell"] * 0.85

    sell_profit = combined["sell"].where(combined["sell"] >= combined["breakeven_sell"])
    sell_loss = combined["sell"].where(combined["sell"] < combined["breakeven_sell"])

    buy_profit = combined["buy"].where(combined["buy"] <= combined["max_buy"])
    buy_loss = combined["buy"].where(combined["buy"] > combined["max_buy"])

    fig_sell = go.Figure()
    fig_sell.add_trace(
        go.Scatter(
            x=combined["timestamp"],
            y=combined["breakeven_sell"],
            mode="lines",
            name="required sell for current buy (buy/0.85)",
            line=dict(color="#ff7f0e", dash="dot"),
        )
    )
    fig_sell.add_trace(
        go.Scatter(
            x=combined["timestamp"],
            y=sell_profit,
            mode="lines",
            name="profit zone",
            line=dict(color="rgba(0,0,0,0)"),
            fill="tonexty",
            fillcolor="rgba(44, 160, 44, 0.2)",
        )
    )
    fig_sell.add_trace(
        go.Scatter(
            x=combined["timestamp"],
            y=combined["breakeven_sell"],
            mode="lines",
            name="required sell for current buy (buy/0.85)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
        )
    )
    fig_sell.add_trace(
        go.Scatter(
            x=combined["timestamp"],
            y=sell_loss,
            mode="lines",
            name="loss zone",
            line=dict(color="rgba(0,0,0,0)"),
            fill="tonexty",
            fillcolor="rgba(214, 39, 40, 0.2)",
        )
    )
    fig_sell.add_trace(
        go.Scatter(
            x=combined["timestamp"],
            y=combined["sell"],
            mode="lines",
            name="current sell (actual+forecast)",
            line=dict(color="#2ca02c"),
        )
    )
    fig_sell.update_layout(
        title="Sell view: sell vs. required sell (buy/0.85)",
        xaxis_title="timestamp",
        yaxis_title="price",
        legend_title="series",
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        margin=dict(b=120),
    )

    fig_buy = go.Figure()
    fig_buy.add_trace(
        go.Scatter(
            x=combined["timestamp"],
            y=combined["max_buy"],
            mode="lines",
            name="max buy for profit (sell*0.85)",
            line=dict(color="#2ca02c", dash="dot"),
        )
    )
    fig_buy.add_trace(
        go.Scatter(
            x=combined["timestamp"],
            y=buy_profit,
            mode="lines",
            name="profit zone",
            line=dict(color="rgba(0,0,0,0)"),
            fill="tonexty",
            fillcolor="rgba(44, 160, 44, 0.2)",
        )
    )
    fig_buy.add_trace(
        go.Scatter(
            x=combined["timestamp"],
            y=combined["max_buy"],
            mode="lines",
            name="max buy for profit (sell*0.85)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
        )
    )
    fig_buy.add_trace(
        go.Scatter(
            x=combined["timestamp"],
            y=buy_loss,
            mode="lines",
            name="loss zone",
            line=dict(color="rgba(0,0,0,0)"),
            fill="tonexty",
            fillcolor="rgba(214, 39, 40, 0.2)",
        )
    )
    fig_buy.add_trace(
        go.Scatter(
            x=combined["timestamp"],
            y=combined["buy"],
            mode="lines",
            name="current buy (actual+forecast)",
            line=dict(color="#ff7f0e"),
        )
    )
    fig_buy.update_layout(
        title="Buy view: buy vs. max buy (sell*0.85)",
        xaxis_title="timestamp",
        yaxis_title="price",
        legend_title="series",
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        margin=dict(b=120),
    )
    return fig_sell, fig_buy


def _build_profitability_summary(
    buy_result: Dict[str, Any],
    sell_result: Dict[str, Any],
    model_name: str,
    actual_points: int,
    buy_override: Dict[str, Any] | None = None,
    sell_override: Dict[str, Any] | None = None,
) -> go.Figure | None:
    buy_context = buy_override or buy_result.get("context") or {}
    sell_context = sell_override or sell_result.get("context") or {}

    buy_hist = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(buy_context.get("timestamps", [])),
            "buy": buy_context.get("values", []),
        }
    )
    sell_hist = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(sell_context.get("timestamps", [])),
            "sell": sell_context.get("values", []),
        }
    )
    if actual_points > 0:
        buy_hist = buy_hist.tail(actual_points)
        sell_hist = sell_hist.tail(actual_points)

    buy_model = next((m for m in buy_result.get("models", []) if m.get("model_name") == model_name), None)
    sell_model = next((m for m in sell_result.get("models", []) if m.get("model_name") == model_name), None)

    if buy_model is None or sell_model is None:
        return None

    buy_future = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(buy_model.get("future", {}).get("timestamps", [])),
            "buy": buy_model.get("future", {}).get("values", []),
        }
    )
    sell_future = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(sell_model.get("future", {}).get("timestamps", [])),
            "sell": sell_model.get("future", {}).get("values", []),
        }
    )

    history = buy_hist.merge(sell_hist, on="timestamp", how="outer").sort_values("timestamp")
    future = buy_future.merge(sell_future, on="timestamp", how="outer").sort_values("timestamp")

    combined = pd.concat([history, future], ignore_index=True).sort_values("timestamp")
    combined["breakeven_sell"] = combined["buy"] / 0.85
    combined["max_buy"] = combined["sell"] * 0.85

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=combined["timestamp"],
            y=combined["buy"],
            mode="lines",
            name="current buy (actual+forecast)",
            line=dict(color="#ff7f0e"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=combined["timestamp"],
            y=combined["sell"],
            mode="lines",
            name="current sell (actual+forecast)",
            line=dict(color="#2ca02c"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=combined["timestamp"],
            y=combined["max_buy"],
            mode="lines",
            name="max buy for profit (sell*0.85)",
            line=dict(color="#2ca02c", dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=combined["timestamp"],
            y=combined["breakeven_sell"],
            mode="lines",
            name="required sell for current buy (buy/0.85)",
            line=dict(color="#ff7f0e", dash="dot"),
        )
    )
    fig.update_layout(
        title="Buy/Sell with required thresholds",
        xaxis_title="timestamp",
        yaxis_title="price",
        legend_title="series",
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        margin=dict(b=120),
    )
    return fig


def render_forecast_tab() -> None:
    st.title("GW2ML: Forecast")

    with st.sidebar:
        st.header("Forecast Configuration")

        days_back = st.number_input(
            "History range (days back)",
            min_value=1,
            max_value=365,
            value=30,
            step=1,
            help="Use the last X days for forecasting.",
        )

        resample_labels = st.multiselect(
            "Resample intervals",
            options=RESAMPLE_OPTIONS,
            default=["5min"],
            help="Run forecast for one or more resampling options.",
        )

        horizon = st.number_input(
            "Forecast horizon (steps)",
            min_value=1,
            value=12,
            step=1,
            help="Minimum horizon will be adjusted to match history length.",
        )

        actual_points = st.number_input(
            "Actual points to show",
            min_value=0,
            max_value=10000,
            value=int(horizon) * 3,
            step=max(1, int(horizon)),
            help="0 = show all actual points.",
        )

        selected_columns = st.multiselect(
            "Forecast fields",
            options=VALUE_COLUMNS,
            default=VALUE_COLUMNS,
            help="Select which fields to forecast.",
        )

        all_models = list_models()
        selected_models = st.multiselect("Models", options=all_models, default=all_models)
        retrain_before = st.checkbox("Retrain before forecast", value=False)
        force_grid_search = st.checkbox(
            "Force grid search",
            value=False,
            help="If enabled, retraining will run a full grid search instead of reusing cached params.",
        )

    st.subheader("Forecast")
    target_item = st.text_input("Forecast item_id", "19976")

    if st.button("Run forecast"):
        start_time = time.time()
        try:
            is_valid, error_msg = _validate_forecast_inputs(target_item, selected_models, int(horizon))
            if not is_valid:
                st.error(f"❌ Validation failed: {error_msg}")
                return
            if not selected_columns:
                st.error("Please select at least one field to forecast.")
                return

            item_id = int(target_item)
            logger.info(
                f"Running forecast for item {item_id} (days_back={days_back})"
            )

            results: Dict[str, Dict[str, Dict[str, Any]]] = {}
            missing_models: set[str] = set()

            with st.status(f"Processing forecast for item {item_id}...", expanded=True) as status:
                resample_choices = _normalize_resample_options(resample_labels)
                if not resample_choices:
                    st.error("Please select at least one resample option.")
                    return

                total_steps = len(resample_choices) * len(selected_columns)
                progress = st.progress(0)
                step_text = st.empty()

                step_idx = 0
                for resample_label, resample_freq in resample_choices:
                    step_text.write(f"📊 Loading history ({resample_label})")
                    series_meta = load_gw2_series(
                        item_id=item_id,
                        days_back=days_back,
                        value_column="buy_unit_price",
                        fill_missing_dates=DEFAULT_CONFIG["data"]["fill_missing_dates"],
                        resample_freq=resample_freq,
                    )
                    history_points = series_meta.num_points
                    horizon_steps = int(horizon)
                    st.caption(
                        f"[{resample_label}] History contains {history_points} points. "
                        f"Forecast horizon: {horizon_steps} steps."
                    )

                    step_text.write(
                        f"🔮 Generating {horizon_steps}-step forecasts ({resample_label})"
                    )

                    results[resample_label] = {}
                    for column in selected_columns:
                        step_idx += 1
                        step_text.write(
                            f"🔁 Forecasting {column} ({resample_label}) "
                            f"[{step_idx}/{total_steps}]"
                        )
                        override_config = _build_override_config(
                            days_back=days_back,
                            value_column=column,
                            horizon=horizon_steps,
                            selected_models=selected_models,
                            resample_freq=resample_freq,
                            force_grid_search=force_grid_search,
                        )
                        result = forecast_item(
                            item_id,
                            override_config=override_config,
                            retrain=retrain_before,
                            include_backtest=False,
                            include_history=True,
                        )
                        results[resample_label][column] = result
                        missing_models.update(result.get("missing_models", []))
                        progress.progress(step_idx / total_steps)

                status.update(label=f"✓ Forecast complete for item {item_id}", state="complete")

            elapsed_time = time.time() - start_time
            st.success(f"Forecast generated in {elapsed_time:.2f} seconds.")

            if missing_models:
                st.warning(
                    f"⚠️ Missing artifacts for: {', '.join(sorted(missing_models))}. "
                    "Forecasts may use zero-shot defaults."
                )
            st.session_state["forecast_results"] = results
            st.session_state["forecast_item_id"] = item_id
            st.session_state["forecast_missing_models"] = sorted(missing_models)
            st.session_state["actual_overrides"] = {}

        except Exception as exc:  # pragma: no cover
            st.error(f"Forecast failed: {exc}")
            st.exception(exc)

    stored_results = st.session_state.get("forecast_results")
    stored_item_id = st.session_state.get("forecast_item_id")
    stored_missing = st.session_state.get("forecast_missing_models", [])
    stored_overrides = st.session_state.get("actual_overrides", {})

    if stored_results and stored_item_id:
        refresh_col_2 = st.columns([1, 3])[0]
        with refresh_col_2:
            refresh_actuals = st.button(
                "Fetch latest actuals",
                help="Pull the latest datapoints from the DB and refresh actual lines.",
            )
        if refresh_actuals:
            resample_choices = _normalize_resample_options(resample_labels)
            overrides: Dict[str, Dict[str, Dict[str, Any]]] = {}
            for resample_label, resample_freq in resample_choices:
                overrides[resample_label] = {}
                for column in selected_columns:
                    overrides[resample_label][column] = _load_actual_context(
                        stored_item_id,
                        days_back=int(days_back),
                        value_column=column,
                        resample_freq=resample_freq,
                    )
            st.session_state["actual_overrides"] = overrides
            stored_overrides = overrides
        if stored_missing:
            st.warning(
                f"⚠️ Missing artifacts for: {', '.join(stored_missing)}. Forecasts may use zero-shot defaults."
            )

        selected_resamples = [label for label, _ in _normalize_resample_options(resample_labels)]
        resample_items = [
            (label, stored_results[label])
            for label in selected_resamples
            if label in stored_results
        ]

        if not resample_items:
            st.info("No stored forecasts for the selected resampling options. Run forecast again.")
            return

        cols = st.columns(len(resample_items))
        for col, (resample_label, resample_results) in zip(cols, resample_items):
            with col:
                st.subheader(f"Resample: {resample_label}")
                for column, result in resample_results.items():
                    if column not in selected_columns:
                        continue
                    st.markdown(f"#### {column}")
                    context = result.get("context") or {}
                    count = len(context.get("timestamps", []))
                    st.caption(f"Loaded {count} data points for {column}.")
                    override_ctx = stored_overrides.get(resample_label, {}).get(column)
                    fig = _plot_history_and_forecast(
                        result,
                        title=f"{column} forecast (item {stored_item_id})",
                        actual_points=int(actual_points),
                        actual_override=override_ctx,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                buy_result = resample_results.get("buy_unit_price")
                sell_result = resample_results.get("sell_unit_price")
                if (
                    buy_result
                    and sell_result
                    and buy_result.get("models")
                    and sell_result.get("models")
                    and "buy_unit_price" in selected_columns
                    and "sell_unit_price" in selected_columns
                ):
                    model_names = [
                        m.get("model_name")
                        for m in buy_result.get("models", [])
                        if m.get("model_name")
                    ]
                    st.subheader("Profitability indicator")
                    ref_model = st.selectbox(
                        "Reference model",
                        options=model_names,
                        index=0,
                        key=f"profit_model_{resample_label}",
                    )
                    figs = _build_profitability_plots(
                        buy_result,
                        sell_result,
                        ref_model,
                        actual_points=int(actual_points),
                        buy_override=stored_overrides.get(resample_label, {}).get("buy_unit_price"),
                        sell_override=stored_overrides.get(resample_label, {}).get("sell_unit_price"),
                    )
                    if figs is not None:
                        fig_summary = _build_profitability_summary(
                            buy_result,
                            sell_result,
                            ref_model,
                            actual_points=int(actual_points),
                            buy_override=stored_overrides.get(resample_label, {}).get("buy_unit_price"),
                            sell_override=stored_overrides.get(resample_label, {}).get("sell_unit_price"),
                        )
                        if fig_summary is not None:
                            st.plotly_chart(fig_summary, use_container_width=True)
                        fig_sell, fig_buy = figs
                        st.plotly_chart(fig_sell, use_container_width=True)
                        st.plotly_chart(fig_buy, use_container_width=True)
                    else:
                        st.info(
                            "Profitability indicator requires matching buy/sell forecasts for the selected model."
                        )


def main() -> None:
    st.set_page_config(page_title="GW2ML Forecast", layout="wide")
    render_forecast_tab()


if __name__ == "__main__":
    main()
