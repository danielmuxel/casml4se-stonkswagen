from __future__ import annotations

import time
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st

from app_header import render_page_header
from gw2ml.metrics.registry import get_metric, list_metrics
from gw2ml.modeling.registry import list_models
from gw2ml.data.loaders import list_items
from gw2ml.pipelines.config import DEFAULT_CONFIG
from gw2ml.pipelines.forecast import forecast_item
from gw2ml.utils import get_logger

logger = get_logger("evaluation_app")

ALLOWED_CONFIG_KEYS = {"data", "split", "forecast", "metric", "models", "train"}
VALUE_COLUMNS = [
    "buy_unit_price",
    "sell_unit_price",
    "buy_quantity",
    "sell_quantity",
]
RESAMPLE_OPTIONS = ["5min", "15min", "30min", "1h", "4h", "12h", "1d"]

CACHE_TIMEOUT = 60 * 60 * 24 * 7  # 7 days

MODEL_COLOR_MAP = {
    "ARIMA": "#ff7f0e",
    "ExponentialSmoothing": "#2ca02c",
    "XGBoost": "#d62728",
    "Chronos2": "#9467bd",
    "Chronos": "#9467bd",
}
FALLBACK_MODEL_COLORS = ["#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


@st.cache_data(ttl=CACHE_TIMEOUT, persist="disk", show_spinner=True)
def cached_list_items(limit: int = 30000):
    return list_items(limit=limit)


def _model_color(model_name: str) -> str:
    if model_name in MODEL_COLOR_MAP:
        return MODEL_COLOR_MAP[model_name]
    if "Chronos" in model_name:
        return MODEL_COLOR_MAP["Chronos"]
    import hashlib

    digest = hashlib.md5(model_name.encode("utf-8")).hexdigest()
    idx = int(digest, 16) % len(FALLBACK_MODEL_COLORS)
    return FALLBACK_MODEL_COLORS[idx]


def _format_resample_minutes(label: str) -> str:
    if label.endswith("min"):
        return label
    if label.endswith("h"):
        hours = int(label.replace("h", ""))
        return f"{hours * 60}min"
    if label.endswith("d"):
        days = int(label.replace("d", ""))
        return f"{days * 1440}min"
    return label


def _validate_eval_inputs(
    item_id_text: str, selected_models: List[str], days_back: int
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

    if days_back < 3:
        return False, "Need at least 3 days of data for train/val/test split."

    try:
        from darts import TimeSeries  # noqa: F401
    except ImportError as e:
        return False, f"Required dependency 'darts' not available: {e}"

    return True, None


def _sanitize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in cfg.items() if key in ALLOWED_CONFIG_KEYS}


def _normalize_resample_options(labels: List[str]) -> List[tuple[str, str | None]]:
    choices: List[tuple[str, str | None]] = []
    for label in labels:
        choices.append((label, label))
    return choices


def _build_override_config(
    days_back: int,
    value_column: str,
    horizon: int,
    selected_models: List[str],
    primary_metric: str,
    metrics: List[str],
    resample_freq: str | None,
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
        "metric": {"primary": primary_metric, "metrics": metrics},
        "train": {"force_grid_search": force_grid_search},
        "models": [{"name": name, "grid": None} for name in selected_models],
    }
    return _sanitize_config(override)


def _render_evaluation_result(
    result: Dict[str, Any],
    item_id: int,
    metrics: List[str],
    *,
    heading: str | None = None,
    legend_bottom: bool = False,
) -> None:
    if heading:
        st.markdown(f"#### {heading}")

    models_payload = result.get("models", [])
    missing_models = result.get("missing_models", [])

    if models_payload:
        successful_models = [m.get("model_name") for m in models_payload]
        st.caption(f"Models: {', '.join(successful_models)}")

        failed_backtests = [
            m.get("model_name")
            for m in models_payload
            if m.get("history", {}).get("metric") != m.get("history", {}).get("metric")
        ]
        if failed_backtests:
            st.warning(
                f"⚠️ Backtest failed for: {', '.join(failed_backtests)} "
                "(possible hardware incompatibility)."
            )

    if missing_models:
        st.warning(
            f"⚠️ Missing artifacts for: {', '.join(missing_models)}. Retrain to generate them."
        )

    if not models_payload and not missing_models:
        st.error("❌ All models failed to generate evaluation results. Check logs for details.")
        return

    future_rows = []
    for m in models_payload:
        f_ts = m.get("future", {}).get("timestamps", [])
        f_vals = m.get("future", {}).get("values", [])
        for ts, val in zip(f_ts, f_vals):
            future_rows.append({"timestamp": ts, "value": val, "model": m.get("model_name")})
    if future_rows:
        df_future = pd.DataFrame(future_rows)
        models_unique = df_future["model"].unique().tolist()
        color_map = {m: _model_color(m) for m in models_unique}
        fig_future = px.line(
            df_future,
            x="timestamp",
            y="value",
            color="model",
            title=f"Future forecasts for item {item_id}",
            color_discrete_map=color_map,
        )
        if legend_bottom:
            fig_future.update_layout(
                legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5)
            )
        st.plotly_chart(fig_future, use_container_width=True)

    hist_rows = []
    actual_rows = None
    for m in models_payload:
        h_ts = m.get("history", {}).get("timestamps", [])
        h_fc = m.get("history", {}).get("forecast", [])
        h_act = m.get("history", {}).get("actual", [])
        if h_ts and h_act and actual_rows is None:
            actual_rows = [
                {"timestamp": ts, "actual": val}
                for ts, val in zip(h_ts, h_act)
                if val is not None
            ]
        for ts, val in zip(h_ts, h_fc):
            hist_rows.append({"timestamp": ts, "forecast": val, "model": m.get("model_name")})
    if hist_rows:
        df_hist_fc = pd.DataFrame(hist_rows)
        models_unique = df_hist_fc["model"].unique().tolist()
        color_map = {m: _model_color(m) for m in models_unique}
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
        if legend_bottom:
            fig_h.update_layout(
                legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5)
            )
        st.plotly_chart(fig_h, use_container_width=True)

    if models_payload:
        metrics_data = []
        for m in models_payload:
            h_fc = m.get("history", {}).get("forecast", [])
            h_act = m.get("history", {}).get("actual", [])

            row = {"Model": m.get("model_name")}

            if h_fc and h_act and len(h_fc) == len(h_act):
                from darts import TimeSeries

                try:
                    ts_actual = TimeSeries.from_values(h_act)
                    ts_forecast = TimeSeries.from_values(h_fc)

                    for metric_name in metrics:
                        try:
                            metric_fn = get_metric(metric_name)
                            value = metric_fn(ts_actual, ts_forecast)
                            row[metric_name.upper()] = (
                                round(float(value), 4) if not pd.isna(value) else "NaN"
                            )
                        except Exception:
                            row[metric_name.upper()] = "Error"
                except Exception:
                    for metric_name in metrics:
                        row[metric_name.upper()] = "Error"
            else:
                for metric_name in metrics:
                    row[metric_name.upper()] = "N/A"

            metrics_data.append(row)

        if metrics_data:
            df_metrics = pd.DataFrame(metrics_data)
            st.subheader("Model Performance Metrics")

            metric_cols = [col for col in df_metrics.columns if col != "Model"]

            def highlight_min(s: pd.Series) -> list[str]:
                """Highlight the minimum value in a Series (lowest error is best)."""
                if s.name not in metric_cols:
                    return [""] * len(s)
                numeric_s = pd.to_numeric(s, errors="coerce")
                if numeric_s.isna().all():
                    return [""] * len(s)
                is_min = numeric_s == numeric_s.min()
                return [
                    "background-color: #2E8B57; font-weight: bold" if v else ""
                    for v in is_min
                ]

            styled_df = df_metrics.style.apply(highlight_min)
            st.dataframe(styled_df, use_container_width=True, hide_index=True)


def render_evaluation_tab() -> None:
    render_page_header("GW2ML: Evaluation & Backtest")

    with st.sidebar:
        st.header("Evaluation Configuration")

        days_back = st.number_input(
            "History range (days back)",
            min_value=1,
            max_value=365,
            value=30,
            step=1,
            help="Use the last X days for evaluation.",
        )

        resample_labels = st.multiselect(
            "Resample intervals",
            options=RESAMPLE_OPTIONS,
            default=["5min"],
            help="Run evaluation for one or more resampling options.",
        )

        selected_columns = st.multiselect(
            "Forecast fields",
            options=VALUE_COLUMNS,
            default=VALUE_COLUMNS,
            help="Select which fields to evaluate.",
        )
        horizon = st.number_input(
            "Forecast horizon (steps)",
            min_value=1,
            max_value=288,
            value=12,
            step=1,
        )

        all_models = list_models()
        selected_models = st.multiselect("Models", options=all_models, default=all_models)

        available_metrics = list_metrics()
        primary_metric = st.selectbox("Primary metric", options=available_metrics, index=0)
        metrics = st.multiselect("Metrics to compute", options=available_metrics, default=available_metrics)

        retrain_before = st.checkbox("Retrain before evaluation", value=False)
        force_grid_search = st.checkbox(
            "Force grid search",
            value=False,
            help="If enabled, retraining will run a full grid search instead of reusing cached params.",
        )

    st.subheader("Evaluation")
    items = cached_list_items()
    selected_items = st.multiselect(
        "Search for items",
        options=items,
        default=[],
        format_func=lambda x: f"[{x['item_id']}] {x['item_name']}",
        help="Type to search and select one or more items",
    )
    if st.button("Run evaluation"):
        start_time = time.time()
        try:
            if not selected_items:
                st.error("Please select at least one item to evaluate.")
                return
            is_valid, error_msg = _validate_eval_inputs(str(selected_items[0]["item_id"]), selected_models, days_back)
            if not is_valid:
                st.error(f"❌ Validation failed: {error_msg}")
                return

            item_ids = [int(item["item_id"]) for item in selected_items]
            item_name_map = {
                int(item["item_id"]): item.get("item_name", str(item["item_id"]))
                for item in selected_items
            }

            with st.status("Processing evaluations...", expanded=True) as status:
                if retrain_before:
                    st.write(f"🔄 Retraining models: {', '.join(selected_models)}")

                st.write(f"📊 Loading {days_back} days of data...")
                st.write(f"🔮 Generating {horizon}-step forecast...")
                st.write("📈 Running backtest validation...")

                resample_choices = _normalize_resample_options(resample_labels)
                if not resample_choices:
                    st.error("Please select at least one resample option.")
                    return

                if not selected_columns:
                    st.error("Please select at least one field to evaluate.")
                    return

                results: Dict[int, Dict[str, Dict[str, Dict[str, Any]]]] = {}
                total_steps = len(item_ids) * len(resample_choices) * len(selected_columns)
                progress = st.progress(0)
                step_idx = 0
                for item_id in item_ids:
                    item_label = item_name_map.get(item_id, str(item_id))
                    results[item_id] = {}
                    for label, freq in resample_choices:
                        st.write(f"⚙️ Evaluating {item_label} ({label})")
                        results[item_id][label] = {}
                        for value_column in selected_columns:
                            step_idx += 1
                            payload = _build_override_config(
                                days_back,
                                value_column,
                                horizon,
                                selected_models,
                                primary_metric,
                                metrics,
                                freq,
                                force_grid_search,
                            )
                            result = forecast_item(
                                item_id,
                                override_config=payload,
                                retrain=retrain_before,
                                include_backtest=True,
                                include_history=False,
                            )
                            results[item_id][label][value_column] = result
                            progress.progress(step_idx / total_steps)

                status.update(label="✓ Evaluations complete", state="complete")

            elapsed_time = time.time() - start_time
            st.success(f"Evaluation generated in {elapsed_time:.2f} seconds.")
            if results:
                for item_id in item_ids:
                    item_results = results.get(item_id)
                    if not item_results:
                        continue
                    item_label = item_name_map.get(item_id, str(item_id))
                    with st.expander(f"[{item_id}] {item_label}", expanded=False):
                        cols = st.columns(len(item_results))
                        for col, (label, per_field) in zip(cols, item_results.items()):
                            with col:
                                minutes_label = _format_resample_minutes(label)
                                st.subheader(f"Resample: {label} ({minutes_label})")
                                for value_column, result in per_field.items():
                                    st.markdown(f"#### {value_column} ({minutes_label})")
                                    _render_evaluation_result(
                                        result,
                                        item_id,
                                        metrics,
                                        heading=None,
                                        legend_bottom=len(item_results) >= 2,
                                    )

        except Exception as exc:  # pragma: no cover
            st.error(f"Evaluation failed: {exc}")
            st.exception(exc)


def main() -> None:
    st.set_page_config(page_title="GW2ML Evaluation", layout="wide")
    render_evaluation_tab()


if __name__ == "__main__":
    main()
