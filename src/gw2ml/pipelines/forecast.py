from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

from gw2ml.data.loaders import load_gw2_series
from gw2ml.metrics.registry import get_metric
from gw2ml.pipelines.config import DEFAULT_CONFIG, Config, get_artifacts_dir, merge_config


def _load_best_artifact(item_id: int, config: Config) -> tuple[Any, Dict[str, Any]]:
    artifacts_root = get_artifacts_dir(config)
    item_dir = artifacts_root / str(item_id)
    model_path = item_dir / "best_model.pkl"
    metadata_path = item_dir / "best_metadata.json"

    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"No trained artifact found for item {item_id} in {item_dir}")

    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text())
    return model, metadata


def _load_all_artifacts(
    item_id: int, config: Config, allowed_models: list[str] | None = None
) -> list[tuple[str, Any, Dict[str, Any]]]:
    artifacts_root = get_artifacts_dir(config)
    item_dir = artifacts_root / str(item_id)
    entries: list[tuple[str, Any, Dict[str, Any]]] = []

    if not item_dir.exists():
        return entries

    # Per-model subdirectories
    for sub in item_dir.iterdir():
        if not sub.is_dir():
            continue
        model_path = sub / "best_model.pkl"
        metadata_path = sub / "best_metadata.json"
        if model_path.exists() and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            model_name = metadata.get("model_name") or sub.name
            if allowed_models and model_name not in allowed_models:
                continue
            model = joblib.load(model_path)
            entries.append((model_name, model, metadata))

    # Fallback to legacy single best if no subdirs found
    if not entries:
        try:
            model, metadata = _load_best_artifact(item_id, config)
            model_name = metadata.get("model_name", "best")
            if allowed_models and model_name not in allowed_models:
                return []
            entries.append((model_name, model, metadata))
        except FileNotFoundError:
            pass

    return entries


def forecast_item(item_id: int, override_config: Config | None = None, retrain: bool = False) -> Dict[str, Any]:
    """
    Load the latest trained model for an item and produce:
    - future forecast for the next horizon steps
    - a short backtest (historical forecasts) for quick performance view
    """
    config = merge_config(DEFAULT_CONFIG, override_config)
    horizon = int(config["forecast"]["horizon"])
    primary_metric = config.get("metric", {}).get("primary", "mape")
    metric_fn = get_metric(primary_metric)

    # Optional retrain before forecasting (full hyperparameter search + artifacts update)
    if retrain:
        from gw2ml.pipelines.train import train_items

        train_items([item_id], override_config=override_config)

    series_meta = load_gw2_series(
        item_id=item_id,
        days_back=config["data"]["days_back"],
        value_column=config["data"]["value_column"],
        fill_missing_dates=config["data"]["fill_missing_dates"],
    )

    requested_models = [m.get("name") for m in config.get("models", []) if m.get("name")]

    def _to_df(ts_obj):
        # Convert TimeSeries to DataFrame using to_dataframe() method
        try:
            df_local = ts_obj.to_dataframe()
            if df_local.shape[1] == 1 and "value" not in df_local.columns:
                df_local.columns = ["value"]
            return df_local
        except Exception:
            pass
        # Fallback: build from values and time_index.
        values = ts_obj.values(copy=False)
        index = ts_obj.time_index
        df_fallback = pd.DataFrame(values, index=index, columns=["value"] if values.shape[1] == 1 else None)
        if "value" not in df_fallback.columns and df_fallback.shape[1] == 1:
            df_fallback.columns = ["value"]
        return df_fallback

    models_payload: list[Dict[str, Any]] = []
    loaded = _load_all_artifacts(item_id, config, allowed_models=requested_models or None)
    missing_models = []
    if requested_models:
        found_names = {name for name, _, _ in loaded}
        missing_models = [m for m in requested_models if m not in found_names]

    # If retrain was requested or artifacts are missing, retrain then reload artifacts.
    if retrain or missing_models:
        if retrain or missing_models:
            from gw2ml.pipelines.train import train_items

            train_items([item_id], override_config=override_config)
            loaded = _load_all_artifacts(item_id, config, allowed_models=requested_models or None)
            if requested_models:
                found_names = {name for name, _, _ in loaded}
                missing_models = [m for m in requested_models if m not in found_names]

    if not loaded:
        raise FileNotFoundError(
            f"No artifacts available for item {item_id}. "
            f"Selected models={requested_models or 'all'}. "
            f"Missing={missing_models}. Consider retraining."
        )

    for model_name, model_obj, metadata in loaded:
        # Refit on latest data
        model_obj.fit(series_meta.series)
        last_ts = series_meta.series.time_index[-1]

        forecast = model_obj.predict(n=horizon)
        df_future = _to_df(forecast)
        df_future.index = pd.to_datetime(df_future.index).tz_localize(None)
        last_ts_naive = pd.to_datetime(last_ts).tz_localize(None)
        df_future = df_future[df_future.index > last_ts_naive]

        # Determine if model supports retrain=False (Global models like XGBoost)
        # Local models (ARIMA, ExponentialSmoothing) require retrain=True
        try:
            # Try with retrain=False first (faster for Global models)
            hist_forecast = model_obj.historical_forecasts(
                series=series_meta.series,
                start=0.8,
                forecast_horizon=horizon,
                stride=1,
                retrain=False,
                verbose=False,
            )
        except (ValueError, NotImplementedError):
            # Fallback to retrain=True for Local models
            hist_forecast = model_obj.historical_forecasts(
                series=series_meta.series,
                start=0.8,
                forecast_horizon=horizon,
                stride=1,
                retrain=True,
                verbose=False,
            )
        actual_series = series_meta.series.slice_intersect(hist_forecast)
        hist_metric = float(metric_fn(actual_series, hist_forecast)) if len(hist_forecast) > 0 else float("nan")

        df_hist_fc = _to_df(hist_forecast)
        df_hist_fc.index = pd.to_datetime(df_hist_fc.index).tz_localize(None)
        df_actual = _to_df(actual_series)
        df_actual.index = pd.to_datetime(df_actual.index).tz_localize(None)
        df_hist_fc = df_hist_fc.reindex(df_actual.index)

        models_payload.append(
            {
                "model_name": model_name,
                "params": metadata.get("params"),
                "trained_at": metadata.get("trained_at"),
                "future": {
                    "timestamps": df_future.index.astype(str).tolist(),
                    "values": df_future["value"].tolist(),
                },
                "history": {
                    "timestamps": df_actual.index.astype(str).tolist(),
                    "forecast": df_hist_fc.iloc[:, 0].tolist(),
                    "actual": df_actual.iloc[:, 0].tolist(),
                    "metric": hist_metric,
                },
            }
        )

    return {
        "item_id": item_id,
        "primary_metric": primary_metric,
        "forecast_horizon": horizon,
        "models": models_payload,
        "missing_models": missing_models,
    }


__all__ = ["forecast_item"]

