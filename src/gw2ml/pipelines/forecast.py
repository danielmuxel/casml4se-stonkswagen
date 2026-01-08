from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

from gw2ml.data.loaders import load_gw2_series
from gw2ml.evaluation.backtest import walk_forward_backtest
from gw2ml.metrics.registry import get_metric
from gw2ml.pipelines.config import DEFAULT_CONFIG, Config, get_artifacts_dir, merge_config

logger = logging.getLogger(__name__)


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
    logger.info(f"Starting forecast for item {item_id}")
    config = merge_config(DEFAULT_CONFIG, override_config)
    horizon = int(config["forecast"]["horizon"])
    backtest_split = float(config["forecast"].get("backtest_split", 0.8))  # Default to 80/20 split
    primary_metric = config.get("metric", {}).get("primary", "mape")
    metric_fn = get_metric(primary_metric)

    logger.info(f"  Forecast horizon: {horizon} steps")
    logger.info(f"  Backtest split: {backtest_split:.0%} train / {(1-backtest_split):.0%} test")

    # Optional retrain before forecasting (full hyperparameter search + artifacts update)
    if retrain:
        logger.info("  Retraining requested - running full training pipeline first...")
        from gw2ml.pipelines.train import train_items

        train_items([item_id], override_config=override_config)

    series_meta = load_gw2_series(
        item_id=item_id,
        days_back=config["data"]["days_back"],
        value_column=config["data"]["value_column"],
        fill_missing_dates=config["data"]["fill_missing_dates"],
        resample_freq=config["data"].get("resample_freq"),
    )
    logger.info(f"  Loaded {series_meta.num_points} data points")

    requested_models = [m.get("name") for m in config.get("models", []) if m.get("name")]
    logger.info(f"  Requested models: {requested_models or 'all available'}")

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
    logger.info(f"  Found {len(loaded)} trained model(s): {[name for name, _, _ in loaded]}")

    missing_models = []
    if requested_models:
        found_names = {name for name, _, _ in loaded}
        missing_models = [m for m in requested_models if m not in found_names]
        if missing_models:
            logger.warning(f"  Missing artifacts for: {missing_models}")

    # If retrain was requested or artifacts are missing, retrain then reload artifacts.
    if retrain or missing_models:
        logger.info("  Triggering retraining...")
        if retrain or missing_models:
            from gw2ml.pipelines.train import train_items

            train_items([item_id], override_config=override_config)
            loaded = _load_all_artifacts(item_id, config, allowed_models=requested_models or None)
            if requested_models:
                found_names = {name for name, _, _ in loaded}
                missing_models = [m for m in requested_models if m not in found_names]

    if not loaded:
        error_msg = (
            f"No trained artifacts available for item {item_id}. "
            f"Selected models={requested_models or 'all'}. "
            f"Missing={missing_models}. Train models first before forecasting."
        )
        logger.error(f"  ✗ {error_msg}")
        raise FileNotFoundError(error_msg)

    for idx, (model_name, model_obj, metadata) in enumerate(loaded, 1):
        logger.info(f"  [{idx}/{len(loaded)}] Processing {model_name}...")

        # STEP 1: Split data into train/test for proper evaluation
        split_idx = int(len(series_meta.series) * backtest_split)
        train_series = series_meta.series[:split_idx]
        test_series = series_meta.series[split_idx:]

        logger.debug(f"    Data split: train={len(train_series)}, test={len(test_series)}")

        # STEP 2: Evaluate on test data (model trained ONLY on train data)
        hist_forecast = None
        actual_series = None
        hist_metric = float("nan")
        try:
            # Determine stride for backtest
            # stride=1: Iterative 1-step-ahead (model retrains after each step, can "react" to data)
            # stride=horizon: True multi-step-ahead (model commits to N-step forecast, more realistic)
            # stride=horizon//N: Compromise - retrains every N steps for balance
            backtest_stride_config = config["forecast"].get("backtest_stride", "auto")
            if backtest_stride_config == "auto":
                # Use stride = max(horizon//4, 6) for balance between density and realism
                # This means the model retrains every ~30 min (6 steps = 30 min at 5-min intervals)
                # rather than every 5 minutes (stride=1) or every 2 hours (stride=horizon)
                backtest_stride = max(horizon // 4, 6)
                logger.info(f"  Using auto stride={backtest_stride} (horizon={horizon}, ~{backtest_stride*5}min between retrains)")
            else:
                backtest_stride = int(backtest_stride_config)

            logger.debug(f"    Running train/test evaluation (stride={backtest_stride}, horizon={horizon})...")
            backtest_result = walk_forward_backtest(
                model_class=model_obj.__class__,
                model_params=metadata.get("params", {}),
                series=series_meta.series,  # Full series for context
                train_series=train_series,  # Train on this ONLY
                test_series=test_series,  # Evaluate on this
                forecast_horizon=horizon,
                stride=backtest_stride,
                verbose=False,
            )
            hist_forecast = backtest_result.forecasts
            actual_series = backtest_result.actuals
            hist_metric = float(metric_fn(actual_series, hist_forecast)) if len(hist_forecast) > 0 else float("nan")
            logger.info(f"    {model_name} test set {primary_metric}: {hist_metric:.4f} (train={backtest_result.train_size}, test={backtest_result.test_size})")
        except Exception as exc:
            exc_msg = str(exc)
            if "MPS" in exc_msg or "mps" in exc_msg or "float32" in exc_msg or "dtype" in exc_msg or "_model_call" in exc_msg:
                logger.warning(f"    ⚠️ {model_name} backtest failed: Hardware incompatibility (MPS/GPU issue)")
            else:
                logger.warning(f"    ⚠️ {model_name} backtest failed: {exc}")
            logger.info(f"    Continuing with future forecast only (backtest unavailable)")

        # STEP 3: Retrain on ALL data for production forecast
        try:
            logger.debug(f"    Retraining {model_name} on ALL data for production forecast...")
            model_obj.fit(series_meta.series)
            last_ts = series_meta.series.time_index[-1]

            logger.debug(f"    Generating {horizon}-step future forecast...")
            forecast = model_obj.predict(n=horizon)
        except Exception as exc:
            exc_msg = str(exc)
            if "MPS" in exc_msg or "mps" in exc_msg or "float32" in exc_msg or "dtype" in exc_msg:
                logger.error(f"    ✗ {model_name} production forecast failed: Hardware incompatibility (MPS/GPU issue)")
            else:
                logger.error(f"    ✗ {model_name} production forecast failed: {exc}")
            logger.warning(f"    Skipping {model_name}")
            continue

        df_future = _to_df(forecast)
        df_future.index = pd.to_datetime(df_future.index).tz_localize(None)
        last_ts_naive = pd.to_datetime(last_ts).tz_localize(None)
        df_future = df_future[df_future.index > last_ts_naive]

        # Handle backtest results (may be None if backtest failed)
        if hist_forecast is not None and actual_series is not None:
            df_hist_fc = _to_df(hist_forecast)
            df_hist_fc.index = pd.to_datetime(df_hist_fc.index).tz_localize(None)
            df_actual = _to_df(actual_series)
            df_actual.index = pd.to_datetime(df_actual.index).tz_localize(None)
            # Note: No reindexing needed - slice_intersect already aligned them
            # Reindexing can introduce forward-fill artifacts that look like data leakage
        else:
            # Backtest failed, create empty dataframes
            df_hist_fc = pd.DataFrame()
            df_actual = pd.DataFrame()

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
                    "timestamps": df_actual.index.astype(str).tolist() if not df_actual.empty else [],
                    "forecast": df_hist_fc.iloc[:, 0].tolist() if not df_hist_fc.empty else [],
                    "actual": df_actual.iloc[:, 0].tolist() if not df_actual.empty else [],
                    "metric": hist_metric,
                },
            }
        )

    if models_payload:
        logger.info(f"✓ Forecast complete for item {item_id} - {len(models_payload)} model(s) succeeded")
    else:
        logger.warning(f"⚠️ Forecast complete for item {item_id} but no models produced forecasts")

    # Determine what stride was used for backtest
    backtest_stride_config = config["forecast"].get("backtest_stride", "auto")
    if backtest_stride_config == "auto":
        actual_stride = max(horizon // 4, 6)
    else:
        actual_stride = int(backtest_stride_config)

    return {
        "item_id": item_id,
        "primary_metric": primary_metric,
        "forecast_horizon": horizon,
        "models": models_payload,
        "missing_models": missing_models,
        "backtest_config": {
            "stride": actual_stride,
            "horizon": horizon,
            "mode": backtest_stride_config,
        },
    }


__all__ = ["forecast_item"]

