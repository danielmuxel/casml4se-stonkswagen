from __future__ import annotations

import json
import time
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import joblib
from darts import TimeSeries

from gw2ml.data.loaders import GW2Series, load_gw2_series
from gw2ml.metrics.registry import get_metric
from gw2ml.modeling.registry import get_default_grid, get_model
from gw2ml.pipelines.config import DEFAULT_CONFIG, Config, get_artifacts_dir, merge_config
from gw2ml.utils import get_logger

logger = get_logger("pipelines.train")


def _iter_param_grid(grid: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    if not grid:
        yield {}
        return
    keys = list(grid.keys())
    values = [v if isinstance(v, list) else [v] for v in grid.values()]
    for combo in product(*values):
        yield {key: value for key, value in zip(keys, combo)}


def _compute_metrics(metric_names: List[str], y_true: TimeSeries, y_pred: TimeSeries) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for name in metric_names:
        metric_fn = get_metric(name)
        metrics[name] = metric_fn(y_true, y_pred)
    return metrics


def _split_series(series: GW2Series, split_cfg: Dict[str, Any]) -> Tuple[TimeSeries, TimeSeries | None, TimeSeries | None]:
    train_ratio = split_cfg.get("train", 0.7)
    val_ratio = split_cfg.get("val", 0.15)
    test_days = split_cfg.get("test_days")
    val_days = split_cfg.get("val_days")

    if test_days:
        parts = series.split_days(test_days=test_days, val_days=val_days)
        if len(parts) == 2:
            train_ts, test_ts = parts
            return train_ts, None, test_ts
        train_ts, val_ts, test_ts = parts
        return train_ts, val_ts, test_ts

    parts = series.split(train=train_ratio, val=val_ratio)
    if len(parts) == 2:
        train_ts, test_ts = parts
        return train_ts, None, test_ts
    train_ts, val_ts, test_ts = parts
    return train_ts, val_ts, test_ts


def _save_artifact(model: Any, metadata: Dict[str, Any], target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    model_path = target_dir / "best_model.pkl"
    metadata_path = target_dir / "best_metadata.json"
    joblib.dump(model, model_path)
    metadata_path.write_text(json.dumps(metadata, indent=2))


def _maybe_upload_s3(config: Config, item_dir: Path) -> None:
    if not config["paths"].get("use_s3"):
        return
    prefix = config["paths"].get("s3_prefix")
    bucket = config["paths"].get("bucket_name") or "ost-s3"
    if not prefix:
        return
    try:
        from gw2ml.data.s3_sync import upload_folder_to_s3
    except Exception:
        return
    try:
        upload_folder_to_s3(
            local_folder=str(item_dir),
            s3_folder_prefix=prefix if prefix.endswith("/") else f"{prefix}/",
            unix_timestamp=int(time.time()),
            bucket_name=bucket,
        )
    except Exception:
        # If upload fails, keep local artifacts; avoid raising to keep pipeline running.
        return


def train_items(item_ids: List[int], override_config: Config | None = None) -> List[Dict[str, Any]]:
    """
    Train and select the best model per item id.

    Returns a list of metadata dictionaries (one per item).
    """
    logger.info(f"Training items: {item_ids}")
    config = merge_config(DEFAULT_CONFIG, override_config)
    artifacts_root = get_artifacts_dir(config)
    metric_cfg = config.get("metric", {})
    primary_metric = metric_cfg.get("primary", "mape")
    metric_names: List[str] = metric_cfg.get("metrics", [primary_metric])

    logger.info(f"Starting training for {len(item_ids)} item(s): {item_ids}")
    logger.info(f"Models to train: {[m['name'] for m in config['models']]}")

    results: List[Dict[str, Any]] = []

    for idx, item_id in enumerate(item_ids, 1):
        logger.info(f"[{idx}/{len(item_ids)}] Processing item {item_id}...")
        series_meta = load_gw2_series(
            item_id=item_id,
            days_back=config["data"]["days_back"],
            value_column=config["data"]["value_column"],
            fill_missing_dates=config["data"]["fill_missing_dates"],
            resample_freq=config["data"].get("resample_freq"),
        )
        logger.info(f"  Loaded {series_meta.num_points} data points for item {item_id}")

        if series_meta.num_points < config["data"]["min_points"]:
            logger.warning(f"  Item {item_id} has insufficient data ({series_meta.num_points} < {config['data']['min_points']}), skipping")
            continue

        train_ts, val_ts, test_ts = _split_series(series_meta, config["split"])
        logger.info(f"  Split: train={len(train_ts)}, val={len(val_ts) if val_ts else 0}, test={len(test_ts) if test_ts else 0}")

        per_model_results: List[Dict[str, Any]] = []

        for model_spec in config["models"]:
            model_name = model_spec["name"]
            logger.info(f"  Training {model_name}...")

            try:
                model_cls = get_model(model_name)
                grid = model_spec.get("grid") or get_default_grid(model_name)

                grid_size = sum(1 for _ in _iter_param_grid(grid))
                logger.info(f"    Grid search: {grid_size} parameter combination(s)")

                model_best_score = float("inf")
                model_best: Any = None
                model_best_params: Dict[str, Any] = {}
                model_best_val: Dict[str, float] | None = None
                model_best_test: Dict[str, float] | None = None
            except Exception as exc:
                logger.error(f"    ✗ Failed to initialize {model_name}: {exc}", exc_info=False)
                logger.warning(f"    Skipping {model_name} entirely due to initialization error")
                continue

            for param_idx, params in enumerate(_iter_param_grid(grid), 1):
                if model_name == "ExponentialSmoothing":
                    sp = params.get("seasonal_periods")
                    min_points_needed = 10
                    if sp:
                        min_points_needed = max(min_points_needed, 2 * int(sp))
                    if len(train_ts) < min_points_needed:
                        continue

                if grid_size > 1:
                    logger.debug(f"    [{param_idx}/{grid_size}] Testing params: {params}")

                try:
                    candidate = model_cls(**params)
                    candidate.fit(train_ts)
                except Exception as exc:
                    # Skip failing candidate; e.g., insufficient data, MPS/GPU incompatibility, etc.
                    exc_msg = str(exc)
                    if "MPS" in exc_msg or "mps" in exc_msg or "float32" in exc_msg or "dtype" in exc_msg:
                        logger.warning(f"    Skipped {model_name} with params={params}: Hardware incompatibility (MPS/GPU issue)")
                    elif "data" in exc_msg.lower() or "points" in exc_msg.lower():
                        logger.debug(f"    Skipped {model_name} with params={params}: Insufficient data")
                    else:
                        logger.warning(f"    Skipped {model_name} with params={params}: {exc}")
                    continue

                local_val_ts = val_ts or test_ts
                if local_val_ts is None:
                    continue

                val_pred = candidate.predict(n=len(local_val_ts))
                val_metrics = _compute_metrics(metric_names, local_val_ts, val_pred)
                score = val_metrics.get(primary_metric)
                if score is None:
                    continue

                if score >= model_best_score:
                    continue

                model_best_score = score
                model_best = candidate
                model_best_params = params
                model_best_val = val_metrics

            if model_best is None:
                logger.warning(f"    ✗ No valid model found for {model_name} (all parameter combinations failed)")
                continue

            logger.info(f"    Best {model_name}: {primary_metric}={model_best_score:.4f}, params={model_best_params}")

            try:
                combined_train = train_ts if val_ts is None else TimeSeries.concatenate([train_ts, val_ts])
                logger.debug(f"    Retraining {model_name} on combined train+val data...")
                model_best.fit(combined_train)

                model_best_test_metrics: Dict[str, float] | None = None
                if test_ts is not None:
                    test_pred = model_best.predict(n=len(test_ts))
                    model_best_test_metrics = _compute_metrics(metric_names, test_ts, test_pred)

                per_model_results.append(
                    {
                        "model_name": model_name,
                        "params": model_best_params,
                        "primary_metric": primary_metric,
                        "metrics": {
                            "val": model_best_val,
                            "test": model_best_test_metrics,
                        },
                        "model_obj": model_best,
                    }
                )
            except Exception as exc:
                logger.error(f"    ✗ Failed to retrain {model_name} on combined data: {exc}", exc_info=False)
                logger.warning(f"    Skipping {model_name} (retraining failed - possible hardware incompatibility)")
                continue

        if not per_model_results:
            logger.error(f"  ✗ No models successfully trained for item {item_id} - all models failed!")
            logger.info(f"  Common causes: hardware incompatibility (e.g., MPS/GPU issues), insufficient data, or dependency conflicts")
            continue

        # Pick global best by primary metric using val metric.
        global_best = min(
            per_model_results,
            key=lambda m: float(m["metrics"]["val"].get(primary_metric, float("inf")) if m["metrics"]["val"] else float("inf")),
        )
        logger.info(f"  Best overall model: {global_best['model_name']} ({primary_metric}={global_best['metrics']['val'].get(primary_metric, 'N/A')})")

        item_dir = artifacts_root / str(item_id)
        item_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"  Saving artifacts to {item_dir}")

        # Save per-model artifacts
        for model_entry in per_model_results:
            meta = {
                "item_id": item_id,
                "model_name": model_entry["model_name"],
                "params": model_entry["params"],
                "primary_metric": primary_metric,
                "metrics": model_entry["metrics"],
                "data": {
                    "value_column": config["data"]["value_column"],
                    "days_back": config["data"]["days_back"],
                    "fill_missing_dates": config["data"]["fill_missing_dates"],
                    "num_points": series_meta.num_points,
                    "start_time": series_meta.start_time.isoformat(),
                    "end_time": series_meta.end_time.isoformat(),
                },
                "trained_at": int(time.time()),
            }
            model_dir = item_dir / model_entry["model_name"]
            _save_artifact(model_entry["model_obj"], meta, model_dir)

        # Save global best for compatibility
        global_meta = {
            "item_id": item_id,
            "model_name": global_best["model_name"],
            "params": global_best["params"],
            "primary_metric": primary_metric,
            "metrics": global_best["metrics"],
            "data": {
                "value_column": config["data"]["value_column"],
                "days_back": config["data"]["days_back"],
                "fill_missing_dates": config["data"]["fill_missing_dates"],
                "num_points": series_meta.num_points,
                "start_time": series_meta.start_time.isoformat(),
                "end_time": series_meta.end_time.isoformat(),
            },
            "trained_at": int(time.time()),
            "models": [
                {k: v for k, v in m.items() if k != "model_obj"} for m in per_model_results
            ],
        }

        _save_artifact(global_best["model_obj"], global_meta, item_dir)
        _maybe_upload_s3(config, item_dir)

        results.append(global_meta)
        logger.info(f"  ✓ Item {item_id} complete")

    logger.info(f"Training complete! Processed {len(results)}/{len(item_ids)} item(s)")
    return results


__all__ = ["train_items"]

