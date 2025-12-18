from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from gw2ml.modeling import MedianPriceModel, persist_model
from gw2ml.pipeline.data_preparation import (
    DatasetMeta,
    build_pipeline_context,
    load_latest_validation,
    load_training_snapshot,
)
from gw2ml.pipeline.feature_hooks import apply_augmentors
from gw2ml.paths import PROJECT_ROOT


@dataclass(slots=True)
class TrainArimaConfig:
    """Configuration payload controlling the training pipeline."""

    model_key: str = "arima"
    run_date: datetime = field(default_factory=lambda: datetime.now(UTC))
    item_ids: tuple[int, ...] | None = None
    snapshot_grace_minutes: int = 4
    validation_lookback_days: int = 30
    rows_per_item: int = 2000
    target_column: str = "sell_unit_price"
    output_dir: Path = PROJECT_ROOT / "data" / "train_arima"
    connection_url: str | None = None


@dataclass(slots=True)
class TrainArimaArtifacts:
    train_path: Path
    validation_path: Path
    model_path: Path
    metadata: dict[str, DatasetMeta | dict[str, float]]


def _ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _export_dataframe(df: pd.DataFrame, path: Path) -> Path:
    df.to_csv(path, index=False)
    return path


def _resolve_validation_ids(train_df: pd.DataFrame, explicit_ids: Iterable[int] | None) -> list[int]:
    if explicit_ids:
        return sorted({int(item_id) for item_id in explicit_ids})
    if "item_id" not in train_df.columns:
        return []
    return sorted(train_df["item_id"].dropna().astype(int).unique().tolist())


def _mean_absolute_error(actuals: pd.Series, predictions: pd.Series) -> float:
    if actuals.empty:
        return 0.0
    return float((actuals - predictions).abs().mean())


def _mean_absolute_percentage_error(actuals: pd.Series, predictions: pd.Series) -> float:
    safe_actuals = actuals.replace(0, pd.NA).dropna()
    aligned_predictions = predictions.loc[safe_actuals.index]
    if safe_actuals.empty:
        return 0.0
    percentage_errors = ((safe_actuals - aligned_predictions).abs() / safe_actuals).dropna()
    if percentage_errors.empty:
        return 0.0
    return float(percentage_errors.mean() * 100)


def run(config: TrainArimaConfig | None = None) -> TrainArimaArtifacts:
    """Execute the ARIMA training pipeline."""
    cfg = config or TrainArimaConfig()
    context = build_pipeline_context(
        cfg.model_key,
        run_date=cfg.run_date,
        connection_url=cfg.connection_url,
        lookback_days=cfg.validation_lookback_days,
    )

    train_df, train_meta = load_training_snapshot(
        context,
        item_ids=cfg.item_ids,
        grace_minutes=cfg.snapshot_grace_minutes,
    )
    validation_ids = _resolve_validation_ids(train_df, cfg.item_ids)
    val_df, val_meta = load_latest_validation(
        context,
        item_ids=validation_ids or None,
        lookback_days=cfg.validation_lookback_days,
        rows_per_item=cfg.rows_per_item,
    )

    train_df = apply_augmentors(train_df, context, "training")
    val_df = apply_augmentors(val_df, context, "validation")

    model = MedianPriceModel(value_column=cfg.target_column).fit(train_df)

    evaluated_validation = val_df.dropna(subset=[cfg.target_column])
    predictions = model.predict(evaluated_validation)
    actuals = evaluated_validation[cfg.target_column].astype(float)
    metrics = {
        "mae": _mean_absolute_error(actuals, predictions),
        "mape": _mean_absolute_percentage_error(actuals, predictions),
        "validation_rows": int(len(evaluated_validation)),
    }

    artifact_dir = _ensure_output_dir(cfg.output_dir)
    slug = context.run_date.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    train_path = _export_dataframe(train_df, artifact_dir / f"{cfg.model_key}_train_{slug}.csv")
    validation_path = _export_dataframe(val_df, artifact_dir / f"{cfg.model_key}_validation_{slug}.csv")
    model_path = persist_model(model, model_key=cfg.model_key, run_date=context.run_date)

    metadata = {"training": train_meta, "validation": val_meta, "metrics": metrics}
    return TrainArimaArtifacts(
        train_path=train_path,
        validation_path=validation_path,
        model_path=model_path,
        metadata=metadata,
    )


def _print_summary(artifacts: TrainArimaArtifacts) -> None:
    metrics = artifacts.metadata.get("metrics", {})
    mae = metrics.get("mae", 0.0)
    mape = metrics.get("mape", 0.0)
    print(f"Saved training set to {artifacts.train_path}")
    print(f"Saved validation set to {artifacts.validation_path}")
    print(f"Persisted model artifact to {artifacts.model_path}")
    print(f"Metrics -> MAE: {mae:.2f}, MAPE: {mape:.2f}%")


if __name__ == "__main__":
    summary = run()
    _print_summary(summary)

