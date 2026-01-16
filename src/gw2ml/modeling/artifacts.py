from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from gw2ml.paths import PROJECT_ROOT


def _ensure_utc(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _normalize_model_key(model_key: str) -> str:
    key = (model_key or "").strip()
    if not key:
        raise ValueError("model_key must be non-empty.")
    return key.replace("/", "_").replace("\\", "_")


def _models_root() -> Path:
    env_root = os.getenv("GW2ML_MODELS_DIR")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return PROJECT_ROOT / "models"


def persist_model(model: Any, *, model_key: str, run_date: datetime | None = None) -> Path:
    """Persist a model artifact under the configured models directory."""
    key = _normalize_model_key(model_key)
    timestamp = _ensure_utc(run_date)
    slug = timestamp.strftime("%Y%m%dT%H%M%SZ")
    model_dir = _models_root() / key
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = model_dir / f"model_{slug}.joblib"
    joblib.dump(model, artifact_path)
    (model_dir / "latest.txt").write_text(artifact_path.name)
    return artifact_path


def load_model(model_key: str, *, run_date: datetime | None = None) -> Any:
    """Load a persisted model artifact by key (and optional run date)."""
    key = _normalize_model_key(model_key)
    model_dir = _models_root() / key
    if run_date is not None:
        slug = _ensure_utc(run_date).strftime("%Y%m%dT%H%M%SZ")
        artifact_path = model_dir / f"model_{slug}.joblib"
        if not artifact_path.exists():
            raise FileNotFoundError(f"No artifact found for model_key={model_key} at {slug}.")
        return joblib.load(artifact_path)

    latest_marker = model_dir / "latest.txt"
    if latest_marker.exists():
        candidate = model_dir / latest_marker.read_text().strip()
        if candidate.exists():
            return joblib.load(candidate)

    candidates = sorted(model_dir.glob("model_*.joblib"))
    if not candidates:
        raise FileNotFoundError(f"No artifacts found for model_key={model_key}.")
    return joblib.load(candidates[-1])


__all__ = ["persist_model", "load_model"]
