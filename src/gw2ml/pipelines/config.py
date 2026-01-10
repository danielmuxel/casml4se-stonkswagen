from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from gw2ml.paths import PROJECT_ROOT

Config = Dict[str, Any]

# Default configuration; can be overridden by callers.
DEFAULT_CONFIG: Config = {
    "data": {
        "value_column": "buy_unit_price",
        "days_back": 30,
        "fill_missing_dates": True,
        "min_points": 50,
    },
    "split": {
        "train": 0.7,
        "val": 0.15,
        "test_days": 7,
        "val_days": None,
    },
    "forecast": {"horizon": 12},
    "paths": {
        "artifacts_dir": str((PROJECT_ROOT / "models").resolve()),
        "use_s3": False,
        "s3_prefix": None,
        "bucket_name": "ost-s3",
    },
    "metric": {"primary": "mape", "metrics": ["mape"]},
    "models": [
        {"name": "ARIMA", "grid": None},
        {"name": "ExponentialSmoothing", "grid": None},
        {"name": "XGBoost", "grid": None},
        {"name": "Chronos2", "grid": None},
    ],
}


def get_artifacts_dir(config: Config) -> Path:
    """Return resolved artifacts directory, ensuring it exists."""
    artifacts_dir = Path(config["paths"]["artifacts_dir"]).expanduser().resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return artifacts_dir


def merge_config(base: Config, override: Config | None = None) -> Config:
    """Shallow-merge override into base (one level deep)."""
    if not override:
        return dict(base)
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            merged = dict(result[key])
            merged.update(value)
            result[key] = merged
        else:
            result[key] = value
    return result


__all__ = ["DEFAULT_CONFIG", "get_artifacts_dir", "merge_config", "Config"]

