from __future__ import annotations

from typing import Callable, Dict, List

from darts.metrics import mae, mape, rmse, smape

MetricFn = Callable[[object, object], float]

_REGISTRY: Dict[str, MetricFn] = {}


def register_metric(name: str, fn: MetricFn) -> None:
    """Register or override a metric by name (case-insensitive)."""
    key = name.strip().lower()
    if not key:
        raise ValueError("Metric name must be non-empty.")
    _REGISTRY[key] = fn


def get_metric(name: str) -> MetricFn:
    """Return a registered metric callable."""
    key = name.strip().lower()
    if key not in _REGISTRY:
        raise KeyError(f"Metric '{name}' is not registered.")
    return _REGISTRY[key]


def list_metrics() -> List[str]:
    """List available metric names."""
    return sorted(_REGISTRY.keys())


def _safe_mape(y_true, y_pred) -> float:
    try:
        return float(mape(y_true, y_pred))
    except ValueError:
        # MAPE is undefined if y_true contains zeros.
        return float("nan")


# Default metrics
register_metric("mape", _safe_mape)
register_metric("rmse", lambda y_true, y_pred: float(rmse(y_true, y_pred)))
register_metric("mae", lambda y_true, y_pred: float(mae(y_true, y_pred)))
register_metric("smape", lambda y_true, y_pred: float(smape(y_true, y_pred)))

__all__ = ["register_metric", "get_metric", "list_metrics"]

