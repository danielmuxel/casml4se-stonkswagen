from __future__ import annotations

from typing import Any, Dict, List, Type

from .base import BaseModel

_REGISTRY: Dict[str, Dict[str, Any]] = {}


def register_model(name: str, cls: Type[BaseModel], default_grid: Dict[str, Any] | None = None) -> None:
    """Register or override a model by name (case-insensitive)."""
    key = name.strip()
    if not key:
        raise ValueError("Model name must be non-empty.")
    if not issubclass(cls, BaseModel):
        raise TypeError("Model class must inherit from BaseModel.")
    _REGISTRY[key] = {"cls": cls, "default_grid": default_grid or {}}


def get_model(name: str) -> Type[BaseModel]:
    """Return a registered model class."""
    key = name.strip()
    if key not in _REGISTRY:
        raise KeyError(f"Model '{name}' is not registered.")
    return _REGISTRY[key]["cls"]


def get_default_grid(name: str) -> Dict[str, Any]:
    """Return the default parameter grid for a registered model."""
    key = name.strip()
    if key not in _REGISTRY:
        raise KeyError(f"Model '{name}' is not registered.")
    return dict(_REGISTRY[key]["default_grid"])


def list_models() -> List[str]:
    """List available model names."""
    return sorted(_REGISTRY.keys())


__all__ = ["register_model", "get_model", "get_default_grid", "list_models"]

