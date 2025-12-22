from __future__ import annotations

from typing import Callable, Dict, Iterable, Literal, Protocol

import pandas as pd

from gw2ml.pipeline.data_preparation import PipelineContext

Phase = Literal["training", "validation", "inference"]


class FeatureAugmentor(Protocol):
    """Callable signature for feature hooks that mutate a dataframe."""

    def __call__(self, df: pd.DataFrame, context: PipelineContext) -> pd.DataFrame: ...


AugmentorRegistry = Dict[Phase, list[FeatureAugmentor]]

_registry: AugmentorRegistry = {
    "training": [],
    "validation": [],
    "inference": [],
}


def register_augmentor(phase: Phase, augmentor: FeatureAugmentor) -> None:
    """Register a feature hook for a specific pipeline phase."""
    _registry.setdefault(phase, []).append(augmentor)


def clear_augmentors(phase: Phase | None = None) -> None:
    """Remove augmentors for the selected phase (or all phases)."""
    if phase:
        _registry[phase] = []
        return
    for key in list(_registry.keys()):
        _registry[key] = []


def get_augmentors(phase: Phase) -> tuple[FeatureAugmentor, ...]:
    """Expose registered augmentors without leaking internal lists."""
    return tuple(_registry.get(phase, []))


def apply_augmentors(df: pd.DataFrame, context: PipelineContext, phase: Phase) -> pd.DataFrame:
    """Apply registered augmentors sequentially to the provided dataframe."""
    augmentors = get_augmentors(phase)
    if not augmentors:
        return df
    mutated = df
    for augmentor in augmentors:
        mutated = augmentor(mutated, context)
    return mutated


__all__ = [
    "AugmentorRegistry",
    "FeatureAugmentor",
    "Phase",
    "apply_augmentors",
    "clear_augmentors",
    "get_augmentors",
    "register_augmentor",
]




