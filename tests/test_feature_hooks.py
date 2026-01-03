from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from gw2ml.pipeline.data_preparation import build_pipeline_context
from gw2ml.pipeline.feature_hooks import apply_augmentors, clear_augmentors, register_augmentor


def test_apply_augmentors_runs_in_registration_order() -> None:
    clear_augmentors()
    context = build_pipeline_context("demo", connection_url="sqlite:///:memory:", run_date=datetime.now(UTC))
    source = pd.DataFrame({"value": [1, 2]})

    def augment_add(df: pd.DataFrame, _context) -> pd.DataFrame:
        df = df.copy()
        df["value"] = df["value"] + 1
        return df

    def augment_double(df: pd.DataFrame, _context) -> pd.DataFrame:
        df = df.copy()
        df["value"] = df["value"] * 2
        return df

    register_augmentor("training", augment_add)
    register_augmentor("training", augment_double)

    transformed = apply_augmentors(source, context, "training")
    assert transformed["value"].tolist() == [4, 6]


def test_clear_augmentors_resets_registry() -> None:
    register_augmentor("validation", lambda df, ctx: df)
    clear_augmentors("validation")
    context = build_pipeline_context("demo", connection_url="sqlite:///:memory:", run_date=datetime.now(UTC))
    df = pd.DataFrame({"value": [1]})
    result = apply_augmentors(df, context, "validation")
    assert result.equals(df)

