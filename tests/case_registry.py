"""
Registry of integration-style ML test cases.

Add new entries to `TEST_CASES` so scripts can discover them without scanning
the filesystem. Keys should be short, descriptive names; values are pytest node
ids that the runner can pass to `pytest`.
"""

from __future__ import annotations

TEST_CASES: dict[str, str] = {
    "database_export": "tests/test_database.py::test_fetch_and_export_real_tables",
    "snapshot_export": "tests/test_database.py::test_fetch_prices_snapshot_and_export",
    "feature_hooks": "tests/test_feature_hooks.py::test_apply_augmentors_runs_in_registration_order",
    "model_registry": "tests/test_model_registry.py::test_persist_and_load_model",
}


__all__ = ["TEST_CASES"]


