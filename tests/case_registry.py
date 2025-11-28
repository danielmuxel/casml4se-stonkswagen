"""
Registry of integration-style ML test cases.

Add new entries to `TEST_CASES` so scripts can discover them without scanning
the filesystem. Keys should be short, descriptive names; values are pytest node
ids that the runner can pass to `pytest`.
"""

from __future__ import annotations

from collections.abc import Iterable

TEST_CASES: dict[str, str] = {
    "database_export": "ml/tests/test_database.py::test_fetch_and_export_real_tables",
}


def get_selected_cases(selected: Iterable[str] | None) -> dict[str, str]:
    """
    Return case name → pytest node id for the provided selection.
    """

    if selected is None:
        return TEST_CASES

    chosen = {}
    for name in selected:
        if name not in TEST_CASES:
            available = ", ".join(sorted(TEST_CASES))
            message = f"Unknown test case '{name}'. Available cases: {available}."
            raise KeyError(message)
        chosen[name] = TEST_CASES[name]
    return chosen


__all__ = ["TEST_CASES", "get_selected_cases"]


