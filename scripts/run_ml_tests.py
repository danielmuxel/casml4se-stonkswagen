"""
Run registered ML test cases through pytest.

Usage
-----
```bash
uv run --project ml python scripts/run_ml_tests.py
uv run --project ml python scripts/run_ml_tests.py --cases database_export
ML_TEST_CASES=database_export uv run --project ml python scripts/run_ml_tests.py
```
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable


def _ensure_repo_root_on_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root


def _resolve_selected_cases(raw: str | None, available: dict[str, str]) -> dict[str, str]:
    if raw is None:
        return available

    requested = [entry.strip() for entry in raw.split(",") if entry.strip()]
    if not requested:
        return available

    missing = [name for name in requested if name not in available]
    if missing:
        joined = ", ".join(sorted(available))
        raise KeyError(f"Unknown case(s) {missing}. Available cases: {joined}.")
    return {name: available[name] for name in requested}


def _print_case_overview(cases: dict[str, str]) -> None:
    print("Running ML test cases:")
    for name, node in cases.items():
        print(f"  - {name}: {node}")


def main() -> int:
    repo_root = _ensure_repo_root_on_path()

    try:
        from tests.case_registry import TEST_CASES
    except Exception as exc:  # pragma: no cover
        print(f"Failed to import test case registry: {exc}")
        return 1

    if not TEST_CASES:
        print("No test cases registered. Update tests/case_registry.py first.")
        return 0

    try:
        import pytest
    except ModuleNotFoundError as exc:  # pragma: no cover
        print(f"pytest is required to run the cases. Install it first: {exc}")
        return 1

    selected_raw = os.getenv("ML_TEST_CASES")
    try:
        selected_cases = _resolve_selected_cases(selected_raw, TEST_CASES)
    except KeyError as exc:
        print(exc)
        return 1

    os.chdir(repo_root)
    _print_case_overview(selected_cases)
    pytest_args = ["-q", "-rs", *selected_cases.values()]
    return pytest.main(pytest_args)


if __name__ == "__main__":
    raise SystemExit(main())


