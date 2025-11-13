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


def _ensure_repo_root_on_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root


def main() -> int:
    repo_root = _ensure_repo_root_on_path()

    try:
        from ml.tests.case_registry import TEST_CASES
    except Exception as exc:  # pragma: no cover
        print(f"Failed to import test case registry: {exc}")
        return 1

    if not TEST_CASES:
        print("No test cases registered. Update ml/tests/case_registry.py first.")
        return 0

    try:
        import pytest
    except ModuleNotFoundError as exc:  # pragma: no cover
        print(f"pytest is required to run the cases. Install it first: {exc}")
        return 1

    print("Running all ML test cases:")
    for name, node in TEST_CASES.items():
        print(f"  - {name}: {node}")

    os.chdir(repo_root)
    pytest_args = ["-q", "-rs", *TEST_CASES.values()]
    return pytest.main(pytest_args)


if __name__ == "__main__":
    raise SystemExit(main())


