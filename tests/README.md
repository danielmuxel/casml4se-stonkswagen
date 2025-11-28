# ml/tests

Add automated tests covering pipelines, shared utilities, and model behaviors here. Include regression tests for metrics, data validation checks, and smoke tests for training scripts.

## How to add a new test case

1. **Create the test file**
   - Place the test under this directory (e.g. `tests/test_my_feature.py`).
   - Write tests using pytest. Keep the public scenario you want exposed as a named function (e.g. `def test_my_feature_behaves()`).

2. **Register the case**
   - Open `case_registry.py`.
   - Add a new entry to `TEST_CASES` where the key is a short handle and the value is the pytest node id for your test function:
     ```python
     TEST_CASES = {
         "database_export": "ml/tests/test_database.py::test_fetch_and_export_real_tables",
         "my_feature": "ml/tests/test_my_feature.py::test_my_feature_behaves",
     }
     ```

3. **Run all registered cases**
   - From the repository root execute:
     ```bash
     uv run --project ml python scripts/run_ml_tests.py
     ```
   - The script resolves every entry in `TEST_CASES`, runs them via pytest, and prints results (including skip reasons).

Whenever you add a new integration case, repeat the steps above so the runner picks it up automatically.

