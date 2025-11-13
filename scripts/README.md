# scripts

Collect helper scripts for data ingestion, environment setup, evaluation, and maintenance tasks here. Use descriptive filenames and document any dependencies or usage examples.

## ML test runner

Execute every registered ML case with:

```bash
uv run --project ml python scripts/run_ml_tests.py
```

Add new cases to `ml/tests/case_registry.py`; the runner automatically picks them up.

