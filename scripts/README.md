# scripts

Collect helper scripts for data ingestion, environment setup, evaluation, and maintenance tasks here. Use descriptive filenames and document any dependencies or usage examples.

## ML test runner

Execute every registered ML case with:

```bash
uv run --project ml python scripts/run_ml_tests.py
```

Add new cases to `ml/tests/case_registry.py`; the runner automatically picks them up.




rocm-smi --showproductname

uv pip uninstall torch
uv pip uninstall torchaudio
uv pip uninstall torchvision
uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ --pre torch torchaudio torchvision