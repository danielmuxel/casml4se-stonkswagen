# scripts

Collect helper scripts for data ingestion, environment setup, evaluation, and maintenance tasks here. Use descriptive filenames and document any dependencies or usage examples.

## ML test runner

Execute every registered ML case with:

```bash
uv run --project ml python scripts/run_ml_tests.py
```

Add new cases to `ml/tests/case_registry.py`; the runner automatically picks them up.



 1072  uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
 1073* rocm-smi --showproductname\n
 1074  uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ "rocm[libraries,devel]"\n
 1075  uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ --pre torch torchaudio torchvision\n
 1076  uv pip remove torch
 1077  uv pip purge torch
 1078  uv pip uninstall torch
 1079  uv pip uninstall torchaudio
 1080  uv pip uninstall torchvision
 1081  pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ --pre torch torchaudio torchvision\n
 1082  uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ --pre torch torchaudio torchvision\n