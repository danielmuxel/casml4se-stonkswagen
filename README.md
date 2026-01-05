# casml4se-stonkswagen

Workspace for exploring Guild Wars 2 trading data, training machine-learning models, and preparing frontend dashboards that surface the results.

## Folder Structure

- `apps/` – Nuxt applications that will present dashboards and tools backed by the ML workflows.
- `data/` – Raw and intermediate datasets. Keep large or sensitive files out of version control.
- `docs/` – Reference material, onboarding guides, and database schema notes.
- `infra/` – Deployment scripts, infrastructure-as-code, and platform configuration.
- `models/` – Stored model artifacts, evaluation summaries, and metadata for best-performing runs.
- `notebooks/` – Exploratory Jupyter notebooks and prototype analyses.
- `scripts/` – Automation helpers such as `init_ml_project.sh` for bootstrapping environments.
- `src/` – Python packages (e.g., `gw2ml`) that power data access, feature prep, and modeling.
- `tests/` – Pytest-based integration suites that exercise data exports and pipelines.

## Working with `uv`

The project uses [`uv`](https://docs.astral.sh/uv/) to manage Python dependencies and virtual environments.

1. Install `uv` (once per machine):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Sync dependencies based on your hardware:

   **macOS (or CPU-only):**
   ```bash
   uv sync --group cpu
   ```

   **Linux with AMD GPU (ROCm):**
   ```bash
   uv sync --group rocm --extra-index-url https://rocm.nightlies.amd.com/v2/gfx1151/
   ```

   **Linux with NVIDIA GPU (CUDA):**
   ```bash
   uv sync --group cuda --extra-index-url https://download.pytorch.org/whl/cu121
   ```

   > **Note:** PyTorch dependencies are separated into hardware-specific groups to support different GPU architectures. Choose the group matching your system.

3. Execute commands inside the managed environment:
   ```bash
   uv run python path/to/script.py
   ```

4. Add new packages when you need them:
   ```bash
   uv add package-name
   ```

> Tip: `scripts/init_ml_project.sh` shows an end-to-end example that installs ML dependencies, registers a Jupyter kernel, and verifies the setup with `uv`.

## Running Tests

The repository now uses a single consolidated uv environment—no nested subprojects required.

### Curated integration suite

Run the registered scenarios listed in `tests/case_registry.py`:

```bash
uv run python scripts/run_ml_tests.py
```

Select specific cases by setting `ML_TEST_CASES` (comma-delimited handles from the registry):

```bash
ML_TEST_CASES=database_export,snapshot_export uv run python scripts/run_ml_tests.py
```

### Direct pytest use

For ad-hoc development you can drive pytest directly:

```bash
uv run pytest tests
```

Some integration tests hit live databases; ensure `DB_URL` (or equivalent env credentials) is configured before running.

## Training/Forecast Entry Points

Run inside the repo root (uses `uv`):

- API (dev hot-reload):  
  `uv run fastapi dev apps/api/main.py --host 0.0.0.0 --port 8000`

- API (uvicorn):  
  `uv run uvicorn apps.api.main:app --host 0.0.0.0 --port 8000`

- Streamlit UI:  
  `uv run streamlit run apps/streamlit/forecast_app.py`

These entry points wrap the existing `train_items` and `forecast_item` functions. Configuration exposed in the UI/API allows setting item_ids, days_back, value_column, horizon, models, and metrics; DB and S3 settings are intentionally excluded from the UI/API surface (the pipelines still rely on environment-based DB connection as before).

## `initialize_all.sh`

- Runs `uv sync` for every `pyproject.toml` discovered in the repository, keeping all Python projects aligned with their dependency manifests.
- Usage:
  ```
  ./initialize_all.sh
  ```

## Uploading data to S3

Use `scripts/upload_data_to_s3.py` to send any folder under `DATA_PATH` (configurable via `.env`) to Hetzner object storage.

```bash
uv run python scripts/upload_data_to_s3.py gw2/raw \
  --bucket ost-s3 \
  --workers 8 \
  --filter-ext .csv,.parquet
```

- `folder` accepts absolute paths or paths relative to `DATA_PATH` (which defaults to `data/` at the repo root).
- The script automatically loads `.env`, resolves relative paths against the project root, and preserves subfolder structure in S3.
- Customize the destination with `--prefix` or `--timestamp`, preview uploads with `--dry-run`, and increase throughput via `--workers`.

