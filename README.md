# casml4se-stonkswagen

Workspace for exploring Guild Wars 2 trading data, training machine-learning models, and preparing frontend dashboards that surface the results.

## Folder Structure

- `apps/` – Nuxt applications that will present dashboards and tools backed by the ML workflows.
- `data/` – Raw and intermediate datasets. Keep large or sensitive files out of version control.
- `docs/` – Reference material, onboarding guides, and database schema notes.
- `infra/` – Deployment scripts, infrastructure-as-code, and platform configuration.
- `ml/` – Core Python code: shared utilities, training pipelines, and automated tests.
- `models/` – Stored model artifacts, evaluation summaries, and metadata for best-performing runs.
- `notebooks/` – Exploratory Jupyter notebooks and prototype analyses.
- `scripts/` – Automation helpers such as `init_ml_project.sh` for bootstrapping environments.

## Working with `uv`

The project uses [`uv`](https://docs.astral.sh/uv/) to manage Python dependencies and virtual environments.

1. Install `uv` (once per machine):
   ```
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. Sync dependencies defined in a `pyproject.toml` (for example inside `ml/`, `scripts/`, or `notebooks/`):
   ```
   uv sync
   ```
3. Execute commands inside the managed environment:
   ```
   uv run python path/to/script.py
   ```
4. Add new packages when you need them:
   ```
   uv add package-name
   ```

> Tip: `scripts/init_ml_project.sh` shows an end-to-end example that installs ML dependencies, registers a Jupyter kernel, and verifies the setup with `uv`.


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

