# Stonkswagen Agents Guide

This playbook distills every `README.md` in the repository so an agent can confidently navigate the codebase, extend it, and document the results. Use it as both an onboarding primer and a day-to-day checklist.

## 1. Mission & Quick Start
- Treat `ml/` as the production-grade machine-learning workspace; everything else either feeds it (data/scripts) or publishes it (apps/infra/models).
- Keep bulky or sensitive assets out of git. Prefer scripted download or generation steps recorded under `scripts/` or `docs/`.
- When experimenting, start in `notebooks/`, then promote reusable logic into `ml/common/`, pipelines into `ml/pipelines/`, and final artifacts into `models/`.
- Run the curated ML validation suite before sharing changes:
  ```bash
  uv run --project ml python scripts/run_ml_tests.py
  ```

## 2. Directory Field Guide
| Path | Ownership | When to Touch |
| --- | --- | --- |
| `apps/` | Nuxt 4 frontends that will expose dashboards & ML tooling. | Add UI/UX code once a dashboard or tool needs to surface ML outputs. |
| `data/` | Raw/intermediate datasets (untracked when large/sensitive). | Stage sample data, document sourcing steps, or store lightweight fixtures. |
| `docs/` | System design, experiment reports, onboarding, and API notes. | Capture findings from experiments, playbooks like this, and SOPs. |
| `infra/` | Deployment, IaC, container specs, MLflow tracking setup. | Update when provisioning new environments or adjusting infrastructure. |
| `ml/` | Core ML workspace: shared code, pipelines, configs, tests. | Move any reusable experiment logic here. |
| `ml/common/` | Feature engineering, logging, config, DB helpers. | Implement utilities that multiple pipelines/models share. |
| `ml/models/` | PyTorch Lightning modules, Darts models, training logic. | Define model classes and their training routines. |
| `ml/pipelines/` | End-to-end training/evaluation orchestration with metrics + MLflow logging. | Build reproducible workflows that call into data, models, and trackers. |
| `ml/tests/` | Pytest-based regression, smoke, and data validation suites. | Add coverage for new utilities, pipelines, or model behaviors. |
| `models/` | Exported artifacts, serialized weights, config snapshots, MLflow links. | Store metadata that points to best runs; keep heavy binaries elsewhere. |
| `notebooks/` | Exploratory notebooks and prototype visuals. | Ideate, visualize, and summarize findings before promoting code. |
| `scripts/` | Helper scripts for ingestion, setup, evaluation, maintenance. | Automate any repetitive step and document dependencies inline. |

## 3. Data & Experiment Lifecycle
- **Source responsibly.** Keep the definitive instructions for obtaining datasets in `scripts/` or `docs/`, not inside git-tracked blobs under `data/`.
- **Prototype in notebooks.** Use `notebooks/` for exploration, visual reports, and quick experiments. Once logic is reusable, migrate it into the structured `ml/` packages.
- **Promote artifacts.** When a run produces a champion model, export lightweight descriptors (config dumps, run IDs, metric summaries) into `models/`. Heavy weights belong in an artifact store referenced from those descriptors.
- **Document every step.** Pair each major dataset, experiment series, or release with narrative notes inside `docs/` so future agents can replay the work.

## 4. Core ML Workspace (`ml/`)
### Shared Utilities – `ml/common/`
- Centralize preprocessing, feature engineering, configuration, and logging helpers here so pipelines stay DRY.
- Includes a `database` submodule with a `DatabaseClient` factory plus query helpers. Instantiate the client once and pass it around:
  ```python
  from datetime import datetime, UTC, timedelta
  from ml.common import DatabaseClient, database

  client = DatabaseClient.from_env()
  last_day = datetime.now(UTC) - timedelta(days=1)
  prices_df = database.get_prices(client, item_id=19721, start_time=last_day)
  history_df = database.get_generic_rows(
      client,
      table="gw2tp_historical_prices",
      item_id=19721,
      start_time=last_day,
      time_column="timestamp",
      time_column_unit="milliseconds",
  )
  ```
- Any helper reused across models, pipelines, or services belongs here.

### Pipelines – `ml/pipelines/`
- Orchestrate end-to-end workflows: load data, pick models, compute metrics, and log outcomes to MLflow or equivalent trackers.
- Keep pipelines declarative: inputs, processing blocks, model selection, evaluation, and logging should be explicit so they are easy to automate.

### Models – `ml/models/` vs `models/`
- `ml/models/` houses the *definitions*: PyTorch Lightning modules, Darts models, and specialized training logic grouped by family or feature set.
- `models/` stores the *outputs*: exported weights, configuration snapshots, and references (e.g., MLflow run IDs). Never check in large binaries—store metadata or pointers instead.

### Tests – `ml/tests/`
- Cover pipelines, utilities, and model behaviors with pytest suites (regression, smoke, and validation checks).
- To add a curated test case for the ML runner:
  1. Create the pytest file under `ml/tests/` with a descriptive function (e.g., `def test_my_feature_behaves(): ...`).
  2. Register it inside `ml/tests/case_registry.py`:
     ```python
     TEST_CASES = {
         "database_export": "ml/tests/test_database.py::test_fetch_and_export_real_tables",
         "my_feature": "ml/tests/test_my_feature.py::test_my_feature_behaves",
     }
     ```
  3. Execute the runner to ensure the suite passes:
     ```bash
     uv run --project ml python scripts/run_ml_tests.py
     ```
- The runner resolves every registered node id and prints pass/fail/skip status, so keep `TEST_CASES` curated.

## 5. Automation & Tooling (`scripts/`)
- Collect shell/Python helpers for ingestion, environment bootstrapping, evaluation, or cron-like maintenance.
- Inline usage docs or dependency notes directly in each script, and prefer descriptive filenames so agents can find the right tool quickly.
- The repository already exposes `scripts/run_ml_tests.py`, which drives the curated pytest cases through the UV-managed environment.

## 6. Surfaces & Deployment
- **Frontends (`apps/`):** Nuxt 4 dashboards and operator tools will live here. Keep UI code modular so it can surface model insights, evaluation dashboards, or manual override panels.
- **Infrastructure (`infra/`):** Track deployment manifests, IaC modules, container specs, and MLflow tracking server configuration. Any environment change (QA, staging, prod) should originate here to remain reproducible.

## 7. Documentation Practices (`docs/`)
- Use `docs/` for architectural decisions, experiment retrospectives, onboarding manuals, and API references.
- Organize subfolders by topic (e.g., `docs/database/`) and link to relevant scripts, notebooks, or pipelines so the knowledge graph stays connected.
- Whenever you finish a substantial exploration (data study, model comparison, infra change), record the context, decisions, and next steps here.

## 8. Contribution Checklist for Agents
1. **Define intent** – capture the problem statement inside `docs/` or an issue.
2. **Gather data** – stage or document datasets in `data/` and codify ingestion under `scripts/`.
3. **Prototype** – explore in `notebooks/`, logging observations back into `docs/`.
4. **Productionize** – lift reusable code into `ml/common/`, `ml/models/`, and `ml/pipelines/`.
5. **Validate** – add or update pytest coverage under `ml/tests/`, register curated cases, and run `uv run --project ml python scripts/run_ml_tests.py`.
6. **Package outputs** – store metadata for champion models in `models/` and hook up any necessary dashboards in `apps/`.
7. **Deploy & document** – reflect infra changes under `infra/`, and close the loop with final notes in `docs/`.

Follow this loop and every agent will leave the workspace more orderly—and more reproducible—than they found it.
