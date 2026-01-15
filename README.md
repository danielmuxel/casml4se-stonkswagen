# casml4se-stonkswagen

Workspace for exploring Guild Wars 2 trading data, training machine-learning models, and preparing frontend dashboards that surface the results.

## TL;DR - Quick Start

**Zu faul zum Lesen? Hier sind die wichtigsten Befehle:**

1. **Dependencies installieren** (einmalig, wähle deine Hardware):
   ```bash
   # macOS oder CPU-only:
   uv sync --group cpu

   # Linux mit AMD GPU (ROCm):
   uv sync --group rocm --extra-index-url https://rocm.nightlies.amd.com/v2/gfx1151/

   # Linux mit NVIDIA GPU (CUDA):
   uv sync --group cuda --extra-index-url https://download.pytorch.org/whl/cu121
   ```

2. **`.env` Datei anlegen** (für DB-Verbindung etc.):
   ```bash
   # Erstelle eine .env Datei im Projekt-Root mit mindestens:
   DB_URL=postgresql://user:password@host:port/database
   ```

3. **Backend starten** (in einem Terminal):
   ```bash
   uv run fastapi dev apps/api/main.py --host 0.0.0.0 --port 8000
   ```

4. **Frontend starten** (in einem ANDEREN Terminal):
   ```bash
   uv run streamlit run apps/streamlit/gw2_app.py
   ```

> **Wichtig:** Backend und Frontend sind zwei separate Prozesse und müssen in verschiedenen Terminals gestartet werden. Das Backend läuft auf Port 8000, das Streamlit-Frontend öffnet automatisch einen Browser.

## Folder Structure

- `apps/` – Streamlit/FastAPI applications for dashboards and ML workflows.
- `data/` – Raw and intermediate datasets. Keep large or sensitive files out of version control.
- `docs/` – Reference material, onboarding guides, and database schema notes.
- `infra/` – Deployment scripts, infrastructure-as-code, and platform configuration.
- `models/` – Stored model artifacts, evaluation summaries, and benchmark results.
- `notebooks/` – Exploratory Jupyter notebooks and prototype analyses.
- `scripts/` – Automation helpers for benchmarking, data export, and system checks.
- `src/` – Python packages (e.g., `gw2ml`) that power data access, feature prep, and modeling.
- `tests/` – Pytest-based integration suites that exercise data exports and pipelines.

## Scripts

Alle Scripts befinden sich im `scripts/` Ordner und werden mit `uv run python scripts/<name>.py` ausgeführt.

| Script | Beschreibung | Kategorie |
|--------|--------------|-----------|
| [`benchmark_models.py`](#benchmark_modelspy---modell-performance-vergleich) | Benchmarkt alle Forecasting-Modelle | ML/Performance |
| [`check_pytorch.py`](#check_pytorchpy---pytorch-system-check) | Zeigt PyTorch-Version und Hardware | System |
| [`show_devices.py`](#show_devicespy---pytorch-devices-auflisten) | Listet verfügbare PyTorch-Devices | System |
| [`fetch_all_prices.py`](#fetch_all_pricespy---preisdaten-exportieren) | Exportiert Preisdaten als CSV | Daten |
| [`compute_price_correlations.py`](#compute_price_correlationspy---preis-korrelationen-berechnen) | Berechnet Item-Korrelationen | Analyse |
| [`upload_data_to_s3.py`](#upload_data_to_s3py---daten-zu-s3-hochladen) | Lädt Daten zu S3 hoch | Daten |
| [`run_ml_tests.py`](#run_ml_testspy---ml-tests-ausführen) | Führt ML-Testfälle aus | Testing |

---

### `benchmark_models.py` - Modell-Performance-Vergleich

Benchmarkt alle Forecasting-Modelle (ARIMA, ExponentialSmoothing, XGBoost, Chronos2) und erstellt einen Performance-Report.

```bash
# Standard-Benchmark (speichert nach models/benchmark.json)
uv run python scripts/benchmark_models.py

# Mit echten Daten (Mystic Coin)
uv run python scripts/benchmark_models.py --item-id 19976 --days 30

# Eigener Output-Pfad
uv run python scripts/benchmark_models.py --output results/benchmark.json

# Ohne Speichern (nur Ausgabe)
uv run python scripts/benchmark_models.py --no-save
```

**Output:** Tabelle mit Fit/Predict/Backtest-Zeiten und relativer Performance (schnellstes Modell = 1.0x).

[↑ Zurück zur Übersicht](#scripts)

---

### `check_pytorch.py` - PyTorch System-Check

Zeigt PyTorch-Version, verfügbare Hardware-Beschleuniger und Systemressourcen.

```bash
uv run python scripts/check_pytorch.py
```

**Unterstützt:** macOS (Apple Silicon & Intel), Linux/Windows (NVIDIA CUDA), AMD ROCm.

[↑ Zurück zur Übersicht](#scripts)

---

### `show_devices.py` - PyTorch Devices auflisten

Listet alle verfügbaren PyTorch-Devices (CPU, CUDA, MPS, ROCm).

```bash
uv run python scripts/show_devices.py
```

[↑ Zurück zur Übersicht](#scripts)

---

### `fetch_all_prices.py` - Preisdaten exportieren

Exportiert alle GW2-Preisdaten aus der Datenbank als CSV-Dateien.

```bash
# Alle Daten exportieren
uv run python scripts/fetch_all_prices.py

# Nur handelbare Items
uv run python scripts/fetch_all_prices.py --tradable-only

# Mit Datumsfilter
uv run python scripts/fetch_all_prices.py --start-date 2024-01-01

# Mit mehr Parallel-Workern
uv run python scripts/fetch_all_prices.py --workers 12

# Liste handelbarer Items exportieren
uv run python scripts/fetch_all_prices.py --export-tradable-items
```

**Output:** CSV-Dateien in `data/gw2/snapshots/<timestamp>/`.

[↑ Zurück zur Übersicht](#scripts)

---

### `compute_price_correlations.py` - Preis-Korrelationen berechnen

Berechnet Item-zu-Item-Korrelationen und Cluster aus gecachten Preis-Snapshots.

```bash
uv run python scripts/compute_price_correlations.py
```

**Modi:**
- Atemporal (distributional features): Korrelationen basierend auf Preisverteilungen
- Temporal (time-aligned): Korrelationen basierend auf zeitlich ausgerichteten Preisverläufen

[↑ Zurück zur Übersicht](#scripts)

---

### `upload_data_to_s3.py` - Daten zu S3 hochladen

Lädt Daten zu Hetzner Object Storage (S3-kompatibel) hoch.

```bash
uv run python scripts/upload_data_to_s3.py gw2/raw \
  --bucket ost-s3 \
  --workers 8 \
  --filter-ext .csv,.parquet
```

[↑ Zurück zur Übersicht](#scripts)

---

### `run_ml_tests.py` - ML-Tests ausführen

Führt registrierte ML-Testfälle durch pytest aus.

```bash
# Alle Tests
uv run python scripts/run_ml_tests.py

# Spezifische Tests
uv run python scripts/run_ml_tests.py --cases database_export
ML_TEST_CASES=database_export uv run python scripts/run_ml_tests.py
```

[↑ Zurück zur Übersicht](#scripts)

---

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


