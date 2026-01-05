# Project Overview

This project is a machine learning workspace for analyzing Guild Wars 2 trading data. The main goal is to train and evaluate time series forecasting models to predict item prices. The project is well-structured, with separate directories for data, notebooks, source code, and infrastructure. It uses a variety of models, including traditional statistical models, tree-based models, and deep learning models.

The core technologies used are:
- Python for data analysis and modeling.
- `darts` for time series forecasting.
- `pandas` for data manipulation.
- `mlflow` for experiment tracking.
- `SQLAlchemy` for database interaction.
- `uv` for python environment management.

# Building and Running

The project uses `uv` to manage Python dependencies. The `pyproject.toml` file in the root directory contains a `[dependency-groups.dev]` table with all the necessary packages for development.

## Setup

1.  **Install `uv`:**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
2.  **Install dependencies:**
    ```bash
    uv sync
    ```

## Running Evaluations

The main entry point for running model evaluations is the `scripts/run_eval.sh` script. This script sets up the environment and then executes `notebooks/models/evaluations.py`.

To run the evaluations:

```bash
./scripts/run_eval.sh
```

The evaluation script will:
1.  Load and prepare the data.
2.  Train and evaluate a suite of forecasting models.
3.  Log the results to MLflow.
4.  Print a summary of the results.

## Development Conventions

- The project follows a standard Python project structure.
- The `src` directory contains the main source code, which is organized into sub-packages for data, modeling, and features.
- The `notebooks` directory is used for exploratory analysis and prototyping.
- The `tests` directory contains automated tests.
- The project uses MLflow for experiment tracking, which is a good practice for managing and comparing model performance.
- Database connection details are managed via environment variables and a `.env` file, which is a good security practice.
- The `run_eval.sh` script demonstrates how to set up the `PYTHONPATH` correctly to run scripts from the project root. This is a common convention in Python projects.
