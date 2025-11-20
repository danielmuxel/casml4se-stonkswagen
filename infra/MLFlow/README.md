# MLflow Infrastructure

This folder contains the Docker Compose setup for a production-ready MLflow tracking server backed by Postgres (for metadata) and MinIO (S3-compatible storage for artifacts).

## Services

| Service | URL | Description | Credentials (Default) |
| :--- | :--- | :--- | :--- |
| **MLflow UI** | [http://localhost:5000](http://localhost:5000) | Experiment tracking dashboard | N/A |
| **MinIO Console** | [http://localhost:9001](http://localhost:9001) | Object storage management | `admin` / `admin123` |
| **MinIO API** | `localhost:9000` | S3-compatible API endpoint | `minio` / `minio123` |
| **Postgres** | `localhost:5432` | Backend store for MLflow | `mlflow` / `mlflow` |

## Setup & Running

### 1. Prepare Environment
1. Create the data directories (required for Docker volumes):
   ```bash
   mkdir pgdata minio-data
   ```
2. Create the environment file:
   ```bash
   cp env_example .env
   ```

### 2. Start Services
```bash
docker compose up -d
```

To stop the services:
```bash
docker compose down
```

## Client Configuration (Python)

To log experiments from your Python code, ensure your environment (e.g., `ml` or `notebooks`) is active and configured.

### 1. Environment Variables
Set these variables in your shell or your project's `.env` file:

```bash
# MLflow Server URL
export MLFLOW_TRACKING_URI=http://localhost:5000

# MinIO (S3) Configuration
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID=minio
export AWS_SECRET_ACCESS_KEY=minio123
export AWS_DEFAULT_REGION=us-east-1
export MLFLOW_S3_IGNORE_TLS=true
```

### 3. Usage Example

```python
import mlflow
import os

# Ensure env vars are set
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("my-experiment")

with mlflow.start_run():
    mlflow.log_param("model_type", "test")
    mlflow.log_metric("accuracy", 0.95)
    
    with open("test.txt", "w") as f:
        f.write("Artifact content")
    mlflow.log_artifact("test.txt")
```

## Troubleshooting

**Error: `failed to mount local volume: ... no such file or directory`**
*   **Cause:** The `pgdata` or `minio-data` folders do not exist.
*   **Fix:** Run `mkdir pgdata minio-data` inside the `infra/MLFlow` folder.

