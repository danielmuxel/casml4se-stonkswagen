# infra

Store deployment, infrastructure-as-code, container definitions, and MLflow tracking server configuration in this folder. Use it to capture how the platform is provisioned across environments.

## MLflow Local Setup

Run these steps from `infra/MLFlow` to start the local tracking stack:

1. Copy the environment template so Docker Compose can read required variables:

   ```
   cp .env.example .env
   ```

2. Create the bind-mounted data directories to avoid volume mount errors:

   ```
   mkdir -p pgdata minio-data
   ```

3. Start the services in detached mode:

   ```
   docker compose up -d
   ```

The stack provisions PostgreSQL for the MLflow backend and MinIO for artifact storage. Verify containers are healthy with `docker compose ps`.

