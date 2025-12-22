from __future__ import annotations

import os
import pickle
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from gw2ml.paths import PROJECT_ROOT

MODELS_ENV_VAR = "GW2ML_MODELS_DIR"
DEFAULT_EXTENSION = ".pkl"


def _resolve_registry_root() -> Path:
    override = os.getenv(MODELS_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()
    return PROJECT_ROOT / "models"


def get_model_dir(model_key: str, *, create: bool = True) -> Path:
    directory = _resolve_registry_root() / model_key
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    return directory


def _resolve_filename(filename: str | None, run_date: datetime | None) -> str:
    if filename:
        return filename if filename.endswith(DEFAULT_EXTENSION) else f"{filename}{DEFAULT_EXTENSION}"
    if run_date is None:
        run_date = datetime.now(UTC)
    timestamp = run_date.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}{DEFAULT_EXTENSION}"


def persist_model(
    model: Any,
    *,
    model_key: str,
    run_date: datetime | None = None,
    filename: str | None = None,
) -> Path:
    """Serialize a model artifact under ``models/<model_key>/``."""
    directory = get_model_dir(model_key, create=True)
    target_name = _resolve_filename(filename, run_date)
    target_path = directory / target_name
    with target_path.open("wb") as fp:
        pickle.dump(model, fp)
    return target_path


def _list_artifacts(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix == DEFAULT_EXTENSION),
        key=lambda path: path.name,
        reverse=True,
    )


def load_model(model_key: str, filename: str | None = None) -> Any:
    """Load the latest (or explicitly named) artifact for a model key."""
    directory = get_model_dir(model_key, create=False)
    if filename:
        artifact_path = directory / (filename if filename.endswith(DEFAULT_EXTENSION) else f"{filename}{DEFAULT_EXTENSION}")
        if not artifact_path.exists():
            message = f"Artifact {artifact_path} does not exist."
            raise FileNotFoundError(message)
    else:
        artifacts = _list_artifacts(directory)
        if not artifacts:
            message = f"No artifacts found for model_key={model_key}."
            raise FileNotFoundError(message)
        artifact_path = artifacts[0]

    with artifact_path.open("rb") as fp:
        return pickle.load(fp)


def list_artifacts(model_key: str) -> list[str]:
    """Return artifact filenames available for a given model key."""
    directory = get_model_dir(model_key, create=False)
    return [path.name for path in _list_artifacts(directory)]


def purge_artifacts(model_key: str, keep: int = 5) -> list[Path]:
    """Keep the ``keep`` newest artifacts, deleting the rest."""
    directory = get_model_dir(model_key, create=False)
    artifacts = _list_artifacts(directory)
    obsolete = artifacts[keep:]
    for path in obsolete:
        path.unlink(missing_ok=True)
    return obsolete


__all__ = [
    "DEFAULT_EXTENSION",
    "get_model_dir",
    "list_artifacts",
    "load_model",
    "persist_model",
    "purge_artifacts",
]




