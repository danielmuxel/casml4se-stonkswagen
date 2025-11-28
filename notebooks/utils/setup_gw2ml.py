from __future__ import annotations

import sys
from pathlib import Path

__all__ = ["configure_gw2ml_path", "initialize_notebook_environment"]


def configure_gw2ml_path(src_directory: str = "src") -> Path:
    """Ensure the repository src directory is available on sys.path for notebook imports."""
    current_file = Path(__file__).resolve()
    repo_root = current_file.parents[2]
    src_path = repo_root / src_directory

    if not src_path.exists():
        raise FileNotFoundError(f"Unable to locate src directory at {src_path}")

    src_as_string = str(src_path)
    if src_as_string in sys.path:
        return src_path

    sys.path.insert(0, src_as_string)
    return src_path


def initialize_notebook_environment() -> Path:
    """Configure the import path for gw2ml when running from inside notebooks."""
    configured_path = configure_gw2ml_path()
    return configured_path

