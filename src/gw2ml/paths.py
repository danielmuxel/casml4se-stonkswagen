from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

ROOT_MARKERS = (".git", "pyproject.toml", "uv.lock")


def _detect_project_root(start: Optional[Path] = None) -> Path:
    """Walk upward from start until a repo marker is found."""
    base = (start or Path.cwd()).resolve()
    for candidate in (base, *base.parents):
        if any((candidate / marker).exists() for marker in ROOT_MARKERS):
            return candidate
    return base


def get_project_root(start: Optional[Path] = None) -> Path:
    """Return the repository root, honoring PROJECT_ROOT env override."""
    env_root = os.getenv("PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return _detect_project_root(start)


PROJECT_ROOT = get_project_root()


__all__ = ["get_project_root", "PROJECT_ROOT"]

