from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterable

__all__ = []  # overwritten once the real module is imported


def _ensure_src_path_first(existing_paths: Iterable[str], src_path: str) -> None:
    filtered_paths = [path for path in existing_paths if path != src_path]
    sys.path[:] = [src_path, *filtered_paths]


def _load_real_module() -> ModuleType:
    bridge_path = Path(__file__).resolve()
    repo_root = bridge_path.parents[2]
    src_directory = repo_root / "src"

    if not src_directory.exists():
        message = (
            "Unable to import gw2ml because the repository src directory "
            f"does not exist at: {src_directory}"
        )
        raise ModuleNotFoundError(message)

    src_directory_string = str(src_directory)
    _ensure_src_path_first(sys.path, src_directory_string)

    module_name = __name__
    sys.modules.pop(module_name, None)
    real_module = importlib.import_module(module_name)

    return real_module


def _redirect_to_real_module() -> None:
    real_module = _load_real_module()

    globals().update(real_module.__dict__)
    sys.modules[__name__] = real_module


_redirect_to_real_module()

