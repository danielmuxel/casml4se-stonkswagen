"""Top-level package for the gw2ml project."""

from . import data as database
from .data import DatabaseClient
from .paths import PROJECT_ROOT, get_project_root

__all__ = ["database", "DatabaseClient", "PROJECT_ROOT", "get_project_root"]
