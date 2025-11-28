"""Top-level package for the gw2ml project."""

from . import data as database
from .data import DatabaseClient

__all__ = ["database", "DatabaseClient"]
