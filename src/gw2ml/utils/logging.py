from __future__ import annotations

import logging
import sys


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Sets up a simple logger that outputs to stdout."""
    logger = logging.getLogger("gw2ml")
    
    # Avoid duplicate handlers if setup_logging is called multiple times
    if not logger.handlers:
        logger.setLevel(level)
        
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Returns a logger instance."""
    base_logger = "gw2ml"
    if name:
        return logging.getLogger(f"{base_logger}.{name}")
    return logging.getLogger(base_logger)
