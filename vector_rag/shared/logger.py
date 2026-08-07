"""Shared logging setup used by every OceanMind AI module."""
import logging
from config import Config


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger configured with the project's standard format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(Config.LOG_LEVEL)
    return logger
