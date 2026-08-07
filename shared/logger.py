"""Shared logging setup for OceanMind AI. Every module imports get_logger from here
instead of configuring its own handlers, so log output is consistent project-wide.
"""
import logging

from config import Config


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger configured with the project's standard format.

    Args:
        name: usually __name__ of the calling module.

    Returns:
        A configured logging.Logger instance. Safe to call repeatedly for the
        same name — handlers are only attached once.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(Config.LOG_LEVEL)
    return logger
