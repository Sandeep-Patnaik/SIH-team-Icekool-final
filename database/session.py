"""Engine and session management for OceanMind AI (Module 2).

Every other module that needs raw DB access should go through
database.repository.ProfileRepository instead of importing this directly —
this module exists mainly so ProfileRepository and Alembic have one shared
way to obtain a connection.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from config import Config
from shared.logger import get_logger

logger = get_logger(__name__)

_engine: Engine | None = None
_SessionLocal: sessionmaker | None = None


def get_engine() -> Engine:
    """Return a lazily-created, process-wide SQLAlchemy Engine for Config.DATABASE_URL.

    Returns:
        The shared SQLAlchemy Engine instance.

    Raises:
        RuntimeError: if the engine cannot be created (e.g. malformed DATABASE_URL).
    """
    global _engine
    if _engine is None:
        try:
            # pool_pre_ping avoids handing out stale/dropped connections, which
            # matters for a long-running Streamlit dashboard process (Module 5).
            _engine = create_engine(Config.DATABASE_URL, pool_pre_ping=True, future=True)
            logger.info("Database engine created for %s", _safe_url(Config.DATABASE_URL))
        except Exception as exc:
            logger.error("Failed to create database engine", exc_info=True)
            raise RuntimeError("Could not create database engine") from exc
    return _engine


def get_session() -> Session:
    """Return a new SQLAlchemy Session bound to the shared engine.

    Callers are responsible for closing the session (or use session_scope()
    below for automatic commit/rollback/close).
    """
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), autoflush=False, autocommit=False)
    return _SessionLocal()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Context manager yielding a Session that commits on success, rolls back on error.

    Usage:
        with session_scope() as session:
            session.add(obj)
    """
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        logger.error("Session rolled back due to an error", exc_info=True)
        raise
    finally:
        session.close()


def _safe_url(url: str) -> str:
    """Redact credentials from a DB URL before logging it."""
    if "@" in url and "//" in url:
        scheme, rest = url.split("//", 1)
        creds_and_host = rest.split("@", 1)
        if len(creds_and_host) == 2:
            return f"{scheme}//***@{creds_and_host[1]}"
    return url


if __name__ == "__main__":
    # --- Self-test ---
    # Confirms an engine can be constructed from the configured DATABASE_URL
    # (falls back to a local SQLite file per config.py's default) and that a
    # session can be opened and closed cleanly.
    engine = get_engine()
    logger.info("Engine dialect: %s", engine.dialect.name)
    with session_scope() as s:
        logger.info("Session opened: %s", s)
    logger.info("Self-test passed.")
