"""Shared fixtures for Module 2 (database/) tests.

Uses a throwaway in-memory SQLite database (StaticPool keeps the same
connection alive across the whole test session, since SQLite's ':memory:'
is normally per-connection) so the full test suite runs with no external
Postgres dependency, per Part 0's testing requirements.
"""
from __future__ import annotations

import os

# Config.DATABASE_URL is required with no default (Part 0's canonical
# contract — see config.py), so it must be set before `config` is imported
# by anything below. This placeholder is never actually connected to: the
# sqlite_engine fixture overrides database.session's engine/sessionmaker
# with a throwaway in-memory SQLite engine before every test runs.
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import database.session as session_module
from database.models import Base


@pytest.fixture(autouse=True)
def sqlite_engine():
    """Point database.session at a fresh in-memory SQLite engine for each test.

    Rebuilds the schema before every test and tears down the module-level
    engine/sessionmaker singletons afterward, so tests never leak state into
    each other and never touch whatever Config.DATABASE_URL points at.
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)

    session_module._engine = engine
    session_module._SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    yield engine

    session_module._engine = None
    session_module._SessionLocal = None
    engine.dispose()


@pytest.fixture
def repo():
    """A fresh ProfileRepository bound to the sqlite_engine fixture above."""
    from database.repository import ProfileRepository

    return ProfileRepository()
