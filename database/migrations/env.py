"""Alembic environment for OceanMind AI (Module 2: Database & Query Layer).

Deliberately does NOT read sqlalchemy.url from alembic.ini -- that would mean
duplicating the DB connection string in a second place. Instead this pulls
Config.DATABASE_URL at runtime, the same single source of truth every other
module uses, and points target_metadata at database.models.Base so
`alembic revision --autogenerate` can diff against the real ORM models.
"""
import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config, pool

from alembic import context

# Make the repo root importable (config.py, database/, shared/ all live there)
# since Alembic invokes this file directly rather than via the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import Config  # noqa: E402
from database.models import Base  # noqa: E402

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Point autogenerate at the real ORM models (Float/Profile/Measurement/Report)
# so schema drift between models.py and the live DB is detectable.
target_metadata = Base.metadata

# Always use the project's Config.DATABASE_URL, never alembic.ini's
# sqlalchemy.url placeholder -- keeps DB connection config in one place.
config.set_main_option("sqlalchemy.url", Config.DATABASE_URL)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emits SQL without a live DB connection)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode against a live DB connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
