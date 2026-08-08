"""SQLAlchemy ORM models for OceanMind AI (Module 2: Database & Query Layer).

These classes mirror database/schema.sql exactly — table names, column names,
types, and foreign keys must never drift out of sync between the two, and
`id` / `float_id` must never be renamed since Modules 1, 4, 5, 6 all depend
on them.
"""
from __future__ import annotations

from sqlalchemy import Column
from sqlalchemy import Date
from sqlalchemy import DateTime
from sqlalchemy import Float as SAFloat
from sqlalchemy import ForeignKey
from sqlalchemy import Index
from sqlalchemy import Integer
from sqlalchemy import SmallInteger
from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy import UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Shared declarative base for all OceanMind AI ORM models."""


class Float(Base):
    """An ARGO float (physical instrument), identified by its WMO float_id.

    Named `Float` per Part 2's spec. This does shadow the built-in `float`
    type and `sqlalchemy.Float` within this module's namespace, so the
    SQLAlchemy column type is imported above as `SAFloat` to avoid
    colliding with this class. The underlying table name is 'floats'.
    """

    __tablename__ = "floats"

    float_id = Column(String, primary_key=True)
    deployment_lat = Column(SAFloat)
    deployment_lon = Column(SAFloat)
    deployment_date = Column(Date)
    status = Column(String)

    profiles = relationship("Profile", back_populates="float_", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Float float_id={self.float_id!r} status={self.status!r}>"


class Profile(Base):
    """One ARGO profiling cycle for a given float: a single dive-and-surface event."""

    __tablename__ = "profiles"
    __table_args__ = (
        UniqueConstraint("float_id", "cycle_number", name="ux_profiles_float_cycle"),
        # Mirrors database/schema.sql's ix_profiles_region_date exactly. Without
        # this, Base.metadata.create_all() (used by database/session.py's
        # __main__ self-test, database/repository.py's __main__ self-test, and
        # any local/CI setup that skips Alembic) silently produces a database
        # with no index backing the region+date filter every dashboard page
        # load runs -- a full table scan on profiles as the table grows, even
        # though the "real" schema (schema.sql / the Alembic migration) has it.
        Index("ix_profiles_region_date", "ocean_region", "profile_date"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    float_id = Column(String, ForeignKey("floats.float_id"))
    cycle_number = Column(Integer)
    profile_date = Column(DateTime)
    latitude = Column(SAFloat)
    longitude = Column(SAFloat)
    ocean_region = Column(String)

    float_ = relationship("Float", back_populates="profiles")
    measurements = relationship(
        "Measurement", back_populates="profile", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Profile id={self.id} float_id={self.float_id!r} cycle={self.cycle_number}>"


class Measurement(Base):
    """One depth-level reading within a profile (temperature, salinity, BGC vars)."""

    __tablename__ = "measurements"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # index=True mirrors database/schema.sql's ix_measurements_profile_id.
    # Postgres does NOT auto-index foreign key columns, so without this every
    # get_measurements_for_profile[s]() call -- run once per profile on every
    # map/health/report view -- was a full table scan on measurements. See the
    # ix_profiles_region_date comment above for why this can silently drift
    # from schema.sql/the Alembic migration if only declared there.
    profile_id = Column(Integer, ForeignKey("profiles.id"), index=True)
    pressure_dbar = Column(SAFloat)
    depth_m = Column(SAFloat)
    temperature_c = Column(SAFloat)
    salinity_psu = Column(SAFloat)
    dissolved_oxygen = Column(SAFloat, nullable=True)
    chlorophyll = Column(SAFloat, nullable=True)
    ph = Column(SAFloat, nullable=True)
    qc_flag = Column(SmallInteger)

    profile = relationship("Profile", back_populates="measurements")

    def __repr__(self) -> str:
        return f"<Measurement id={self.id} profile_id={self.profile_id} depth_m={self.depth_m}>"


class Report(Base):
    """A generated Ocean Health report for a region and period (Module 6 output)."""

    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    generated_at = Column(DateTime)
    ocean_region = Column(String)
    period_start = Column(Date)
    period_end = Column(Date)
    file_path = Column(String)
    summary_text = Column(Text)

    def __repr__(self) -> str:
        return f"<Report id={self.id} ocean_region={self.ocean_region!r}>"


if __name__ == "__main__":
    # --- Self-test ---
    # Verifies the models import cleanly and expose the exact table/column
    # names other modules depend on, without touching a real database.
    from shared.logger import get_logger

    logger = get_logger(__name__)

    for model in (Float, Profile, Measurement, Report):
        logger.info(
            "%s: table=%r columns=%s",
            model.__name__,
            model.__tablename__,
            [c.name for c in model.__table__.columns],
        )
