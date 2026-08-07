"""Private helpers for database.repository.ProfileRepository (Module 2).

Split out of repository.py to keep that file under the project's ~300-400
line cap. These are internal implementation details of ProfileRepository —
not part of the locked public API, and not meant to be imported by other
modules. Each function here mirrors what was previously a @staticmethod on
ProfileRepository; behavior is unchanged.
"""
from __future__ import annotations

import math

from sqlalchemy import select
from sqlalchemy.orm import Session

from database.models import Float, Measurement, Profile
from shared.schemas import ProfileRecord

_EARTH_RADIUS_KM = 6371.0


def upsert_float(session: Session, record: ProfileRecord) -> None:
    """Insert the float row if new; a no-op if it already exists.

    Deployment metadata (lat/lon/date/status) isn't carried on
    ProfileRecord, so an existing float row is left untouched rather than
    overwritten with nulls.
    """
    existing = session.get(Float, record.float_id)
    if existing is None:
        session.add(Float(float_id=record.float_id, status="active"))


def upsert_profile(session: Session, record: ProfileRecord) -> int:
    """Insert or update the profiles row for (float_id, cycle_number)."""
    stmt = select(Profile).where(
        Profile.float_id == record.float_id,
        Profile.cycle_number == record.cycle_number,
    )
    existing = session.execute(stmt).scalar_one_or_none()

    if existing is not None:
        existing.profile_date = record.profile_date
        existing.latitude = record.latitude
        existing.longitude = record.longitude
        existing.ocean_region = record.ocean_region
        session.flush()
        return existing.id

    profile = Profile(
        float_id=record.float_id,
        cycle_number=record.cycle_number,
        profile_date=record.profile_date,
        latitude=record.latitude,
        longitude=record.longitude,
        ocean_region=record.ocean_region,
    )
    session.add(profile)
    session.flush()
    return profile.id


def replace_measurements(session: Session, profile_id: int, measurements: list[dict]) -> None:
    """Delete any existing measurements for profile_id and insert the new set."""
    session.query(Measurement).filter(Measurement.profile_id == profile_id).delete()
    for m in measurements:
        session.add(
            Measurement(
                profile_id=profile_id,
                pressure_dbar=m.get("pressure_dbar"),
                depth_m=m.get("depth_m"),
                temperature_c=m.get("temperature_c"),
                salinity_psu=m.get("salinity_psu"),
                dissolved_oxygen=m.get("dissolved_oxygen"),
                chlorophyll=m.get("chlorophyll"),
                ph=m.get("ph"),
                qc_flag=m.get("qc_flag"),
            )
        )


def profile_to_dict(p: Profile) -> dict:
    """Convert a Profile ORM instance into a plain dict."""
    return {
        "id": p.id,
        "float_id": p.float_id,
        "cycle_number": p.cycle_number,
        "profile_date": p.profile_date,
        "latitude": p.latitude,
        "longitude": p.longitude,
        "ocean_region": p.ocean_region,
    }


def measurement_to_dict(m: Measurement) -> dict:
    """Convert a Measurement ORM instance into a plain dict."""
    return {
        "id": m.id,
        "profile_id": m.profile_id,
        "pressure_dbar": m.pressure_dbar,
        "depth_m": m.depth_m,
        "temperature_c": m.temperature_c,
        "salinity_psu": m.salinity_psu,
        "dissolved_oxygen": m.dissolved_oxygen,
        "chlorophyll": m.chlorophyll,
        "ph": m.ph,
        "qc_flag": m.qc_flag,
    }


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points, in kilometers."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(a))


if __name__ == "__main__":
    # --- Self-test ---
    # Exercises haversine_km (the one pure function here that needs no DB)
    # with a known distance: Sydney <-> Melbourne is ~713 km great-circle.
    from shared.logger import get_logger

    logger = get_logger(__name__)

    dist = haversine_km(-33.8688, 151.2093, -37.8136, 144.9631)
    logger.info("haversine_km(Sydney, Melbourne) = %.1f km", dist)
    assert 700 <= dist <= 730, f"Unexpected haversine distance: {dist}"
    logger.info("Self-test passed.")
