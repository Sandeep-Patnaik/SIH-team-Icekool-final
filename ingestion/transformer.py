"""
ingestion/transformer.py

Transforms parsed + QC-cleaned ARGO data into the shared ProfileRecord
contract that Module 2 (database) consumes, attaching an ocean-region
label to each profile along the way.

Owned by: Module 1 — Data Ingestion & ETL.
"""

from __future__ import annotations

from typing import Any

from shared.logger import get_logger
from shared.regions import REGION_NAMES, assign_region
from shared.schemas import ProfileRecord
from ingestion.exceptions import RegionAssignmentError

logger = get_logger(__name__)


def assign_ocean_region(lat: float, lon: float) -> str:
    """
    Map a latitude/longitude pair to one of the exact ocean-region names
    defined in shared/regions.py.

    This function is a thin, fail-loud wrapper around
    shared.regions.assign_region(): it never invents its own region
    spelling and never returns anything outside REGION_NAMES. Where
    shared.regions.assign_region() would silently return "Unclassified"
    (no bounding box matched), this function instead raises
    RegionAssignmentError so the pipeline can log/skip that profile
    explicitly rather than let an unclassified region leak into
    Postgres/the dashboard filters.

    Args:
        lat: latitude in decimal degrees (-90..90).
        lon: longitude in decimal degrees (-180..180).

    Returns:
        One of the exact strings in shared.regions.REGION_NAMES.

    Raises:
        RegionAssignmentError: if lat/lon are out of valid range, or if
            no region bounding box in shared/regions.py contains the
            given coordinates.
    """
    try:
        # Defensive range validation up front — a bad lat/lon should
        # never silently fall through to "Unclassified".
        if lat is None or lon is None:
            raise RegionAssignmentError(
                f"Cannot assign region: lat={lat!r}, lon={lon!r} (missing value)"
            )
        if not (-90.0 <= float(lat) <= 90.0):
            raise RegionAssignmentError(f"Latitude out of range: {lat!r}")
        if not (-180.0 <= float(lon) <= 180.0):
            raise RegionAssignmentError(f"Longitude out of range: {lon!r}")

        region = assign_region(float(lat), float(lon))

        if region not in REGION_NAMES:
            # shared.regions.assign_region() returns "Unclassified" when
            # no bounding box matches — treat that as a hard failure here
            # rather than letting an off-list string reach downstream
            # modules that filter on the exact REGION_NAMES strings.
            raise RegionAssignmentError(
                f"No ocean region matched lat={lat}, lon={lon}"
            )

        return region

    except RegionAssignmentError:
        raise
    except (TypeError, ValueError) as exc:
        logger.error(
            "Failed to assign ocean region for lat=%r, lon=%r", lat, lon, exc_info=True
        )
        raise RegionAssignmentError(
            f"Region assignment failed for lat={lat!r}, lon={lon!r}: {exc}"
        ) from exc


def to_profile_record(
    float_meta: dict[str, Any],
    profile_meta: dict[str, Any],
    measurements: list[dict[str, Any]],
) -> ProfileRecord:
    """
    Build a shared.schemas.ProfileRecord from parsed float metadata,
    profile metadata, and this profile's cleaned measurements.

    This is the single point where ingestion's internal dict shapes
    (from netcdf_parser.py / qc_cleaner.py) get converted into the
    locked Ingestion -> Database contract. All measurement values are
    passed through unchanged — this function does not drop, rename, or
    recompute any measurement field; QC filtering / BGC null-handling
    must already have happened upstream in qc_cleaner.py.

    Args:
        float_meta: dict from NetCDFParser.parse_float_metadata(),
            expected to contain at least "float_id".
        profile_meta: one entry from NetCDFParser.parse_profile_metadata(),
            expected to contain "cycle_number", "profile_date",
            "latitude", "longitude".
        measurements: cleaned per-depth-level measurement dicts (already
            QC-filtered and BGC-normalized) belonging to this profile.

    Returns:
        A populated ProfileRecord ready for ProfileLoader.bulk_insert()
        and ProfileLoader.write_parquet().

    Raises:
        RegionAssignmentError: propagated from assign_ocean_region() if
            this profile's coordinates can't be mapped to a region.
        KeyError: if float_meta/profile_meta is missing a required field
            (fails loudly rather than silently constructing a bad record).
        ValueError: if pydantic validation of ProfileRecord fails (e.g.
            malformed types coming from upstream parsing).
    """
    try:
        float_id = float_meta["float_id"]
        cycle_number = profile_meta["cycle_number"]
        profile_date = profile_meta["profile_date"]
        latitude = profile_meta["latitude"]
        longitude = profile_meta["longitude"]

        # Assign region here so it's captured once, at record-build time,
        # and stamped consistently onto the ProfileRecord for Module 2/5/6.
        ocean_region = assign_ocean_region(latitude, longitude)

        # Preserve every measurement dict exactly as handed to us (already
        # QC-filtered / BGC-normalized upstream) — no field renaming,
        # dropping, or recomputation happens here.
        record = ProfileRecord(
            float_id=float_id,
            cycle_number=cycle_number,
            profile_date=profile_date,
            latitude=latitude,
            longitude=longitude,
            ocean_region=ocean_region,
            measurements=list(measurements),
        )

        logger.info(
            "Built ProfileRecord float_id=%s cycle=%s region=%s (%d measurement rows)",
            float_id,
            cycle_number,
            ocean_region,
            len(measurements),
        )
        return record

    except RegionAssignmentError:
        # Already logged inside assign_ocean_region(); re-raise so the
        # pipeline can catch it and skip just this profile.
        raise
    except KeyError as exc:
        logger.error(
            "Missing required field building ProfileRecord: %s", exc, exc_info=True
        )
        raise
    except Exception as exc:  # noqa: BLE001 - e.g. pydantic ValidationError
        logger.error("Failed to build ProfileRecord", exc_info=True)
        raise ValueError(f"Could not build ProfileRecord: {exc}") from exc


if __name__ == "__main__":
    # --- Self-test ---
    # Exercises both functions against hand-built dicts, no DB/file I/O.
    from datetime import datetime

    logger.info("Running transformer self-test")

    # 1. assign_ocean_region: a coordinate inside the Bay of Bengal box.
    region = assign_ocean_region(15.0, 85.0)
    assert region == "Bay of Bengal", region
    print(f"assign_ocean_region(15.0, 85.0) -> {region}")  # noqa: T201

    # 2. assign_ocean_region: a coordinate with no matching box must raise.
    try:
        assign_ocean_region(0.0, -30.0)  # mid-Atlantic, outside all boxes
        raise AssertionError("expected RegionAssignmentError, none raised")
    except RegionAssignmentError as exc:
        print(f"correctly raised RegionAssignmentError: {exc}")  # noqa: T201

    # 3. to_profile_record: build a full ProfileRecord from sample inputs.
    sample_float_meta = {
        "float_id": "1901234",
        "deployment_lat": 15.0,
        "deployment_lon": 85.0,
        "deployment_date": datetime(2022, 1, 1),
        "status": "ACTIVE",
    }
    sample_profile_meta = {
        "cycle_number": 12,
        "profile_date": datetime(2023, 6, 1, 6, 0, 0),
        "latitude": 15.0,
        "longitude": 85.0,
    }
    sample_measurements = [
        {
            "profile_index": 0,
            "pressure_dbar": 5.0,
            "depth_m": 5.0,
            "temperature_c": 28.1,
            "salinity_psu": 35.1,
            "dissolved_oxygen": 210.5,
            "chlorophyll": None,
            "ph": None,
            "qc_flag": 1,
        },
        {
            "profile_index": 0,
            "pressure_dbar": 50.0,
            "depth_m": 50.0,
            "temperature_c": 22.4,
            "salinity_psu": 35.3,
            "dissolved_oxygen": 0.0,  # genuine zero, must be preserved as-is
            "chlorophyll": 0.4,
            "ph": 8.05,
            "qc_flag": 1,
        },
    ]

    profile_record = to_profile_record(
        sample_float_meta, sample_profile_meta, sample_measurements
    )
    assert profile_record.float_id == "1901234"
    assert profile_record.ocean_region == "Bay of Bengal"
    assert len(profile_record.measurements) == 2
    # Confirm measurement values were preserved unchanged, including the
    # genuine 0.0 dissolved_oxygen reading (not coerced to None).
    assert profile_record.measurements[1]["dissolved_oxygen"] == 0.0
    print("to_profile_record output:", profile_record)  # noqa: T201

    logger.info("transformer self-test passed")