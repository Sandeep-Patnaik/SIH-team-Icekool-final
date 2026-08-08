"""ProfileRepository — the ONLY sanctioned way any other OceanMind AI module
touches the database (Module 2: Database & Query Layer).

Every method here returns plain dicts, never raw SQLAlchemy Row/ORM objects,
so callers in other modules never need to import database.models. This is the
locked API contract published to Modules 1, 4, 5, 6 before the real
implementation even landed — signatures must not change without flagging it
to the whole team.

Private upsert/replace/conversion helpers live in database.repository_helpers
(split out to keep this file under the project's ~300-400 line cap); they are
implementation details of ProfileRepository, not part of the locked API.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timezone

from sqlalchemy import select, text
from sqlalchemy.exc import SQLAlchemyError

from database import repository_helpers as _helpers
from database.exceptions import RecordInsertError, UnsafeQueryError
from database.models import Measurement, Profile, Report
from database.session import session_scope
from shared.logger import get_logger
from shared.schemas import ProfileRecord

logger = get_logger(__name__)

# Only these tables may ever be touched by run_raw_query(); this is the
# defense-in-depth guard alongside Module 4's own sql_guard.py.
_ALLOWED_QUERY_TABLES = {"floats", "profiles", "measurements", "reports"}

# A single leading SELECT (optionally wrapped in whitespace) is the only
# shape run_raw_query() will execute. No semicolons (blocks stacked
# statements), no comment markers (blocks comment-based injection tricks).
_SELECT_ONLY_RE = re.compile(r"^\s*SELECT\b", re.IGNORECASE)
_FORBIDDEN_TOKENS_RE = re.compile(
    r"(;|--|/\*|\*/|\bINSERT\b|\bUPDATE\b|\bDELETE\b|\bDROP\b|\bALTER\b|"
    r"\bTRUNCATE\b|\bCREATE\b|\bGRANT\b|\bREVOKE\b|\bATTACH\b)",
    re.IGNORECASE,
)


class ProfileRepository:
    """Data-access layer for ARGO float/profile/measurement/report data.

    All methods open and close their own session via database.session.session_scope(),
    so callers never manage sessions or transactions themselves.
    """

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def insert_profile_record(self, record: ProfileRecord) -> int:
        """Upsert one ARGO profile (and its measurements) from Module 1's ingestion.

        Upserts on (float_id, cycle_number): re-running ingestion over the same
        file never creates a duplicate profile. The float row is upserted too
        (deployment metadata may arrive/refresh alongside any profile). Existing
        measurements for the profile are replaced wholesale with the new set,
        since a re-ingested profile's measurement list is the source of truth.

        Args:
            record: a validated shared.schemas.ProfileRecord.

        Returns:
            The database id of the (inserted or existing) profiles row.

        Raises:
            RecordInsertError: if the insert/upsert fails for any reason.
        """
        try:
            with session_scope() as session:
                _helpers.upsert_float(session, record)
                profile_id = _helpers.upsert_profile(session, record)
                _helpers.replace_measurements(session, profile_id, record.measurements)
                return profile_id
        except SQLAlchemyError as exc:
            logger.error(
                "insert_profile_record failed for float_id=%s cycle=%s",
                record.float_id,
                record.cycle_number,
                exc_info=True,
            )
            raise RecordInsertError(
                f"Could not insert profile for float_id={record.float_id} "
                f"cycle={record.cycle_number}"
            ) from exc

    def insert_report(
        self,
        region: str,
        period_start: date,
        period_end: date,
        file_path: str,
        summary_text: str,
    ) -> int:
        """Insert a generated Ocean Health report row (Module 6 output).

        Args:
            region: ocean region name (must match shared/regions.py exactly).
            period_start: start date of the reporting period.
            period_end: end date of the reporting period.
            file_path: where the report file (PDF/HTML) was written to disk.
            summary_text: short plain-text summary for dashboard display.

        Returns:
            The database id of the newly inserted reports row.

        Raises:
            RecordInsertError: if the insert fails for any reason.
        """
        try:
            with session_scope() as session:
                report = Report(
                    generated_at=datetime.now(timezone.utc),
                    ocean_region=region,
                    period_start=period_start,
                    period_end=period_end,
                    file_path=file_path,
                    summary_text=summary_text,
                )
                session.add(report)
                session.flush()
                return report.id
        except SQLAlchemyError as exc:
            logger.error("insert_report failed for region=%s", region, exc_info=True)
            raise RecordInsertError(f"Could not insert report for region={region}") from exc

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get_profiles_by_region(self, region: str, start: date, end: date) -> list[dict]:
        """Return profiles in a given ocean region within [start, end], inclusive.

        Args:
            region: exact ocean region name from shared/regions.py.
            start: inclusive start date (compared against profile_date).
            end: inclusive end date (compared against profile_date).

        Returns:
            List of plain dicts, one per matching profile, newest first.
            Empty list if none match or on error — this is a read path used
            live by the dashboard, so it fails soft rather than raising.
        """
        try:
            with session_scope() as session:
                stmt = (
                    select(Profile)
                    .where(Profile.ocean_region == region)
                    .where(Profile.profile_date >= start)
                    .where(Profile.profile_date <= end)
                    .order_by(Profile.profile_date.desc())
                )
                rows = session.execute(stmt).scalars().all()
                return [_helpers.profile_to_dict(p) for p in rows]
        except SQLAlchemyError:
            logger.error(
                "get_profiles_by_region failed for region=%s [%s, %s]",
                region,
                start,
                end,
                exc_info=True,
            )
            return []

    def get_profiles_near(self, lat: float, lon: float, radius_km: float) -> list[dict]:
        """Return profiles within radius_km of (lat, lon), using the haversine formula.

        Filters in Python after a coarse bounding-box pre-filter in SQL, which
        keeps this portable across SQLite (used in tests) and Postgres without
        relying on PostGIS.

        Args:
            lat: query latitude in decimal degrees.
            lon: query longitude in decimal degrees.
            radius_km: search radius in kilometers.

        Returns:
            List of plain dicts, one per matching profile, nearest first.
            Empty list if none match or on error.
        """
        try:
            with session_scope() as session:
                # Coarse bounding box (~1 degree latitude ~= 111 km) to avoid a
                # full table scan before the precise haversine check below.
                deg_pad = max(radius_km / 111.0, 0.01)
                stmt = (
                    select(Profile)
                    .where(Profile.latitude.between(lat - deg_pad, lat + deg_pad))
                    .where(Profile.longitude.between(lon - deg_pad, lon + deg_pad))
                )
                candidates = session.execute(stmt).scalars().all()

                results = []
                for p in candidates:
                    dist = _helpers.haversine_km(lat, lon, p.latitude, p.longitude)
                    if dist <= radius_km:
                        d = _helpers.profile_to_dict(p)
                        d["distance_km"] = round(dist, 3)
                        results.append(d)
                results.sort(key=lambda d: d["distance_km"])
                return results
        except SQLAlchemyError:
            logger.error(
                "get_profiles_near failed for (%s, %s) radius=%s",
                lat,
                lon,
                radius_km,
                exc_info=True,
            )
            return []

    def get_measurements_for_profile(self, profile_id: int) -> list[dict]:
        """Return all depth-level measurements for one profile, shallowest first.

        Args:
            profile_id: the profiles.id to fetch measurements for.

        Returns:
            List of plain dicts, one per measurement. Empty list if the
            profile has no measurements, doesn't exist, or on error.
        """
        try:
            with session_scope() as session:
                stmt = (
                    select(Measurement)
                    .where(Measurement.profile_id == profile_id)
                    .order_by(Measurement.depth_m.asc())
                )
                rows = session.execute(stmt).scalars().all()
                return [_helpers.measurement_to_dict(m) for m in rows]
        except SQLAlchemyError:
            logger.error(
                "get_measurements_for_profile failed for profile_id=%s",
                profile_id,
                exc_info=True,
            )
            return []

    def run_raw_query(self, sql: str, params: dict | None = None) -> list[dict]:
        """Execute a read-only, parameterized SELECT and return plain dicts.

        Used by Module 4's NL->SQL query engine. This is a defense-in-depth
        guard on top of Module 4's own sql_guard.py — this layer never trusts
        that the caller already validated the SQL.

        Args:
            sql: a single SELECT statement, using :named parameters (never
                string-interpolated values).
            params: parameter values for the query's :named placeholders.

        Returns:
            List of plain dicts, one per result row.

        Raises:
            UnsafeQueryError: if sql is not a single, safe SELECT statement
                touching only the allowed tables.
        """
        self._validate_select_only(sql)
        try:
            with session_scope() as session:
                result = session.execute(text(sql), params or {})
                columns = result.keys()
                return [dict(zip(columns, row)) for row in result.fetchall()]
        except SQLAlchemyError as exc:
            logger.error("run_raw_query failed for sql=%r", sql, exc_info=True)
            raise UnsafeQueryError(f"Query execution failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_select_only(sql: str) -> None:
        """Reject anything that isn't a single, safe SELECT on allowed tables."""
        if not sql or not _SELECT_ONLY_RE.match(sql):
            raise UnsafeQueryError("Only a single SELECT statement is permitted.")
        if _FORBIDDEN_TOKENS_RE.search(sql):
            raise UnsafeQueryError(
                "Query contains a forbidden token (statement separator, comment, "
                "or a non-SELECT DML/DDL keyword)."
            )
        referenced = {t.lower() for t in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", sql)}
        if not referenced & _ALLOWED_QUERY_TABLES:
            raise UnsafeQueryError(
                f"Query must reference at least one of {sorted(_ALLOWED_QUERY_TABLES)}."
            )


if __name__ == "__main__":
    # --- Self-test ---
    # Exercises the full repository against the locally configured DB
    # (per config.py's DATABASE_URL) with throwaway data, then cleans up.
    from database.models import Base
    from database.session import get_engine

    Base.metadata.create_all(get_engine())

    repo = ProfileRepository()
    record = ProfileRecord(
        float_id="SELFTEST_0001",
        cycle_number=1,
        profile_date=datetime(2023, 3, 4, 12, 0, 0),
        latitude=12.3,
        longitude=68.1,
        ocean_region="Arabian Sea",
        measurements=[
            {"pressure_dbar": 5.0, "depth_m": 5.0, "temperature_c": 28.4,
             "salinity_psu": 35.1, "dissolved_oxygen": None, "chlorophyll": None,
             "ph": None, "qc_flag": 1},
            {"pressure_dbar": 50.0, "depth_m": 50.0, "temperature_c": 24.1,
             "salinity_psu": 36.0, "dissolved_oxygen": None, "chlorophyll": None,
             "ph": None, "qc_flag": 1},
        ],
    )
    pid = repo.insert_profile_record(record)
    logger.info("Inserted profile id=%s", pid)

    pid_again = repo.insert_profile_record(record)
    assert pid == pid_again, "Upsert should not create a duplicate profile"
    logger.info("Upsert re-run confirmed no duplicate.")

    by_region = repo.get_profiles_by_region("Arabian Sea", date(2023, 1, 1), date(2023, 12, 31))
    logger.info("get_profiles_by_region -> %d profile(s)", len(by_region))

    near = repo.get_profiles_near(12.3, 68.1, radius_km=10)
    logger.info("get_profiles_near -> %d profile(s)", len(near))

    measurements = repo.get_measurements_for_profile(pid)
    logger.info("get_measurements_for_profile -> %d row(s)", len(measurements))

    report_id = repo.insert_report(
        "Arabian Sea", date(2023, 1, 1), date(2023, 3, 31), "/tmp/report.pdf", "Self-test report"
    )
    logger.info("Inserted report id=%s", report_id)

    raw = repo.run_raw_query("SELECT float_id, ocean_region FROM profiles WHERE float_id = :fid",
                              {"fid": "SELFTEST_0001"})
    logger.info("run_raw_query -> %s", raw)

    try:
        repo.run_raw_query("DROP TABLE profiles")
        raise SystemExit("SECURITY BUG: unsafe SQL was not rejected!")
    except UnsafeQueryError:
        logger.info("run_raw_query correctly rejected an unsafe statement.")

    logger.info("Self-test passed.")
