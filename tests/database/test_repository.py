"""Tests for database.repository.ProfileRepository (Module 2).

Covers every method in the locked API contract, including the upsert path
(re-ingestion must never duplicate a profile) and run_raw_query's rejection
of anything that isn't a single, safe SELECT -- defense in depth alongside
Module 4's own sql_guard.py, tested here independently at this layer too.
"""
from __future__ import annotations

from datetime import date, datetime

import pytest

from database.exceptions import UnsafeQueryError
from shared.schemas import ProfileRecord


def make_record(
    float_id: str = "WMO_0001",
    cycle_number: int = 1,
    lat: float = 12.3,
    lon: float = 68.1,
    region: str = "Arabian Sea",
    profile_date: datetime = datetime(2023, 3, 4, 12, 0, 0),
) -> ProfileRecord:
    """Build a small, valid ProfileRecord for use across tests."""
    return ProfileRecord(
        float_id=float_id,
        cycle_number=cycle_number,
        profile_date=profile_date,
        latitude=lat,
        longitude=lon,
        ocean_region=region,
        measurements=[
            {
                "pressure_dbar": 5.0,
                "depth_m": 5.0,
                "temperature_c": 28.4,
                "salinity_psu": 35.1,
                "dissolved_oxygen": None,
                "chlorophyll": None,
                "ph": None,
                "qc_flag": 1,
            },
            {
                "pressure_dbar": 50.0,
                "depth_m": 50.0,
                "temperature_c": 24.1,
                "salinity_psu": 36.0,
                "dissolved_oxygen": 4.2,
                "chlorophyll": 0.5,
                "ph": 8.1,
                "qc_flag": 1,
            },
        ],
    )


# ---------------------------------------------------------------------
# insert_profile_record / upsert behavior
# ---------------------------------------------------------------------

class TestInsertProfileRecord:
    def test_insert_returns_profile_id(self, repo):
        profile_id = repo.insert_profile_record(make_record())
        assert isinstance(profile_id, int)

    def test_upsert_same_float_cycle_does_not_duplicate(self, repo):
        record = make_record()
        first_id = repo.insert_profile_record(record)
        second_id = repo.insert_profile_record(record)
        assert first_id == second_id

        rows = repo.get_profiles_by_region("Arabian Sea", date(2023, 1, 1), date(2023, 12, 31))
        assert len(rows) == 1

    def test_upsert_replaces_measurements(self, repo):
        record = make_record()
        profile_id = repo.insert_profile_record(record)
        assert len(repo.get_measurements_for_profile(profile_id)) == 2

        record.measurements = [record.measurements[0]]  # re-ingest with fewer rows
        repo.insert_profile_record(record)
        assert len(repo.get_measurements_for_profile(profile_id)) == 1

    def test_different_cycle_same_float_creates_new_profile(self, repo):
        first_id = repo.insert_profile_record(make_record(cycle_number=1))
        second_id = repo.insert_profile_record(make_record(cycle_number=2))
        assert first_id != second_id


# ---------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------

class TestGetProfilesByRegion:
    def test_filters_by_region_and_date_range(self, repo):
        repo.insert_profile_record(make_record(region="Arabian Sea"))
        repo.insert_profile_record(
            make_record(float_id="WMO_0002", region="Bay of Bengal")
        )

        rows = repo.get_profiles_by_region("Arabian Sea", date(2023, 1, 1), date(2023, 12, 31))
        assert len(rows) == 1
        assert rows[0]["ocean_region"] == "Arabian Sea"

    def test_outside_date_range_excluded(self, repo):
        repo.insert_profile_record(make_record())
        rows = repo.get_profiles_by_region("Arabian Sea", date(2024, 1, 1), date(2024, 12, 31))
        assert rows == []

    def test_no_match_returns_empty_list(self, repo):
        rows = repo.get_profiles_by_region("Southern Indian Ocean", date(2023, 1, 1), date(2023, 12, 31))
        assert rows == []


class TestGetProfilesNear:
    def test_finds_nearby_profile_within_radius(self, repo):
        repo.insert_profile_record(make_record(lat=12.3, lon=68.1))
        results = repo.get_profiles_near(12.3, 68.1, radius_km=10)
        assert len(results) == 1
        assert results[0]["distance_km"] < 1.0

    def test_excludes_profile_outside_radius(self, repo):
        repo.insert_profile_record(make_record(lat=12.3, lon=68.1))
        results = repo.get_profiles_near(-30.0, 90.0, radius_km=50)
        assert results == []

    def test_results_sorted_nearest_first(self, repo):
        repo.insert_profile_record(make_record(float_id="NEAR", lat=12.30, lon=68.10))
        repo.insert_profile_record(make_record(float_id="FAR", cycle_number=1, lat=12.60, lon=68.40))
        results = repo.get_profiles_near(12.30, 68.10, radius_km=100)
        assert [r["float_id"] for r in results] == ["NEAR", "FAR"]


class TestGetMeasurementsForProfile:
    def test_returns_measurements_shallowest_first(self, repo):
        profile_id = repo.insert_profile_record(make_record())
        rows = repo.get_measurements_for_profile(profile_id)
        assert [r["depth_m"] for r in rows] == [5.0, 50.0]

    def test_unknown_profile_id_returns_empty_list(self, repo):
        assert repo.get_measurements_for_profile(999999) == []

    def test_missing_bgc_fields_stay_none(self, repo):
        profile_id = repo.insert_profile_record(make_record())
        rows = repo.get_measurements_for_profile(profile_id)
        assert rows[0]["dissolved_oxygen"] is None
        assert rows[0]["chlorophyll"] is None
        assert rows[0]["ph"] is None


# ---------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------

class TestInsertReport:
    def test_insert_report_returns_id(self, repo):
        report_id = repo.insert_report(
            "Arabian Sea", date(2023, 1, 1), date(2023, 3, 31), "/reports/q1.pdf", "Q1 summary"
        )
        assert isinstance(report_id, int)


# ---------------------------------------------------------------------
# run_raw_query -- must reject everything except a single safe SELECT
# ---------------------------------------------------------------------

class TestRunRawQuery:
    def test_valid_select_executes(self, repo):
        repo.insert_profile_record(make_record())
        rows = repo.run_raw_query(
            "SELECT float_id, ocean_region FROM profiles WHERE float_id = :fid",
            {"fid": "WMO_0001"},
        )
        assert len(rows) == 1
        assert rows[0]["ocean_region"] == "Arabian Sea"

    @pytest.mark.parametrize(
        "sql",
        [
            "DROP TABLE profiles",
            "DELETE FROM profiles",
            "UPDATE profiles SET ocean_region = 'x'",
            "INSERT INTO profiles (float_id) VALUES ('x')",
            "SELECT * FROM profiles; DROP TABLE profiles;",  # stacked statement
            "SELECT * FROM profiles -- ; DROP TABLE profiles",  # comment injection
            "SELECT * FROM profiles /* comment */",
            "",
            "   ",
        ],
    )
    def test_rejects_unsafe_sql(self, repo, sql):
        with pytest.raises(UnsafeQueryError):
            repo.run_raw_query(sql)

    def test_rejects_query_touching_no_allowed_table(self, repo):
        with pytest.raises(UnsafeQueryError):
            repo.run_raw_query("SELECT * FROM sqlite_master")
