"""Unit tests for intelligence_engine/anomaly_detector.py.

ProfileRepository (Module 2) is fully mocked here so this suite runs
standalone with no database and no other module's code, per the Master
Prompt's testing standard. Threshold values are read directly from
anomaly_detector.py's constants rather than hardcoded, so these tests keep
tracking the real thresholds if they're ever tuned.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest

from intelligence_engine.anomaly_detector import (
    OXYGEN_MODERATE_THRESHOLD,
    SALINITY_MODERATE_THRESHOLD_PSU,
    SEVERE_MULTIPLIER,
    TEMPERATURE_MODERATE_THRESHOLD_C,
    AnomalyDetector,
)
from intelligence_engine.exceptions import AnomalyDetectionError
from intelligence_engine.health_index import (
    BASELINE_SALINITY_PSU,
    BASELINE_TEMPERATURE_C,
    OXYGEN_HEALTHY_RANGE,
)
from shared.regions import REGION_NAMES

TEST_REGION = "Arabian Sea"
TEST_START = date(2024, 1, 1)
TEST_END = date(2024, 1, 31)

BASELINE_TEMP = BASELINE_TEMPERATURE_C[TEST_REGION]
BASELINE_SALINITY = BASELINE_SALINITY_PSU[TEST_REGION]
OXYGEN_LOWER_BOUND = OXYGEN_HEALTHY_RANGE[0]
OXYGEN_MID_RANGE = sum(OXYGEN_HEALTHY_RANGE) / 2


def _mock_repository(profiles: list[dict], measurements_by_profile: dict[int, list[dict]]) -> MagicMock:
    """Build a mocked ProfileRepository returning fixed profiles/measurements.

    Args:
        profiles: Rows to return from get_profiles_by_region().
        measurements_by_profile: Maps profile id -> measurement rows returned
            by get_measurements_for_profile() for that id.

    Returns:
        A MagicMock duck-typed as ProfileRepository (Module 2), with no
        real database access.
    """
    repository = MagicMock()
    repository.get_profiles_by_region.return_value = profiles
    repository.get_measurements_for_profile.side_effect = (
        lambda profile_id: measurements_by_profile.get(profile_id, [])
    )
    return repository


def _measurement(temperature_c: float, salinity_psu: float, dissolved_oxygen: float) -> dict:
    """Build a single measurement row with all three monitored fields set."""
    return {
        "temperature_c": temperature_c,
        "salinity_psu": salinity_psu,
        "dissolved_oxygen": dissolved_oxygen,
    }


def _flags_for(parameter: str, flags: list[dict]) -> list[dict]:
    """Filter detected flags down to a single monitored parameter."""
    return [flag for flag in flags if flag["parameter"] == parameter]


class TestDetectionAgainstBaseline:
    """Verifies flags are (or aren't) raised relative to the fixed historical baseline."""

    def test_no_flags_when_all_measurements_at_baseline(self) -> None:
        """Measurements exactly at baseline, mid-range oxygen -> zero anomaly flags."""
        measurements = [
            _measurement(BASELINE_TEMP, BASELINE_SALINITY, OXYGEN_MID_RANGE),
            _measurement(BASELINE_TEMP, BASELINE_SALINITY, OXYGEN_MID_RANGE),
        ]
        repository = _mock_repository([{"id": 1}], {1: measurements})
        detector = AnomalyDetector(repository)

        flags = detector.detect(TEST_REGION, TEST_START, TEST_END)

        assert flags == []

    def test_deviation_just_below_threshold_raises_no_flag(self) -> None:
        """A deviation strictly less than the moderate threshold must not be flagged."""
        below_threshold_temp = BASELINE_TEMP + (TEMPERATURE_MODERATE_THRESHOLD_C - 0.01)
        measurements = [_measurement(below_threshold_temp, BASELINE_SALINITY, OXYGEN_MID_RANGE)]
        repository = _mock_repository([{"id": 1}], {1: measurements})
        detector = AnomalyDetector(repository)

        flags = detector.detect(TEST_REGION, TEST_START, TEST_END)

        assert _flags_for("temperature", flags) == []

    def test_measurements_pooled_across_all_profiles_in_region(self) -> None:
        """Measurements from every returned profile are pooled before checking deviation."""
        measurements_by_profile = {
            1: [_measurement(BASELINE_TEMP + 5.0, BASELINE_SALINITY, OXYGEN_MID_RANGE)],
            2: [_measurement(BASELINE_TEMP + 5.0, BASELINE_SALINITY, OXYGEN_MID_RANGE)],
        }
        repository = _mock_repository([{"id": 1}, {"id": 2}], measurements_by_profile)
        detector = AnomalyDetector(repository)

        flags = detector.detect(TEST_REGION, TEST_START, TEST_END)

        assert repository.get_measurements_for_profile.call_count == 2
        assert len(_flags_for("temperature", flags)) == 1

    def test_multiple_parameters_can_flag_simultaneously(self) -> None:
        """Temperature, salinity, and oxygen anomalies must all surface independently."""
        measurements = [
            _measurement(
                BASELINE_TEMP + 5.0,
                BASELINE_SALINITY + 5.0,
                OXYGEN_LOWER_BOUND - OXYGEN_MODERATE_THRESHOLD,
            )
        ]
        repository = _mock_repository([{"id": 1}], {1: measurements})
        detector = AnomalyDetector(repository)

        flags = detector.detect(TEST_REGION, TEST_START, TEST_END)

        parameters_flagged = {flag["parameter"] for flag in flags}
        assert parameters_flagged == {"temperature", "salinity", "dissolved_oxygen"}


class TestModerateVsSevereThresholds:
    """Verifies the moderate/severe severity boundary for temperature and salinity."""

    def test_temperature_at_moderate_threshold_is_moderate(self) -> None:
        """Deviation exactly at the moderate threshold -> flagged moderate, not severe."""
        temp = BASELINE_TEMP + TEMPERATURE_MODERATE_THRESHOLD_C
        measurements = [_measurement(temp, BASELINE_SALINITY, OXYGEN_MID_RANGE)]
        repository = _mock_repository([{"id": 1}], {1: measurements})
        detector = AnomalyDetector(repository)

        flags = _flags_for("temperature", detector.detect(TEST_REGION, TEST_START, TEST_END))

        assert len(flags) == 1
        assert flags[0]["severity"] == "moderate"

    def test_temperature_at_severe_threshold_is_severe(self) -> None:
        """Deviation exactly at moderate_threshold * SEVERE_MULTIPLIER -> flagged severe."""
        temp = BASELINE_TEMP + TEMPERATURE_MODERATE_THRESHOLD_C * SEVERE_MULTIPLIER
        measurements = [_measurement(temp, BASELINE_SALINITY, OXYGEN_MID_RANGE)]
        repository = _mock_repository([{"id": 1}], {1: measurements})
        detector = AnomalyDetector(repository)

        flags = _flags_for("temperature", detector.detect(TEST_REGION, TEST_START, TEST_END))

        assert len(flags) == 1
        assert flags[0]["severity"] == "severe"

    def test_salinity_at_moderate_threshold_is_moderate(self) -> None:
        """Deviation exactly at the moderate threshold -> flagged moderate, not severe."""
        salinity = BASELINE_SALINITY + SALINITY_MODERATE_THRESHOLD_PSU
        measurements = [_measurement(BASELINE_TEMP, salinity, OXYGEN_MID_RANGE)]
        repository = _mock_repository([{"id": 1}], {1: measurements})
        detector = AnomalyDetector(repository)

        flags = _flags_for("salinity", detector.detect(TEST_REGION, TEST_START, TEST_END))

        assert len(flags) == 1
        assert flags[0]["severity"] == "moderate"

    def test_salinity_at_severe_threshold_is_severe(self) -> None:
        """Deviation exactly at moderate_threshold * SEVERE_MULTIPLIER -> flagged severe."""
        salinity = BASELINE_SALINITY + SALINITY_MODERATE_THRESHOLD_PSU * SEVERE_MULTIPLIER
        measurements = [_measurement(BASELINE_TEMP, salinity, OXYGEN_MID_RANGE)]
        repository = _mock_repository([{"id": 1}], {1: measurements})
        detector = AnomalyDetector(repository)

        flags = _flags_for("salinity", detector.detect(TEST_REGION, TEST_START, TEST_END))

        assert len(flags) == 1
        assert flags[0]["severity"] == "severe"

    def test_deviation_direction_is_recorded_correctly(self) -> None:
        """A below-baseline deviation must still be captured (as a negative signed value)."""
        temp = BASELINE_TEMP - TEMPERATURE_MODERATE_THRESHOLD_C * SEVERE_MULTIPLIER
        measurements = [_measurement(temp, BASELINE_SALINITY, OXYGEN_MID_RANGE)]
        repository = _mock_repository([{"id": 1}], {1: measurements})
        detector = AnomalyDetector(repository)

        flags = _flags_for("temperature", detector.detect(TEST_REGION, TEST_START, TEST_END))

        assert len(flags) == 1
        assert flags[0]["severity"] == "severe"
        assert flags[0]["deviation"] < 0
        assert "below" in flags[0]["message"]


class TestOxygenAnomalyDetection:
    """Verifies dissolved-oxygen deficit detection, separate from the baseline checks."""

    def test_no_flag_when_oxygen_in_healthy_range(self) -> None:
        """Mid-range oxygen must never be flagged."""
        measurements = [_measurement(BASELINE_TEMP, BASELINE_SALINITY, OXYGEN_MID_RANGE)]
        repository = _mock_repository([{"id": 1}], {1: measurements})
        detector = AnomalyDetector(repository)

        flags = _flags_for(
            "dissolved_oxygen", detector.detect(TEST_REGION, TEST_START, TEST_END)
        )

        assert flags == []

    def test_no_flag_just_below_deficit_threshold(self) -> None:
        """A deficit strictly less than OXYGEN_MODERATE_THRESHOLD must not be flagged."""
        oxygen = OXYGEN_LOWER_BOUND - (OXYGEN_MODERATE_THRESHOLD - 1.0)
        measurements = [_measurement(BASELINE_TEMP, BASELINE_SALINITY, oxygen)]
        repository = _mock_repository([{"id": 1}], {1: measurements})
        detector = AnomalyDetector(repository)

        flags = _flags_for(
            "dissolved_oxygen", detector.detect(TEST_REGION, TEST_START, TEST_END)
        )

        assert flags == []

    def test_oxygen_deficit_at_moderate_threshold_is_moderate(self) -> None:
        """Deficit exactly at OXYGEN_MODERATE_THRESHOLD -> flagged moderate."""
        oxygen = OXYGEN_LOWER_BOUND - OXYGEN_MODERATE_THRESHOLD
        measurements = [_measurement(BASELINE_TEMP, BASELINE_SALINITY, oxygen)]
        repository = _mock_repository([{"id": 1}], {1: measurements})
        detector = AnomalyDetector(repository)

        flags = _flags_for(
            "dissolved_oxygen", detector.detect(TEST_REGION, TEST_START, TEST_END)
        )

        assert len(flags) == 1
        assert flags[0]["severity"] == "moderate"

    def test_oxygen_deficit_at_severe_threshold_is_severe(self) -> None:
        """Deficit exactly at OXYGEN_MODERATE_THRESHOLD * SEVERE_MULTIPLIER -> flagged severe."""
        oxygen = OXYGEN_LOWER_BOUND - OXYGEN_MODERATE_THRESHOLD * SEVERE_MULTIPLIER
        measurements = [_measurement(BASELINE_TEMP, BASELINE_SALINITY, oxygen)]
        repository = _mock_repository([{"id": 1}], {1: measurements})
        detector = AnomalyDetector(repository)

        flags = _flags_for(
            "dissolved_oxygen", detector.detect(TEST_REGION, TEST_START, TEST_END)
        )

        assert len(flags) == 1
        assert flags[0]["severity"] == "severe"

    def test_oxygen_excess_above_range_is_not_flagged(self) -> None:
        """Only a deficit below the lower bound is checked - excess above it is not an anomaly here."""
        oxygen = OXYGEN_HEALTHY_RANGE[1] + 500.0
        measurements = [_measurement(BASELINE_TEMP, BASELINE_SALINITY, oxygen)]
        repository = _mock_repository([{"id": 1}], {1: measurements})
        detector = AnomalyDetector(repository)

        flags = _flags_for(
            "dissolved_oxygen", detector.detect(TEST_REGION, TEST_START, TEST_END)
        )

        assert flags == []


class TestInvalidRegionHandling:
    """Verifies unknown regions are rejected before any repository call is made."""

    def test_unknown_region_raises_anomaly_detection_error(self) -> None:
        """A region not in shared.regions.REGION_NAMES must raise, not silently proceed."""
        repository = _mock_repository([], {})
        detector = AnomalyDetector(repository)

        with pytest.raises(AnomalyDetectionError, match="Unknown ocean region"):
            detector.detect("Atlantis", TEST_START, TEST_END)

        repository.get_profiles_by_region.assert_not_called()

    def test_all_documented_regions_have_baselines(self) -> None:
        """Every region in shared.regions.REGION_NAMES must be usable without a KeyError."""
        for region in REGION_NAMES:
            assert region in BASELINE_TEMPERATURE_C
            assert region in BASELINE_SALINITY_PSU

    def test_profile_fetch_failure_raises_anomaly_detection_error(self) -> None:
        """A repository-level exception fetching profiles must be wrapped, not leaked."""
        repository = MagicMock()
        repository.get_profiles_by_region.side_effect = ConnectionError("db unreachable")
        detector = AnomalyDetector(repository)

        with pytest.raises(AnomalyDetectionError):
            detector.detect(TEST_REGION, TEST_START, TEST_END)

    def test_measurement_fetch_failure_raises_anomaly_detection_error(self) -> None:
        """A repository-level exception fetching measurements must be wrapped, not leaked."""
        repository = MagicMock()
        repository.get_profiles_by_region.return_value = [{"id": 1}]
        repository.get_measurements_for_profile.side_effect = ConnectionError("db unreachable")
        detector = AnomalyDetector(repository)

        with pytest.raises(AnomalyDetectionError):
            detector.detect(TEST_REGION, TEST_START, TEST_END)


class TestEmptyDatasetBehavior:
    """Verifies a region/period with no data returns an empty flag list, not an error."""

    def test_no_profiles_returns_empty_flag_list(self) -> None:
        """No profiles at all for the region/period -> empty flags, no exception."""
        repository = _mock_repository([], {})
        detector = AnomalyDetector(repository)

        flags = detector.detect(TEST_REGION, TEST_START, TEST_END)

        assert flags == []

    def test_profiles_with_no_measurements_returns_empty_flag_list(self) -> None:
        """Profiles exist but none have measurement rows -> same empty-flags outcome."""
        repository = _mock_repository([{"id": 1}, {"id": 2}], {1: [], 2: []})
        detector = AnomalyDetector(repository)

        flags = detector.detect(TEST_REGION, TEST_START, TEST_END)

        assert flags == []

    def test_measurements_with_all_monitored_fields_null_returns_empty_flag_list(self) -> None:
        """Rows present but every monitored field null -> nothing to compare, no flags, no crash."""
        measurements = [
            {"temperature_c": None, "salinity_psu": None, "dissolved_oxygen": None},
        ]
        repository = _mock_repository([{"id": 1}], {1: measurements})
        detector = AnomalyDetector(repository)

        flags = detector.detect(TEST_REGION, TEST_START, TEST_END)

        assert flags == []


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))