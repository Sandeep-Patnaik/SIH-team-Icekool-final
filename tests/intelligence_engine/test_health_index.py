"""Unit tests for intelligence_engine/health_index.py.

ProfileRepository (Module 2) is fully mocked here so this suite runs
standalone with no database and no other module's code, per the Master
Prompt's testing standard. Every expected value below is hand-computed
against the documented 30/30/25/15 formula and the linear-decay component
formulas in health_index.py's module docstring - none of these numbers are
just "whatever the code currently returns".
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest

from intelligence_engine.exceptions import HealthIndexError
from intelligence_engine.health_index import (
    BASELINE_SALINITY_PSU,
    BASELINE_TEMPERATURE_C,
    OXYGEN_HEALTHY_RANGE,
    OXYGEN_MAX_DEFICIT,
    OXYGEN_MAX_EXCESS,
    SALINITY_MAX_DEVIATION_PSU,
    TEMPERATURE_MAX_DEVIATION_C,
    OceanHealthCalculator,
)
from shared.regions import REGION_NAMES

TEST_REGION = "Arabian Sea"
TEST_START = date(2024, 1, 1)
TEST_END = date(2024, 1, 31)


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


class TestHealthScoreCalculation:
    """Verifies compute() produces correct component and overall scores."""

    def test_at_baseline_in_range_complete_data_scores_100(self) -> None:
        """Measurements exactly at baseline, mid-range oxygen, no gaps -> perfect score."""
        measurements = [
            {
                "temperature_c": BASELINE_TEMPERATURE_C[TEST_REGION],
                "salinity_psu": BASELINE_SALINITY_PSU[TEST_REGION],
                "dissolved_oxygen": sum(OXYGEN_HEALTHY_RANGE) / 2,
            },
            {
                "temperature_c": BASELINE_TEMPERATURE_C[TEST_REGION],
                "salinity_psu": BASELINE_SALINITY_PSU[TEST_REGION],
                "dissolved_oxygen": sum(OXYGEN_HEALTHY_RANGE) / 2,
            },
        ]
        repository = _mock_repository([{"id": 1}], {1: measurements})
        calculator = OceanHealthCalculator(repository)

        result = calculator.compute(TEST_REGION, TEST_START, TEST_END)

        assert result.ocean_region == TEST_REGION
        assert result.period_start == TEST_START
        assert result.period_end == TEST_END
        assert result.contributing_factors == {
            "temperature_score": 100.0,
            "salinity_score": 100.0,
            "oxygen_score": 100.0,
            "completeness_score": 100.0,
        }
        assert result.score == 100.0
        assert result.recommendation == ""
        repository.get_profiles_by_region.assert_called_once_with(TEST_REGION, TEST_START, TEST_END)

    def test_measurements_pulled_from_all_profiles_in_region(self) -> None:
        """Measurements from every returned profile are pooled before scoring, not just one."""
        measurements_by_profile = {
            1: [{"temperature_c": 27.0, "salinity_psu": 36.0, "dissolved_oxygen": 250.0}],
            2: [{"temperature_c": 27.0, "salinity_psu": 36.0, "dissolved_oxygen": 250.0}],
        }
        repository = _mock_repository([{"id": 1}, {"id": 2}], measurements_by_profile)
        calculator = OceanHealthCalculator(repository)

        result = calculator.compute(TEST_REGION, TEST_START, TEST_END)

        assert repository.get_measurements_for_profile.call_count == 2
        assert result.score == 100.0


class TestWeightingFormula:
    """Verifies the 30/30/25/15 weighted combination against a hand-computed value."""

    def test_weighted_overall_score_matches_manual_calculation(self) -> None:
        """Mixed deviations across all four components must combine per the fixed weights."""
        baseline_temp = BASELINE_TEMPERATURE_C[TEST_REGION]
        baseline_salinity = BASELINE_SALINITY_PSU[TEST_REGION]

        # Two fully-populated rows (temp/salinity 50% off max deviation, oxygen 50%
        # into its deficit range) plus one row missing dissolved_oxygen only.
        measurements = [
            {
                "temperature_c": baseline_temp + TEMPERATURE_MAX_DEVIATION_C / 2,
                "salinity_psu": baseline_salinity + SALINITY_MAX_DEVIATION_PSU / 2,
                "dissolved_oxygen": OXYGEN_HEALTHY_RANGE[0] - OXYGEN_MAX_DEFICIT / 2,
            },
            {
                "temperature_c": baseline_temp + TEMPERATURE_MAX_DEVIATION_C / 2,
                "salinity_psu": baseline_salinity + SALINITY_MAX_DEVIATION_PSU / 2,
                "dissolved_oxygen": OXYGEN_HEALTHY_RANGE[0] - OXYGEN_MAX_DEFICIT / 2,
            },
            {
                "temperature_c": baseline_temp + TEMPERATURE_MAX_DEVIATION_C / 2,
                "salinity_psu": baseline_salinity + SALINITY_MAX_DEVIATION_PSU / 2,
                "dissolved_oxygen": None,
            },
        ]
        repository = _mock_repository([{"id": 1}], {1: measurements})
        calculator = OceanHealthCalculator(repository)

        result = calculator.compute(TEST_REGION, TEST_START, TEST_END)

        # Hand-computed expected components:
        # temperature_score = salinity_score = oxygen_score = 50.0 (each exactly
        # halfway between "at baseline"/"in range" and the zero-score threshold).
        # completeness_score = 100 * 8/9 (9 required-field slots, 1 missing).
        expected_completeness = 100.0 * 8 / 9
        expected_overall = round(
            (30.0 * 50.0 + 30.0 * 50.0 + 25.0 * 50.0 + 15.0 * expected_completeness) / 100.0,
            2,
        )

        assert result.contributing_factors["temperature_score"] == 50.0
        assert result.contributing_factors["salinity_score"] == 50.0
        assert result.contributing_factors["oxygen_score"] == 50.0
        assert result.contributing_factors["completeness_score"] == round(expected_completeness, 2)
        assert result.score == expected_overall
        assert result.score == 55.83

    def test_weights_sum_to_100_percent_of_scale(self) -> None:
        """A component-independent sanity check: uniform component scores pass through unchanged."""
        # If every component were exactly X, weighted average must also be X -
        # this fails immediately if the weights don't sum to 100.
        from intelligence_engine.health_index import (
            WEIGHT_COMPLETENESS,
            WEIGHT_OXYGEN,
            WEIGHT_SALINITY,
            WEIGHT_TEMPERATURE,
        )

        assert WEIGHT_TEMPERATURE + WEIGHT_SALINITY + WEIGHT_OXYGEN + WEIGHT_COMPLETENESS == 100.0


class TestInvalidRegionHandling:
    """Verifies unknown regions are rejected before any repository call is made."""

    def test_unknown_region_raises_health_index_error(self) -> None:
        """A region not in shared.regions.REGION_NAMES must raise, not silently proceed."""
        repository = _mock_repository([], {})
        calculator = OceanHealthCalculator(repository)

        with pytest.raises(HealthIndexError, match="Unknown ocean region"):
            calculator.compute("Atlantis", TEST_START, TEST_END)

        repository.get_profiles_by_region.assert_not_called()

    def test_all_documented_regions_are_accepted(self) -> None:
        """Every region in shared.regions.REGION_NAMES must have a baseline entry."""
        for region in REGION_NAMES:
            assert region in BASELINE_TEMPERATURE_C
            assert region in BASELINE_SALINITY_PSU

    def test_repository_failure_raises_health_index_error(self) -> None:
        """A repository-level exception must be wrapped, not leaked to the caller."""
        repository = MagicMock()
        repository.get_profiles_by_region.side_effect = ConnectionError("db unreachable")
        calculator = OceanHealthCalculator(repository)

        with pytest.raises(HealthIndexError):
            calculator.compute(TEST_REGION, TEST_START, TEST_END)


class TestEmptyDatasetHandling:
    """Verifies a region/period with no data degrades to a zero score, not an error."""

    def test_no_profiles_returns_zero_score(self) -> None:
        """No profiles at all for the region/period -> all-zero score, no exception."""
        repository = _mock_repository([], {})
        calculator = OceanHealthCalculator(repository)

        result = calculator.compute(TEST_REGION, TEST_START, TEST_END)

        assert result.score == 0.0
        assert result.contributing_factors == {
            "temperature_score": 0.0,
            "salinity_score": 0.0,
            "oxygen_score": 0.0,
            "completeness_score": 0.0,
        }

    def test_profiles_with_no_measurements_returns_zero_score(self) -> None:
        """Profiles exist but none have measurement rows -> same zero-score outcome."""
        repository = _mock_repository([{"id": 1}, {"id": 2}], {1: [], 2: []})
        calculator = OceanHealthCalculator(repository)

        result = calculator.compute(TEST_REGION, TEST_START, TEST_END)

        assert result.score == 0.0
        assert all(value == 0.0 for value in result.contributing_factors.values())


class TestBoundaryConditions:
    """Verifies scoring at the exact edges of each component's decay range."""

    def test_temperature_deviation_at_max_scores_zero(self) -> None:
        """Deviation exactly equal to TEMPERATURE_MAX_DEVIATION_C -> temperature_score == 0."""
        measurements = [
            {
                "temperature_c": BASELINE_TEMPERATURE_C[TEST_REGION] + TEMPERATURE_MAX_DEVIATION_C,
                "salinity_psu": BASELINE_SALINITY_PSU[TEST_REGION],
                "dissolved_oxygen": sum(OXYGEN_HEALTHY_RANGE) / 2,
            }
        ]
        repository = _mock_repository([{"id": 1}], {1: measurements})
        calculator = OceanHealthCalculator(repository)

        result = calculator.compute(TEST_REGION, TEST_START, TEST_END)

        assert result.contributing_factors["temperature_score"] == 0.0

    def test_salinity_deviation_at_max_scores_zero(self) -> None:
        """Deviation exactly equal to SALINITY_MAX_DEVIATION_PSU -> salinity_score == 0."""
        measurements = [
            {
                "temperature_c": BASELINE_TEMPERATURE_C[TEST_REGION],
                "salinity_psu": BASELINE_SALINITY_PSU[TEST_REGION] + SALINITY_MAX_DEVIATION_PSU,
                "dissolved_oxygen": sum(OXYGEN_HEALTHY_RANGE) / 2,
            }
        ]
        repository = _mock_repository([{"id": 1}], {1: measurements})
        calculator = OceanHealthCalculator(repository)

        result = calculator.compute(TEST_REGION, TEST_START, TEST_END)

        assert result.contributing_factors["salinity_score"] == 0.0

    @pytest.mark.parametrize("oxygen_value", [OXYGEN_HEALTHY_RANGE[0], OXYGEN_HEALTHY_RANGE[1]])
    def test_oxygen_at_inclusive_range_bounds_scores_100(self, oxygen_value: float) -> None:
        """The healthy range's own lower/upper bound must score 100 (inclusive), not decayed."""
        measurements = [
            {
                "temperature_c": BASELINE_TEMPERATURE_C[TEST_REGION],
                "salinity_psu": BASELINE_SALINITY_PSU[TEST_REGION],
                "dissolved_oxygen": oxygen_value,
            }
        ]
        repository = _mock_repository([{"id": 1}], {1: measurements})
        calculator = OceanHealthCalculator(repository)

        result = calculator.compute(TEST_REGION, TEST_START, TEST_END)

        assert result.contributing_factors["oxygen_score"] == 100.0

    def test_oxygen_deficit_at_max_scores_zero(self) -> None:
        """Deficit exactly equal to OXYGEN_MAX_DEFICIT below the lower bound -> oxygen_score == 0."""
        measurements = [
            {
                "temperature_c": BASELINE_TEMPERATURE_C[TEST_REGION],
                "salinity_psu": BASELINE_SALINITY_PSU[TEST_REGION],
                "dissolved_oxygen": OXYGEN_HEALTHY_RANGE[0] - OXYGEN_MAX_DEFICIT,
            }
        ]
        repository = _mock_repository([{"id": 1}], {1: measurements})
        calculator = OceanHealthCalculator(repository)

        result = calculator.compute(TEST_REGION, TEST_START, TEST_END)

        assert result.contributing_factors["oxygen_score"] == 0.0

    def test_oxygen_excess_at_max_scores_zero(self) -> None:
        """Excess exactly equal to OXYGEN_MAX_EXCESS above the upper bound -> oxygen_score == 0."""
        measurements = [
            {
                "temperature_c": BASELINE_TEMPERATURE_C[TEST_REGION],
                "salinity_psu": BASELINE_SALINITY_PSU[TEST_REGION],
                "dissolved_oxygen": OXYGEN_HEALTHY_RANGE[1] + OXYGEN_MAX_EXCESS,
            }
        ]
        repository = _mock_repository([{"id": 1}], {1: measurements})
        calculator = OceanHealthCalculator(repository)

        result = calculator.compute(TEST_REGION, TEST_START, TEST_END)

        assert result.contributing_factors["oxygen_score"] == 0.0

    def test_completeness_with_all_fields_missing_scores_zero(self) -> None:
        """Rows present but every required field null -> completeness_score == 0, not a crash."""
        measurements = [
            {"temperature_c": None, "salinity_psu": None, "dissolved_oxygen": None},
            {"temperature_c": None, "salinity_psu": None, "dissolved_oxygen": None},
        ]
        repository = _mock_repository([{"id": 1}], {1: measurements})
        calculator = OceanHealthCalculator(repository)

        result = calculator.compute(TEST_REGION, TEST_START, TEST_END)

        assert result.contributing_factors["completeness_score"] == 0.0
        # With every field null, the baseline/oxygen components also have no
        # data to average, so they fall back to zero too - overall score 0.
        assert result.score == 0.0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))