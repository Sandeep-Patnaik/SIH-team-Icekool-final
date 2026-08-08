"""Ocean Health Index calculation for OceanMind AI.

Formula (fixed, documented here so a "why is this score 62?" question in the
demo has a one-line answer)
--------------------------------------------------------------------------
score = 0.30 * temperature_score
      + 0.30 * salinity_score
      + 0.25 * oxygen_score
      + 0.15 * completeness_score

Each component is normalized independently to the 0-100 range and stored
verbatim in ``OceanHealthScore.contributing_factors`` under the keys
``temperature_score``, ``salinity_score``, ``oxygen_score``,
``completeness_score`` (these exact keys are consumed by
``recommendations.py``'s ``FACTOR_LABELS`` mapping - do not rename them).

Component definitions
----------------------
temperature_score / salinity_score
    100 at zero deviation from the region's fixed historical baseline
    (``BASELINE_TEMPERATURE_C`` / ``BASELINE_SALINITY_PSU``), decaying
    linearly to 0 at or beyond a fixed maximum-deviation constant
    (``TEMPERATURE_MAX_DEVIATION_C`` / ``SALINITY_MAX_DEVIATION_PSU``).
    This mirrors the deviation-from-baseline logic anomaly_detector.py uses
    for flagging, so the two modules never disagree about what "normal" is.

oxygen_score
    100 anywhere inside ``OXYGEN_HEALTHY_RANGE`` (inclusive). Below the
    lower bound, decays linearly to 0 at ``OXYGEN_MAX_DEFICIT`` below it.
    Above the upper bound, decays linearly to 0 at ``OXYGEN_MAX_EXCESS``
    above it (excess dissolved oxygen can indicate bloom conditions, so it
    is penalized too, just more gently than a deficit).

completeness_score
    Percentage of non-null values across the required measurement fields
    (``REQUIRED_MEASUREMENT_FIELDS``) over all measurement rows retrieved
    for the region/period. A simple, defensible stand-in for "how much of
    the expected data did we actually get".

If no measurements are available for the region/period at all, every
component is reported as 0.0 and the overall score is 0.0 rather than
raising - a region with no data is unambiguously the worst case, not an
error.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

from shared.logger import get_logger
from shared.regions import REGION_NAMES
from shared.schemas import OceanHealthScore
from intelligence_engine.exceptions import HealthIndexError

if TYPE_CHECKING:
    from database.repository import ProfileRepository

logger = get_logger(__name__)

# --- Fixed historical baselines (°C / PSU), one per shared.regions.REGION_NAMES entry ---
# Placeholder reference values for the demo - refine with real climatology before submission.
BASELINE_TEMPERATURE_C: dict[str, float] = {
    "Arabian Sea": 27.0,
    "Bay of Bengal": 28.0,
    "Equatorial Indian Ocean": 28.5,
    "Southern Indian Ocean": 10.0,
}

BASELINE_SALINITY_PSU: dict[str, float] = {
    "Arabian Sea": 36.0,
    "Bay of Bengal": 33.0,
    "Equatorial Indian Ocean": 34.5,
    "Southern Indian Ocean": 34.0,
}

# Dissolved oxygen "healthy" band (micromol/kg). Inclusive; scored 100 throughout.
OXYGEN_HEALTHY_RANGE: tuple[float, float] = (200.0, 300.0)

# Deviation at which the corresponding component score reaches 0.
TEMPERATURE_MAX_DEVIATION_C = 5.0
SALINITY_MAX_DEVIATION_PSU = 3.0
OXYGEN_MAX_DEFICIT = 100.0  # micromol/kg below OXYGEN_HEALTHY_RANGE[0]
OXYGEN_MAX_EXCESS = 150.0  # micromol/kg above OXYGEN_HEALTHY_RANGE[1]

# Fixed weights (must sum to 100); documented in the module docstring above.
WEIGHT_TEMPERATURE = 30.0
WEIGHT_SALINITY = 30.0
WEIGHT_OXYGEN = 25.0
WEIGHT_COMPLETENESS = 15.0

REQUIRED_MEASUREMENT_FIELDS: tuple[str, ...] = (
    "temperature_c",
    "salinity_psu",
    "dissolved_oxygen",
)


class OceanHealthCalculator:
    """Computes a region/period Ocean Health Index per the fixed weighted formula."""

    def __init__(self, repository: "ProfileRepository") -> None:
        """Initialize the calculator.

        Args:
            repository: A ProfileRepository (Module 2) instance, or any
                duck-typed object exposing get_profiles_by_region() and
                get_measurements_for_profile().
        """
        self._repository = repository

    def compute(self, region: str, start: date, end: date) -> OceanHealthScore:
        """Compute the Ocean Health Index for a region and period.

        Args:
            region: Ocean region name; must match shared.regions.REGION_NAMES.
            start: Inclusive period start date.
            end: Inclusive period end date.

        Returns:
            An OceanHealthScore with contributing_factors populated and
            recommendation left as an empty string - RecommendationEngine
            (recommendations.py) fills that in afterward; the numeric score
            here never depends on it.

        Raises:
            HealthIndexError: If the region is invalid or profile/measurement
                data cannot be fetched from the repository.
        """
        if region not in REGION_NAMES:
            raise HealthIndexError(f"Unknown ocean region '{region}'")

        try:
            profiles = self._repository.get_profiles_by_region(region, start, end)
        except Exception as exc:
            logger.error(
                "Failed to fetch profiles for health index region=%s", region, exc_info=True,
            )
            raise HealthIndexError(f"Could not fetch profiles for {region}") from exc

        measurements: list[dict[str, Any]] = []
        for profile in profiles:
            profile_id = profile["id"] if isinstance(profile, dict) else profile.id
            try:
                measurements.extend(self._repository.get_measurements_for_profile(profile_id))
            except Exception as exc:
                logger.error(
                    "Failed to fetch measurements for profile_id=%s", profile_id, exc_info=True,
                )
                raise HealthIndexError(
                    f"Could not fetch measurements for profile {profile_id}"
                ) from exc

        if not measurements:
            logger.info(
                "No measurements available for %s between %s and %s; reporting zero score",
                region, start, end,
            )
            contributing_factors = {
                "temperature_score": 0.0,
                "salinity_score": 0.0,
                "oxygen_score": 0.0,
                "completeness_score": 0.0,
            }
            return OceanHealthScore(
                ocean_region=region,
                period_start=start,
                period_end=end,
                score=0.0,
                contributing_factors=contributing_factors,
                recommendation="",
            )

        temperature_score = self._score_against_baseline(
            self._mean_of_field(measurements, "temperature_c"),
            BASELINE_TEMPERATURE_C[region],
            TEMPERATURE_MAX_DEVIATION_C,
        )
        salinity_score = self._score_against_baseline(
            self._mean_of_field(measurements, "salinity_psu"),
            BASELINE_SALINITY_PSU[region],
            SALINITY_MAX_DEVIATION_PSU,
        )
        oxygen_score = self._score_oxygen(self._mean_of_field(measurements, "dissolved_oxygen"))
        completeness_score = self._score_completeness(measurements)

        contributing_factors = {
            "temperature_score": round(temperature_score, 2),
            "salinity_score": round(salinity_score, 2),
            "oxygen_score": round(oxygen_score, 2),
            "completeness_score": round(completeness_score, 2),
        }

        overall_score = (
            WEIGHT_TEMPERATURE * temperature_score
            + WEIGHT_SALINITY * salinity_score
            + WEIGHT_OXYGEN * oxygen_score
            + WEIGHT_COMPLETENESS * completeness_score
        ) / 100.0

        logger.info(
            "Computed health index for region=%s: score=%.2f factors=%s",
            region, overall_score, contributing_factors,
        )

        return OceanHealthScore(
            ocean_region=region,
            period_start=start,
            period_end=end,
            score=round(overall_score, 2),
            contributing_factors=contributing_factors,
            recommendation="",
        )

    @staticmethod
    def _mean_of_field(measurements: list[dict[str, Any]], field: str) -> float | None:
        """Return the mean of a numeric field across measurement rows, ignoring None."""
        values = [row[field] for row in measurements if row.get(field) is not None]
        if not values:
            return None
        return sum(values) / len(values)

    @staticmethod
    def _score_against_baseline(
        mean_value: float | None, baseline: float, max_deviation: float
    ) -> float:
        """Score 100 at zero deviation from baseline, decaying linearly to 0 at max_deviation."""
        if mean_value is None:
            return 0.0
        deviation = abs(mean_value - baseline)
        if deviation >= max_deviation:
            return 0.0
        return 100.0 * (1.0 - deviation / max_deviation)

    @staticmethod
    def _score_oxygen(mean_value: float | None) -> float:
        """Score 100 inside OXYGEN_HEALTHY_RANGE, decaying linearly outside it."""
        if mean_value is None:
            return 0.0
        lower_bound, upper_bound = OXYGEN_HEALTHY_RANGE
        if lower_bound <= mean_value <= upper_bound:
            return 100.0
        if mean_value < lower_bound:
            deficit = lower_bound - mean_value
            if deficit >= OXYGEN_MAX_DEFICIT:
                return 0.0
            return 100.0 * (1.0 - deficit / OXYGEN_MAX_DEFICIT)
        excess = mean_value - upper_bound
        if excess >= OXYGEN_MAX_EXCESS:
            return 0.0
        return 100.0 * (1.0 - excess / OXYGEN_MAX_EXCESS)

    @staticmethod
    def _score_completeness(measurements: list[dict[str, Any]]) -> float:
        """Score 100 if every required field is present on every measurement row."""
        total_expected = len(measurements) * len(REQUIRED_MEASUREMENT_FIELDS)
        if total_expected == 0:
            return 0.0
        present = sum(
            1
            for row in measurements
            for field in REQUIRED_MEASUREMENT_FIELDS
            if row.get(field) is not None
        )
        return 100.0 * present / total_expected


if __name__ == "__main__":
    # --- Self-test ---
    from datetime import date as _date

    class _FakeProfileRepository:
        """Fake ProfileRepository for demo/self-test purposes only."""

        def __init__(self, measurement_rows: list[dict]) -> None:
            self._measurement_rows = measurement_rows

        def get_profiles_by_region(self, region: str, start: _date, end: _date) -> list[dict]:
            return [{"id": 1}]

        def get_measurements_for_profile(self, profile_id: int) -> list[dict]:
            return self._measurement_rows

    calculator = OceanHealthCalculator(
        repository=_FakeProfileRepository(
            [
                # At-baseline, in-range, fully complete -> expect a near-100 score.
                {
                    "temperature_c": 27.0,
                    "salinity_psu": 36.0,
                    "dissolved_oxygen": 250.0,
                    "chlorophyll": 0.5,
                },
                {
                    "temperature_c": 27.0,
                    "salinity_psu": 36.0,
                    "dissolved_oxygen": 250.0,
                    "chlorophyll": 0.4,
                },
            ]
        )
    )
    healthy_score = calculator.compute("Arabian Sea", _date(2024, 1, 1), _date(2024, 1, 31))
    assert healthy_score.ocean_region == "Arabian Sea"
    assert healthy_score.contributing_factors["temperature_score"] == 100.0
    assert healthy_score.contributing_factors["salinity_score"] == 100.0
    assert healthy_score.contributing_factors["oxygen_score"] == 100.0
    assert healthy_score.contributing_factors["completeness_score"] == 100.0
    assert healthy_score.score == 100.0

    degraded_calculator = OceanHealthCalculator(
        repository=_FakeProfileRepository(
            [
                # 2.5C off baseline, 1.5 PSU off, oxygen 50 below healthy range,
                # oxygen field missing on one row -> partial completeness.
                {"temperature_c": 29.5, "salinity_psu": 37.5, "dissolved_oxygen": 150.0},
                {"temperature_c": 29.5, "salinity_psu": 37.5, "dissolved_oxygen": None},
            ]
        )
    )
    degraded_score = degraded_calculator.compute(
        "Arabian Sea", _date(2024, 1, 1), _date(2024, 1, 31)
    )
    assert degraded_score.contributing_factors["temperature_score"] == 50.0
    assert degraded_score.contributing_factors["salinity_score"] == 50.0
    assert degraded_score.contributing_factors["completeness_score"] == round(
        100.0 * 5 / 6, 2
    )

    empty_calculator = OceanHealthCalculator(repository=_FakeProfileRepository([]))
    empty_score = empty_calculator.compute("Bay of Bengal", _date(2024, 1, 1), _date(2024, 1, 31))
    assert empty_score.score == 0.0
    assert all(value == 0.0 for value in empty_score.contributing_factors.values())

    try:
        calculator.compute("Not A Real Region", _date(2024, 1, 1), _date(2024, 1, 31))
        raise AssertionError("Expected HealthIndexError for an invalid region")
    except HealthIndexError:
        pass

    logger.info(
        "Self-test scores: healthy=%.2f degraded=%.2f empty=%.2f",
        healthy_score.score, degraded_score.score, empty_score.score,
    )