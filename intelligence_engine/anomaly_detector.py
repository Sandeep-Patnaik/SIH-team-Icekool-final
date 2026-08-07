"""Anomaly detection for OceanMind AI.

Flags ocean regions/periods where the mean of a monitored parameter deviates
from a fixed historical baseline by more than a documented threshold. This is
intentionally simple and defensible for a demo: it reuses the same baseline
reference values as health_index.py (temperature, salinity) plus dissolved
oxygen and chlorophyll, rather than introducing a second, undocumented notion
of "normal".

Output is a list of plain dicts (not a shared/schemas.py contract) because
anomaly flags are consumed only within this module (report_builder.py) to
build report narratives; they are not a cross-module hand-off contract.
Each flag dict has keys: parameter, mean_value, baseline, deviation,
severity ("moderate" | "severe"), message.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

from shared.logger import get_logger
from shared.regions import REGION_NAMES
from intelligence_engine.exceptions import AnomalyDetectionError
from intelligence_engine.health_index import (
    BASELINE_TEMPERATURE_C,
    BASELINE_SALINITY_PSU,
    OXYGEN_HEALTHY_RANGE,
)

if TYPE_CHECKING:
    from database.repository import ProfileRepository

logger = get_logger(__name__)

# Deviation thresholds: moderate flag past this, severe flag past 2x this.
TEMPERATURE_MODERATE_THRESHOLD_C = 1.0
SALINITY_MODERATE_THRESHOLD_PSU = 0.75
OXYGEN_MODERATE_THRESHOLD = 20.0  # micromol/kg below the healthy range's lower bound

SEVERE_MULTIPLIER = 2.0


class AnomalyDetector:
    """Flags regions/periods with unusual temperature/salinity/oxygen deviation."""

    def __init__(self, repository: "ProfileRepository") -> None:
        """Initialize the detector.

        Args:
            repository: A ProfileRepository (Module 2) instance, or any
                duck-typed object exposing get_profiles_by_region() and
                get_measurements_for_profile().
        """
        self._repository = repository

    def detect(self, region: str, start: date, end: date) -> list[dict[str, Any]]:
        """Detect anomalies for a region and period against fixed baselines.

        Args:
            region: Ocean region name; must match shared.regions.REGION_NAMES.
            start: Inclusive period start date.
            end: Inclusive period end date.

        Returns:
            A list of flag dicts (possibly empty if nothing deviates beyond
            the moderate threshold).

        Raises:
            AnomalyDetectionError: If the region is invalid or data cannot be
                fetched.
        """
        if region not in REGION_NAMES:
            raise AnomalyDetectionError(f"Unknown ocean region '{region}'")

        try:
            profiles = self._repository.get_profiles_by_region(region, start, end)
        except Exception as exc:
            logger.error(
                "Failed to fetch profiles for anomaly detection region=%s", region, exc_info=True,
            )
            raise AnomalyDetectionError(f"Could not fetch profiles for {region}") from exc

        measurements: list[dict[str, Any]] = []
        for profile in profiles:
            profile_id = profile["id"] if isinstance(profile, dict) else profile.id
            try:
                measurements.extend(self._repository.get_measurements_for_profile(profile_id))
            except Exception as exc:
                logger.error(
                    "Failed to fetch measurements for profile_id=%s", profile_id, exc_info=True,
                )
                raise AnomalyDetectionError(
                    f"Could not fetch measurements for profile {profile_id}"
                ) from exc

        if not measurements:
            logger.info("No measurements available for %s; reporting no anomalies", region)
            return []

        flags: list[dict[str, Any]] = []
        flags.extend(
            self._check_baseline_deviation(
                measurements, "temperature_c", BASELINE_TEMPERATURE_C[region],
                TEMPERATURE_MODERATE_THRESHOLD_C, "temperature",
            )
        )
        flags.extend(
            self._check_baseline_deviation(
                measurements, "salinity_psu", BASELINE_SALINITY_PSU[region],
                SALINITY_MODERATE_THRESHOLD_PSU, "salinity",
            )
        )
        flags.extend(self._check_oxygen_deficit(measurements))

        logger.info("Detected %d anomaly flag(s) for region=%s", len(flags), region)
        return flags

    @staticmethod
    def _mean_of_field(measurements: list[dict[str, Any]], field: str) -> float | None:
        """Return the mean of a numeric field across measurement rows, ignoring None."""
        values = [row[field] for row in measurements if row.get(field) is not None]
        if not values:
            return None
        return sum(values) / len(values)

    @classmethod
    def _check_baseline_deviation(
        cls,
        measurements: list[dict[str, Any]],
        field: str,
        baseline: float,
        moderate_threshold: float,
        label: str,
    ) -> list[dict[str, Any]]:
        """Flag if mean(field) deviates from baseline beyond moderate_threshold."""
        mean_value = cls._mean_of_field(measurements, field)
        if mean_value is None:
            return []
        deviation = mean_value - baseline
        abs_deviation = abs(deviation)
        if abs_deviation < moderate_threshold:
            return []
        severity = "severe" if abs_deviation >= moderate_threshold * SEVERE_MULTIPLIER else "moderate"
        direction = "above" if deviation > 0 else "below"
        return [
            {
                "parameter": label,
                "mean_value": round(mean_value, 2),
                "baseline": baseline,
                "deviation": round(deviation, 2),
                "severity": severity,
                "message": (
                    f"Mean {label} ({mean_value:.2f}) is {abs_deviation:.2f} {direction} "
                    f"the historical baseline ({baseline:.2f}) - flagged as {severity}."
                ),
            }
        ]

    @classmethod
    def _check_oxygen_deficit(cls, measurements: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Flag if mean dissolved oxygen falls below the healthy range's lower bound."""
        mean_value = cls._mean_of_field(measurements, "dissolved_oxygen")
        if mean_value is None:
            return []
        lower_bound = OXYGEN_HEALTHY_RANGE[0]
        deficit = lower_bound - mean_value
        if deficit < OXYGEN_MODERATE_THRESHOLD:
            return []
        severity = "severe" if deficit >= OXYGEN_MODERATE_THRESHOLD * SEVERE_MULTIPLIER else "moderate"
        return [
            {
                "parameter": "dissolved_oxygen",
                "mean_value": round(mean_value, 2),
                "baseline": lower_bound,
                "deviation": round(-deficit, 2),
                "severity": severity,
                "message": (
                    f"Mean dissolved oxygen ({mean_value:.2f}) is {deficit:.2f} below the "
                    f"healthy range lower bound ({lower_bound:.2f}) - flagged as {severity}."
                ),
            }
        ]


if __name__ == "__main__":
    # --- Self-test ---
    from datetime import date as _date

    class _FakeProfileRepository:
        """Fake ProfileRepository for demo/self-test purposes only."""

        def get_profiles_by_region(self, region: str, start: _date, end: _date) -> list[dict]:
            return [{"id": 1}]

        def get_measurements_for_profile(self, profile_id: int) -> list[dict]:
            # Deliberately anomalous: temperature far above baseline, low oxygen.
            return [
                {"temperature_c": 29.5, "salinity_psu": 36.0, "dissolved_oxygen": 100.0},
                {"temperature_c": 29.8, "salinity_psu": 36.1, "dissolved_oxygen": 95.0},
            ]

    detector = AnomalyDetector(repository=_FakeProfileRepository())
    detected_flags = detector.detect("Arabian Sea", _date(2024, 1, 1), _date(2024, 1, 31))
    assert len(detected_flags) >= 2
    logger.info("Self-test anomaly flags: %s", detected_flags)