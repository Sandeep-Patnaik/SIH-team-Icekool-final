"""Custom exceptions for the Intelligence Engine module (OceanMind AI).

Every public entry point in intelligence_engine/ wraps its DB/file/network
calls in try/except and re-raises one of these instead of letting a raw
exception (or a bare except: pass) escape, per the project's non-negotiable
error-handling standard. All exceptions here derive from
IntelligenceEngineError so calling code (e.g. dashboard/) can catch the
whole module's failures with a single except clause when it doesn't need to
distinguish the source.
"""

from __future__ import annotations


class IntelligenceEngineError(Exception):
    """Base class for all errors raised by the intelligence_engine module."""


class HealthIndexError(IntelligenceEngineError):
    """Raised by health_index.py when a health score cannot be computed.

    Covers an unknown/invalid ocean region and failures fetching profile or
    measurement data from the repository.
    """


class AnomalyDetectionError(IntelligenceEngineError):
    """Raised by anomaly_detector.py when anomaly detection cannot complete.

    Covers an unknown/invalid ocean region and failures fetching profile or
    measurement data from the repository.
    """


class ReportBuildError(IntelligenceEngineError):
    """Raised by report_builder.py when a report cannot be generated.

    Covers an unknown/invalid ocean region, PDF rendering failures, output
    directory creation failures, and failures persisting the report row via
    ProfileRepository.insert_report().
    """


class RecommendationError(IntelligenceEngineError):
    """Reserved for recommendations.py.

    Not currently raised: RecommendationEngine.generate() is documented to
    never raise, degrading to a templated fallback on any internal failure
    instead. Defined here for hierarchy completeness and in case a future
    caller needs to distinguish a recommendation-specific failure mode.
    """


if __name__ == "__main__":
    # --- Self-test ---
    import logging

    logging.basicConfig(level=logging.INFO)
    test_logger = logging.getLogger(__name__)

    # Every subclass must be catchable as IntelligenceEngineError.
    for exception_class in (
        HealthIndexError,
        AnomalyDetectionError,
        ReportBuildError,
        RecommendationError,
    ):
        try:
            raise exception_class("self-test failure")
        except IntelligenceEngineError as exc:
            assert isinstance(exc, exception_class)
            test_logger.info("OK: %s caught as IntelligenceEngineError (%s)", exception_class.__name__, exc)
        else:
            raise AssertionError(f"{exception_class.__name__} did not raise")

    test_logger.info("Self-test passed: all intelligence_engine exceptions defined and catchable.")