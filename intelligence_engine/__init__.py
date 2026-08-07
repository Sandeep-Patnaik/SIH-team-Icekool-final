"""Intelligence Engine module for OceanMind AI.

Owns the Ocean Health Index, AI-generated recommendations, anomaly
detection, and automated report generation:

- health_index.py       OceanHealthCalculator - computes OceanHealthScore
                         (shared.schemas) per region/period.
- recommendations.py    RecommendationEngine - plain-language phrasing of a
                         computed score; LLM-assisted with a templated
                         fallback, never affects the numeric score.
- anomaly_detector.py   AnomalyDetector - flags baseline deviations for
                         report narratives.
- report_builder.py     ReportBuilder - orchestrates the three above into a
                         PDF report and persists a reports row via
                         database.repository.ProfileRepository.
- exceptions.py         IntelligenceEngineError hierarchy shared by all of
                         the above.

This package intentionally exposes no re-exports at the top level; import
each class from its owning submodule, e.g.:

    from intelligence_engine.health_index import OceanHealthCalculator
    from intelligence_engine.recommendations import RecommendationEngine
    from intelligence_engine.anomaly_detector import AnomalyDetector
    from intelligence_engine.report_builder import ReportBuilder
    from intelligence_engine.exceptions import IntelligenceEngineError
"""