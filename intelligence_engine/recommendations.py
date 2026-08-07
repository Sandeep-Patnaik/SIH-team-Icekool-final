"""Plain-language recommendation generation for OceanMind AI.

The OceanHealthScore's numeric value and contributing_factors are already
final by the time they reach this module - RecommendationEngine only chooses
*wording*. An optional LLM call may be used to phrase the recommendation more
naturally, but every code path has a templated fallback that requires no
network access, so a missing API key or a network failure never blocks
report generation.
"""

from __future__ import annotations

from shared.logger import get_logger
from shared.schemas import OceanHealthScore
from config import Config

logger = get_logger(__name__)

# Score bands used by the templated (non-LLM) fallback.
SCORE_BAND_EXCELLENT = 80.0
SCORE_BAND_GOOD = 60.0
SCORE_BAND_FAIR = 40.0

FACTOR_LABELS = {
    "temperature_score": "temperature stability",
    "salinity_score": "salinity stability",
    "oxygen_score": "dissolved oxygen levels",
    "completeness_score": "data coverage",
}


class RecommendationEngine:
    """Generates a plain-language recommendation string from an OceanHealthScore."""

    def __init__(self, use_llm: bool = True) -> None:
        """Initialize the recommendation engine.

        Args:
            use_llm: If True, attempt an LLM-assisted phrasing first and fall
                back to the template on any failure. If False, always use
                the template (useful for tests and offline demos).
        """
        self._use_llm = use_llm

    def generate(self, score: OceanHealthScore) -> str:
        """Generate a plain-language recommendation for a computed health score.

        Args:
            score: The OceanHealthScore to explain (numeric fields are read
                only, never modified).

        Returns:
            A human-readable recommendation string. This method never
            raises - any internal failure degrades to the template.
        """
        if self._use_llm and Config.LLM_API_KEY:
            try:
                return self._generate_with_llm(score)
            except Exception as exc:
                logger.error(
                    "LLM-assisted recommendation failed for region=%s; falling back to template",
                    score.ocean_region, exc_info=True,
                )
        return self._generate_template(score)

    def _generate_template(self, score: OceanHealthScore) -> str:
        """Build a deterministic, template-based recommendation (no network calls)."""
        band = self._band_label(score.score)
        weakest_factor = min(score.contributing_factors, key=score.contributing_factors.get)
        weakest_label = FACTOR_LABELS.get(weakest_factor, weakest_factor)
        weakest_value = score.contributing_factors[weakest_factor]

        return (
            f"{score.ocean_region} scored {score.score:.1f}/100 ({band}) for the period "
            f"{score.period_start} to {score.period_end}. The weakest contributing factor is "
            f"{weakest_label} at {weakest_value:.1f}/100. Prioritize additional monitoring and "
            f"mitigation focused on {weakest_label} in this region before the next assessment "
            f"cycle."
        )

    def _generate_with_llm(self, score: OceanHealthScore) -> str:
        """Attempt an LLM-assisted rephrasing of the templated recommendation.

        Uses a minimal, provider-agnostic client call gated behind
        Config.LLM_PROVIDER/LLM_API_KEY. Any exception here is caught by
        generate() and triggers the template fallback - this method is
        allowed to raise freely.
        """
        base_text = self._generate_template(score)
        prompt = (
            "Rewrite the following ocean-health recommendation for a decision-support "
            "dashboard. Keep every number exactly as given, keep it to 2-3 sentences, "
            "and do not invent new facts.\n\n" + base_text
        )

        if Config.LLM_PROVIDER == "openai":
            from openai import OpenAI  # local import: optional dependency, only needed here

            client = OpenAI(api_key=Config.LLM_API_KEY)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
            )
            text = response.choices[0].message.content
            if not text or not text.strip():
                raise ValueError("LLM returned an empty recommendation")
            logger.info("Generated LLM-assisted recommendation for region=%s", score.ocean_region)
            return text.strip()

        # Other providers (qwen / llama via MCP) are wired by Module 4's client
        # pattern; not duplicated here to avoid two competing LLM-call
        # implementations. Unsupported provider -> raise so generate() falls
        # back to the template.
        raise NotImplementedError(f"LLM provider '{Config.LLM_PROVIDER}' not wired in this module")

    @staticmethod
    def _band_label(score_value: float) -> str:
        """Map a numeric score to a qualitative band label."""
        if score_value >= SCORE_BAND_EXCELLENT:
            return "Excellent"
        if score_value >= SCORE_BAND_GOOD:
            return "Good"
        if score_value >= SCORE_BAND_FAIR:
            return "Fair"
        return "Poor"


if __name__ == "__main__":
    # --- Self-test ---
    from datetime import date as _date

    sample_score = OceanHealthScore(
        ocean_region="Arabian Sea",
        period_start=_date(2024, 1, 1),
        period_end=_date(2024, 1, 31),
        score=62.0,
        contributing_factors={
            "temperature_score": 80.0,
            "salinity_score": 75.0,
            "oxygen_score": 40.0,
            "completeness_score": 90.0,
        },
        recommendation="",
    )

    # use_llm=False forces the template path so this self-test needs no API key.
    engine = RecommendationEngine(use_llm=False)
    recommendation_text = engine.generate(sample_score)
    assert "Arabian Sea" in recommendation_text
    assert "dissolved oxygen" in recommendation_text
    logger.info("Self-test recommendation: %s", recommendation_text)