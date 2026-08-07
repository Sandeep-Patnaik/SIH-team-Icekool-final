"""Unit tests for intelligence_engine/recommendations.py.

No real network or LLM provider is ever contacted here: the `openai` import
inside RecommendationEngine._generate_with_llm is a local import, so it is
faked out via a stub module injected into sys.modules for the duration of
each test that needs it. Config.LLM_API_KEY / Config.LLM_PROVIDER are
monkeypatched rather than read from the environment, so this suite runs
standalone with no .env and no other module's code, per the Master Prompt's
testing standard.
"""

from __future__ import annotations

import sys
import types
from datetime import date
from unittest.mock import MagicMock

import pytest

from config import Config
from intelligence_engine.recommendations import (
    FACTOR_LABELS,
    SCORE_BAND_EXCELLENT,
    SCORE_BAND_FAIR,
    SCORE_BAND_GOOD,
    RecommendationEngine,
)
from shared.schemas import OceanHealthScore

TEST_REGION = "Arabian Sea"
TEST_START = date(2024, 1, 1)
TEST_END = date(2024, 1, 31)


def _build_score(overall_score: float, weakest_factor: str = "oxygen_score") -> OceanHealthScore:
    """Build an OceanHealthScore with all factors healthy except one weakest factor.

    Args:
        overall_score: The top-level score value to embed.
        weakest_factor: Which contributing_factors key should hold the
            minimum value, so tests can assert it gets called out by name.

    Returns:
        A fully-populated OceanHealthScore with an empty recommendation,
        matching what health_index.py hands to RecommendationEngine.
    """
    contributing_factors = {
        "temperature_score": 90.0,
        "salinity_score": 85.0,
        "oxygen_score": 95.0,
        "completeness_score": 92.0,
    }
    contributing_factors[weakest_factor] = 20.0
    return OceanHealthScore(
        ocean_region=TEST_REGION,
        period_start=TEST_START,
        period_end=TEST_END,
        score=overall_score,
        contributing_factors=contributing_factors,
        recommendation="",
    )


@pytest.fixture
def stub_openai_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Inject a fake `openai` module into sys.modules so no real package is needed.

    The fake module's OpenAI class is a MagicMock whose instances respond to
    `.chat.completions.create(...)` with a configurable return value, letting
    individual tests control the "LLM response" without any network access.
    """
    fake_module = types.ModuleType("openai")
    fake_module.OpenAI = MagicMock()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    return fake_module


def _set_llm_response_text(stub_openai_module: types.ModuleType, text: str | None) -> None:
    """Configure the stubbed OpenAI client to return a fixed response body."""
    fake_response = MagicMock()
    fake_message = MagicMock()
    fake_message.content = text
    fake_response.choices = [MagicMock(message=fake_message)]
    client_instance = stub_openai_module.OpenAI.return_value  # type: ignore[attr-defined]
    client_instance.chat.completions.create.return_value = fake_response


def _set_llm_raises(stub_openai_module: types.ModuleType, exc: Exception) -> None:
    """Configure the stubbed OpenAI client to raise on chat.completions.create()."""
    client_instance = stub_openai_module.OpenAI.return_value  # type: ignore[attr-defined]
    client_instance.chat.completions.create.side_effect = exc


class TestTemplateGeneration:
    """Verifies the deterministic, no-network template path."""

    def test_template_used_when_use_llm_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """use_llm=False must never attempt an LLM call, even with an API key configured."""
        monkeypatch.setattr(Config, "LLM_API_KEY", "irrelevant-key")
        engine = RecommendationEngine(use_llm=False)
        score = _build_score(overall_score=72.0)

        result = engine.generate(score)

        assert TEST_REGION in result
        assert "72.0" in result

    def test_template_used_when_no_api_key_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """use_llm=True but no API key must fall straight to the template, skipping the LLM path."""
        monkeypatch.setattr(Config, "LLM_API_KEY", "")
        engine = RecommendationEngine(use_llm=True)
        score = _build_score(overall_score=55.0)

        result = engine.generate(score)

        assert TEST_REGION in result
        assert "55.0" in result

    def test_template_names_the_weakest_contributing_factor(self) -> None:
        """The template must call out the factor with the lowest numeric value, by its label."""
        engine = RecommendationEngine(use_llm=False)
        score = _build_score(overall_score=60.0, weakest_factor="salinity_score")

        result = engine.generate(score)

        assert FACTOR_LABELS["salinity_score"] in result
        # Sanity check: the other factors' labels should not be the ones called
        # out as the weakest (they're all higher-valued in _build_score).
        assert FACTOR_LABELS["oxygen_score"] not in result.split("weakest contributing factor is")[1].split(".")[0]

    def test_template_includes_period_dates(self) -> None:
        """The template must surface the period start/end dates for context."""
        engine = RecommendationEngine(use_llm=False)
        score = _build_score(overall_score=50.0)

        result = engine.generate(score)

        assert str(TEST_START) in result
        assert str(TEST_END) in result


class TestLlmFallbackBehavior:
    """Verifies LLM-assisted phrasing is used when available, and never blocks on failure."""

    def test_llm_response_used_when_call_succeeds(
        self, monkeypatch: pytest.MonkeyPatch, stub_openai_module: types.ModuleType
    ) -> None:
        """A successful, non-empty LLM response must be returned verbatim (stripped)."""
        monkeypatch.setattr(Config, "LLM_API_KEY", "test-key")
        monkeypatch.setattr(Config, "LLM_PROVIDER", "openai")
        _set_llm_response_text(stub_openai_module, "  Arabian Sea conditions look concerning.  ")
        engine = RecommendationEngine(use_llm=True)
        score = _build_score(overall_score=45.0)

        result = engine.generate(score)

        assert result == "Arabian Sea conditions look concerning."

    def test_falls_back_to_template_when_llm_raises(
        self, monkeypatch: pytest.MonkeyPatch, stub_openai_module: types.ModuleType
    ) -> None:
        """generate() must never raise: an LLM-side exception degrades to the template."""
        monkeypatch.setattr(Config, "LLM_API_KEY", "test-key")
        monkeypatch.setattr(Config, "LLM_PROVIDER", "openai")
        _set_llm_raises(stub_openai_module, TimeoutError("upstream timed out"))
        engine = RecommendationEngine(use_llm=True)
        score = _build_score(overall_score=45.0)

        result = engine.generate(score)

        # Falls back to the deterministic template, which always names the region.
        assert TEST_REGION in result
        assert "45.0" in result

    def test_falls_back_to_template_when_llm_returns_empty_text(
        self, monkeypatch: pytest.MonkeyPatch, stub_openai_module: types.ModuleType
    ) -> None:
        """An empty/whitespace-only LLM response must also degrade to the template, not crash."""
        monkeypatch.setattr(Config, "LLM_API_KEY", "test-key")
        monkeypatch.setattr(Config, "LLM_PROVIDER", "openai")
        _set_llm_response_text(stub_openai_module, "   ")
        engine = RecommendationEngine(use_llm=True)
        score = _build_score(overall_score=63.0)

        result = engine.generate(score)

        assert TEST_REGION in result
        assert "63.0" in result

    def test_falls_back_to_template_for_unsupported_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A non-openai LLM_PROVIDER (not wired in this module) must degrade gracefully."""
        monkeypatch.setattr(Config, "LLM_API_KEY", "test-key")
        monkeypatch.setattr(Config, "LLM_PROVIDER", "qwen")
        engine = RecommendationEngine(use_llm=True)
        score = _build_score(overall_score=38.0)

        result = engine.generate(score)

        assert TEST_REGION in result
        assert "38.0" in result

    def test_llm_never_called_when_use_llm_false(self, stub_openai_module: types.ModuleType) -> None:
        """use_llm=False must skip the LLM branch entirely, even if a client were available."""
        engine = RecommendationEngine(use_llm=False)
        score = _build_score(overall_score=70.0)

        engine.generate(score)

        stub_openai_module.OpenAI.assert_not_called()  # type: ignore[attr-defined]


class TestScoreBands:
    """Verifies the qualitative band label thresholds: Excellent / Good / Fair / Poor."""

    @pytest.mark.parametrize(
        "score_value, expected_band",
        [
            (100.0, "Excellent"),
            (SCORE_BAND_EXCELLENT, "Excellent"),
            (SCORE_BAND_EXCELLENT - 0.1, "Good"),
            (SCORE_BAND_GOOD, "Good"),
            (SCORE_BAND_GOOD - 0.1, "Fair"),
            (SCORE_BAND_FAIR, "Fair"),
            (SCORE_BAND_FAIR - 0.1, "Poor"),
            (0.0, "Poor"),
        ],
    )
    def test_band_label_thresholds(self, score_value: float, expected_band: str) -> None:
        """Each boundary value must map to the documented band, inclusive on the lower edge."""
        assert RecommendationEngine._band_label(score_value) == expected_band

    @pytest.mark.parametrize(
        "score_value, expected_band",
        [(85.0, "Excellent"), (65.0, "Good"), (45.0, "Fair"), (10.0, "Poor")],
    )
    def test_generated_text_contains_band_label(self, score_value: float, expected_band: str) -> None:
        """The band word actually produced by the templated recommendation must match."""
        engine = RecommendationEngine(use_llm=False)
        score = _build_score(overall_score=score_value)

        result = engine.generate(score)

        assert f"({expected_band})" in result


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))