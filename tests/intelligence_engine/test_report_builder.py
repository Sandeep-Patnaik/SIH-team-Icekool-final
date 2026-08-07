"""Unit tests for intelligence_engine/report_builder.py.

ProfileRepository (Module 2) is mocked throughout. OceanHealthCalculator and
AnomalyDetector (this module's own siblings) are patched out at the
report_builder module level so these tests exercise only ReportBuilder's own
orchestration logic - filename/path construction, PDF rendering hand-off,
error wrapping, and the insert_report() call - rather than re-testing the
health-index formula or anomaly thresholds already covered by their own test
files. PDF rendering itself is faked (filesystem write only, no real
matplotlib/reportlab call) so this suite has no heavy-dependency requirement
and runs standalone, per the Master Prompt's testing standard.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from config import Config
from intelligence_engine.exceptions import ReportBuildError
from intelligence_engine.report_builder import ReportBuilder
from shared.schemas import OceanHealthScore

TEST_REGION = "Arabian Sea"
TEST_START = date(2024, 1, 1)
TEST_END = date(2024, 1, 31)

EXPECTED_FILENAME = "arabian_sea_20240101_20240131.pdf"
EXPECTED_RELATIVE_PATH = Path("data") / "reports" / EXPECTED_FILENAME


def _fake_score(recommendation: str = "") -> OceanHealthScore:
    """Build a fresh OceanHealthScore, matching what OceanHealthCalculator.compute() returns."""
    return OceanHealthScore(
        ocean_region=TEST_REGION,
        period_start=TEST_START,
        period_end=TEST_END,
        score=72.5,
        contributing_factors={
            "temperature_score": 80.0,
            "salinity_score": 75.0,
            "oxygen_score": 65.0,
            "completeness_score": 90.0,
        },
        recommendation=recommendation,
    )


@pytest.fixture
def reports_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point Config.REPORTS_DIR at an isolated temp directory for the test."""
    isolated_dir = tmp_path / "reports"
    monkeypatch.setattr(Config, "REPORTS_DIR", isolated_dir)
    return isolated_dir


@pytest.fixture
def render_pdf_recorder(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, ...]]:
    """Replace ReportBuilder._render_pdf with a fake that writes bytes and records its call.

    Avoids any dependency on matplotlib/reportlab actually being installed
    or producing a real PDF; the test only needs to confirm ReportBuilder
    hands the render step the right arguments and that a file lands on disk.
    """
    calls: list[tuple[Any, ...]] = []

    def _fake_render_pdf(output_path: Path, score: Any, anomaly_flags: list, summary_text: str) -> None:
        calls.append((output_path, score, anomaly_flags, summary_text))
        output_path.write_bytes(b"%PDF-FAKE-CONTENT")

    monkeypatch.setattr(ReportBuilder, "_render_pdf", staticmethod(_fake_render_pdf))
    return calls


@pytest.fixture
def patched_health_calculator(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch OceanHealthCalculator (as imported into report_builder.py) with a controllable mock."""
    instance = MagicMock()
    instance.compute.return_value = _fake_score()
    calculator_class = MagicMock(return_value=instance)
    monkeypatch.setattr("intelligence_engine.report_builder.OceanHealthCalculator", calculator_class)
    return instance


@pytest.fixture
def patched_anomaly_detector(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch AnomalyDetector (as imported into report_builder.py) with a controllable mock."""
    instance = MagicMock()
    instance.detect.return_value = []
    detector_class = MagicMock(return_value=instance)
    monkeypatch.setattr("intelligence_engine.report_builder.AnomalyDetector", detector_class)
    return instance


@pytest.fixture
def stub_recommendation_engine() -> MagicMock:
    """A recommendation engine stub whose generate() returns fixed, predictable text."""
    engine = MagicMock()
    engine.generate.return_value = "Prioritize monitoring in this region."
    return engine


@pytest.fixture
def mock_repository() -> MagicMock:
    """A mocked ProfileRepository exposing only insert_report (the surface ReportBuilder uses)."""
    repository = MagicMock()
    repository.insert_report.return_value = 1
    return repository


@pytest.fixture
def builder(
    mock_repository: MagicMock,
    stub_recommendation_engine: MagicMock,
    patched_health_calculator: MagicMock,
    patched_anomaly_detector: MagicMock,
    reports_dir: Path,
    render_pdf_recorder: list,
) -> ReportBuilder:
    """A fully-wired ReportBuilder with every collaborator mocked/faked/isolated."""
    return ReportBuilder(repository=mock_repository, recommendation_engine=stub_recommendation_engine)


class TestPdfGeneration:
    """Verifies the PDF render step is invoked correctly and produces a file on disk."""

    def test_pdf_file_is_written_to_disk(
        self, builder: ReportBuilder, reports_dir: Path
    ) -> None:
        """After generate(), the PDF must exist at Config.REPORTS_DIR / filename."""
        builder.generate(TEST_REGION, TEST_START, TEST_END)

        output_file = reports_dir / EXPECTED_FILENAME
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_render_pdf_called_with_score_flags_and_summary(
        self, builder: ReportBuilder, render_pdf_recorder: list
    ) -> None:
        """The render step must receive the scored OceanHealthScore, flags, and summary text."""
        builder.generate(TEST_REGION, TEST_START, TEST_END)

        assert len(render_pdf_recorder) == 1
        output_path, score, anomaly_flags, summary_text = render_pdf_recorder[0]
        assert output_path.name == EXPECTED_FILENAME
        assert score.ocean_region == TEST_REGION
        assert anomaly_flags == []
        assert "Prioritize monitoring in this region." in summary_text

    def test_recommendation_is_attached_to_score_before_rendering(
        self, builder: ReportBuilder, render_pdf_recorder: list, stub_recommendation_engine: MagicMock
    ) -> None:
        """RecommendationEngine.generate() output must be set on score.recommendation pre-render."""
        builder.generate(TEST_REGION, TEST_START, TEST_END)

        _, rendered_score, _, _ = render_pdf_recorder[0]
        assert rendered_score.recommendation == "Prioritize monitoring in this region."

    def test_pdf_render_failure_raises_report_build_error(
        self,
        mock_repository: MagicMock,
        stub_recommendation_engine: MagicMock,
        patched_health_calculator: MagicMock,
        patched_anomaly_detector: MagicMock,
        reports_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A rendering exception must be wrapped as ReportBuildError, not leaked."""
        def _broken_render(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("matplotlib exploded")

        monkeypatch.setattr(ReportBuilder, "_render_pdf", staticmethod(_broken_render))
        builder = ReportBuilder(repository=mock_repository, recommendation_engine=stub_recommendation_engine)

        with pytest.raises(ReportBuildError):
            builder.generate(TEST_REGION, TEST_START, TEST_END)

        mock_repository.insert_report.assert_not_called()


class TestReportInsertion:
    """Verifies the reports-table row is inserted with the correct fields."""

    def test_insert_report_called_with_expected_fields(
        self, builder: ReportBuilder, mock_repository: MagicMock
    ) -> None:
        """insert_report() must receive region, dates, the relative file_path, and summary_text."""
        builder.generate(TEST_REGION, TEST_START, TEST_END)

        mock_repository.insert_report.assert_called_once()
        (inserted_row,), _ = mock_repository.insert_report.call_args
        assert inserted_row["ocean_region"] == TEST_REGION
        assert inserted_row["period_start"] == TEST_START
        assert inserted_row["period_end"] == TEST_END
        assert inserted_row["file_path"] == str(EXPECTED_RELATIVE_PATH)
        assert "Prioritize monitoring in this region." in inserted_row["summary_text"]

    def test_insert_report_includes_anomaly_flags_in_summary(
        self,
        mock_repository: MagicMock,
        stub_recommendation_engine: MagicMock,
        patched_health_calculator: MagicMock,
        patched_anomaly_detector: MagicMock,
        reports_dir: Path,
        render_pdf_recorder: list,
    ) -> None:
        """When anomalies are detected, the summary text must enumerate them."""
        patched_anomaly_detector.detect.return_value = [
            {"parameter": "temperature", "severity": "severe", "message": "Temp way off baseline."}
        ]
        builder = ReportBuilder(repository=mock_repository, recommendation_engine=stub_recommendation_engine)

        builder.generate(TEST_REGION, TEST_START, TEST_END)

        (inserted_row,), _ = mock_repository.insert_report.call_args
        assert "1 anomaly flag(s) detected" in inserted_row["summary_text"]
        assert "Temp way off baseline." in inserted_row["summary_text"]

    def test_insert_report_failure_raises_report_build_error(
        self, builder: ReportBuilder, mock_repository: MagicMock
    ) -> None:
        """A repository-level exception on insert must be wrapped as ReportBuildError."""
        mock_repository.insert_report.side_effect = ConnectionError("db unreachable")

        with pytest.raises(ReportBuildError):
            builder.generate(TEST_REGION, TEST_START, TEST_END)


class TestGeneratedFilePath:
    """Verifies the documented file_path convention: relative to project root, fixed pattern."""

    def test_filename_follows_region_slug_and_date_pattern(self) -> None:
        """_build_filename must lowercase/underscore the region and use YYYYMMDD dates."""
        filename = ReportBuilder._build_filename(TEST_REGION, TEST_START, TEST_END)

        assert filename == EXPECTED_FILENAME

    def test_returned_path_is_relative_not_absolute(
        self, builder: ReportBuilder
    ) -> None:
        """generate() must return a path relative to the project root, per the documented convention."""
        result_path = builder.generate(TEST_REGION, TEST_START, TEST_END)

        assert not result_path.is_absolute()
        assert result_path == EXPECTED_RELATIVE_PATH

    def test_returned_path_matches_stored_file_path(
        self, builder: ReportBuilder, mock_repository: MagicMock
    ) -> None:
        """The path returned to the caller and the path persisted to the DB must be identical."""
        result_path = builder.generate(TEST_REGION, TEST_START, TEST_END)

        (inserted_row,), _ = mock_repository.insert_report.call_args
        assert inserted_row["file_path"] == str(result_path)


class TestFailureHandling:
    """Verifies error paths: invalid region, directory creation failure, anomaly-detector failure."""

    def test_invalid_region_raises_before_any_computation(
        self,
        mock_repository: MagicMock,
        stub_recommendation_engine: MagicMock,
        patched_health_calculator: MagicMock,
        patched_anomaly_detector: MagicMock,
        reports_dir: Path,
    ) -> None:
        """An unknown region must raise immediately, before touching the calculator or repository."""
        builder = ReportBuilder(repository=mock_repository, recommendation_engine=stub_recommendation_engine)

        with pytest.raises(ReportBuildError, match="Unknown ocean region"):
            builder.generate("Atlantis", TEST_START, TEST_END)

        patched_health_calculator.compute.assert_not_called()
        mock_repository.insert_report.assert_not_called()

    def test_reports_directory_creation_failure_raises_report_build_error(
        self,
        builder: ReportBuilder,
        mock_repository: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An OSError creating Config.REPORTS_DIR must be wrapped as ReportBuildError."""
        def _raising_mkdir(self: Path, *args: Any, **kwargs: Any) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(Path, "mkdir", _raising_mkdir)

        with pytest.raises(ReportBuildError, match="reports output directory"):
            builder.generate(TEST_REGION, TEST_START, TEST_END)

        mock_repository.insert_report.assert_not_called()

    def test_anomaly_detection_failure_does_not_block_report_generation(
        self,
        mock_repository: MagicMock,
        stub_recommendation_engine: MagicMock,
        patched_health_calculator: MagicMock,
        patched_anomaly_detector: MagicMock,
        reports_dir: Path,
        render_pdf_recorder: list,
    ) -> None:
        """An AnomalyDetector failure must degrade to zero flags, not abort the whole report."""
        patched_anomaly_detector.detect.side_effect = RuntimeError("anomaly service down")
        builder = ReportBuilder(repository=mock_repository, recommendation_engine=stub_recommendation_engine)

        result_path = builder.generate(TEST_REGION, TEST_START, TEST_END)

        assert result_path == EXPECTED_RELATIVE_PATH
        mock_repository.insert_report.assert_called_once()
        (inserted_row,), _ = mock_repository.insert_report.call_args
        assert "No significant anomalies detected" in inserted_row["summary_text"]

    def test_health_score_computation_failure_propagates(
        self,
        mock_repository: MagicMock,
        stub_recommendation_engine: MagicMock,
        patched_health_calculator: MagicMock,
        patched_anomaly_detector: MagicMock,
        reports_dir: Path,
    ) -> None:
        """A health-index computation failure is not caught by generate() - it propagates as-is.

        This documents current behavior: unlike the anomaly-detector call, the
        health_calculator.compute() call in report_builder.py is not wrapped in
        its own try/except, so whatever exception it raises reaches the caller
        unmodified rather than being converted to ReportBuildError.
        """
        patched_health_calculator.compute.side_effect = ValueError("no profiles for region")
        builder = ReportBuilder(repository=mock_repository, recommendation_engine=stub_recommendation_engine)

        with pytest.raises(ValueError, match="no profiles for region"):
            builder.generate(TEST_REGION, TEST_START, TEST_END)

        mock_repository.insert_report.assert_not_called()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))