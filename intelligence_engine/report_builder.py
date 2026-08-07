"""PDF report generation for OceanMind AI.

file_path convention (Module 5 depends on this)
-------------------------------------------------
ReportBuilder writes PDFs under ``Config.REPORTS_DIR`` (``data/reports/``)
using the filename pattern::

    {region_slug}_{start:%Y%m%d}_{end:%Y%m%d}.pdf

e.g. ``arabian_sea_20240101_20240131.pdf``.

The value stored in ``reports.file_path`` (and returned by ``generate()``) is
the path **relative to the project root**, e.g.
``data/reports/arabian_sea_20240101_20240131.pdf`` - not an absolute
filesystem path. This is so the same DB row is valid whether opened from a
developer's laptop, a teammate's checkout, or the deployed demo box: Module 5
should join it onto its own known project root (or serve it from a static
files route rooted at the project root) rather than treating it as directly
openable. Internally this class resolves the relative path against
``config.BASE_DIR`` (via ``Config.REPORTS_DIR``) purely to know where to
write the file on disk.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from config import Config
from shared.logger import get_logger
from shared.regions import REGION_NAMES
from shared.schemas import OceanHealthScore
from intelligence_engine.exceptions import ReportBuildError
from intelligence_engine.health_index import OceanHealthCalculator
from intelligence_engine.recommendations import RecommendationEngine
from intelligence_engine.anomaly_detector import AnomalyDetector

if TYPE_CHECKING:
    from database.repository import ProfileRepository

logger = get_logger(__name__)


class ReportBuilder:
    """Assembles a per-region ocean health PDF report and persists its record.

    Orchestrates OceanHealthCalculator, RecommendationEngine, and
    AnomalyDetector, renders a chart of the contributing factors plus a
    narrative section, writes the PDF to disk, and inserts a row into the
    reports table via ProfileRepository.insert_report().
    """

    def __init__(
        self,
        repository: "ProfileRepository",
        recommendation_engine: RecommendationEngine | None = None,
    ) -> None:
        """Initialize the report builder.

        Args:
            repository: A ProfileRepository (Module 2) instance, or any
                duck-typed object exposing get_profiles_by_region(),
                get_measurements_for_profile(), and insert_report().
            recommendation_engine: Optional pre-configured RecommendationEngine;
                defaults to one with LLM-assisted phrasing enabled.
        """
        self._repository = repository
        self._health_calculator = OceanHealthCalculator(repository)
        self._anomaly_detector = AnomalyDetector(repository)
        self._recommendation_engine = recommendation_engine or RecommendationEngine()

    def generate(self, region: str, start: date, end: date) -> Path:
        """Generate a PDF report for a region/period and persist its record.

        Args:
            region: Ocean region name; must match shared.regions.REGION_NAMES.
            start: Inclusive period start date.
            end: Inclusive period end date.

        Returns:
            The path (relative to the project root) of the written PDF file,
            as also stored in reports.file_path.

        Raises:
            ReportBuildError: If the region is invalid, the score cannot be
                computed, the PDF cannot be written, or the DB insert fails.
        """
        if region not in REGION_NAMES:
            raise ReportBuildError(f"Unknown ocean region '{region}'")

        score = self._health_calculator.compute(region, start, end)
        score.recommendation = self._recommendation_engine.generate(score)

        try:
            anomaly_flags = self._anomaly_detector.detect(region, start, end)
        except Exception as exc:
            logger.error(
                "Anomaly detection failed for region=%s; continuing without flags",
                region, exc_info=True,
            )
            anomaly_flags = []

        try:
            Config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.error("Could not create reports directory %s", Config.REPORTS_DIR, exc_info=True)
            raise ReportBuildError("Could not create reports output directory") from exc

        filename = self._build_filename(region, start, end)
        absolute_path = Config.REPORTS_DIR / filename
        relative_path = Path("data") / "reports" / filename

        summary_text = self._build_summary_text(score, anomaly_flags)

        try:
            self._render_pdf(absolute_path, score, anomaly_flags, summary_text)
        except Exception as exc:
            logger.error("Failed to render PDF report at %s", absolute_path, exc_info=True)
            raise ReportBuildError(f"Could not render PDF report for {region}") from exc

        try:
            self._repository.insert_report(
                {
                    "generated_at": datetime.now(),
                    "ocean_region": region,
                    "period_start": start,
                    "period_end": end,
                    "file_path": str(relative_path),
                    "summary_text": summary_text,
                }
            )
        except Exception as exc:
            logger.error(
                "Failed to insert report row for region=%s file_path=%s",
                region, relative_path, exc_info=True,
            )
            raise ReportBuildError(f"Could not persist report record for {region}") from exc

        logger.info("Generated report for region=%s at %s", region, relative_path)
        return relative_path

    @staticmethod
    def _build_filename(region: str, start: date, end: date) -> str:
        """Build the report filename per the documented file_path convention."""
        region_slug = region.lower().replace(" ", "_")
        return f"{region_slug}_{start:%Y%m%d}_{end:%Y%m%d}.pdf"

    @staticmethod
    def _build_summary_text(score: OceanHealthScore, anomaly_flags: list[dict]) -> str:
        """Build the plain-text summary stored in reports.summary_text."""
        lines = [score.recommendation]
        if anomaly_flags:
            lines.append(f"{len(anomaly_flags)} anomaly flag(s) detected:")
            lines.extend(f"- {flag['message']}" for flag in anomaly_flags)
        else:
            lines.append("No significant anomalies detected for this period.")
        return "\n".join(lines)

    @staticmethod
    def _render_pdf(
        output_path: Path, score: OceanHealthScore, anomaly_flags: list[dict], summary_text: str
    ) -> None:
        """Render the chart + narrative PDF to output_path using matplotlib + reportlab."""
        import io
        import matplotlib
        matplotlib.use("Agg")  # headless backend, no display required
        import matplotlib.pyplot as plt
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.pdfgen import canvas as pdf_canvas
        from reportlab.lib.utils import ImageReader

        factors = score.contributing_factors
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(list(factors.keys()), list(factors.values()), color="#2b6cb0")
        ax.set_ylim(0, 100)
        ax.set_ylabel("Score (0-100)")
        ax.set_title(f"{score.ocean_region} - Contributing Factors")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        fig.tight_layout()

        chart_buffer = io.BytesIO()
        fig.savefig(chart_buffer, format="png", dpi=150)
        plt.close(fig)
        chart_buffer.seek(0)

        canvas = pdf_canvas.Canvas(str(output_path), pagesize=A4)
        page_width, page_height = A4
        margin = 2 * cm
        cursor_y = page_height - margin

        canvas.setFont("Helvetica-Bold", 16)
        canvas.drawString(margin, cursor_y, f"Ocean Health Report - {score.ocean_region}")
        cursor_y -= 0.8 * cm

        canvas.setFont("Helvetica", 11)
        canvas.drawString(
            margin, cursor_y, f"Period: {score.period_start} to {score.period_end}"
        )
        cursor_y -= 0.6 * cm
        canvas.drawString(margin, cursor_y, f"Overall score: {score.score:.1f} / 100")
        cursor_y -= 1.0 * cm

        chart_image = ImageReader(chart_buffer)
        chart_width = page_width - 2 * margin
        chart_height = chart_width * (3.5 / 6.0)
        canvas.drawImage(
            chart_image, margin, cursor_y - chart_height, width=chart_width, height=chart_height
        )
        cursor_y -= chart_height + 1.0 * cm

        canvas.setFont("Helvetica-Bold", 12)
        canvas.drawString(margin, cursor_y, "Summary & Recommendation")
        cursor_y -= 0.6 * cm

        canvas.setFont("Helvetica", 10)
        text_object = canvas.beginText(margin, cursor_y)
        text_object.setLeading(14)
        max_chars_per_line = 95
        for paragraph in summary_text.split("\n"):
            while len(paragraph) > max_chars_per_line:
                text_object.textLine(paragraph[:max_chars_per_line])
                paragraph = paragraph[max_chars_per_line:]
            text_object.textLine(paragraph)
        canvas.drawText(text_object)

        canvas.showPage()
        canvas.save()


if __name__ == "__main__":
    # --- Self-test ---
    from datetime import date as _date

    class _FakeProfileRepository:
        """Fake ProfileRepository for demo/self-test purposes only."""

        def __init__(self) -> None:
            self.inserted_reports: list[dict] = []

        def get_profiles_by_region(self, region: str, start: _date, end: _date) -> list[dict]:
            return [{"id": 1}, {"id": 2}]

        def get_measurements_for_profile(self, profile_id: int) -> list[dict]:
            return [
                {"temperature_c": 26.9, "salinity_psu": 36.0, "dissolved_oxygen": 190.0},
                {"temperature_c": 27.1, "salinity_psu": 35.8, "dissolved_oxygen": 185.0},
            ]

        def insert_report(self, report: dict) -> int:
            self.inserted_reports.append(report)
            return len(self.inserted_reports)

    fake_repository = _FakeProfileRepository()
    builder = ReportBuilder(
        repository=fake_repository,
        recommendation_engine=RecommendationEngine(use_llm=False),
    )
    output_relative_path = builder.generate("Arabian Sea", _date(2024, 1, 1), _date(2024, 1, 31))
    # Config.REPORTS_DIR is already the absolute equivalent of data/reports/,
    # so resolve against its parent's parent (the project root) here.
    absolute_output_path = Config.REPORTS_DIR / output_relative_path.name

    assert absolute_output_path.exists()
    assert absolute_output_path.stat().st_size > 0
    assert len(fake_repository.inserted_reports) == 1
    logger.info(
        "Self-test report written to %s (%d bytes)",
        absolute_output_path, absolute_output_path.stat().st_size,
    )