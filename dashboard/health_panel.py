"""Ocean health scoring, recommendations and report generation panels.

Renders the composite Ocean Health Index as a gauge, the KPI summary, the
contributing sub-scores, the intelligence engine's recommendations, and the
generated reports list with download controls.

Backend contracts
-----------------
* ``OceanHealthCalculator.compute(...)`` -- composite score and factors
* ``ReportGenerator`` -- existing report generation implementation

Both return types are owned by the backend, so :func:`normalise_health` adapts
whatever shape arrives into the :class:`HealthView` these panels render.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Final, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.export_utils import render_binary_download, render_export_bar
from dashboard.profile_plots import contributing_factors, health_gauge
from dashboard.styles import OCEAN, pill, render_kpi_row, section_header
from dashboard.utils import (
    SESSION_REPORTS,
    backend_regions_for,
    format_datetime,
    format_number,
    score_to_label,
    score_to_tone,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Backend binding
# --------------------------------------------------------------------------- #

try:
    from database.repository import ProfileRepository  # type: ignore[import-not-found]
    from intelligence_engine.health_index import (  # type: ignore[import-not-found]
        OceanHealthCalculator as _BackendHealthCalculator,
    )
    from intelligence_engine.recommendations import RecommendationEngine  # type: ignore[import-not-found]

    HEALTH_BACKEND_AVAILABLE: Final[bool] = True

    class OceanHealthCalculator:  # type: ignore[no-redef]
        """Adapts the backend's region/period ``OceanHealthCalculator`` to the
        dashboard's ``compute(profiles, region, start_date, end_date)`` call.

        The real calculator fetches its own data from ``ProfileRepository``
        given a single exact backend region name and date range -- it does
        not take a ``profiles`` DataFrame (that argument is accepted here
        only to preserve the dashboard's existing call signature and is
        otherwise unused). When the dashboard's region key maps to more than
        one backend region (e.g. "Indian Ocean", "Global"), the per-region
        scores are averaged and their contributing factors are combined.
        """

        def __init__(self) -> None:
            self._repository = ProfileRepository()
            self._calculator = _BackendHealthCalculator(self._repository)

        def compute(
            self,
            profiles: Optional[pd.DataFrame] = None,
            region: Optional[str] = None,
            start_date: Optional[date] = None,
            end_date: Optional[date] = None,
        ) -> Dict[str, Any]:
            start = start_date or (date.today() - timedelta(days=365))
            end = end_date or date.today()

            scores = []
            for backend_region in backend_regions_for(region):
                scores.append(self._calculator.compute(backend_region, start, end))

            if not scores:
                return {
                    "score": 0.0,
                    "factors": {},
                    "recommendations": [],
                    "region": region,
                    "computed_at": datetime.now(timezone.utc),
                }

            overall_score = sum(s.score for s in scores) / len(scores)

            factor_totals: Dict[str, List[float]] = {}
            for s in scores:
                for name, value in s.contributing_factors.items():
                    factor_totals.setdefault(name, []).append(value)
            factors = {name: round(sum(values) / len(values), 2) for name, values in factor_totals.items()}

            recommendation_engine = RecommendationEngine()
            recommendations = []
            for s in scores:
                if not s.recommendation:
                    s.recommendation = recommendation_engine.generate(s)
                recommendations.append(
                    {
                        "title": f"{s.ocean_region} recommendation",
                        "detail": s.recommendation,
                        "priority": "High" if s.score < 50 else ("Medium" if s.score < 75 else "Low"),
                    }
                )

            return {
                "score": round(overall_score, 1),
                "factors": factors,
                "recommendations": recommendations,
                "region": region,
                "computed_at": datetime.now(timezone.utc),
            }

except Exception:  # noqa: BLE001 - covers ImportError *and* a misconfigured
    # backend (e.g. DATABASE_URL not set, which raises KeyError at import
    # time from config.py) so the dashboard degrades to demo mode instead
    # of crashing.
    HEALTH_BACKEND_AVAILABLE = False

    class OceanHealthCalculator:  # type: ignore[no-redef]
        """Interface-compatible stub for the backend's health calculator.

        Mirrors ``OceanHealthCalculator.compute(...)`` so that restoring the
        import is the only change required. Scores are derived from simple
        descriptive statistics of the supplied frame -- this is **not** a
        reimplementation of the backend's scientific model.
        """

        def compute(
            self,
            profiles: Optional[pd.DataFrame] = None,
            region: Optional[str] = None,
            start_date: Optional[date] = None,
            end_date: Optional[date] = None,
        ) -> Dict[str, Any]:
            """Return an illustrative health assessment.

            Args:
                profiles: Observations to assess.
                region: Region the assessment covers.
                start_date: Inclusive lower date bound.
                end_date: Inclusive upper date bound.

            Returns:
                A mapping with ``score``, ``factors`` and ``recommendations``.
            """
            factors: Dict[str, float] = {
                "thermal_stability": 72.0,
                "salinity_balance": 68.0,
                "oxygen_availability": 54.0,
                "biological_productivity": 63.0,
                "acidification": 47.0,
                "data_coverage": 81.0,
            }

            if isinstance(profiles, pd.DataFrame) and not profiles.empty:
                if "temperature" in profiles.columns:
                    warmth = float(profiles["temperature"].mean())
                    factors["thermal_stability"] = float(np.clip(100.0 - (warmth - 18.0) * 4.0, 5.0, 98.0))
                if "oxygen" in profiles.columns:
                    oxygen = float(profiles["oxygen"].mean())
                    factors["oxygen_availability"] = float(np.clip(oxygen / 2.4, 5.0, 98.0))
                if "ph" in profiles.columns:
                    acidity = float(profiles["ph"].mean())
                    factors["acidification"] = float(np.clip((acidity - 7.6) * 160.0, 5.0, 98.0))
                if "float_id" in profiles.columns:
                    coverage = profiles["float_id"].nunique()
                    factors["data_coverage"] = float(np.clip(45.0 + coverage * 2.4, 5.0, 99.0))

            score = float(np.mean(list(factors.values())))
            return {
                "score": round(score, 1),
                "factors": {name: round(value, 1) for name, value in factors.items()},
                "recommendations": _demo_recommendations(factors),
                "region": region,
                "computed_at": datetime.now(timezone.utc),
            }


try:
    from database.repository import ProfileRepository as _ReportProfileRepository  # type: ignore[import-not-found]
    from intelligence_engine.report_builder import ReportBuilder  # type: ignore[import-not-found]
    from config import Config as _ReportConfig  # type: ignore[import-not-found]

    REPORT_BACKEND_AVAILABLE: Final[bool] = True


    class ReportGenerator:  # type: ignore[no-redef]
        """Adapts the backend's region/period ``ReportBuilder`` to the
        dashboard's ``generate(profiles, title, region) -> bytes`` call.

        The real builder writes a PDF to disk (rooted at ``Config.REPORTS_DIR``)
        and returns its path *relative to the project root* -- it doesn't take
        a title or return bytes directly, and it needs a single exact backend
        region name plus a date range rather than a profiles DataFrame. This
        wrapper resolves those: the date range comes from ``profiles["time"]``
        (falling back to a 1-year window), the region is the first backend
        region matched by ``backend_regions_for()``, and the returned bytes are
        read back from the PDF the builder wrote.
        """

        def __init__(self) -> None:
            self._repository = _ReportProfileRepository()
            self._builder = ReportBuilder(self._repository)

        def generate(
            self,
            profiles: Optional[pd.DataFrame] = None,
            title: str = "Ocean Intelligence Report",
            region: Optional[str] = None,
        ) -> bytes:
            backend_region = backend_regions_for(region)[0]

            start: date
            end: date
            if isinstance(profiles, pd.DataFrame) and not profiles.empty and "time" in profiles.columns:
                stamps = pd.to_datetime(profiles["time"], errors="coerce").dropna()
                if not stamps.empty:
                    start = stamps.min().date()
                    end = stamps.max().date()
                else:
                    start, end = date.today() - timedelta(days=365), date.today()
            else:
                start, end = date.today() - timedelta(days=365), date.today()

            relative_path = self._builder.generate(backend_region, start, end)
            # ReportBuilder.generate() returns a path string ("data/reports/...")
            # that does not actually match where it writes the file (it writes
            # under Config.REPORTS_DIR, i.e. "<project_root>/reports/..." --
            # see the "reports" vs "data/reports" mismatch in report_builder.py).
            # report_builder.py's own __main__ self-test works around this the
            # same way: by re-joining just the filename onto Config.REPORTS_DIR
            # rather than trusting the returned path's directory part.
            absolute_path = _ReportConfig.REPORTS_DIR / relative_path.name
            try:
                return absolute_path.read_bytes()
            except OSError as exc:
                logger.error("Could not read generated report at %s", absolute_path, exc_info=True)
                raise RuntimeError(f"Report was generated but could not be read back from {absolute_path}") from exc

except Exception:  # noqa: BLE001 - covers ImportError *and* a misconfigured
    # backend (e.g. DATABASE_URL not set, which raises KeyError at import
    # time from config.py) so the dashboard degrades to demo mode instead
    # of crashing.
    REPORT_BACKEND_AVAILABLE = False

    class ReportGenerator:  # type: ignore[no-redef]
        """Interface-compatible stub for the backend's report generator.

        Produces a plain-text summary so the Reports tab is demonstrable. The
        real generator's output format is confirmed in Integration Notes.
        """

        def generate(
            self,
            profiles: Optional[pd.DataFrame] = None,
            title: str = "Ocean Intelligence Report",
            region: Optional[str] = None,
        ) -> bytes:
            """Render a text report describing the supplied observations.

            Args:
                profiles: Observations covered by the report.
                title: Report title.
                region: Region the report covers.

            Returns:
                The report encoded as UTF-8 bytes.
            """
            stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
            rows = 0 if profiles is None or profiles.empty else len(profiles)
            floats = (
                0
                if profiles is None or profiles.empty or "float_id" not in profiles.columns
                else profiles["float_id"].nunique()
            )
            lines = [
                "OceanMind AI - Ocean Intelligence Report",
                "=" * 46,
                f"Title      : {title}",
                f"Region     : {region or 'All regions'}",
                f"Generated  : {stamp}",
                f"Floats     : {floats}",
                f"Observations: {rows:,}",
                "",
                "NOTE: Generated in demo mode. Figures are synthetic and must not",
                "be presented as scientific findings.",
            ]
            return "\n".join(lines).encode("utf-8")


def _demo_recommendations(factors: Mapping[str, float]) -> List[Dict[str, str]]:
    """Derive illustrative recommendations from the weakest sub-scores.

    Args:
        factors: Sub-score mapping.

    Returns:
        A list of recommendation mappings ordered by severity.
    """
    catalogue = {
        "oxygen_availability": (
            "Expand oxygen monitoring in the OMZ",
            "Dissolved oxygen is the weakest indicator. Prioritise BGC-float "
            "deployments between 200 m and 800 m to resolve the minimum zone.",
        ),
        "acidification": (
            "Increase carbonate chemistry sampling",
            "pH is trending towards the lower bound. Add pH and alkalinity "
            "sensors on the next deployment cycle.",
        ),
        "thermal_stability": (
            "Track marine heatwave risk",
            "Surface warming is elevated. Increase sampling cadence in the "
            "upper 100 m to detect stratification changes early.",
        ),
        "salinity_balance": (
            "Investigate freshwater influence",
            "Salinity variance is high near river outflows. Cross-check with "
            "monsoon discharge records.",
        ),
        "biological_productivity": (
            "Correlate chlorophyll with nutrient supply",
            "Productivity is below the regional baseline. Compare chlorophyll "
            "maxima against upwelling indices.",
        ),
        "data_coverage": (
            "Close spatial coverage gaps",
            "Float density is uneven. Plan deployments to fill under-sampled "
            "grid cells before the next reporting cycle.",
        ),
    }
    ordered = sorted(factors.items(), key=lambda item: item[1])[:4]
    priorities = ("High", "High", "Medium", "Low")
    return [
        {
            "title": catalogue.get(name, (name.replace("_", " ").title(), ""))[0],
            "detail": catalogue.get(name, ("", "Review this indicator."))[1],
            "priority": priorities[index],
            "factor": name,
            "score": f"{value:.0f}",
        }
        for index, (name, value) in enumerate(ordered)
    ]


# --------------------------------------------------------------------------- #
# Response adaptation
# --------------------------------------------------------------------------- #


@dataclass
class HealthView:
    """The renderable form of a health assessment.

    Attributes:
        score: Composite index on a 0-100 scale.
        factors: Sub-score per contributing indicator.
        recommendations: Ordered recommendation mappings.
        region: Region the assessment covers.
        computed_at: When the assessment was produced.
        raw: The untouched backend response.
    """

    score: float
    factors: Dict[str, float] = field(default_factory=dict)
    recommendations: List[Dict[str, str]] = field(default_factory=list)
    region: Optional[str] = None
    computed_at: Optional[datetime] = None
    raw: Any = None


def normalise_health(raw: Any) -> HealthView:
    """Adapt any ``OceanHealthCalculator.compute()`` return shape.

    Args:
        raw: Whatever the calculator returned -- mapping, object or number.

    Returns:
        A populated :class:`HealthView`. Unreadable payloads yield a zero score
        rather than raising.
    """
    if isinstance(raw, HealthView):
        return raw

    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return HealthView(score=float(raw), raw=raw)

    def _lookup(keys: Sequence[str]) -> Any:
        for key in keys:
            if isinstance(raw, Mapping) and key in raw:
                return raw[key]
            value = getattr(raw, key, None)
            if value is not None and not callable(value):
                return value
        return None

    score_value = _lookup(("score", "health_score", "index", "value", "overall"))
    try:
        score = float(score_value) if score_value is not None else 0.0
    except (TypeError, ValueError):
        score = 0.0

    factors_value = _lookup(("factors", "components", "indicators", "sub_scores"))
    factors: Dict[str, float] = {}
    if isinstance(factors_value, Mapping):
        for name, value in factors_value.items():
            try:
                factors[str(name)] = float(value)
            except (TypeError, ValueError):
                continue

    recommendations_value = _lookup(("recommendations", "actions", "advice"))
    recommendations: List[Dict[str, str]] = []
    if isinstance(recommendations_value, Sequence) and not isinstance(recommendations_value, (str, bytes)):
        for item in recommendations_value:
            if isinstance(item, Mapping):
                recommendations.append({str(k): str(v) for k, v in item.items()})
            else:
                recommendations.append({"title": str(item), "detail": "", "priority": "Medium"})

    computed = _lookup(("computed_at", "generated_at", "timestamp"))
    computed_at: Optional[datetime] = None
    if computed is not None:
        try:
            computed_at = pd.to_datetime(computed).to_pydatetime()
        except (ValueError, TypeError):
            computed_at = None

    return HealthView(
        score=score,
        factors=factors,
        recommendations=recommendations,
        region=_lookup(("region",)),
        computed_at=computed_at,
        raw=raw,
    )


# --------------------------------------------------------------------------- #
# Backend access
# --------------------------------------------------------------------------- #


@st.cache_resource(show_spinner=False)
def get_health_calculator() -> OceanHealthCalculator:
    """Return the shared ocean health calculator instance."""
    return OceanHealthCalculator()


@st.cache_resource(show_spinner=False)
def get_report_generator() -> ReportGenerator:
    """Return the shared report generator instance."""
    return ReportGenerator()


@st.cache_data(show_spinner="Computing ocean health index...", ttl=600)
def compute_health(
    _profiles: pd.DataFrame,
    *,
    region: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> HealthView:
    """Run the health calculation and adapt its response.

    Cached on ``(region, start_date, end_date)`` -- the leading underscore on
    ``_profiles`` tells Streamlit not to hash that argument, since
    ``OceanHealthCalculator.compute()`` fetches its own data from the
    repository and never actually reads it (see the class docstring above).

    Without this cache, every rerun of the Ocean Health/Reports pages
    recomputed the score from scratch: for a multi-region key like "Indian
    Ocean" that means one backend call *per underlying region*, each doing
    its own DB-bound aggregation over the full date range -- tens of seconds
    per region, repeated on every single widget interaction.

    Args:
        _profiles: Observations to assess (unused by the real calculator;
            present only to preserve the call signature).
        region: Region the assessment covers.
        start_date: Inclusive lower date bound.
        end_date: Inclusive upper date bound.

    Returns:
        The adapted assessment. Backend failures return a zero-score view with
        the error surfaced as a recommendation rather than raising.
    """
    try:
        raw = get_health_calculator().compute(
            profiles=_profiles, region=region, start_date=start_date, end_date=end_date
        )
    except Exception as exc:  # noqa: BLE001 - surface backend failures in the UI
        logger.exception("OceanHealthCalculator.compute failed")
        return HealthView(
            score=0.0,
            recommendations=[
                {
                    "title": "Health calculation unavailable",
                    "detail": f"{type(exc).__name__}: {exc}",
                    "priority": "High",
                }
            ],
        )
    return normalise_health(raw)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def render_health_summary(view: HealthView, profiles: pd.DataFrame) -> None:
    """Render the gauge, KPI cards and contributing-factor chart.

    Args:
        view: The adapted health assessment.
        profiles: Observations the assessment was computed from.
    """
    section_header(
        "Ocean Health Index",
        "A composite indicator combining thermal, chemical, biological and "
        "coverage sub-scores from the intelligence engine.",
    )

    left, right = st.columns([1, 1.35], gap="large")
    with left:
        st.plotly_chart(health_gauge(view.score), width="stretch")
        tone = score_to_tone(view.score)
        st.markdown(
            f"{pill(score_to_label(view.score), tone)} "
            f"<span style='color:{OCEAN['text_muted']};font-size:.82rem;margin-left:.5rem'>"
            f"assessed {format_datetime(view.computed_at) if view.computed_at else 'just now'}"
            "</span>",
            unsafe_allow_html=True,
        )
    with right:
        st.plotly_chart(contributing_factors(view.factors), width="stretch")

    weakest = min(view.factors.items(), key=lambda item: item[1]) if view.factors else None
    strongest = max(view.factors.items(), key=lambda item: item[1]) if view.factors else None

    render_kpi_row(
        [
            {
                "label": "Composite Index",
                "value": format_number(view.score, decimals=1),
                "delta": score_to_label(view.score),
                "delta_tone": score_to_tone(view.score),
                "caption": "0-100 scale",
            },
            {
                "label": "Weakest Indicator",
                "value": weakest[0].replace("_", " ").title() if weakest else "--",
                "delta": f"{weakest[1]:.0f}/100" if weakest else None,
                "delta_tone": "bad",
                "caption": "Prioritise this factor",
            },
            {
                "label": "Strongest Indicator",
                "value": strongest[0].replace("_", " ").title() if strongest else "--",
                "delta": f"{strongest[1]:.0f}/100" if strongest else None,
                "delta_tone": "good",
                "caption": "Performing well",
            },
            {
                "label": "Observations Assessed",
                "value": format_number(len(profiles), decimals=0, compact=True),
                "caption": f"{profiles['float_id'].nunique() if 'float_id' in profiles.columns else 0} floats",
            },
        ]
    )


def render_recommendations(view: HealthView) -> None:
    """Render the recommendation panel.

    Args:
        view: The adapted health assessment.
    """
    section_header(
        "Recommendations",
        "Actions proposed by the intelligence engine, ordered by priority.",
    )

    if not view.recommendations:
        st.info("The intelligence engine returned no recommendations for this selection.")
        return

    tone_by_priority = {"high": "bad", "medium": "warn", "low": "good"}
    for index, recommendation in enumerate(view.recommendations):
        priority = str(recommendation.get("priority", "Medium"))
        tone = tone_by_priority.get(priority.lower(), "warn")
        with st.container(border=True):
            header = st.columns([4, 1])
            with header[0]:
                st.markdown(f"**{recommendation.get('title', f'Recommendation {index + 1}')}**")
            with header[1]:
                st.markdown(
                    f"<div style='text-align:right'>{pill(priority, tone)}</div>",
                    unsafe_allow_html=True,
                )
            detail = recommendation.get("detail") or recommendation.get("description")
            if detail:
                st.caption(detail)


def render_reports(profiles: pd.DataFrame, *, region: Optional[str] = None) -> None:
    """Render report generation, the generated-report list and exports.

    Args:
        profiles: Observations the report should cover.
        region: Region the report covers.
    """
    section_header(
        "Reports & Export",
        "Generate a formatted intelligence report, or export the underlying "
        "observations as CSV, NetCDF or ASCII.",
    )

    st.session_state.setdefault(SESSION_REPORTS, [])
    reports: List[Dict[str, Any]] = st.session_state[SESSION_REPORTS]

    controls = st.columns([2, 1])
    with controls[0]:
        title = st.text_input(
            "Report title",
            value=f"{region or 'Ocean'} Intelligence Report",
            key="om_report_title",
        )
    with controls[1]:
        st.write("")
        generate = st.button("Generate report", type="primary", width="stretch")

    if generate:
        with st.spinner("Generating report..."):
            payload = _generate_report(profiles, title=title, region=region)
        if payload is not None:
            reports.insert(
                0,
                {
                    "title": title,
                    "region": region or "All regions",
                    "created_at": datetime.now(timezone.utc),
                    "payload": payload,
                    "observations": int(len(profiles)),
                    # The real ReportBuilder always writes a PDF; the demo
                    # stub always writes plain text -- pick the download's
                    # extension/mime accordingly rather than hardcoding .txt.
                    "extension": "pdf" if REPORT_BACKEND_AVAILABLE else "txt",
                    "mime": "application/pdf" if REPORT_BACKEND_AVAILABLE else "text/plain",
                },
            )
            st.success(f"Report generated ({len(payload):,} bytes).")

    if reports:
        st.markdown("##### Generated reports")
        for index, report in enumerate(reports):
            with st.container(border=True):
                columns = st.columns([3, 1.4, 1.3])
                with columns[0]:
                    st.markdown(f"**{report['title']}**")
                    st.caption(
                        f"{report['region']} - {report['observations']:,} observations - "
                        f"{format_datetime(report['created_at'], fmt='%d %b %Y %H:%M UTC')}"
                    )
                with columns[1]:
                    st.markdown(pill("Ready", "good"), unsafe_allow_html=True)
                with columns[2]:
                    render_binary_download(
                        report["payload"],
                        base_name="oceanmind_report",
                        extension=report.get("extension", "txt"),
                        label="Download",
                        mime=report.get("mime", "text/plain"),
                        key=f"om_report_dl_{index}",
                    )
    else:
        st.caption("No reports generated yet. Use the button above to create one.")

    st.divider()
    st.markdown("##### Export observations")
    render_export_bar(
        profiles,
        base_name=f"oceanmind_{(region or 'all').lower().replace(' ', '_')}",
        key_prefix="health_export",
        caption="Download the filtered dataset backing this assessment.",
    )


def _generate_report(
    profiles: pd.DataFrame,
    *,
    title: str,
    region: Optional[str],
) -> Optional[bytes]:
    """Invoke the report generator and normalise its output to bytes.

    The backend's return type is unconfirmed, so ``bytes``, ``str`` and file
    paths are all handled.

    Args:
        profiles: Observations the report covers.
        title: Report title.
        region: Region the report covers.

    Returns:
        The report bytes, or ``None`` when generation failed.
    """
    try:
        result = get_report_generator().generate(profiles=profiles, title=title, region=region)
    except Exception as exc:  # noqa: BLE001 - surface backend failures in the UI
        logger.exception("ReportGenerator.generate failed")
        st.error(f"Report generation failed: {type(exc).__name__}: {exc}")
        return None

    if isinstance(result, bytes):
        return result
    if isinstance(result, str):
        # Could be report text, or a path the generator wrote to.
        try:
            with open(result, "rb") as handle:
                return handle.read()
        except OSError:
            return result.encode("utf-8")
    if result is None:
        st.warning("The report generator returned nothing.")
        return None
    return str(result).encode("utf-8")


def render_health_panel(
    profiles: pd.DataFrame,
    *,
    region: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> HealthView:
    """Render the complete Ocean Health tab.

    Args:
        profiles: Observations for the active filters.
        region: Region the assessment covers.
        start_date: Inclusive lower date bound.
        end_date: Inclusive upper date bound.

    Returns:
        The computed health assessment, so callers may reuse it.
    """
    if not HEALTH_BACKEND_AVAILABLE:
        st.info(
            "Demo mode: the intelligence engine is not importable, so scores are "
            "illustrative. Connect the backend for real assessments.",
            icon=":material/info:",
        )

    view = compute_health(profiles, region=region, start_date=start_date, end_date=end_date)
    render_health_summary(view, profiles)
    st.divider()
    render_recommendations(view)
    return view
