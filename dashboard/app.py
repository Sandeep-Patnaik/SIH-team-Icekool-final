"""Streamlit entry point for the OceanMind AI dashboard.

Owns page configuration, theming, sidebar navigation, the global filter set and
routing between the four workspaces. Deliberately contains no business logic and
no plotting code -- every view is delegated to its own module.

Run from the backend root::

    streamlit run dashboard/app.py
"""

from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, Final, List

# Allow `streamlit run dashboard/app.py` from any working directory: Streamlit
# places the script's own folder on sys.path, not its parent, so the absolute
# `dashboard.*` imports below would otherwise fail.
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

import pandas as pd  # noqa: E402  - must follow the sys.path bootstrap
import streamlit as st  # noqa: E402

from dashboard import APP_NAME, APP_TAGLINE, __version__  # noqa: E402
from dashboard import chat_panel, health_panel, map_view, profile_plots  # noqa: E402
from dashboard.export_utils import render_export_bar  # noqa: E402
from dashboard.styles import (  # noqa: E402
    apply_theme,
    logo_svg_markup,
    pill,
    render_ambient_strip,
    render_hero,
    render_kpi_row,
    section_header,
)  # noqa: E402
from dashboard.utils import (  # noqa: E402
    DEFAULT_REGION,
    REGIONS,
    SESSION_FILTERS,
    VARIABLES,
    available_variables,
    ensure_session_defaults,
    filter_profiles,
    format_datetime,
    format_number,
    normalise_date_range,
    summarise_profiles,
    variable_label,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PAGES: Final[Dict[str, str]] = {
    "Explore Ocean": ":material/public:",
    "AI Workspace": ":material/smart_toy:",
    "Ocean Health": ":material/waves:",
    "Reports": ":material/description:",
}

#: Observation rows fetched per page load. The backend join is one DB call
#: per *profile* (see ``map_view._profile_rows_to_frame``), so this caps how
#: many of those calls a single wide filter can trigger -- without it, a
#: broad date range across "Indian Ocean" could walk hundreds of profiles
#: before the page renders anything.
DEFAULT_QUERY_LIMIT: Final[int] = 20_000


# --------------------------------------------------------------------------- #
# Page configuration
# --------------------------------------------------------------------------- #


def configure_page() -> None:
    """Apply Streamlit page configuration and the dashboard theme."""
    st.set_page_config(
        page_title=f"{APP_NAME} - {APP_TAGLINE}",
        page_icon=":ocean:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_theme()


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #


def render_brand() -> None:
    """Render the product wordmark at the top of the sidebar."""
    st.markdown(
        f"""
        <div class="om-brand">
          {logo_svg_markup(34, css_class="om-brand__mark")}
          <div>
            <div class="om-brand__name">{APP_NAME}</div>
            <div class="om-brand__tagline">{APP_TAGLINE}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_backend_status() -> None:
    """Show whether the dashboard is bound to the live backend or to stubs."""
    connected = (
        map_view.BACKEND_AVAILABLE
        and chat_panel.BACKEND_AVAILABLE
        and health_panel.HEALTH_BACKEND_AVAILABLE
        and health_panel.REPORT_BACKEND_AVAILABLE
    )
    if connected:
        st.markdown(pill("Backend: connected", "good"), unsafe_allow_html=True)
        return

    st.markdown(pill("Backend: offline (demo data)", "warn"), unsafe_allow_html=True)
    with st.expander("Why am I in demo mode?", expanded=False):
        rows = [
            ("database.repository", map_view.BACKEND_AVAILABLE),
            ("llm_query_engine.query_engine", chat_panel.BACKEND_AVAILABLE),
            ("intelligence_engine.health_index", health_panel.HEALTH_BACKEND_AVAILABLE),
            ("intelligence_engine.report_builder", health_panel.REPORT_BACKEND_AVAILABLE),
        ]
        for module, available in rows:
            st.markdown(f"{'&#9989;' if available else '&#10060;'} `{module}`", unsafe_allow_html=True)
        st.caption(
            "Place `dashboard/` at the backend root, or set PYTHONPATH to the folder "
            "containing these packages, and set the DATABASE_URL environment variable. "
            "Figures shown meanwhile are synthetic."
        )


def render_sidebar() -> None:
    """Render the sidebar: brand, the global filter set and backend status.

    Workspace navigation lives in :func:`render_topnav` at the top of the
    main content area instead -- see that function's docstring for why.
    """
    with st.sidebar:
        render_brand()
        st.markdown("###### Filters")
        filters = st.session_state[SESSION_FILTERS]

        filters["region"] = st.selectbox(
            "Region",
            options=list(REGIONS),
            index=list(REGIONS).index(filters.get("region", DEFAULT_REGION)),
            key="om_region",
        )

        date_bounds = map_view.available_date_bounds()
        date_input_kwargs: Dict[str, Any] = {}
        if date_bounds is not None:
            data_min, data_max = date_bounds
            # Widen (never narrow) around the current selection so existing
            # filter state always stays valid, and always allow up to today.
            date_input_kwargs["min_value"] = min(data_min, filters["date_start"])
            date_input_kwargs["max_value"] = max(data_max, filters["date_end"], date.today())

        date_value = st.date_input(
            "Date range",
            value=(filters["date_start"], filters["date_end"]),
            key="om_dates",
            **date_input_kwargs,
        )
        filters["date_start"], filters["date_end"] = normalise_date_range(date_value)

        filters["depth_max"] = float(
            st.slider(
                "Maximum depth (m)",
                min_value=100,
                max_value=2000,
                value=int(filters.get("depth_max", 2000)),
                step=100,
                key="om_depth",
            )
        )

        filters["variables"] = st.multiselect(
            "Variables",
            options=list(VARIABLES),
            default=filters.get("variables", ["temperature", "salinity"]),
            format_func=lambda name: variable_label(name, with_unit=False),
            key="om_variables",
        )

        st.divider()
        render_backend_status()
        st.caption(f"v{__version__}")

    return None


def render_topnav() -> str:
    """Render the four workspaces as a big-icon top navigation bar.

    Deliberately four ``st.button`` widgets rather than ``st.tabs``: Streamlit
    executes the body of every ``st.tabs`` block on each rerun regardless of
    which tab is visible, which would mean loading the map, chat engine and
    health/report panels on every single interaction. A button only ever
    triggers the branch the caller chooses to run on the *next* rerun, and
    each carries a real ``:material/...:`` icon (rendered as scalable font
    glyphs, sized up via CSS) rather than emoji baked into plain text.

    Returns:
        The name of the selected workspace.
    """
    active = st.session_state.setdefault("om_page", next(iter(PAGES)))

    st.markdown('<div class="om-topnav">', unsafe_allow_html=True)
    columns = st.columns(len(PAGES), gap="small")
    for column, (name, icon) in zip(columns, PAGES.items()):
        with column:
            if st.button(
                name,
                icon=icon,
                key=f"om_nav_{name}",
                width="stretch",
                type="primary" if name == active else "secondary",
            ):
                st.session_state["om_page"] = name
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    return st.session_state["om_page"]


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #


def load_filtered_profiles(filters: Dict[str, Any]) -> pd.DataFrame:
    """Load profiles for the active filters and apply client-side refinement.

    Args:
        filters: The active filter mapping from session state.

    Returns:
        The filtered observation table.
    """
    frame = map_view.load_profiles(
        region=filters["region"],
        start_date=filters["date_start"],
        end_date=filters["date_end"],
        limit=DEFAULT_QUERY_LIMIT,
    )
    return filter_profiles(
        frame,
        region=filters["region"],
        start=filters["date_start"],
        end=filters["date_end"],
        max_depth=filters["depth_max"],
    )


def render_kpi_summary(frame: pd.DataFrame) -> None:
    """Render the shared KPI row shown at the top of every data page.

    Args:
        frame: The filtered observation table.
    """
    summary = summarise_profiles(frame)
    render_kpi_row(
        [
            {
                "label": "Active Floats",
                "value": format_number(summary["floats"], decimals=0),
                "caption": "Reporting in range",
            },
            {
                "label": "Profiles",
                "value": format_number(summary["profiles"], decimals=0, compact=True),
                "caption": "Float-cycle pairs",
            },
            {
                "label": "Observations",
                "value": format_number(summary["observations"], decimals=0, compact=True),
                "caption": "Depth-level records",
            },
            {
                "label": "Mean Temperature",
                "value": format_number(summary["mean_temperature"], decimals=2, unit="degC"),
                "caption": "Across all depths",
            },
            {
                "label": "Latest Profile",
                "value": format_datetime(summary["latest_time"]),
                "caption": "Most recent transmission",
            },
        ]
    )


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #


def page_explore(frame: pd.DataFrame, filters: Dict[str, Any]) -> None:
    """Render the Explore Ocean workspace.

    Args:
        frame: The filtered observation table.
        filters: The active filter mapping.
    """
    summary = summarise_profiles(frame)
    render_hero(
        "Explore Ocean",
        "Locate ARGO floats, follow their trajectories and inspect the water column "
        "beneath them.",
        pills=[
            pill(f"{format_number(summary['floats'], decimals=0)} floats in view", "accent"),
            pill(filters["region"], "good"),
        ],
    )
    render_kpi_summary(frame)
    st.divider()

    map_view.render_map_panel(frame, region=filters["region"])
    map_view.render_nearby_lookup()
    st.divider()

    section_header(
        "Water Column Profiles",
        "Depth profiles for the selected variables, averaged per float across cycles.",
    )

    selected = [name for name in filters["variables"] if name in available_variables(frame)]
    if not selected:
        st.info("Select at least one variable in the sidebar to draw depth profiles.")
    else:
        for row_start in range(0, len(selected), 2):
            row = selected[row_start : row_start + 2]
            columns = st.columns(len(row), gap="large")
            for column, variable in zip(columns, row):
                with column:
                    st.plotly_chart(
                        profile_plots.depth_profile(frame, variable),
                        width="stretch",
                    )

    st.divider()
    section_header(
        "Relationships & Trends",
        "Water-mass structure and how the surface layer has changed over the window.",
    )
    left, right = st.columns(2, gap="large")
    with left:
        st.plotly_chart(profile_plots.ts_diagram(frame), width="stretch")
    with right:
        trend_variable = selected[0] if selected else "temperature"
        st.plotly_chart(profile_plots.time_series(frame, trend_variable), width="stretch")

    if "oxygen" in available_variables(frame):
        st.plotly_chart(profile_plots.oxygen_trend(frame), width="stretch")

    st.divider()
    section_header("Observations", "The filtered records backing every chart above.")
    st.dataframe(frame.head(500), width="stretch", height=330)
    st.caption(f"Showing up to 500 of {len(frame):,} observations.")
    render_export_bar(
        frame,
        base_name=f"argo_{filters['region'].lower().replace(' ', '_')}",
        key_prefix="explore_export",
    )


def page_ai_workspace(frame: pd.DataFrame) -> None:
    """Render the AI Workspace.

    Args:
        frame: The filtered observation table, shown as available context.
    """
    render_hero(
        "AI Workspace",
        "Query the ARGO archive in natural language. The engine generates SQL, "
        "executes it and explains the result.",
        pills=[pill("Backend: connected", "good") if chat_panel.BACKEND_AVAILABLE else pill("Demo mode", "warn")],
    )
    render_kpi_summary(frame)
    st.divider()
    chat_panel.render_chat_panel()


def page_ocean_health(frame: pd.DataFrame, filters: Dict[str, Any]) -> None:
    """Render the Ocean Health workspace.

    Args:
        frame: The filtered observation table.
        filters: The active filter mapping.
    """
    render_hero(
        "Ocean Health",
        "A composite assessment of ecosystem condition for the selected region "
        "and period, with prioritised recommendations.",
        pills=[pill(filters["region"], "accent")],
    )
    health_panel.render_health_panel(
        frame,
        region=filters["region"],
        start_date=filters["date_start"],
        end_date=filters["date_end"],
    )
    st.divider()
    section_header(
        "Supporting Distributions",
        "How the measured variables are distributed across the current selection.",
    )
    variables = available_variables(frame)[:3]
    if variables:
        columns = st.columns(len(variables), gap="large")
        for column, variable in zip(columns, variables):
            with column:
                st.plotly_chart(
                    profile_plots.variable_distribution(frame, variable),
                    width="stretch",
                )
    else:
        st.info("No measured variables are available for the current filters.")


def page_reports(frame: pd.DataFrame, filters: Dict[str, Any]) -> None:
    """Render the Reports workspace.

    Args:
        frame: The filtered observation table.
        filters: The active filter mapping.
    """
    render_hero(
        "Reports",
        "Generate formatted intelligence reports and export the underlying "
        "observations in the formats used by marine data centres.",
    )
    render_kpi_summary(frame)
    st.divider()
    health_panel.render_reports(frame, region=filters["region"])


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> None:
    """Compose and route the dashboard."""
    configure_page()
    ensure_session_defaults()

    render_sidebar()
    page = render_topnav()
    render_ambient_strip()
    filters = st.session_state[SESSION_FILTERS]

    frame = load_filtered_profiles(filters)

    if frame.empty:
        render_hero(page, "No observations match the current filters.")
        st.warning(
            "Widen the region, extend the date range or increase the maximum depth.",
            icon=":material/filter_alt_off:",
        )
        return

    if len(frame) >= DEFAULT_QUERY_LIMIT:
        st.caption(
            f"Showing the first {DEFAULT_QUERY_LIMIT:,} observations for this filter -- "
            "narrow the date range or region for the complete set."
        )

    if page == "Explore Ocean":
        page_explore(frame, filters)
    elif page == "AI Workspace":
        page_ai_workspace(frame)
    elif page == "Ocean Health":
        page_ocean_health(frame, filters)
    elif page == "Reports":
        page_reports(frame, filters)
    else:  # pragma: no cover - the radio constrains the value
        logger.warning("Unknown page %r; falling back to Explore Ocean", page)
        page_explore(frame, filters)


main()
