"""OceanMind AI dashboard package.

The Streamlit presentation tier for the OceanMind AI ocean-intelligence
platform. It consumes the completed backend (``database``, ``vector_rag``,
``llm_query_engine``, ``intelligence_engine``) through direct Python imports
and contains no business logic of its own.

Launch with::

    streamlit run dashboard/app.py

Modules
-------
app
    Streamlit entry point: navigation, filters, layout and routing.
map_view
    Folium map of ARGO float positions and trajectories.
profile_plots
    Pure Plotly figure builders.
chat_panel
    Natural-language AI workspace over the query engine.
health_panel
    Ocean Health Index, recommendations and reports.
export_utils
    CSV, NetCDF and ASCII export helpers.
styles
    Dark ocean theme, palette and custom Streamlit CSS.
utils
    Formatting, dates, filtering and colour helpers.
"""

from __future__ import annotations

__all__ = ["__version__", "APP_NAME", "APP_TAGLINE"]

__version__ = "1.0.0"

APP_NAME = "OceanMind AI"
APP_TAGLINE = "Ocean Intelligence Platform"
