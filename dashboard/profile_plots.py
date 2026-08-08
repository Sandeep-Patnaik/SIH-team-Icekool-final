"""Plotly figure builders for the OceanMind AI dashboard.

Every function in this module is pure: it accepts a :class:`pandas.DataFrame`
and returns a :class:`plotly.graph_objects.Figure`. Nothing here touches
Streamlit, session state or the backend, which keeps the visualisations
independently testable and reusable across tabs.

Oceanographic convention is respected throughout: depth increases downwards, so
every depth axis is reversed.
"""

from __future__ import annotations

import logging
from typing import Dict, Final, List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dashboard.styles import CATEGORICAL, OCEAN, PLOTLY_TEMPLATE, SEQUENTIAL
from dashboard.utils import colour_for_index, variable_label

logger = logging.getLogger(__name__)

DEPTH_COLUMN: Final[str] = "depth"
FLOAT_COLUMN: Final[str] = "float_id"
TIME_COLUMN: Final[str] = "time"

#: Height applied to every figure unless a caller overrides it.
DEFAULT_HEIGHT: Final[int] = 430


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _empty_figure(message: str, *, height: int = DEFAULT_HEIGHT) -> go.Figure:
    """Build a themed placeholder shown when a chart has no data to draw.

    Returning a figure rather than ``None`` keeps the page layout stable, so
    panels never collapse when a filter excludes everything.

    Args:
        message: Explanation rendered in the centre of the empty plot.
        height: Figure height in pixels.

    Returns:
        A figure containing only the message.
    """
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        showarrow=False,
        font=dict(color=OCEAN["text_muted"], size=13),
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
    )
    figure.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return figure


#: Above this many points, scatter charts are randomly downsampled before
#: plotting. Keeps plain SVG ``go.Scatter`` traces responsive and readable
#: without depending on WebGL (``go.Scattergl``), which some lab/demo
#: machines and remote sessions report as unsupported -- and a literal
#: point-per-observation plot is unreadable overplotted noise well before
#: it becomes a WebGL performance problem anyway.
_SCATTER_POINT_CAP: Final[int] = 8000


def _downsample(frame: pd.DataFrame, *, cap: int = _SCATTER_POINT_CAP) -> pd.DataFrame:
    """Randomly subsample a frame for scatter plotting, if it's large.

    Args:
        frame: Source observations.
        cap: Maximum rows to keep.

    Returns:
        ``frame`` unchanged if it's within ``cap``, otherwise a reproducible
        random sample of ``cap`` rows.
    """
    if len(frame) <= cap:
        return frame
    return frame.sample(n=cap, random_state=0)


def _finalise(
    figure: go.Figure,
    *,
    title: str,
    height: int = DEFAULT_HEIGHT,
    show_legend: bool = True,
) -> go.Figure:
    """Apply the shared layout contract to a figure.

    Args:
        figure: Figure to finalise.
        title: Chart title.
        height: Figure height in pixels.
        show_legend: Whether the legend is displayed.

    Returns:
        The same figure, themed and sized.
    """
    figure.update_layout(
        template=PLOTLY_TEMPLATE,
        title=title,
        height=height,
        showlegend=show_legend,
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    return figure


def _has_columns(frame: Optional[pd.DataFrame], columns: Sequence[str]) -> bool:
    """Report whether a frame is usable and carries every required column.

    Args:
        frame: Candidate table.
        columns: Required column names.

    Returns:
        ``True`` when the frame is non-empty and complete.
    """
    if frame is None or frame.empty:
        return False
    return set(columns) <= set(frame.columns)


def _profile_traces(
    frame: pd.DataFrame,
    variable: str,
    *,
    max_floats: int = 12,
) -> List[go.Scatter]:
    """Build one depth-profile trace per float.

    Args:
        frame: Observations containing ``depth``, ``variable`` and optionally
            ``float_id``.
        variable: Column plotted on the x axis.
        max_floats: Cap on distinct floats drawn, keeping the chart legible.

    Returns:
        A list of scatter traces ordered by float identifier.
    """
    if FLOAT_COLUMN not in frame.columns:
        ordered = frame.sort_values(DEPTH_COLUMN)
        return [
            go.Scatter(
                x=ordered[variable],
                y=ordered[DEPTH_COLUMN],
                mode="lines",
                name=variable_label(variable, with_unit=False),
                line=dict(color=OCEAN["accent"], width=2),
                hovertemplate=(
                    f"{variable_label(variable)}: %{{x:.3f}}<br>Depth: %{{y:.0f}} m<extra></extra>"
                ),
            )
        ]

    traces: List[go.Scatter] = []
    float_ids = sorted(frame[FLOAT_COLUMN].astype(str).unique())[:max_floats]
    for index, float_id in enumerate(float_ids):
        subset = frame[frame[FLOAT_COLUMN].astype(str) == float_id]
        # Average duplicate depths across cycles so one float draws one clean line.
        ordered = subset.groupby(DEPTH_COLUMN, as_index=False)[variable].mean()
        ordered = ordered.sort_values(DEPTH_COLUMN)
        traces.append(
            go.Scatter(
                x=ordered[variable],
                y=ordered[DEPTH_COLUMN],
                mode="lines",
                name=f"Float {float_id}",
                line=dict(color=colour_for_index(index, CATEGORICAL), width=2),
                hovertemplate=(
                    f"Float {float_id}<br>{variable_label(variable)}: %{{x:.3f}}"
                    "<br>Depth: %{y:.0f} m<extra></extra>"
                ),
            )
        )
    return traces


# --------------------------------------------------------------------------- #
# Depth profiles
# --------------------------------------------------------------------------- #


def depth_profile(
    frame: pd.DataFrame,
    variable: str = "temperature",
    *,
    title: Optional[str] = None,
    height: int = DEFAULT_HEIGHT,
    max_floats: int = 12,
) -> go.Figure:
    """Plot a variable against depth, one line per float.

    Args:
        frame: Observations with ``depth`` and ``variable`` columns.
        variable: Column to place on the x axis.
        title: Override for the chart title.
        height: Figure height in pixels.
        max_floats: Maximum distinct floats drawn.

    Returns:
        A depth profile figure with the depth axis reversed.
    """
    if not _has_columns(frame, [DEPTH_COLUMN, variable]):
        return _empty_figure(f"No {variable_label(variable, with_unit=False).lower()} data available", height=height)

    figure = go.Figure(data=_profile_traces(frame, variable, max_floats=max_floats))
    figure.update_xaxes(title=variable_label(variable))
    figure.update_yaxes(title="Depth (m)", autorange="reversed")
    return _finalise(
        figure,
        title=title or f"{variable_label(variable, with_unit=False)} vs Depth",
        height=height,
    )


def temperature_depth_profile(frame: pd.DataFrame, **kwargs: object) -> go.Figure:
    """Plot temperature against depth.

    Args:
        frame: Observations with ``depth`` and ``temperature``.
        **kwargs: Forwarded to :func:`depth_profile`.

    Returns:
        The temperature profile figure.
    """
    return depth_profile(frame, "temperature", **kwargs)  # type: ignore[arg-type]


def salinity_depth_profile(frame: pd.DataFrame, **kwargs: object) -> go.Figure:
    """Plot salinity against depth.

    Args:
        frame: Observations with ``depth`` and ``salinity``.
        **kwargs: Forwarded to :func:`depth_profile`.

    Returns:
        The salinity profile figure.
    """
    return depth_profile(frame, "salinity", **kwargs)  # type: ignore[arg-type]


def pressure_profile(frame: pd.DataFrame, *, height: int = DEFAULT_HEIGHT) -> go.Figure:
    """Plot the pressure-depth relationship for the selected floats.

    Args:
        frame: Observations with ``depth`` and ``pressure``.
        height: Figure height in pixels.

    Returns:
        A pressure profile figure.
    """
    return depth_profile(frame, "pressure", title="Pressure vs Depth", height=height)


def oxygen_trend(frame: pd.DataFrame, *, height: int = DEFAULT_HEIGHT) -> go.Figure:
    """Plot the dissolved oxygen profile, highlighting the oxygen minimum zone.

    Args:
        frame: Observations with ``depth`` and ``oxygen``.
        height: Figure height in pixels.

    Returns:
        A figure with the mean oxygen profile and its minimum annotated.
    """
    if not _has_columns(frame, [DEPTH_COLUMN, "oxygen"]):
        return _empty_figure("No dissolved oxygen data available", height=height)

    profile = frame.groupby(DEPTH_COLUMN, as_index=False)["oxygen"].mean().sort_values(DEPTH_COLUMN)

    if profile["oxygen"].notna().sum() == 0:
        # The column exists (e.g. from a multi-float selection) but this
        # float/period combination has no actual oxygen sensor readings --
        # not every ARGO float carries an oxygen sensor. idxmin() on an
        # all-NaN series raises ValueError, so bail out to the same
        # "no data" figure used when the column is missing entirely.
        return _empty_figure("No dissolved oxygen data available", height=height)

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=profile["oxygen"],
            y=profile[DEPTH_COLUMN],
            mode="lines",
            name="Mean dissolved oxygen",
            line=dict(color=OCEAN["teal"], width=2.5),
            fill="tozerox",
            fillcolor="rgba(45,212,191,0.10)",
            hovertemplate="O2: %{x:.1f} umol/kg<br>Depth: %{y:.0f} m<extra></extra>",
        )
    )

    minimum = profile.loc[profile["oxygen"].idxmin()]
    figure.add_annotation(
        x=float(minimum["oxygen"]),
        y=float(minimum[DEPTH_COLUMN]),
        text=f"OMZ core ~{minimum[DEPTH_COLUMN]:.0f} m",
        showarrow=True,
        arrowhead=2,
        arrowcolor=OCEAN["warn"],
        font=dict(color=OCEAN["warn"], size=11),
        ax=52,
        ay=0,
    )

    figure.update_xaxes(title=variable_label("oxygen"))
    figure.update_yaxes(title="Depth (m)", autorange="reversed")
    return _finalise(figure, title="Dissolved Oxygen Profile", height=height, show_legend=False)


# --------------------------------------------------------------------------- #
# Comparison and relationship plots
# --------------------------------------------------------------------------- #


def float_comparison(
    frame: pd.DataFrame,
    variable: str = "temperature",
    *,
    float_ids: Optional[Sequence[str]] = None,
    height: int = DEFAULT_HEIGHT,
) -> go.Figure:
    """Compare the same variable across selected floats.

    Args:
        frame: Observations with ``float_id``, ``depth`` and ``variable``.
        variable: Column to compare.
        float_ids: Restrict the comparison to these floats. ``None`` uses all.
        height: Figure height in pixels.

    Returns:
        An overlaid depth profile figure, one line per float.
    """
    if not _has_columns(frame, [FLOAT_COLUMN, DEPTH_COLUMN, variable]):
        return _empty_figure("Float comparison needs float, depth and variable columns", height=height)

    subset = frame
    if float_ids:
        wanted = {str(value) for value in float_ids}
        subset = frame[frame[FLOAT_COLUMN].astype(str).isin(wanted)]
    if subset.empty:
        return _empty_figure("No observations for the selected floats", height=height)

    figure = go.Figure(data=_profile_traces(subset, variable, max_floats=10))
    figure.update_xaxes(title=variable_label(variable))
    figure.update_yaxes(title="Depth (m)", autorange="reversed")
    return _finalise(
        figure,
        title=f"Float Comparison - {variable_label(variable, with_unit=False)}",
        height=height,
    )


def ts_diagram(frame: pd.DataFrame, *, height: int = DEFAULT_HEIGHT) -> go.Figure:
    """Draw a temperature-salinity diagram coloured by depth.

    The T-S diagram is the standard way oceanographers identify water masses.

    Args:
        frame: Observations with ``temperature``, ``salinity`` and ``depth``.
        height: Figure height in pixels.

    Returns:
        A scatter figure with a depth colour bar.
    """
    if not _has_columns(frame, ["temperature", "salinity", DEPTH_COLUMN]):
        return _empty_figure("T-S diagram needs temperature, salinity and depth", height=height)

    plotted = _downsample(frame)
    figure = go.Figure(
        data=go.Scatter(
            x=plotted["salinity"],
            y=plotted["temperature"],
            mode="markers",
            marker=dict(
                size=5,
                color=plotted[DEPTH_COLUMN],
                colorscale=[[index / (len(SEQUENTIAL) - 1), colour] for index, colour in enumerate(SEQUENTIAL)],
                reversescale=True,
                showscale=True,
                colorbar=dict(
                    title=dict(text="Depth (m)", font=dict(color=OCEAN["text_muted"], size=11)),
                    tickfont=dict(color=OCEAN["text_muted"], size=10),
                    outlinewidth=0,
                    thickness=12,
                ),
                opacity=0.78,
            ),
            hovertemplate=(
                "Salinity: %{x:.3f} PSU<br>Temperature: %{y:.2f} degC"
                "<br>Depth: %{marker.color:.0f} m<extra></extra>"
            ),
        )
    )
    figure.update_xaxes(title=variable_label("salinity"))
    figure.update_yaxes(title=variable_label("temperature"))
    return _finalise(figure, title="Temperature-Salinity Diagram", height=height, show_legend=False)


def scatter_plot(
    frame: pd.DataFrame,
    x: str,
    y: str,
    *,
    colour_by: Optional[str] = None,
    height: int = DEFAULT_HEIGHT,
    title: Optional[str] = None,
) -> go.Figure:
    """Plot an arbitrary relationship between two measured variables.

    Args:
        frame: Source observations.
        x: Column for the x axis.
        y: Column for the y axis.
        colour_by: Optional numeric column driving marker colour.
        height: Figure height in pixels.
        title: Override for the chart title.

    Returns:
        A scatter figure.
    """
    if not _has_columns(frame, [x, y]):
        return _empty_figure(f"Columns '{x}' and '{y}' are not both available", height=height)

    plotted = _downsample(frame)
    marker: Dict[str, object] = {"size": 5, "opacity": 0.75, "color": OCEAN["accent"]}
    if colour_by and colour_by in frame.columns:
        marker = {
            "size": 5,
            "opacity": 0.8,
            "color": plotted[colour_by],
            "colorscale": [
                [index / (len(SEQUENTIAL) - 1), colour] for index, colour in enumerate(SEQUENTIAL)
            ],
            "showscale": True,
            "colorbar": dict(
                title=dict(text=variable_label(colour_by, with_unit=False), font=dict(size=11)),
                thickness=12,
                outlinewidth=0,
            ),
        }

    figure = go.Figure(
        data=go.Scatter(x=plotted[x], y=plotted[y], mode="markers", marker=marker)
    )
    figure.update_xaxes(title=variable_label(x))
    figure.update_yaxes(title=variable_label(y))
    return _finalise(
        figure,
        title=title or f"{variable_label(y, with_unit=False)} vs {variable_label(x, with_unit=False)}",
        height=height,
        show_legend=False,
    )


# --------------------------------------------------------------------------- #
# Time series
# --------------------------------------------------------------------------- #


def time_series(
    frame: pd.DataFrame,
    variable: str = "temperature",
    *,
    surface_only: bool = True,
    surface_depth: float = 50.0,
    height: int = DEFAULT_HEIGHT,
) -> go.Figure:
    """Plot a variable through time, with a rolling mean overlay.

    Args:
        frame: Observations with ``time`` and ``variable`` columns.
        variable: Column to trend.
        surface_only: Restrict to the near-surface layer, which is what
            climate indicators are normally reported against.
        surface_depth: Depth cut-off in metres for ``surface_only``.
        height: Figure height in pixels.

    Returns:
        A time series figure with observed values and a 7-point rolling mean.
    """
    if not _has_columns(frame, [TIME_COLUMN, variable]):
        return _empty_figure("No time-stamped observations available", height=height)

    subset = frame
    if surface_only and DEPTH_COLUMN in frame.columns:
        subset = frame[frame[DEPTH_COLUMN] <= surface_depth]
    if subset.empty:
        return _empty_figure("No observations in the selected depth band", height=height)

    series = (
        subset.assign(_time=pd.to_datetime(subset[TIME_COLUMN], errors="coerce"))
        .dropna(subset=["_time"])
        .groupby("_time", as_index=False)[variable]
        .mean()
        .sort_values("_time")
    )
    if series.empty:
        return _empty_figure("No parseable timestamps available", height=height)

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=series["_time"],
            y=series[variable],
            mode="lines+markers",
            name="Observed",
            line=dict(color=OCEAN["accent"], width=1.6),
            marker=dict(size=5),
            hovertemplate="%{x|%d %b %Y}<br>%{y:.3f}<extra></extra>",
        )
    )

    if len(series) >= 5:
        window = max(3, min(7, len(series) // 2))
        figure.add_trace(
            go.Scatter(
                x=series["_time"],
                y=series[variable].rolling(window, min_periods=1, center=True).mean(),
                mode="lines",
                name=f"{window}-point rolling mean",
                line=dict(color=OCEAN["warn"], width=2.4, dash="dot"),
                hovertemplate="%{x|%d %b %Y}<br>%{y:.3f}<extra></extra>",
            )
        )

    figure.update_xaxes(title="Date")
    figure.update_yaxes(title=variable_label(variable))
    depth_note = f" (upper {surface_depth:.0f} m)" if surface_only else ""
    return _finalise(
        figure,
        title=f"{variable_label(variable, with_unit=False)} Trend{depth_note}",
        height=height,
    )


def variable_distribution(
    frame: pd.DataFrame,
    variable: str = "temperature",
    *,
    height: int = 320,
) -> go.Figure:
    """Show the distribution of a variable across the current selection.

    Args:
        frame: Source observations.
        variable: Column to summarise.
        height: Figure height in pixels.

    Returns:
        A histogram figure with the mean marked.
    """
    if not _has_columns(frame, [variable]):
        return _empty_figure(f"No {variable} data available", height=height)

    values = pd.to_numeric(frame[variable], errors="coerce").dropna()
    if values.empty:
        return _empty_figure(f"No numeric {variable} values", height=height)

    figure = go.Figure(
        data=go.Histogram(
            x=values,
            nbinsx=42,
            marker=dict(color=OCEAN["accent_deep"], line=dict(color=OCEAN["border"], width=1)),
            hovertemplate="%{x}<br>Count: %{y}<extra></extra>",
        )
    )
    figure.add_vline(
        x=float(values.mean()),
        line=dict(color=OCEAN["warn"], width=2, dash="dash"),
        annotation_text=f"mean {values.mean():.2f}",
        annotation_font=dict(color=OCEAN["warn"], size=11),
    )
    figure.update_xaxes(title=variable_label(variable))
    figure.update_yaxes(title="Observations")
    return _finalise(
        figure,
        title=f"{variable_label(variable, with_unit=False)} Distribution",
        height=height,
        show_legend=False,
    )


def health_gauge(score: float, *, height: int = 300, label: str = "Ocean Health Index") -> go.Figure:
    """Render the composite ocean health score as a gauge.

    Args:
        score: Health score on a 0-100 scale. Values outside the range are
            clamped rather than rejected.
        height: Figure height in pixels.
        label: Text shown beneath the number.

    Returns:
        A Plotly indicator gauge figure.
    """
    from dashboard.utils import score_to_colour  # local import avoids a cycle at module load

    clamped = float(np.clip(score, 0.0, 100.0))
    colour = score_to_colour(clamped)

    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=clamped,
            number=dict(font=dict(color=OCEAN["text_primary"], size=44), suffix=""),
            title=dict(text=label, font=dict(color=OCEAN["text_muted"], size=13)),
            gauge=dict(
                axis=dict(
                    range=[0, 100],
                    tickcolor=OCEAN["text_muted"],
                    tickfont=dict(color=OCEAN["text_muted"], size=10),
                ),
                bar=dict(color=colour, thickness=0.26),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
                steps=[
                    dict(range=[0, 35], color="rgba(248,113,113,0.16)"),
                    dict(range=[35, 55], color="rgba(251,191,36,0.16)"),
                    dict(range=[55, 75], color="rgba(45,212,191,0.16)"),
                    dict(range=[75, 100], color="rgba(52,211,153,0.16)"),
                ],
                threshold=dict(
                    line=dict(color=OCEAN["text_primary"], width=3),
                    thickness=0.78,
                    value=clamped,
                ),
            ),
        )
    )
    figure.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        margin=dict(l=24, r=24, t=48, b=16),
    )
    return figure


def contributing_factors(factors: Dict[str, float], *, height: int = 330) -> go.Figure:
    """Plot the indicators contributing to the composite health score.

    Args:
        factors: Mapping of factor name to a 0-100 sub-score.
        height: Figure height in pixels.

    Returns:
        A horizontal bar chart ordered worst-first, so risks read at the top.
    """
    from dashboard.utils import score_to_colour  # local import avoids a cycle at module load

    if not factors:
        return _empty_figure("No contributing factors reported", height=height)

    ordered = sorted(factors.items(), key=lambda item: item[1])
    names = [name.replace("_", " ").title() for name, _ in ordered]
    values = [float(value) for _, value in ordered]

    figure = go.Figure(
        data=go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker=dict(color=[score_to_colour(value) for value in values]),
            text=[f"{value:.0f}" for value in values],
            textposition="outside",
            textfont=dict(color=OCEAN["text_secondary"], size=11),
            hovertemplate="%{y}: %{x:.1f}/100<extra></extra>",
        )
    )
    figure.update_xaxes(title="Sub-score (0-100)", range=[0, 108])
    figure.update_yaxes(title="")
    return _finalise(figure, title="Contributing Factors", height=height, show_legend=False)
