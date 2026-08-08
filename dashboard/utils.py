"""Shared helper functions for the OceanMind AI dashboard.

Formatting, date handling, colour mapping, session-state defaults and the
synthetic ARGO generator that powers demo mode when the backend is not
importable. Nothing here contains business logic -- scientific computation
belongs to the backend's intelligence engine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Final, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.styles import OCEAN

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #

SESSION_FILTERS: Final[str] = "om_filters"
SESSION_PROFILES: Final[str] = "om_profiles"
SESSION_CHAT: Final[str] = "om_chat_history"
SESSION_HEALTH: Final[str] = "om_health"
SESSION_REPORTS: Final[str] = "om_reports"

DEFAULT_LOOKBACK_DAYS: Final[int] = 90


# --------------------------------------------------------------------------- #
# Regions
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Region:
    """A named geographic bounding box used by the region filter.

    Attributes:
        name: Human readable region name.
        lat_min: Southern boundary in degrees north.
        lat_max: Northern boundary in degrees north.
        lon_min: Western boundary in degrees east.
        lon_max: Eastern boundary in degrees east.
        zoom: Sensible initial Folium zoom level for this extent.
    """

    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    zoom: int = 4

    @property
    def center(self) -> Tuple[float, float]:
        """Return the ``(latitude, longitude)`` midpoint of the box."""
        return ((self.lat_min + self.lat_max) / 2.0, (self.lon_min + self.lon_max) / 2.0)

    @property
    def bounds(self) -> List[List[float]]:
        """Return ``[[south, west], [north, east]]`` for map fitting."""
        return [[self.lat_min, self.lon_min], [self.lat_max, self.lon_max]]

    def contains(self, latitude: float, longitude: float) -> bool:
        """Report whether a coordinate falls inside this region.

        Args:
            latitude: Degrees north.
            longitude: Degrees east.

        Returns:
            ``True`` when the point lies within the bounding box.
        """
        return (
            self.lat_min <= latitude <= self.lat_max
            and self.lon_min <= longitude <= self.lon_max
        )


REGIONS: Final[Mapping[str, Region]] = {
    "Indian Ocean": Region("Indian Ocean", -45.0, 30.0, 20.0, 120.0, zoom=3),
    "Arabian Sea": Region("Arabian Sea", 5.0, 25.0, 50.0, 78.0, zoom=5),
    "Bay of Bengal": Region("Bay of Bengal", 5.0, 23.0, 78.0, 100.0, zoom=5),
    "Equatorial Indian Ocean": Region("Equatorial Indian Ocean", -10.0, 5.0, 45.0, 100.0, zoom=4),
    "Southern Ocean": Region("Southern Ocean", -60.0, -35.0, 20.0, 120.0, zoom=3),
    "Global": Region("Global", -70.0, 70.0, -180.0, 180.0, zoom=2),
}

DEFAULT_REGION: Final[str] = "Indian Ocean"

#: The exact ocean_region strings the backend (shared/regions.py,
#: intelligence_engine, ProfileRepository) knows about. The dashboard's own
#: REGIONS above are a superset used purely for map framing/zoom (they add
#: "Indian Ocean", "Southern Ocean" and "Global" as broader viewing options
#: that don't correspond to a single backend region).
BACKEND_REGION_NAMES: Final[Tuple[str, ...]] = (
    "Arabian Sea",
    "Bay of Bengal",
    "Equatorial Indian Ocean",
    "Southern Indian Ocean",
)

#: Maps a dashboard region key to the single backend region name it stands
#: in for. Keys not present here ("Indian Ocean", "Southern Ocean", "Global")
#: don't correspond to one backend region -- see backend_regions_for().
_UI_TO_BACKEND_REGION: Final[Mapping[str, str]] = {
    "Arabian Sea": "Arabian Sea",
    "Bay of Bengal": "Bay of Bengal",
    "Equatorial Indian Ocean": "Equatorial Indian Ocean",
    "Southern Ocean": "Southern Indian Ocean",
}


def backend_regions_for(ui_region: Optional[str]) -> List[str]:
    """Resolve a dashboard region key to the backend region name(s) it covers.

    Args:
        ui_region: A key from :data:`REGIONS` (or ``None``).

    Returns:
        ``[backend_region]`` for a direct match (e.g. ``"Arabian Sea"``), or
        every :data:`BACKEND_REGION_NAMES` entry for a broad selection such
        as ``"Indian Ocean"``, ``"Southern Ocean"`` (a much larger area than
        the backend's "Southern Indian Ocean" box), ``"Global"``, or an
        unrecognised/``None`` region.
    """
    if ui_region in _UI_TO_BACKEND_REGION:
        return [_UI_TO_BACKEND_REGION[ui_region]]
    return list(BACKEND_REGION_NAMES)

#: Measured variables the dashboard can plot, mapped to display metadata.
VARIABLES: Final[Mapping[str, Dict[str, str]]] = {
    "temperature": {"label": "Temperature", "unit": "degC", "symbol": "T"},
    "salinity": {"label": "Salinity", "unit": "PSU", "symbol": "S"},
    "pressure": {"label": "Pressure", "unit": "dbar", "symbol": "P"},
    "oxygen": {"label": "Dissolved Oxygen", "unit": "umol/kg", "symbol": "O2"},
    "chlorophyll": {"label": "Chlorophyll-a", "unit": "mg/m3", "symbol": "Chl"},
    "ph": {"label": "pH", "unit": "total scale", "symbol": "pH"},
}


def variable_label(variable: str, *, with_unit: bool = True) -> str:
    """Return a display label for a measured variable.

    Args:
        variable: Column name such as ``"temperature"``.
        with_unit: Whether to append the unit in parentheses.

    Returns:
        A label such as ``"Temperature (degC)"``. Unknown variables are
        title-cased rather than rejected.
    """
    meta = VARIABLES.get(variable.lower())
    if meta is None:
        return variable.replace("_", " ").title()
    return f"{meta['label']} ({meta['unit']})" if with_unit else meta["label"]


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #


def format_number(
    value: Optional[float],
    *,
    decimals: int = 2,
    unit: str = "",
    compact: bool = False,
) -> str:
    """Format a numeric value for display, tolerating ``None`` and ``NaN``.

    Args:
        value: The number to render.
        decimals: Decimal places for the non-compact form.
        unit: Optional unit appended after a thin space.
        compact: Abbreviate thousands/millions as ``K``/``M``.

    Returns:
        A display string, or ``"--"`` when the value is missing.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "--"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"

    if compact:
        for threshold, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
            if abs(number) >= threshold:
                text = f"{number / threshold:.1f}{suffix}"
                break
        else:
            text = f"{number:,.0f}"
    else:
        text = f"{number:,.{decimals}f}"

    return f"{text} {unit}".strip() if unit else text


def format_coordinate(latitude: float, longitude: float, *, decimals: int = 3) -> str:
    """Format a coordinate pair in hemisphere notation.

    Args:
        latitude: Degrees north (negative for south).
        longitude: Degrees east (negative for west).
        decimals: Decimal places to show.

    Returns:
        A string such as ``"12.345 N, 72.100 E"``.
    """
    lat_hemisphere = "N" if latitude >= 0 else "S"
    lon_hemisphere = "E" if longitude >= 0 else "W"
    return (
        f"{abs(latitude):.{decimals}f} {lat_hemisphere}, "
        f"{abs(longitude):.{decimals}f} {lon_hemisphere}"
    )


def format_datetime(value: Any, *, fmt: str = "%d %b %Y") -> str:
    """Format a timestamp defensively.

    Args:
        value: Anything pandas can coerce to a timestamp.
        fmt: ``strftime`` pattern.

    Returns:
        The formatted date, or ``"--"`` when the value cannot be parsed.
    """
    if value is None:
        return "--"
    try:
        stamp = pd.to_datetime(value)
    except (ValueError, TypeError):
        return "--"
    if pd.isna(stamp):
        return "--"
    return stamp.strftime(fmt)


def format_delta(current: Optional[float], previous: Optional[float], *, unit: str = "") -> str:
    """Render a signed change between two readings.

    Args:
        current: Latest value.
        previous: Value to compare against.
        unit: Optional unit suffix.

    Returns:
        A string such as ``"+0.42 degC"``, or ``"--"`` if either side is missing.
    """
    if current is None or previous is None:
        return "--"
    try:
        change = float(current) - float(previous)
    except (TypeError, ValueError):
        return "--"
    sign = "+" if change >= 0 else ""
    return f"{sign}{change:,.2f} {unit}".strip()


# --------------------------------------------------------------------------- #
# Dates
# --------------------------------------------------------------------------- #


def default_date_range(days: int = DEFAULT_LOOKBACK_DAYS) -> Tuple[date, date]:
    """Return a sensible default ``(start, end)`` window ending today.

    Args:
        days: Width of the window in days.

    Returns:
        A tuple of dates suitable for ``st.date_input``.
    """
    end = datetime.now(timezone.utc).date()
    return end - timedelta(days=max(days, 1)), end


def normalise_date_range(value: Any, fallback_days: int = DEFAULT_LOOKBACK_DAYS) -> Tuple[date, date]:
    """Coerce a Streamlit date-input result into an ordered ``(start, end)`` pair.

    ``st.date_input`` returns a single date while the user is mid-selection, so
    every caller needs this guard.

    Args:
        value: Raw widget value: a date, or a sequence of one or two dates.
        fallback_days: Window width used when only one date is available.

    Returns:
        An ordered pair of dates.
    """
    if isinstance(value, (list, tuple)):
        if len(value) >= 2 and value[0] and value[1]:
            start, end = value[0], value[1]
            return (start, end) if start <= end else (end, start)
        if len(value) == 1 and value[0]:
            return value[0] - timedelta(days=fallback_days), value[0]
        return default_date_range(fallback_days)
    if isinstance(value, date):
        return value - timedelta(days=fallback_days), value
    return default_date_range(fallback_days)


# --------------------------------------------------------------------------- #
# Colour helpers
# --------------------------------------------------------------------------- #


def score_to_colour(score: float) -> str:
    """Map a 0-100 health score to a semantic palette colour.

    Args:
        score: Ocean health score.

    Returns:
        A hex colour drawn from :data:`dashboard.styles.OCEAN`.
    """
    if score >= 75:
        return OCEAN["good"]
    if score >= 55:
        return OCEAN["teal"]
    if score >= 35:
        return OCEAN["warn"]
    return OCEAN["bad"]


def score_to_label(score: float) -> str:
    """Describe a 0-100 health score in words.

    Args:
        score: Ocean health score.

    Returns:
        One of ``"Excellent"``, ``"Good"``, ``"Moderate"``, ``"Poor"`` or
        ``"Critical"``.
    """
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Good"
    if score >= 50:
        return "Moderate"
    if score >= 30:
        return "Poor"
    return "Critical"


def score_to_tone(score: float) -> str:
    """Map a 0-100 score to a :mod:`dashboard.styles` tone name.

    Args:
        score: Ocean health score.

    Returns:
        ``"good"``, ``"warn"`` or ``"bad"``.
    """
    if score >= 70:
        return "good"
    if score >= 45:
        return "warn"
    return "bad"


def colour_for_index(index: int, palette: Sequence[str]) -> str:
    """Pick a palette colour by position, wrapping around the sequence.

    Args:
        index: Series position.
        palette: Ordered colour sequence.

    Returns:
        A hex colour string.
    """
    return palette[index % len(palette)] if palette else OCEAN["accent"]


# --------------------------------------------------------------------------- #
# DataFrame helpers
# --------------------------------------------------------------------------- #


def filter_profiles(
    frame: pd.DataFrame,
    *,
    region: Optional[str] = None,
    start: Optional[date] = None,
    end: Optional[date] = None,
    max_depth: Optional[float] = None,
    float_ids: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Apply the dashboard's standard filters to a profile table.

    Every filter is optional and silently skipped when the required column is
    absent, so partially populated backend results still render.

    Args:
        frame: Profile observations.
        region: Key into :data:`REGIONS`.
        start: Inclusive lower date bound.
        end: Inclusive upper date bound.
        max_depth: Drop observations deeper than this, in metres.
        float_ids: Restrict to these float identifiers.

    Returns:
        A filtered copy. The input is never mutated.
    """
    if frame is None or frame.empty:
        return pd.DataFrame(columns=frame.columns if frame is not None else [])

    result = frame.copy()

    if region and region in REGIONS and {"latitude", "longitude"} <= set(result.columns):
        box = REGIONS[region]
        result = result[
            result["latitude"].between(box.lat_min, box.lat_max)
            & result["longitude"].between(box.lon_min, box.lon_max)
        ]

    if "time" in result.columns and (start or end):
        stamps = pd.to_datetime(result["time"], errors="coerce")
        if start:
            result = result[stamps >= pd.Timestamp(start)]
            stamps = stamps.loc[result.index]
        if end:
            result = result[stamps <= pd.Timestamp(end) + pd.Timedelta(days=1)]

    if max_depth is not None and "depth" in result.columns:
        result = result[result["depth"] <= max_depth]

    if float_ids and "float_id" in result.columns:
        result = result[result["float_id"].astype(str).isin({str(f) for f in float_ids})]

    return result.reset_index(drop=True)


def summarise_profiles(frame: pd.DataFrame) -> Dict[str, Any]:
    """Compute the headline counts shown in the KPI row.

    Args:
        frame: Filtered profile observations.

    Returns:
        A mapping with ``floats``, ``profiles``, ``observations``,
        ``max_depth``, ``mean_temperature``, ``mean_salinity`` and
        ``latest_time``. Missing inputs yield ``None`` rather than raising.
    """
    if frame is None or frame.empty:
        return {
            "floats": 0,
            "profiles": 0,
            "observations": 0,
            "max_depth": None,
            "mean_temperature": None,
            "mean_salinity": None,
            "latest_time": None,
        }

    def _mean(column: str) -> Optional[float]:
        return float(frame[column].mean()) if column in frame.columns else None

    profiles = 0
    if {"float_id", "cycle_number"} <= set(frame.columns):
        profiles = int(frame.groupby(["float_id", "cycle_number"]).ngroup().nunique())
    elif "float_id" in frame.columns:
        profiles = int(frame["float_id"].nunique())

    return {
        "floats": int(frame["float_id"].nunique()) if "float_id" in frame.columns else 0,
        "profiles": profiles,
        "observations": int(len(frame)),
        "max_depth": float(frame["depth"].max()) if "depth" in frame.columns else None,
        "mean_temperature": _mean("temperature"),
        "mean_salinity": _mean("salinity"),
        "latest_time": frame["time"].max() if "time" in frame.columns else None,
    }


def available_variables(frame: pd.DataFrame) -> List[str]:
    """List the plottable variables actually present in a table.

    Args:
        frame: Profile observations.

    Returns:
        Known variable column names present in ``frame``, in canonical order.
    """
    if frame is None or frame.empty:
        return []
    return [name for name in VARIABLES if name in frame.columns]


def ensure_session_defaults() -> None:
    """Populate the dashboard's session-state keys on first run.

    Safe to call on every rerun; existing values are never overwritten.
    """
    start, end = default_date_range()
    defaults: Dict[str, Any] = {
        SESSION_FILTERS: {
            "region": DEFAULT_REGION,
            "date_start": start,
            "date_end": end,
            "depth_max": 2000.0,
            "variables": ["temperature", "salinity"],
        },
        SESSION_CHAT: [],
        SESSION_HEALTH: None,
        SESSION_REPORTS: [],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def get_filters() -> Dict[str, Any]:
    """Return the active filter set, initialising defaults if necessary.

    Returns:
        The mutable filter mapping held in session state.
    """
    ensure_session_defaults()
    return st.session_state[SESSION_FILTERS]


# --------------------------------------------------------------------------- #
# Demo data (used only when the backend is not importable)
# --------------------------------------------------------------------------- #


def generate_demo_profiles(
    *,
    n_floats: int = 18,
    n_levels: int = 40,
    n_cycles: int = 4,
    region: str = DEFAULT_REGION,
    seed: int = 20260808,
) -> pd.DataFrame:
    """Generate physically plausible synthetic ARGO profiles for demo mode.

    Produces a realistic mixed layer, thermocline and deep structure so the
    charts look scientifically credible while the backend is unavailable. The
    values are **synthetic and must never be presented as findings**.

    Args:
        n_floats: Number of distinct floats.
        n_levels: Depth levels sampled per profile.
        n_cycles: Profiles (cycles) per float.
        region: Key into :data:`REGIONS` bounding the float positions.
        seed: Seed making the output reproducible across reruns.

    Returns:
        A tidy observation-per-row DataFrame with the columns the dashboard
        expects from :class:`ProfileRepository`.
    """
    rng = np.random.default_rng(seed)
    box = REGIONS.get(region, REGIONS[DEFAULT_REGION])
    depths = np.linspace(0.0, 2000.0, n_levels)
    now = datetime.now(timezone.utc)

    records: List[Dict[str, Any]] = []
    for float_index in range(n_floats):
        float_id = f"29{27000 + float_index * 137:05d}"
        base_lat = rng.uniform(box.lat_min + 1.0, box.lat_max - 1.0)
        base_lon = rng.uniform(box.lon_min + 1.0, box.lon_max - 1.0)
        surface_temp = rng.uniform(26.0, 30.5)
        surface_salt = rng.uniform(33.4, 36.4)
        thermocline = rng.uniform(90.0, 190.0)

        for cycle in range(n_cycles):
            # Floats drift slowly between cycles.
            latitude = base_lat + rng.normal(0.0, 0.22) * (cycle + 1)
            longitude = base_lon + rng.normal(0.0, 0.28) * (cycle + 1)
            stamp = now - timedelta(days=(n_cycles - cycle) * 10, hours=float(rng.integers(0, 24)))

            sigmoid = 1.0 / (1.0 + np.exp(-(depths - thermocline) / 55.0))
            temperature = surface_temp - (surface_temp - 3.2) * sigmoid
            temperature += rng.normal(0.0, 0.12, n_levels)

            salinity = surface_salt + 0.85 * sigmoid - 0.35 * np.exp(-depths / 120.0)
            salinity += rng.normal(0.0, 0.035, n_levels)

            # Oxygen minimum zone between roughly 200 m and 800 m.
            oxygen = 40.0 + 180.0 * np.exp(-((depths - 30.0) ** 2) / (2 * 110.0**2))
            oxygen += 90.0 * (depths / 2000.0) ** 1.6
            oxygen += rng.normal(0.0, 4.0, n_levels)

            chlorophyll = 0.85 * np.exp(-((depths - 55.0) ** 2) / (2 * 38.0**2))
            chlorophyll = np.clip(chlorophyll + rng.normal(0.0, 0.02, n_levels), 0.0, None)

            ph = 8.09 - 0.30 * sigmoid + rng.normal(0.0, 0.012, n_levels)
            pressure = depths * 1.0068 + rng.normal(0.0, 0.4, n_levels)

            for level in range(n_levels):
                records.append(
                    {
                        "float_id": float_id,
                        "cycle_number": cycle + 1,
                        "time": stamp,
                        "latitude": round(float(latitude), 4),
                        "longitude": round(float(longitude), 4),
                        "depth": round(float(depths[level]), 1),
                        "pressure": round(float(pressure[level]), 2),
                        "temperature": round(float(temperature[level]), 3),
                        "salinity": round(float(salinity[level]), 3),
                        "oxygen": round(float(oxygen[level]), 2),
                        "chlorophyll": round(float(chlorophyll[level]), 4),
                        "ph": round(float(ph[level]), 3),
                    }
                )

    frame = pd.DataFrame.from_records(records)
    frame["time"] = pd.to_datetime(frame["time"]).dt.tz_localize(None)
    logger.info("Generated %d synthetic observations for demo mode", len(frame))
    return frame
