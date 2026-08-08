"""Interactive Folium map of ARGO float positions and trajectories.

Renders the Explore tab's geospatial view: one marker per float at its most
recent position, an optional trajectory polyline per float, clustered markers
for dense regions, and a layer control to toggle base maps and overlays.

Backend contract
----------------
Profiles are read through the existing repository interface:

* ``ProfileRepository.get_profiles_by_region()``
* ``ProfileRepository.get_profiles_near()``

When the backend cannot be imported, a signature-identical stub serves
synthetic profiles so the view still renders in demo mode.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, Final, List, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st

from dashboard.styles import OCEAN, section_header
from dashboard.utils import (
    DEFAULT_REGION,
    REGIONS,
    backend_regions_for,
    format_coordinate,
    format_datetime,
    format_number,
    generate_demo_profiles,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Optional mapping dependencies
# --------------------------------------------------------------------------- #

try:
    import folium
    from folium.plugins import Fullscreen, MarkerCluster

    FOLIUM_AVAILABLE: Final[bool] = True
except ImportError:  # pragma: no cover - environment dependent
    folium = None  # type: ignore[assignment]
    MarkerCluster = None  # type: ignore[assignment]
    Fullscreen = None  # type: ignore[assignment]
    FOLIUM_AVAILABLE = False

try:
    from streamlit_folium import st_folium

    STREAMLIT_FOLIUM_AVAILABLE: Final[bool] = True
except ImportError:  # pragma: no cover - environment dependent
    st_folium = None  # type: ignore[assignment]
    STREAMLIT_FOLIUM_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Backend binding
# --------------------------------------------------------------------------- #

try:
    from database.repository import ProfileRepository as _BackendProfileRepository  # type: ignore[import-not-found]

    BACKEND_AVAILABLE: Final[bool] = True

    def _profile_rows_to_frame(
        repository: "_BackendProfileRepository",
        profile_rows: List[Dict[str, Any]],
        *,
        distance_col: bool = False,
    ) -> pd.DataFrame:
        """Flatten profile + measurement rows into the dashboard's wide table.

        ``ProfileRepository.get_profiles_by_region()``/``get_profiles_near()``
        return one row per *profile* (float_id, cycle_number, profile_date,
        latitude, longitude, id, ...). The dashboard needs one row per
        *depth-level observation* (with temperature/salinity/oxygen/etc.), so
        this joins each profile against
        ``ProfileRepository.get_measurements_for_profile()`` and renames the
        backend's measurement columns to the names every dashboard module
        (utils.VARIABLES, map_view, profile_plots) expects.
        """
        records: List[Dict[str, Any]] = []
        for profile in profile_rows:
            profile_id = profile.get("id")
            try:
                measurements = repository.get_measurements_for_profile(profile_id) if profile_id is not None else []
            except Exception:  # noqa: BLE001 - one bad profile shouldn't blank the page
                logger.exception("get_measurements_for_profile failed for profile_id=%s", profile_id)
                measurements = []

            base: Dict[str, Any] = {
                "float_id": profile.get("float_id"),
                "cycle_number": profile.get("cycle_number"),
                "time": profile.get("profile_date"),
                "latitude": profile.get("latitude"),
                "longitude": profile.get("longitude"),
                "ocean_region": profile.get("ocean_region"),
            }
            if distance_col and "distance_km" in profile:
                base["distance_km"] = profile["distance_km"]

            if not measurements:
                # Keep the float/position visible on the map even without
                # depth-level readings yet.
                records.append(dict(base))
                continue

            for m in measurements:
                records.append(
                    {
                        **base,
                        "depth": m.get("depth_m"),
                        "pressure": m.get("pressure_dbar"),
                        "temperature": m.get("temperature_c"),
                        "salinity": m.get("salinity_psu"),
                        "oxygen": m.get("dissolved_oxygen"),
                        "chlorophyll": m.get("chlorophyll"),
                        "ph": m.get("ph"),
                    }
                )
        return pd.DataFrame.from_records(records)

    class ProfileRepository:  # type: ignore[no-redef]
        """Adapts the backend's locked ``ProfileRepository`` (Module 2) to the
        wide, per-observation shape the dashboard renders.

        The real repository is profile-level (one row per dive/surface
        cycle); measurements live in a separate call. This wrapper performs
        that join, maps dashboard region keys onto the backend's exact
        ``shared.regions.REGION_NAMES`` strings, and applies ``limit`` after
        fetching -- the real repository has no limit parameter.
        """

        def __init__(self) -> None:
            self._backend = _BackendProfileRepository()

        def get_profiles_by_region(
            self,
            region: Optional[str] = None,
            start_date: Optional[date] = None,
            end_date: Optional[date] = None,
            limit: Optional[int] = None,
        ) -> pd.DataFrame:
            """Fetch profiles+measurements for a region and date window.

            Args:
                region: A dashboard region key (see dashboard.utils.REGIONS);
                    mapped onto one or more exact backend region names.
                start_date: Inclusive lower date bound (defaults to a wide
                    historical window when omitted, since the backend
                    requires both bounds).
                end_date: Inclusive upper date bound (defaults to today).
                limit: Maximum observations returned, applied client-side.

            Returns:
                Observation-per-row DataFrame, or an empty frame on error.
            """
            start = start_date or date(2000, 1, 1)
            end = end_date or date.today()

            frames: List[pd.DataFrame] = []
            for backend_region in backend_regions_for(region):
                try:
                    profiles = self._backend.get_profiles_by_region(backend_region, start, end)
                except Exception:  # noqa: BLE001 - surface backend failures as empty state
                    logger.exception(
                        "ProfileRepository.get_profiles_by_region failed for region=%s", backend_region
                    )
                    continue
                if profiles:
                    frames.append(_profile_rows_to_frame(self._backend, profiles))

            if not frames:
                return pd.DataFrame()
            frame = pd.concat(frames, ignore_index=True)
            if limit is not None:
                frame = frame.head(int(limit))
            return frame

        def get_profiles_near(
            self,
            latitude: float,
            longitude: float,
            radius_km: float = 200.0,
            limit: Optional[int] = None,
        ) -> pd.DataFrame:
            """Fetch profiles+measurements within a radius of a coordinate.

            Args:
                latitude: Centre latitude in degrees north.
                longitude: Centre longitude in degrees east.
                radius_km: Search radius in kilometres.
                limit: Maximum observations returned, applied client-side.

            Returns:
                Observation-per-row DataFrame, or an empty frame on error.
            """
            try:
                profiles = self._backend.get_profiles_near(latitude, longitude, radius_km)
            except Exception:  # noqa: BLE001 - surface backend failures as empty state
                logger.exception("ProfileRepository.get_profiles_near failed")
                return pd.DataFrame()

            frame = _profile_rows_to_frame(self._backend, profiles, distance_col=True)
            if limit is not None:
                frame = frame.head(int(limit))
            return frame

except Exception:  # noqa: BLE001 - covers ImportError *and* a misconfigured
    # backend (e.g. DATABASE_URL not set, which raises KeyError at import
    # time from config.py) so the dashboard degrades to demo mode instead
    # of crashing.
    BACKEND_AVAILABLE = False

    class ProfileRepository:  # type: ignore[no-redef]
        """Interface-compatible stub for the backend's profile repository.

        Mirrors the real signatures exactly so that restoring the import is the
        only change required to switch to live data. Every method returns
        synthetic observations and is **not** a reimplementation of backend
        logic.
        """

        def get_profiles_by_region(
            self,
            region: Optional[str] = None,
            start_date: Optional[date] = None,
            end_date: Optional[date] = None,
            limit: Optional[int] = None,
        ) -> pd.DataFrame:
            """Return synthetic profiles bounded by a named region.

            Args:
                region: Region name.
                start_date: Inclusive lower date bound.
                end_date: Inclusive upper date bound.
                limit: Maximum observations returned.

            Returns:
                A synthetic observation-per-row DataFrame.
            """
            frame = generate_demo_profiles(region=region or DEFAULT_REGION)
            if start_date is not None:
                frame = frame[frame["time"] >= pd.Timestamp(start_date)]
            if end_date is not None:
                frame = frame[frame["time"] <= pd.Timestamp(end_date) + pd.Timedelta(days=1)]
            if limit is not None:
                frame = frame.head(int(limit))
            return frame.reset_index(drop=True)

        def get_profiles_near(
            self,
            latitude: float,
            longitude: float,
            radius_km: float = 200.0,
            limit: Optional[int] = None,
        ) -> pd.DataFrame:
            """Return synthetic profiles within a radius of a coordinate.

            Args:
                latitude: Centre latitude in degrees north.
                longitude: Centre longitude in degrees east.
                radius_km: Search radius in kilometres.
                limit: Maximum observations returned.

            Returns:
                A synthetic observation-per-row DataFrame.
            """
            frame = generate_demo_profiles()
            # Rough degree-space filter; the real repository uses PostGIS.
            degrees = max(radius_km, 1.0) / 111.0
            frame = frame[
                (frame["latitude"] - latitude).abs().le(degrees)
                & (frame["longitude"] - longitude).abs().le(degrees)
            ]
            if limit is not None:
                frame = frame.head(int(limit))
            return frame.reset_index(drop=True)


def available_date_bounds() -> Optional[Tuple[date, date]]:
    """Return the real ``(earliest, latest)`` profile_date in the database.

    Used to size the sidebar's date picker to the data that's actually
    present (e.g. archival ARGO floats from the early 2000s), instead of
    Streamlit's ``st.date_input`` default of "10 years before the current
    value", which silently hides anything older.

    Returns:
        ``(min_date, max_date)``, or ``None`` in demo mode / on any query
        failure (callers should fall back to a fixed window in that case).
    """
    if not BACKEND_AVAILABLE:
        return None
    try:
        repository = _BackendProfileRepository()
        rows = repository.run_raw_query(
            "SELECT MIN(profile_date) AS min_date, MAX(profile_date) AS max_date FROM profiles"
        )
    except Exception:  # noqa: BLE001 - a bad bounds query shouldn't block the picker
        logger.exception("available_date_bounds query failed")
        return None

    if not rows or rows[0].get("min_date") is None or rows[0].get("max_date") is None:
        return None

    min_raw, max_raw = rows[0]["min_date"], rows[0]["max_date"]
    try:
        min_date = pd.to_datetime(min_raw).date()
        max_date = pd.to_datetime(max_raw).date()
    except (TypeError, ValueError):
        logger.warning("available_date_bounds could not parse %r / %r", min_raw, max_raw)
        return None
    return min_date, max_date


BASE_LAYERS: Final[Dict[str, Dict[str, str]]] = {
    "Ocean Basemap": {
        "tiles": (
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}"
        ),
        "attr": "Esri Ocean Basemap",
    },
    "Dark Matter": {
        "tiles": "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        "attr": "CARTO",
    },
    "Satellite": {
        "tiles": (
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        "attr": "Esri World Imagery",
    },
}


# --------------------------------------------------------------------------- #
# Data access
# --------------------------------------------------------------------------- #


@st.cache_resource(show_spinner=False)
def get_repository() -> ProfileRepository:
    """Return the shared profile repository instance.

    Held in ``st.cache_resource`` so the underlying database connection is
    created once per server rather than on every Streamlit rerun.

    Returns:
        The repository, real or stubbed depending on backend availability.
    """
    return ProfileRepository()


@st.cache_data(show_spinner="Loading ARGO profiles...", ttl=600)
def load_profiles(
    region: str,
    start_date: date,
    end_date: date,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Fetch profiles for a region and date window.

    Args:
        region: Region name from :data:`dashboard.utils.REGIONS`.
        start_date: Inclusive lower date bound.
        end_date: Inclusive upper date bound.
        limit: Maximum observations to request.

    Returns:
        Observation-per-row profiles, or an empty frame if the backend errors.
    """
    repository = get_repository()
    try:
        frame = repository.get_profiles_by_region(
            region=region, start_date=start_date, end_date=end_date, limit=limit
        )
    except Exception:  # noqa: BLE001 - surface backend failures as empty state
        logger.exception("ProfileRepository.get_profiles_by_region failed")
        return pd.DataFrame()
    return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)


def latest_float_positions(frame: pd.DataFrame) -> pd.DataFrame:
    """Reduce observations to one row per float at its most recent cycle.

    Args:
        frame: Observation-per-row profiles.

    Returns:
        One row per float carrying position, time and summary measurements.
    """
    required = {"float_id", "latitude", "longitude"}
    if frame is None or frame.empty or not required <= set(frame.columns):
        return pd.DataFrame()

    sort_columns = [column for column in ("time", "cycle_number") if column in frame.columns]
    ordered = frame.sort_values(sort_columns) if sort_columns else frame

    aggregations: Dict[str, Any] = {"latitude": "last", "longitude": "last"}
    for column in ("time", "cycle_number"):
        if column in ordered.columns:
            aggregations[column] = "last"
    for column in ("temperature", "salinity", "oxygen"):
        if column in ordered.columns:
            aggregations[column] = "mean"
    if "depth" in ordered.columns:
        aggregations["depth"] = "max"

    positions = ordered.groupby("float_id", as_index=False).agg(aggregations)
    return positions


def float_trajectories(frame: pd.DataFrame) -> Dict[str, List[Tuple[float, float]]]:
    """Build an ordered coordinate path per float.

    Args:
        frame: Observation-per-row profiles.

    Returns:
        A mapping of float identifier to its ordered ``(lat, lon)`` path.
        Floats with fewer than two distinct positions are omitted.
    """
    required = {"float_id", "latitude", "longitude"}
    if frame is None or frame.empty or not required <= set(frame.columns):
        return {}

    sort_columns = [column for column in ("time", "cycle_number") if column in frame.columns]
    columns = ["float_id", "latitude", "longitude"] + sort_columns
    unique = frame[columns].drop_duplicates()
    if sort_columns:
        unique = unique.sort_values(["float_id", *sort_columns])

    paths: Dict[str, List[Tuple[float, float]]] = {}
    for float_id, group in unique.groupby("float_id"):
        points = list(zip(group["latitude"].astype(float), group["longitude"].astype(float)))
        if len(points) > 1:
            paths[str(float_id)] = points
    return paths


# --------------------------------------------------------------------------- #
# Map construction
# --------------------------------------------------------------------------- #


def _popup_html(row: pd.Series) -> str:
    """Build the HTML shown when a float marker is clicked.

    Args:
        row: One row from :func:`latest_float_positions`.

    Returns:
        An HTML fragment sized for a Folium popup.
    """
    lines = [
        f'<div style="font-family:sans-serif;min-width:212px;color:#0F1E33">',
        f'<div style="font-size:14px;font-weight:700;margin-bottom:6px">'
        f'Float {row.get("float_id", "unknown")}</div>',
        f'<div style="font-size:12px;line-height:1.65">',
        f'<b>Position:</b> {format_coordinate(float(row["latitude"]), float(row["longitude"]))}<br>',
    ]
    if "time" in row and pd.notna(row["time"]):
        lines.append(f'<b>Last profile:</b> {format_datetime(row["time"])}<br>')
    if "cycle_number" in row and pd.notna(row["cycle_number"]):
        lines.append(f'<b>Cycle:</b> {int(row["cycle_number"])}<br>')
    if "temperature" in row and pd.notna(row["temperature"]):
        lines.append(f'<b>Mean temp:</b> {format_number(row["temperature"], decimals=2, unit="degC")}<br>')
    if "salinity" in row and pd.notna(row["salinity"]):
        lines.append(f'<b>Mean salinity:</b> {format_number(row["salinity"], decimals=2, unit="PSU")}<br>')
    if "depth" in row and pd.notna(row["depth"]):
        lines.append(f'<b>Max depth:</b> {format_number(row["depth"], decimals=0, unit="m")}')
    lines.append("</div></div>")
    return "".join(lines)


def build_map(
    frame: pd.DataFrame,
    *,
    region: str = DEFAULT_REGION,
    show_trajectories: bool = True,
    cluster_markers: bool = True,
) -> "folium.Map":
    """Construct the Folium map for the current selection.

    Args:
        frame: Observation-per-row profiles.
        region: Region name used to centre and zoom the initial view.
        show_trajectories: Draw a polyline per float across its cycles.
        cluster_markers: Group nearby markers into expandable clusters.

    Returns:
        A configured :class:`folium.Map`.

    Raises:
        RuntimeError: If Folium is not installed.
    """
    if not FOLIUM_AVAILABLE:
        raise RuntimeError("Folium is not installed; install 'folium' to build maps.")

    box = REGIONS.get(region, REGIONS[DEFAULT_REGION])
    ocean = BASE_LAYERS["Ocean Basemap"]

    fmap = folium.Map(
        location=list(box.center),
        zoom_start=box.zoom,
        tiles=ocean["tiles"],
        attr=ocean["attr"],
        name="Ocean Basemap",
        control_scale=True,
        prefer_canvas=True,
    )
    for name, layer in BASE_LAYERS.items():
        if name != "Ocean Basemap":
            folium.TileLayer(tiles=layer["tiles"], attr=layer["attr"], name=name).add_to(fmap)

    positions = latest_float_positions(frame)

    if not positions.empty:
        marker_layer = folium.FeatureGroup(name="ARGO Floats", show=True)
        container = MarkerCluster().add_to(marker_layer) if cluster_markers else marker_layer

        for _, row in positions.iterrows():
            folium.CircleMarker(
                location=[float(row["latitude"]), float(row["longitude"])],
                radius=7,
                color=OCEAN["accent"],
                weight=2,
                fill=True,
                fill_color=OCEAN["accent"],
                fill_opacity=0.72,
                popup=folium.Popup(_popup_html(row), max_width=280),
                tooltip=f"Float {row.get('float_id', '')}",
            ).add_to(container)

        marker_layer.add_to(fmap)

        if show_trajectories:
            trajectory_layer = folium.FeatureGroup(name="Float Trajectories", show=True)
            for float_id, points in float_trajectories(frame).items():
                folium.PolyLine(
                    locations=points,
                    color=OCEAN["teal"],
                    weight=2,
                    opacity=0.62,
                    tooltip=f"Trajectory - Float {float_id}",
                ).add_to(trajectory_layer)
                folium.CircleMarker(
                    location=points[0],
                    radius=3.5,
                    color=OCEAN["warn"],
                    fill=True,
                    fill_opacity=0.9,
                    tooltip=f"Deployment - Float {float_id}",
                ).add_to(trajectory_layer)
            trajectory_layer.add_to(fmap)

        fmap.fit_bounds(
            [
                [positions["latitude"].min(), positions["longitude"].min()],
                [positions["latitude"].max(), positions["longitude"].max()],
            ],
            padding=(28, 28),
        )
    else:
        fmap.fit_bounds(box.bounds)

    if Fullscreen is not None:
        Fullscreen(position="topright").add_to(fmap)
    folium.LayerControl(collapsed=True, position="topright").add_to(fmap)
    return fmap


# --------------------------------------------------------------------------- #
# Streamlit rendering
# --------------------------------------------------------------------------- #


def _render_fallback_map(frame: pd.DataFrame) -> None:
    """Render a basic scatter map when Folium is unavailable.

    Uses Streamlit's built-in map so the Explore tab still shows float
    positions rather than an error page.

    Args:
        frame: Observation-per-row profiles.
    """
    st.warning(
        "Folium is not installed, so the interactive map is unavailable. "
        "Install it with `pip install folium streamlit-folium` for trajectories, "
        "clustering and layer controls. Showing basic positions meanwhile.",
        icon=":material/map:",
    )
    positions = latest_float_positions(frame)
    if positions.empty:
        st.info("No float positions to display for the current filters.")
        return
    st.map(positions[["latitude", "longitude"]], size=24, color="#22D3EE")


def render_map_panel(
    frame: pd.DataFrame,
    *,
    region: str = DEFAULT_REGION,
    height: int = 520,
    key: str = "om_map",
) -> Optional[Dict[str, Any]]:
    """Render the map panel with its display controls.

    Args:
        frame: Observation-per-row profiles for the active filters.
        region: Region name used for the initial view.
        height: Map height in pixels.
        key: Streamlit widget key namespace.

    Returns:
        The ``st_folium`` interaction payload (last clicked object, current
        bounds), or ``None`` when the interactive map could not be rendered.
    """
    section_header(
        "ARGO Float Positions",
        "Latest reported position per float. Click a marker for profile details; "
        "toggle overlays from the layer control.",
    )

    if not FOLIUM_AVAILABLE or not STREAMLIT_FOLIUM_AVAILABLE:
        _render_fallback_map(frame)
        return None

    controls = st.columns([1, 1, 2])
    with controls[0]:
        show_trajectories = st.toggle("Trajectories", value=True, key=f"{key}_traj")
    with controls[1]:
        cluster_markers = st.toggle("Cluster markers", value=True, key=f"{key}_cluster")
    with controls[2]:
        positions = latest_float_positions(frame)
        st.caption(
            f"{len(positions)} float(s) - {len(frame):,} observations in view"
            if not positions.empty
            else "No floats match the current filters."
        )

    try:
        fmap = build_map(
            frame,
            region=region,
            show_trajectories=show_trajectories,
            cluster_markers=cluster_markers,
        )
    except RuntimeError:
        logger.exception("Map construction failed")
        _render_fallback_map(frame)
        return None

    return st_folium(fmap, height=height, use_container_width=True, key=f"{key}_canvas")


def render_nearby_lookup(*, key: str = "om_nearby") -> Optional[pd.DataFrame]:
    """Render the proximity search that wraps ``get_profiles_near()``.

    Args:
        key: Streamlit widget key namespace.

    Returns:
        The profiles found near the requested coordinate, or ``None`` when the
        user has not run a search this session.
    """
    with st.expander("Search profiles near a coordinate", expanded=False):
        columns = st.columns([1, 1, 1, 1])
        with columns[0]:
            latitude = st.number_input("Latitude", -90.0, 90.0, 12.5, 0.5, key=f"{key}_lat")
        with columns[1]:
            longitude = st.number_input("Longitude", -180.0, 180.0, 72.0, 0.5, key=f"{key}_lon")
        with columns[2]:
            radius = st.number_input("Radius (km)", 10.0, 2000.0, 250.0, 10.0, key=f"{key}_radius")
        with columns[3]:
            st.write("")
            search = st.button("Search", key=f"{key}_go", width="stretch")

        if not search:
            return None

        try:
            found = get_repository().get_profiles_near(
                latitude=latitude, longitude=longitude, radius_km=radius
            )
        except Exception:  # noqa: BLE001 - surface backend failures in the UI
            logger.exception("ProfileRepository.get_profiles_near failed")
            st.error("The proximity search failed. Check the backend connection and logs.")
            return None

        if found is None or len(found) == 0:
            st.info(f"No profiles found within {radius:.0f} km of that position.")
            return None

        st.success(f"Found {len(found):,} observations within {radius:.0f} km.")
        st.dataframe(found.head(200), width="stretch", height=280)
        return found
