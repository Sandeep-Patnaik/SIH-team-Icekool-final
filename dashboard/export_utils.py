"""Export and download helpers for the OceanMind AI dashboard.

This module owns every "get the data out of the dashboard" concern: turning an
in-memory :class:`pandas.DataFrame` into the interchange formats marine
researchers actually exchange -- CSV, CF-styled NetCDF and delimited/aligned
ASCII -- and rendering the Streamlit controls that hand those bytes to the user.

Design notes
------------
* The serialisation functions (:func:`to_csv_bytes`, :func:`to_ascii_bytes`,
  :func:`to_netcdf_bytes`) are **pure**: DataFrame in, ``bytes`` out. They carry
  no Streamlit dependency and can be unit tested head-less.
* Caching lives at the rendering layer (:func:`render_export_bar`), so the
  expensive NetCDF encode runs once per unique frame rather than on every
  Streamlit rerun.
* ``xarray`` is an optional dependency. When it is absent the NetCDF export
  degrades to a disabled control with an explanatory tooltip instead of raising
  at import time.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, Final, Mapping, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st

try:  # Optional dependency -- only required for the NetCDF exporter.
    import xarray as xr

    _XARRAY_IMPORT_ERROR: Optional[str] = None
except ImportError as exc:  # pragma: no cover - environment dependent
    xr = None  # type: ignore[assignment]
    _XARRAY_IMPORT_ERROR = str(exc)


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

MIME_CSV: Final[str] = "text/csv"
MIME_NETCDF: Final[str] = "application/x-netcdf"
MIME_ASCII: Final[str] = "text/plain"

DEFAULT_FLOAT_FORMAT: Final[str] = "%.4f"
DEFAULT_TIMESTAMP_FORMAT: Final[str] = "%Y%m%dT%H%M%SZ"

INSTITUTION: Final[str] = "OceanMind AI"
CF_CONVENTIONS: Final[str] = "CF-1.8"

#: NetCDF engines in order of preference. ``netcdf4``/``h5netcdf`` produce
#: NETCDF4 files with proper string support; ``scipy`` is a NETCDF3 last resort.
_NETCDF_ENGINE_CANDIDATES: Final[Tuple[Tuple[str, str], ...]] = (
    ("netcdf4", "netCDF4"),
    ("h5netcdf", "h5netcdf"),
    ("scipy", "scipy"),
)

#: Minimal CF metadata for the ARGO variables the dashboard surfaces. Columns
#: absent from this mapping are still exported, just without rich attributes.
CF_VARIABLE_ATTRIBUTES: Final[Mapping[str, Mapping[str, str]]] = {
    "temperature": {
        "standard_name": "sea_water_temperature",
        "long_name": "Sea water temperature",
        "units": "degree_Celsius",
    },
    "salinity": {
        "standard_name": "sea_water_practical_salinity",
        "long_name": "Sea water practical salinity",
        "units": "psu",
    },
    "pressure": {
        "standard_name": "sea_water_pressure",
        "long_name": "Sea water pressure",
        "units": "decibar",
    },
    "depth": {
        "standard_name": "depth",
        "long_name": "Depth below sea surface",
        "units": "m",
        "positive": "down",
    },
    "latitude": {
        "standard_name": "latitude",
        "long_name": "Latitude of the profile",
        "units": "degrees_north",
    },
    "longitude": {
        "standard_name": "longitude",
        "long_name": "Longitude of the profile",
        "units": "degrees_east",
    },
    "oxygen": {
        "standard_name": "moles_of_oxygen_per_unit_mass_in_sea_water",
        "long_name": "Dissolved oxygen",
        "units": "micromole kg-1",
    },
    "chlorophyll": {
        "standard_name": "mass_concentration_of_chlorophyll_in_sea_water",
        "long_name": "Chlorophyll-a concentration",
        "units": "mg m-3",
    },
    "ph": {
        "standard_name": "sea_water_ph_reported_on_total_scale",
        "long_name": "Sea water pH on the total scale",
        "units": "1",
    },
    "float_id": {"long_name": "ARGO float platform identifier"},
    "cycle_number": {"long_name": "ARGO profile cycle number", "units": "1"},
    "time": {"standard_name": "time", "long_name": "Profile measurement time"},
}


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class ExportError(RuntimeError):
    """Raised when a dataset cannot be serialised into the requested format."""


class ExportUnavailableError(ExportError):
    """Raised when an export format's optional dependency is not installed."""


# --------------------------------------------------------------------------- #
# Capability probing
# --------------------------------------------------------------------------- #


def is_netcdf_available() -> bool:
    """Return ``True`` when the environment can write NetCDF files.

    Returns:
        ``True`` if both ``xarray`` and at least one supported NetCDF backend
        engine are importable, ``False`` otherwise.
    """
    if xr is None:
        return False
    return _find_netcdf_engine() is not None


def _find_netcdf_engine() -> Optional[str]:
    """Return the name of the best available xarray NetCDF engine.

    Returns:
        The engine name (``"netcdf4"``, ``"h5netcdf"`` or ``"scipy"``), or
        ``None`` when no supported backend is installed.
    """
    from importlib.util import find_spec

    for engine, module_name in _NETCDF_ENGINE_CANDIDATES:
        try:
            if find_spec(module_name) is not None:
                return engine
        except (ImportError, ValueError):  # pragma: no cover - defensive
            continue
    return None


def netcdf_unavailable_reason() -> str:
    """Explain, in user-facing language, why NetCDF export is disabled.

    Returns:
        A short sentence naming the missing dependency. Returns an empty string
        when NetCDF export is in fact available.
    """
    if xr is None:
        detail = f" ({_XARRAY_IMPORT_ERROR})" if _XARRAY_IMPORT_ERROR else ""
        return f"NetCDF export requires the 'xarray' package{detail}."
    if _find_netcdf_engine() is None:
        return (
            "NetCDF export requires a backend engine. "
            "Install 'netCDF4' (recommended), 'h5netcdf' or 'scipy'."
        )
    return ""


# --------------------------------------------------------------------------- #
# Filenames
# --------------------------------------------------------------------------- #


def build_filename(
    prefix: str,
    extension: str,
    *,
    timestamp: Optional[datetime] = None,
) -> str:
    """Compose a timestamped, filesystem-safe export filename.

    Args:
        prefix: Logical dataset name, e.g. ``"argo_profiles"``. Characters that
            are awkward in filenames are replaced with underscores.
        extension: File extension with or without a leading dot.
        timestamp: Moment to stamp into the name. Defaults to "now" in UTC.

    Returns:
        A name such as ``"argo_profiles_20260808T101500Z.csv"``.
    """
    moment = timestamp or datetime.now(timezone.utc)
    safe_prefix = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_" for char in prefix.strip()
    ).strip("_")
    safe_prefix = safe_prefix or "oceanmind_export"
    suffix = extension if extension.startswith(".") else f".{extension}"
    return f"{safe_prefix}_{moment.strftime(DEFAULT_TIMESTAMP_FORMAT)}{suffix}"


# --------------------------------------------------------------------------- #
# Serialisers (pure -- no Streamlit dependency)
# --------------------------------------------------------------------------- #


def to_csv_bytes(
    frame: pd.DataFrame,
    *,
    include_index: bool = False,
    float_format: Optional[str] = DEFAULT_FLOAT_FORMAT,
) -> bytes:
    """Serialise a DataFrame to UTF-8 encoded CSV.

    Args:
        frame: Table to serialise.
        include_index: Whether the DataFrame index becomes a leading column.
        float_format: ``printf``-style format applied to floating point
            columns, or ``None`` to keep pandas' full repr.

    Returns:
        The CSV document as UTF-8 bytes.

    Raises:
        ExportError: If pandas cannot serialise the frame.
    """
    _validate_frame(frame)
    try:
        text = frame.to_csv(index=include_index, float_format=float_format)
    except (ValueError, TypeError) as exc:
        raise ExportError(f"Unable to serialise dataset to CSV: {exc}") from exc
    logger.debug("Serialised %d rows to CSV", len(frame))
    return text.encode("utf-8")


def to_ascii_bytes(
    frame: pd.DataFrame,
    *,
    delimiter: Optional[str] = None,
    float_format: str = DEFAULT_FLOAT_FORMAT,
    title: str = "OceanMind AI data export",
    extra_header: Optional[Mapping[str, str]] = None,
) -> bytes:
    """Serialise a DataFrame to a commented, 7-bit ASCII text table.

    The output opens with a ``#``-prefixed metadata block (the convention used
    by most oceanographic ASCII products) followed by the data table itself.

    Args:
        frame: Table to serialise.
        delimiter: Column separator. ``None`` -- the default -- produces a
            whitespace-aligned fixed-width table.
        float_format: ``printf``-style format for floating point values.
        title: Human readable title written into the header block.
        extra_header: Additional ``key: value`` pairs appended to the header.

    Returns:
        The ASCII document as bytes. Any non-ASCII character present in the
        data is replaced with ``?`` so the result honours its "ASCII" contract.

    Raises:
        ExportError: If pandas cannot render the frame.
    """
    _validate_frame(frame)

    header: Dict[str, str] = {
        "title": title,
        "institution": INSTITUTION,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": str(len(frame)),
        "columns": ", ".join(str(column) for column in frame.columns),
        "format": "fixed-width ASCII" if delimiter is None else f"delimited ({delimiter!r})",
    }
    if extra_header:
        header.update({str(key): str(value) for key, value in extra_header.items()})

    header_lines = [f"# {key}: {value}" for key, value in header.items()]

    try:
        if delimiter is None:
            body = frame.to_string(index=False, float_format=lambda v: float_format % v)
        else:
            body = frame.to_csv(index=False, sep=delimiter, float_format=float_format)
    except (ValueError, TypeError) as exc:
        raise ExportError(f"Unable to serialise dataset to ASCII: {exc}") from exc

    document = "\n".join(header_lines) + "\n#\n" + body.rstrip("\n") + "\n"
    logger.debug("Serialised %d rows to ASCII", len(frame))
    return document.encode("ascii", errors="replace")


def to_netcdf_bytes(
    frame: pd.DataFrame,
    *,
    title: str = "OceanMind AI ARGO profile export",
    global_attributes: Optional[Mapping[str, str]] = None,
) -> bytes:
    """Serialise a DataFrame to a CF-styled NetCDF document.

    Every column becomes a variable over a single ``obs`` dimension. Recognised
    ARGO variables receive CF ``standard_name``/``units`` attributes; object
    columns are cast to strings so they survive the encoding.

    Args:
        frame: Table to serialise.
        title: Value of the NetCDF ``title`` global attribute.
        global_attributes: Additional global attributes merged into the file.

    Returns:
        The NetCDF file as bytes.

    Raises:
        ExportUnavailableError: If ``xarray`` or a NetCDF engine is missing.
        ExportError: If the dataset cannot be encoded.
    """
    _validate_frame(frame)

    if xr is None or (engine := _find_netcdf_engine()) is None:
        raise ExportUnavailableError(netcdf_unavailable_reason())

    dataset = _build_dataset(frame, title=title, global_attributes=global_attributes)

    handle, path = tempfile.mkstemp(suffix=".nc", prefix="oceanmind_")
    os.close(handle)
    try:
        dataset.to_netcdf(path, engine=engine)
        with open(path, "rb") as stream:
            payload = stream.read()
    except (ValueError, TypeError, OSError) as exc:
        raise ExportError(
            f"Unable to encode dataset as NetCDF using the '{engine}' engine: {exc}"
        ) from exc
    finally:
        dataset.close()
        try:
            os.unlink(path)
        except OSError:  # pragma: no cover - best-effort cleanup
            logger.warning("Could not remove temporary NetCDF file %s", path)

    logger.debug("Serialised %d rows to NetCDF (%d bytes)", len(frame), len(payload))
    return payload


def _build_dataset(
    frame: pd.DataFrame,
    *,
    title: str,
    global_attributes: Optional[Mapping[str, str]],
) -> "xr.Dataset":
    """Convert a flat DataFrame into a CF-annotated :class:`xarray.Dataset`.

    Args:
        frame: Table to convert.
        title: Value of the ``title`` global attribute.
        global_attributes: Extra global attributes merged over the defaults.

    Returns:
        A dataset with one ``obs``-dimensioned variable per DataFrame column.
    """
    prepared = frame.reset_index(drop=True).copy()
    for column in prepared.columns:
        if prepared[column].dtype == object:
            prepared[column] = prepared[column].astype(str)

    dataset = xr.Dataset(
        data_vars={
            str(column): ("obs", prepared[column].to_numpy())
            for column in prepared.columns
        },
        coords={"obs": prepared.index.to_numpy()},
    )

    for column in dataset.data_vars:
        attributes = CF_VARIABLE_ATTRIBUTES.get(str(column).lower())
        if attributes:
            dataset[column].attrs.update(dict(attributes))

    dataset["obs"].attrs.update(
        {"long_name": "Observation index", "cf_role": "profile_id"}
    )

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    dataset.attrs.update(
        {
            "title": title,
            "institution": INSTITUTION,
            "source": "OceanMind AI dashboard export",
            "Conventions": CF_CONVENTIONS,
            "date_created": now,
            "history": f"{now}: exported from the OceanMind AI dashboard",
        }
    )
    if global_attributes:
        dataset.attrs.update({str(k): str(v) for k, v in global_attributes.items()})

    return dataset


def _validate_frame(frame: pd.DataFrame) -> None:
    """Guard the serialisers against unusable input.

    Args:
        frame: Candidate table.

    Raises:
        ExportError: If ``frame`` is not a DataFrame or carries no columns.
    """
    if not isinstance(frame, pd.DataFrame):
        raise ExportError(f"Expected a pandas DataFrame, received {type(frame).__name__}.")
    if frame.columns.empty:
        raise ExportError("Cannot export a dataset with no columns.")


# --------------------------------------------------------------------------- #
# Format registry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ExportSpec:
    """Describes one downloadable representation of a dataset.

    Attributes:
        key: Stable identifier used to select the format, e.g. ``"csv"``.
        label: Text shown on the Streamlit download button.
        extension: Filename extension, without the leading dot.
        mime: MIME type advertised to the browser.
        builder: Pure callable turning a DataFrame into bytes.
        requires_optional_dependency: ``True`` when the format may be
            unavailable at runtime and must be capability-checked first.
    """

    key: str
    label: str
    extension: str
    mime: str
    builder: Callable[[pd.DataFrame], bytes]
    requires_optional_dependency: bool = False

    def is_available(self) -> bool:
        """Report whether this format can be produced in this environment."""
        if not self.requires_optional_dependency:
            return True
        return is_netcdf_available()

    def unavailable_reason(self) -> str:
        """Explain why the format is unavailable, or return an empty string."""
        return "" if self.is_available() else netcdf_unavailable_reason()


EXPORT_FORMATS: Final[Mapping[str, ExportSpec]] = {
    "csv": ExportSpec(
        key="csv",
        label="Download CSV",
        extension="csv",
        mime=MIME_CSV,
        builder=to_csv_bytes,
    ),
    "netcdf": ExportSpec(
        key="netcdf",
        label="Download NetCDF",
        extension="nc",
        mime=MIME_NETCDF,
        builder=to_netcdf_bytes,
        requires_optional_dependency=True,
    ),
    "ascii": ExportSpec(
        key="ascii",
        label="Download ASCII",
        extension="txt",
        mime=MIME_ASCII,
        builder=to_ascii_bytes,
    ),
}

DEFAULT_FORMATS: Final[Tuple[str, ...]] = ("csv", "netcdf", "ascii")


# --------------------------------------------------------------------------- #
# Streamlit rendering
# --------------------------------------------------------------------------- #


@st.cache_data(show_spinner=False, max_entries=32)
def _serialise_cached(format_key: str, frame: pd.DataFrame) -> bytes:
    """Serialise ``frame`` once per unique (format, content) pair.

    Streamlit reruns the whole script on every interaction; caching here keeps
    the comparatively expensive NetCDF encode off the hot path.

    Args:
        format_key: Key into :data:`EXPORT_FORMATS`.
        frame: Table to serialise.

    Returns:
        The encoded document as bytes.
    """
    return EXPORT_FORMATS[format_key].builder(frame)


def render_export_bar(
    frame: Optional[pd.DataFrame],
    base_name: str,
    *,
    formats: Sequence[str] = DEFAULT_FORMATS,
    key_prefix: str = "export",
    caption: Optional[str] = None,
    empty_message: str = "No records match the current filters -- nothing to export.",
) -> None:
    """Render a horizontal row of download buttons for a dataset.

    Unavailable formats render as disabled buttons carrying a tooltip that
    names the missing dependency, so the UI stays stable across environments.

    Args:
        frame: Dataset to offer for download. ``None`` or empty renders an
            informational caption instead of buttons.
        base_name: Filename stem, e.g. ``"argo_profiles_arabian_sea"``.
        formats: Ordered format keys to offer, drawn from
            :data:`EXPORT_FORMATS`. Unknown keys are skipped with a warning.
        key_prefix: Namespace for the Streamlit widget keys. Must be unique per
            call site on a page.
        caption: Optional line of helper text rendered above the buttons.
        empty_message: Text shown when there is nothing to export.
    """
    if frame is None or frame.empty:
        st.caption(empty_message)
        return

    specs = [EXPORT_FORMATS[key] for key in formats if key in EXPORT_FORMATS]
    for unknown in (key for key in formats if key not in EXPORT_FORMATS):
        logger.warning("Ignoring unknown export format %r", unknown)
    if not specs:
        st.caption(empty_message)
        return

    if caption:
        st.caption(caption)

    columns = st.columns(len(specs))
    for column, spec in zip(columns, specs):
        with column:
            _render_single_export(spec, frame, base_name, key_prefix)


def _render_single_export(
    spec: ExportSpec,
    frame: pd.DataFrame,
    base_name: str,
    key_prefix: str,
) -> None:
    """Render one download button, handling unavailability and encode errors.

    Args:
        spec: The export format to render.
        frame: Dataset to serialise.
        base_name: Filename stem passed to :func:`build_filename`.
        key_prefix: Namespace for the Streamlit widget key.
    """
    widget_key = f"{key_prefix}_{spec.key}"

    if not spec.is_available():
        st.button(
            spec.label,
            key=widget_key,
            disabled=True,
            help=spec.unavailable_reason(),
            width="stretch",
        )
        return

    try:
        payload = _serialise_cached(spec.key, frame)
    except ExportError as exc:
        logger.exception("Export to %s failed", spec.key)
        st.button(
            spec.label,
            key=widget_key,
            disabled=True,
            help=str(exc),
            width="stretch",
        )
        return

    st.download_button(
        label=spec.label,
        data=payload,
        file_name=build_filename(base_name, spec.extension),
        mime=spec.mime,
        key=widget_key,
        help=f"{len(frame):,} records - {_format_size(len(payload))}",
        width="stretch",
    )


def render_binary_download(
    payload: bytes,
    base_name: str,
    extension: str,
    *,
    label: str,
    mime: str,
    key: str,
    help_text: Optional[str] = None,
    full_width: bool = True,
) -> None:
    """Offer an already-encoded artefact (such as a generated report) for download.

    Used by the Reports tab, where the bytes come from the backend's report
    generator rather than from a DataFrame.

    Args:
        payload: The encoded document.
        base_name: Filename stem passed to :func:`build_filename`.
        extension: Filename extension, with or without a leading dot.
        label: Button text.
        mime: MIME type advertised to the browser.
        key: Streamlit widget key, unique within the page.
        help_text: Tooltip. Defaults to the payload size.
        full_width: Whether the button fills its container.
    """
    st.download_button(
        label=label,
        data=payload,
        file_name=build_filename(base_name, extension),
        mime=mime,
        key=key,
        help=help_text or _format_size(len(payload)),
        width="stretch" if full_width else "content",
    )


def _format_size(num_bytes: int) -> str:
    """Render a byte count as a compact human readable string.

    Args:
        num_bytes: Size in bytes.

    Returns:
        A string such as ``"12.4 KB"``.
    """
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0
    return f"{size:.1f} GB"  # pragma: no cover - unreachable
