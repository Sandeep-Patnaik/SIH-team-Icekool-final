"""
ingestion/netcdf_parser.py

Parses raw ARGO NetCDF (.nc) files into plain Python dict structures that
downstream ingestion stages (qc_cleaner, transformer, loader) can consume
without any of them needing to know about xarray/netCDF4 internals.

Owned by: Module 1 — Data Ingestion & ETL.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import xarray as xr

from shared.logger import get_logger
from ingestion.exceptions import MalformedNetCDFError, MissingVariableError

logger = get_logger(__name__)

# ARGO NetCDF variable names this parser looks for. Core physical variables
# are required; BGC variables are optional and vary float-to-float.
REQUIRED_FLOAT_META_VARS: tuple[str, ...] = ("PLATFORM_NUMBER",)
REQUIRED_PROFILE_VARS: tuple[str, ...] = ("LATITUDE", "LONGITUDE", "JULD")
REQUIRED_MEASUREMENT_VARS: tuple[str, ...] = ("PRES", "TEMP", "PSAL")
OPTIONAL_BGC_VARS: dict[str, str] = {
    "DOXY": "dissolved_oxygen",
    "CHLA": "chlorophyll",
    "PH_IN_SITU_TOTAL": "ph",
}


class NetCDFParser:
    """
    Parses ARGO NetCDF datasets into dicts consumed by the rest of the
    ingestion pipeline.

    An instance is stateless and safe to reuse across many files; all
    methods take an already-opened xarray.Dataset so that pipeline.py
    controls the file open/close lifecycle (and can catch open errors
    separately from parse errors).
    """

    def __init__(self) -> None:
        """Initialize the parser. Currently holds no state."""
        logger.debug("NetCDFParser initialized")

    def parse_float_metadata(self, ds: xr.Dataset) -> dict[str, Any]:
        """
        Extract float-level (deployment) metadata from an ARGO dataset.

        Args:
            ds: An opened xarray.Dataset for a single ARGO NetCDF file.

        Returns:
            dict with keys: float_id, deployment_lat, deployment_lon,
            deployment_date, status.

        Raises:
            MalformedNetCDFError: if required variables are missing or
                cannot be coerced to expected types.
        """
        try:
            self._require_variables(ds, REQUIRED_FLOAT_META_VARS)

            float_id = self._extract_scalar_str(ds["PLATFORM_NUMBER"])
            if not float_id:
                raise MalformedNetCDFError(
                    "PLATFORM_NUMBER present but empty/unparseable"
                )

            deployment_lat = self._first_valid_float(ds, "LATITUDE")
            deployment_lon = self._first_valid_float(ds, "LONGITUDE")
            deployment_date = self._first_valid_datetime(ds, "JULD")

            status = self._extract_scalar_str(ds.get("PLATFORM_STATUS")) or "UNKNOWN"

            metadata = {
                "float_id": float_id,
                "deployment_lat": deployment_lat,
                "deployment_lon": deployment_lon,
                "deployment_date": deployment_date,
                "status": status,
            }
            logger.info("Parsed float metadata for float_id=%s", float_id)
            return metadata

        except MalformedNetCDFError:
            raise
        except Exception as exc:  # noqa: BLE001 - convert everything to our type
            logger.error("Failed to parse float metadata", exc_info=True)
            raise MalformedNetCDFError(f"float metadata parse failed: {exc}") from exc

    def parse_profile_metadata(self, ds: xr.Dataset) -> list[dict[str, Any]]:
        """
        Extract one metadata record per profile (ARGO "cycle") in the dataset.

        Args:
            ds: An opened xarray.Dataset for a single ARGO NetCDF file.

        Returns:
            list of dicts, one per cycle, each with keys: cycle_number,
            profile_date, latitude, longitude.

        Raises:
            MalformedNetCDFError: if required profile variables are missing
                or the N_PROF dimension cannot be determined.
        """
        try:
            self._require_variables(ds, REQUIRED_PROFILE_VARS)

            n_prof = self._profile_count(ds)
            cycle_numbers = self._as_1d(ds.get("CYCLE_NUMBER"), n_prof, fill=None)
            lats = self._as_1d(ds["LATITUDE"], n_prof)
            lons = self._as_1d(ds["LONGITUDE"], n_prof)
            juld = self._as_1d(ds["JULD"], n_prof)

            profiles: list[dict[str, Any]] = []
            for i in range(n_prof):
                cycle_number = (
                    int(cycle_numbers[i])
                    if cycle_numbers[i] is not None and not pd.isna(cycle_numbers[i])
                    else i
                )
                profile_date = self._to_python_datetime(juld[i])
                latitude = self._to_python_float(lats[i])
                longitude = self._to_python_float(lons[i])

                if latitude is None or longitude is None:
                    logger.warning(
                        "Skipping profile index %d: missing lat/lon", i
                    )
                    continue

                profiles.append(
                    {
                        "cycle_number": cycle_number,
                        "profile_date": profile_date,
                        "latitude": latitude,
                        "longitude": longitude,
                    }
                )

            if not profiles:
                raise MalformedNetCDFError("No valid profiles found in dataset")

            logger.info("Parsed %d profile(s) from dataset", len(profiles))
            return profiles

        except MalformedNetCDFError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to parse profile metadata", exc_info=True)
            raise MalformedNetCDFError(
                f"profile metadata parse failed: {exc}"
            ) from exc

    def parse_measurements(self, ds: xr.Dataset) -> list[dict[str, Any]]:
        """
        Extract per-depth-level measurements across all profiles in the dataset.

        Core physical variables (pressure, temperature, salinity) are
        required per level. BGC variables (dissolved oxygen, chlorophyll,
        pH) are included only where present in the dataset; missing BGC
        sensors yield None for that field rather than raising or crashing.

        Args:
            ds: An opened xarray.Dataset for a single ARGO NetCDF file.

        Returns:
            list of dicts, one per (profile, depth-level) pair, each with
            keys: profile_index, pressure_dbar, depth_m, temperature_c,
            salinity_psu, dissolved_oxygen, chlorophyll, ph, qc_flag.
            profile_index is the position within this file's N_PROF
            dimension (0-based) so the transformer stage can align it back
            to the corresponding entry from parse_profile_metadata().

        Raises:
            MalformedNetCDFError: if required core variables are missing
                or dimensions are inconsistent.
        """
        try:
            self._require_variables(ds, REQUIRED_MEASUREMENT_VARS)

            n_prof = self._profile_count(ds)
            n_levels = self._level_count(ds)

            pres = self._as_2d(ds["PRES"], n_prof, n_levels)
            temp = self._as_2d(ds["TEMP"], n_prof, n_levels)
            psal = self._as_2d(ds["PSAL"], n_prof, n_levels)
            qc = self._as_2d(ds.get("PRES_QC"), n_prof, n_levels, fill=1)

            bgc_arrays: dict[str, Optional[np.ndarray]] = {}
            for nc_name, out_name in OPTIONAL_BGC_VARS.items():
                if nc_name in ds.variables:
                    bgc_arrays[out_name] = self._as_2d(ds[nc_name], n_prof, n_levels)
                else:
                    bgc_arrays[out_name] = None

            measurements: list[dict[str, Any]] = []
            for p in range(n_prof):
                for lvl in range(n_levels):
                    pressure = self._to_python_float(pres[p, lvl])
                    if pressure is None:
                        # No pressure reading means no usable level at all.
                        continue

                    temperature = self._to_python_float(temp[p, lvl])
                    salinity = self._to_python_float(psal[p, lvl])
                    qc_flag = self._to_python_int(qc[p, lvl], default=1)

                    record = {
                        "profile_index": p,
                        "pressure_dbar": pressure,
                        "depth_m": self._pressure_to_depth_m(pressure),
                        "temperature_c": temperature,
                        "salinity_psu": salinity,
                        "qc_flag": qc_flag,
                    }
                    for out_name, arr in bgc_arrays.items():
                        record[out_name] = (
                            self._to_python_float(arr[p, lvl])
                            if arr is not None
                            else None
                        )

                    measurements.append(record)

            if not measurements:
                raise MalformedNetCDFError(
                    "No usable measurement levels found in dataset"
                )

            logger.info(
                "Parsed %d measurement level(s) across %d profile(s)",
                len(measurements),
                n_prof,
            )
            return measurements

        except MalformedNetCDFError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to parse measurements", exc_info=True)
            raise MalformedNetCDFError(f"measurement parse failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _require_variables(ds: xr.Dataset, names: tuple[str, ...]) -> None:
        """Raise MissingVariableError if any of `names` is absent from `ds`."""
        missing = [n for n in names if n not in ds.variables]
        if missing:
            raise MissingVariableError(
                f"Missing required variable(s): {', '.join(missing)}"
            )

    @staticmethod
    def _profile_count(ds: xr.Dataset) -> int:
        """Return the size of the N_PROF dimension, defaulting to 1 if absent."""
        if "N_PROF" in ds.dims:
            return int(ds.dims["N_PROF"])
        return 1

    @staticmethod
    def _level_count(ds: xr.Dataset) -> int:
        """Return the size of the N_LEVELS dimension."""
        for dim_name in ("N_LEVELS", "N_LEVEL"):
            if dim_name in ds.dims:
                return int(ds.dims[dim_name])
        raise MissingVariableError("Could not determine depth-levels dimension")

    @staticmethod
    def _as_1d(
        var: Optional[xr.DataArray], n_prof: int, fill: Any = np.nan
    ) -> np.ndarray:
        """Coerce a variable to a flat length-n_prof numpy array, filling if absent."""
        if var is None:
            return np.full((n_prof,), fill, dtype=object)
        values = np.asarray(var.values).reshape(-1)
        if values.shape[0] < n_prof:
            pad = np.full((n_prof - values.shape[0],), fill, dtype=object)
            values = np.concatenate([values.astype(object), pad])
        return values

    @staticmethod
    def _as_2d(
        var: Optional[xr.DataArray], n_prof: int, n_levels: int, fill: Any = np.nan
    ) -> np.ndarray:
        """Coerce a variable to an (n_prof, n_levels) numpy array, filling if absent."""
        if var is None:
            return np.full((n_prof, n_levels), fill, dtype=float)
        values = np.asarray(var.values)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        rows, cols = values.shape
        out = np.full((n_prof, n_levels), fill, dtype=float)
        out[: min(rows, n_prof), : min(cols, n_levels)] = values[
            : min(rows, n_prof), : min(cols, n_levels)
        ]
        return out

    @staticmethod
    def _first_valid_float(ds: xr.Dataset, name: str) -> Optional[float]:
        """Return the first non-NaN float value of `name`, or None."""
        if name not in ds.variables:
            return None
        values = np.asarray(ds[name].values).reshape(-1)
        for v in values:
            f = NetCDFParser._to_python_float(v)
            if f is not None:
                return f
        return None

    @staticmethod
    def _first_valid_datetime(ds: xr.Dataset, name: str) -> Optional[Any]:
        """Return the first valid timestamp of `name` as a Python datetime, or None."""
        if name not in ds.variables:
            return None
        values = np.asarray(ds[name].values).reshape(-1)
        for v in values:
            dt = NetCDFParser._to_python_datetime(v)
            if dt is not None:
                return dt
        return None

    @staticmethod
    def _extract_scalar_str(var: Optional[xr.DataArray]) -> Optional[str]:
        """Extract a clean scalar string from a (possibly byte-encoded) NetCDF var."""
        if var is None:
            return None
        try:
            values = np.asarray(var.values).reshape(-1)
            if values.size == 0:
                return None
            raw = values[0]
            if isinstance(raw, bytes):
                return raw.decode("utf-8", errors="ignore").strip()
            text = str(raw).strip()
            return text or None
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _to_python_float(value: Any) -> Optional[float]:
        """Convert a numpy/xarray scalar to a Python float, or None if invalid."""
        try:
            if value is None:
                return None
            f = float(value)
            if np.isnan(f):
                return None
            return f
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_python_int(value: Any, default: Optional[int] = None) -> Optional[int]:
        """Convert a numpy/xarray scalar to a Python int, or `default` if invalid."""
        try:
            if value is None:
                return default
            f = float(value)
            if np.isnan(f):
                return default
            return int(f)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _to_python_datetime(value: Any) -> Optional[Any]:
        """Convert a numpy datetime64/JULD-like scalar to a Python datetime, or None."""
        try:
            ts = pd.to_datetime(value)
            if pd.isna(ts):
                return None
            return ts.to_pydatetime()
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _pressure_to_depth_m(pressure_dbar: float) -> float:
        """
        Approximate depth in meters from pressure in decibars using the
        standard oceanographic rule-of-thumb (1 dbar ~= 1 m at mid-latitudes).
        Sufficient for dashboard/report purposes; not a geophysical-grade
        conversion.
        """
        return round(pressure_dbar * 1.0, 2)


if __name__ == "__main__":
    # --- Self-test ---
    # Builds a tiny synthetic ARGO-like dataset in memory (no file I/O,
    # no network) and exercises all three parse methods, including a
    # missing-BGC-sensor case.
    logger.info("Running NetCDFParser self-test with synthetic dataset")

    n_prof, n_levels = 2, 3
    synthetic_ds = xr.Dataset(
        data_vars={
            "PLATFORM_NUMBER": ("N_PROF", np.array([b"1901234"] * n_prof)),
            "PLATFORM_STATUS": ("N_PROF", np.array([b"ACTIVE"] * n_prof)),
            "CYCLE_NUMBER": ("N_PROF", np.array([1, 2])),
            "LATITUDE": ("N_PROF", np.array([12.5, 12.6])),
            "LONGITUDE": ("N_PROF", np.array([65.0, 65.2])),
            "JULD": (
                "N_PROF",
                np.array(
                    ["2023-01-01T00:00:00", "2023-02-01T00:00:00"],
                    dtype="datetime64[ns]",
                ),
            ),
            "PRES": (
                ("N_PROF", "N_LEVELS"),
                np.array([[5.0, 50.0, 100.0], [5.0, 50.0, np.nan]]),
            ),
            "TEMP": (
                ("N_PROF", "N_LEVELS"),
                np.array([[28.1, 22.4, 18.0], [27.9, 22.1, np.nan]]),
            ),
            "PSAL": (
                ("N_PROF", "N_LEVELS"),
                np.array([[35.1, 35.3, 35.5], [35.0, 35.2, np.nan]]),
            ),
            "PRES_QC": (
                ("N_PROF", "N_LEVELS"),
                np.array([[1, 1, 4], [1, 1, 9]]),
            ),
            # DOXY intentionally omitted to simulate a core-only float
            # (no dissolved-oxygen sensor).
        }
    )

    parser = NetCDFParser()

    float_meta = parser.parse_float_metadata(synthetic_ds)
    assert float_meta["float_id"] == "1901234"
    print("float_metadata:", float_meta)  # noqa: T201 - self-test output only

    profiles = parser.parse_profile_metadata(synthetic_ds)
    assert len(profiles) == n_prof
    print("profile_metadata:", profiles)  # noqa: T201

    measurements = parser.parse_measurements(synthetic_ds)
    assert all(m["dissolved_oxygen"] is None for m in measurements)
    assert any(m["qc_flag"] == 4 for m in measurements)
    print(f"parsed {len(measurements)} measurement levels")  # noqa: T201

    logger.info("NetCDFParser self-test passed")