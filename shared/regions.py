"""Shared ocean-region definitions for OceanMind AI (Part 0).

Owned collectively — pasted verbatim into every module's local repo copy,
not redeclared. Module 1 (Ingestion) calls assign_region() to tag each
profile on the way in; Module 5 (Dashboard) filters its map by
REGION_NAMES; Module 6 (Intelligence Engine) groups by REGION_NAMES when
computing Ocean Health Indices. All three depend on these exact strings
and this exact REGION_BOUNDS shape — do not restructure.
"""
from __future__ import annotations

REGION_NAMES: list[str] = [
    "Arabian Sea",
    "Bay of Bengal",
    "Equatorial Indian Ocean",
    "Southern Indian Ocean",
]

# (lat_min, lat_max, lon_min, lon_max) — refine with real boundaries before
# the demo. Kept as plain tuples (not nested dicts) per Part 0's canonical
# shared/regions.py, since assign_region() below unpacks them positionally.
REGION_BOUNDS: dict[str, tuple[float, float, float, float]] = {
    "Arabian Sea": (8.0, 25.0, 50.0, 78.0),
    "Bay of Bengal": (5.0, 22.0, 78.0, 100.0),
    "Equatorial Indian Ocean": (-10.0, 8.0, 50.0, 100.0),
    "Southern Indian Ocean": (-40.0, -10.0, 20.0, 120.0),
}


def assign_region(lat: float, lon: float) -> str:
    """Map a lat/lon pair to one of REGION_NAMES; returns 'Unclassified' if no box matches.

    Args:
        lat: latitude in decimal degrees.
        lon: longitude in decimal degrees.

    Returns:
        The matching region name from REGION_NAMES, or "Unclassified" if
        the point doesn't fall inside any defined bounding box.
    """
    for name, (lat_min, lat_max, lon_min, lon_max) in REGION_BOUNDS.items():
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return name
    return "Unclassified"


if __name__ == "__main__":
    # --- Self-test ---
    # Verifies every REGION_NAMES entry has bounds and that a few known
    # points classify as expected.
    from shared.logger import get_logger

    logger = get_logger(__name__)

    for name in REGION_NAMES:
        assert name in REGION_BOUNDS, f"Missing REGION_BOUNDS entry for {name!r}"
    logger.info("All %d region names have bounds.", len(REGION_NAMES))

    samples = [
        (15.0, 65.0, "Arabian Sea"),
        (15.0, 90.0, "Bay of Bengal"),
        (0.0, 80.0, "Equatorial Indian Ocean"),
        (-30.0, 70.0, "Southern Indian Ocean"),
        (80.0, 80.0, "Unclassified"),
    ]
    for lat, lon, expected in samples:
        result = assign_region(lat, lon)
        assert result == expected, f"assign_region({lat}, {lon}) -> {result!r}, expected {expected!r}"
        logger.info("assign_region(%s, %s) -> %s (expected)", lat, lon, result)

    logger.info("Self-test passed.")
