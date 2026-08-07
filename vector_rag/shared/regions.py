"""Shared ocean-region names and bounding boxes.

Module 1 assigns these, Module 6 groups by them, Module 5 filters the map
by them — same exact strings everywhere.
"""

REGION_NAMES = [
    "Arabian Sea",
    "Bay of Bengal",
    "Equatorial Indian Ocean",
    "Southern Indian Ocean",
]

REGION_BOUNDS = {
    # (lat_min, lat_max, lon_min, lon_max) — refine with real boundaries before the demo
    "Arabian Sea": (8.0, 25.0, 50.0, 78.0),
    "Bay of Bengal": (5.0, 22.0, 78.0, 100.0),
    "Equatorial Indian Ocean": (-10.0, 8.0, 50.0, 100.0),
    "Southern Indian Ocean": (-40.0, -10.0, 20.0, 120.0),
}


def assign_region(lat: float, lon: float) -> str:
    """Map a lat/lon pair to one of REGION_NAMES; returns 'Unclassified' if no box matches."""
    for name, (lat_min, lat_max, lon_min, lon_max) in REGION_BOUNDS.items():
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return name
    return "Unclassified"
