"""
Custom exceptions for the OceanMind AI ingestion module.

These are raised by ingestion/netcdf_parser.py, ingestion/transformer.py,
and caught/logged (not fatal to the whole batch) by ingestion/pipeline.py
so that one broken file never stops a full run.
"""


class MalformedNetCDFError(Exception):
    """
    Raised when a .nc file cannot be opened or is structurally broken.

    Covers cases such as corrupt files, unreadable headers, or files
    that don't conform to the expected ARGO NetCDF structure.
    """

    DEFAULT_MESSAGE = "The NetCDF file is malformed or could not be read."

    def __init__(self, message: str | None = None) -> None:
        """
        Initialize the exception.

        Args:
            message: Optional description of what went wrong (e.g. the
                offending file path). Falls back to DEFAULT_MESSAGE if
                not provided.
        """
        super().__init__(message or self.DEFAULT_MESSAGE)


class MissingVariableError(Exception):
    """
    Raised when a required (non-BGC) variable is absent from a dataset.

    Note: missing BGC variables (dissolved_oxygen, chlorophyll, ph) are
    NOT an error case — those are set to None. This exception is only
    for variables the pipeline cannot proceed without.
    """

    DEFAULT_MESSAGE = "A required variable is missing from the NetCDF dataset."

    def __init__(self, message: str | None = None) -> None:
        """
        Initialize the exception.

        Args:
            message: Optional description of what went wrong (e.g. the
                name of the missing variable). Falls back to
                DEFAULT_MESSAGE if not provided.
        """
        super().__init__(message or self.DEFAULT_MESSAGE)


class RegionAssignmentError(Exception):
    """
    Raised when a profile's lat/lon cannot be mapped to a known ocean region.

    Region names must exactly match shared/regions.py's REGION_NAMES so
    downstream modules (dashboard filters, intelligence engine grouping)
    stay consistent.
    """

    DEFAULT_MESSAGE = "Unable to assign an ocean region for the given coordinates."

    def __init__(self, message: str | None = None) -> None:
        """
        Initialize the exception.

        Args:
            message: Optional description of what went wrong (e.g. the
                offending lat/lon pair). Falls back to DEFAULT_MESSAGE
                if not provided.
        """
        super().__init__(message or self.DEFAULT_MESSAGE)


if __name__ == "__main__":
    # --- Self-test / demo ---
    # Verify default messages and custom messages both work as expected.
    try:
        raise MalformedNetCDFError()
    except MalformedNetCDFError as exc:
        print(f"MalformedNetCDFError (default): {exc}")  # noqa: T201

    try:
        raise MissingVariableError("Missing required variable: TEMP")
    except MissingVariableError as exc:
        print(f"MissingVariableError (custom): {exc}")  # noqa: T201

    try:
        raise RegionAssignmentError()
    except RegionAssignmentError as exc:
        print(f"RegionAssignmentError (default): {exc}")  # noqa: T201