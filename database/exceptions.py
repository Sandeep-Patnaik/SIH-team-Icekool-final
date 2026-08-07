"""Custom exceptions for OceanMind AI's database layer (Module 2)."""


class DatabaseError(Exception):
    """Base class for all database-layer errors raised by ProfileRepository."""


class RecordInsertError(DatabaseError):
    """Raised when a float/profile/measurement/report insert or upsert fails."""


class UnsafeQueryError(DatabaseError):
    """Raised by run_raw_query() when given anything other than a single
    parameterized SELECT statement. Defense in depth alongside Module 4's
    own sql_guard.py check — this layer never trusts the caller.
    """
