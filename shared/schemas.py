"""Shared Pydantic data contracts for OceanMind AI.

Owned collectively (Part 0). If a module genuinely needs a new field, add it here
and call it out in that module's integration notes — never redeclare a lookalike
class inside your own module folder.
"""
from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel


class ProfileRecord(BaseModel):
    """Ingestion -> Database contract: one ARGO profile with its depth-level measurements."""

    float_id: str
    cycle_number: int
    profile_date: datetime
    latitude: float
    longitude: float
    ocean_region: Optional[str] = None
    measurements: list[dict]


class QueryResult(BaseModel):
    """LLM Query Engine -> Dashboard contract."""

    natural_language_query: str
    generated_sql: Optional[str] = None
    result_rows: list[dict] = []
    summary_answer: str


class OceanHealthScore(BaseModel):
    """Intelligence Engine -> Dashboard contract."""

    ocean_region: str
    period_start: date
    period_end: date
    score: float
    contributing_factors: dict[str, float]
    recommendation: str
