"""
ingestion/qc_cleaner.py

Applies ARGO quality-control rules to parsed measurement dicts before they
reach transformer.py / loader.py.

Owned by: Module 1 — Data Ingestion & ETL.

QC flag policy (ARGO Reference Table 2 convention)
---------------------------------------------------
Only `qc_flag == 1` ("good data") is kept as valid. Every other flag value
is DROPPED (the whole measurement-level record is removed), including:
    2 -> probably good data
    3 -> probably bad data (potentially correctable)
    4 -> bad data
    5 -> value changed
    8 -> estimated value
    9 -> missing value
This project takes a conservative "drop, don't flag" stance rather than
"keep but flag", because the dashboard/report/RAG layers downstream have
no notion of a QC flag on a measurement — anything they see is assumed
trustworthy. This must be called out clearly for the demo Q&A: we are
NOT presenting flag-2/3 data with a caveat, we simply omit it.
"""

from __future__ import annotations

from typing import Any, Optional

from shared.logger import get_logger

logger = get_logger(__name__)

# The only QC flag value treated as valid/keepable.
VALID_QC_FLAG: int = 1

# BGC measurement fields that may be legitimately absent on core-only
# floats (no BGC sensor package). These must always resolve to None,
# never NaN or 0.0, when missing.
BGC_FIELDS: tuple[str, ...] = ("dissolved_oxygen", "chlorophyll", "ph")


def apply_qc_flags(measurements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Filter a list of measurement dicts down to only ARGO-good (qc_flag == 1)
    readings, dropping every other record entirely.

    Args:
        measurements: list of per-depth-level measurement dicts as produced
            by NetCDFParser.parse_measurements(). Each dict is expected to
            contain a "qc_flag" key (int), though a missing/unparseable
            flag is treated defensively rather than crashing the batch.

    Returns:
        A new list containing only the measurement dicts whose qc_flag is
        exactly VALID_QC_FLAG (1). Order is preserved. Never mutates the
        input list or its dicts in place.

    Raises:
        None directly — a malformed individual record is logged and
        skipped rather than raised, so one bad record cannot abort an
        entire file's worth of otherwise-good data.
    """
    cleaned: list[dict[str, Any]] = []
    dropped_count = 0

    for idx, record in enumerate(measurements):
        try:
            qc_flag = record.get("qc_flag")

            # Defensive coercion: some sources may hand us a numpy scalar,
            # a string, or an outright missing key.
            if qc_flag is None:
                logger.warning(
                    "Measurement at index %d missing qc_flag; dropping", idx
                )
                dropped_count += 1
                continue

            qc_flag_int = int(qc_flag)

            if qc_flag_int == VALID_QC_FLAG:
                cleaned.append(dict(record))  # shallow copy, don't mutate input
            else:
                dropped_count += 1

        except (TypeError, ValueError) as exc:
            logger.error(
                "Unparseable qc_flag at measurement index %d: %r",
                idx,
                record.get("qc_flag"),
                exc_info=True,
            )
            dropped_count += 1
            continue

    logger.info(
        "QC filtering: kept %d / %d measurement(s), dropped %d",
        len(cleaned),
        len(measurements),
        dropped_count,
    )
    return cleaned


def handle_missing_bgc(measurements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Normalize BGC (biogeochemical) fields on each measurement dict so that
    any missing/invalid sensor reading is represented as None.

    This guards against three failure modes that would otherwise leak
    downstream: a raw NaN (which is not valid JSON and breaks the
    Pydantic ProfileRecord / Postgres insert), a sentinel 0.0 (which would
    be silently misread as "zero dissolved oxygen" instead of "no
    sensor"), or the key being absent entirely from the dict.

    Args:
        measurements: list of per-depth-level measurement dicts, typically
            already passed through apply_qc_flags().

    Returns:
        A new list of dicts (input is not mutated) where every dict is
        guaranteed to contain all of BGC_FIELDS, each either a valid float
        or exactly None.
    """
    normalized: list[dict[str, Any]] = []

    for idx, record in enumerate(measurements):
        try:
            new_record = dict(record)  # shallow copy, don't mutate input

            for field in BGC_FIELDS:
                new_record[field] = _clean_bgc_value(new_record.get(field))

            normalized.append(new_record)

        except Exception:  # noqa: BLE001 - never let one bad record abort the batch
            logger.error(
                "Failed to normalize BGC fields at measurement index %d",
                idx,
                exc_info=True,
            )
            continue

    logger.info("Normalized BGC fields on %d measurement(s)", len(normalized))
    return normalized


def _clean_bgc_value(value: Any) -> Optional[float]:
    """
    Coerce a single BGC field value to a valid float or None.

    Treats None, NaN, and non-numeric/unparseable values all as "sensor
    absent" -> None. A genuine 0.0 reading (sensor present, value
    happens to be zero) is preserved as 0.0, not confused with "missing".

    Args:
        value: the raw value from a measurement dict; may be None, a
            float, an int, a numpy scalar, or occasionally a string.

    Returns:
        A Python float if the value is a valid finite number, else None.
    """
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None

    # NaN != NaN is the standard float NaN check; avoids importing numpy
    # here just for this.
    if f != f:
        return None

    return f


if __name__ == "__main__":
    # --- Self-test ---
    # Hand-built measurement dicts, no DB/file I/O required.
    logger.info("Running qc_cleaner self-test with synthetic measurements")

    sample_measurements: list[dict[str, Any]] = [
        {
            "profile_index": 0,
            "pressure_dbar": 5.0,
            "depth_m": 5.0,
            "temperature_c": 28.1,
            "salinity_psu": 35.1,
            "dissolved_oxygen": 210.5,
            "chlorophyll": None,  # sensor absent
            "ph": float("nan"),  # unparseable / sensor absent
            "qc_flag": 1,  # keep
        },
        {
            "profile_index": 0,
            "pressure_dbar": 50.0,
            "depth_m": 50.0,
            "temperature_c": 22.4,
            "salinity_psu": 35.3,
            "dissolved_oxygen": 0.0,  # genuine zero reading, must survive
            "chlorophyll": 0.4,
            "ph": 8.05,
            "qc_flag": 4,  # bad data -> drop
        },
        {
            "profile_index": 1,
            "pressure_dbar": 100.0,
            "depth_m": 100.0,
            "temperature_c": 18.0,
            "salinity_psu": 35.5,
            "dissolved_oxygen": 205.0,
            "chlorophyll": 0.2,
            "ph": 8.02,
            "qc_flag": 9,  # missing value -> drop
        },
        {
            "profile_index": 1,
            "pressure_dbar": 150.0,
            "depth_m": 150.0,
            "temperature_c": 16.5,
            "salinity_psu": 35.6,
            # BGC keys entirely absent -> must normalize to None, not KeyError
            "qc_flag": 1,  # keep
        },
    ]

    qc_passed = apply_qc_flags(sample_measurements)
    assert len(qc_passed) == 2, f"expected 2 kept, got {len(qc_passed)}"
    assert all(m["qc_flag"] == 1 for m in qc_passed)
    print(f"apply_qc_flags kept {len(qc_passed)} record(s)")  # noqa: T201

    fully_cleaned = handle_missing_bgc(qc_passed)
    assert fully_cleaned[0]["chlorophyll"] is None
    assert fully_cleaned[0]["ph"] is None
    assert fully_cleaned[1]["dissolved_oxygen"] is None
    assert fully_cleaned[1]["chlorophyll"] is None
    assert fully_cleaned[1]["ph"] is None
    print("handle_missing_bgc output:", fully_cleaned)  # noqa: T201

    # Confirm the original input lists/dicts were never mutated in place.
    assert sample_measurements[0]["qc_flag"] == 1
    assert sample_measurements[1]["qc_flag"] == 4

    logger.info("qc_cleaner self-test passed")