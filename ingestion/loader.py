"""
ingestion/loader.py

Loads transformed ProfileRecord objects into PostgreSQL (via Module 2's
locked ProfileRepository API) and mirrors them to Parquet for Module 3's
RAG pipeline to embed from without hitting Postgres directly.

Owned by: Module 1 — Data Ingestion & ETL.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from config import Config
from shared.logger import get_logger
from shared.schemas import ProfileRecord
from database.repository import ProfileRepository

logger = get_logger(__name__)


class ProfileLoader:
    """
    Loads a batch of ProfileRecord objects into their two destinations:
    PostgreSQL (floats/profiles/measurements, via ProfileRepository) and
    a per-run Parquet mirror under Config.PROCESSED_DATA_DIR.

    The ProfileRepository instance is injected rather than constructed
    internally, so tests can pass a mock/stub repository and never touch
    a real database (per the project's "mock other modules in tests"
    testing standard).
    """

    def __init__(self, repository: Optional[ProfileRepository] = None) -> None:
        """
        Initialize the loader.

        Args:
            repository: an instance implementing Module 2's locked
                ProfileRepository API (specifically
                insert_profile_record()). If omitted, a default
                ProfileRepository() is constructed, which will use
                Config.DATABASE_URL to connect.
        """
        self._repository: ProfileRepository = repository or ProfileRepository()
        logger.debug("ProfileLoader initialized with repository=%r", self._repository)

    def bulk_insert(self, records: list[ProfileRecord]) -> int:
        """
        Insert a batch of ProfileRecord objects into PostgreSQL via
        Module 2's ProfileRepository.insert_profile_record().

        Each record is inserted individually so that one bad record
        (e.g. a constraint violation) does not abort the whole batch —
        the failure is logged and counted as skipped, and insertion
        continues with the remaining records. Upsert-on-(float_id,
        cycle_number) semantics are implemented inside
        ProfileRepository.insert_profile_record() (Module 2's locked
        API); this method does not attempt its own duplicate detection.

        Args:
            records: list of ProfileRecord objects to insert, typically
                produced by transformer.to_profile_record().

        Returns:
            The number of records successfully inserted (or upserted).

        Raises:
            None directly — per-record failures are caught, logged with
            exc_info=True, and skipped rather than raised, so a single
            malformed record cannot fail an entire ingestion run.
        """
        inserted_count = 0
        skipped_count = 0

        for record in records:
            try:
                self._repository.insert_profile_record(record)
                inserted_count += 1
            except Exception:  # noqa: BLE001 - any DB/network error from Module 2
                skipped_count += 1
                logger.error(
                    "Failed to insert profile float_id=%s cycle_number=%s",
                    getattr(record, "float_id", "?"),
                    getattr(record, "cycle_number", "?"),
                    exc_info=True,
                )
                continue

        logger.info(
            "bulk_insert complete: %d inserted, %d skipped (of %d total)",
            inserted_count,
            skipped_count,
            len(records),
        )
        return inserted_count

    def write_parquet(self, records: list[ProfileRecord], run_id: str) -> Path:
        """
        Write a batch of ProfileRecord objects to a single Parquet file
        under Config.PROCESSED_DATA_DIR, one file per ingestion run.

        This is Module 3's embedding source: it reads this Parquet file
        directly rather than querying Postgres. The nested
        `measurements` list-of-dicts field is serialized to a JSON
        string column so it survives a flat Parquet schema; Module 3
        should `json.loads()` it back out.

        Args:
            records: list of ProfileRecord objects to persist.
            run_id: identifier for this ingestion run (e.g. a UTC
                timestamp string or UUID), used to build the output
                filename `ingestion_{run_id}.parquet`. Generating run_id
                itself is pipeline.py's responsibility, not this method's.

        Returns:
            The Path to the written Parquet file.

        Raises:
            OSError: if the output directory can't be created or the
                file can't be written.
            ValueError: if records is empty (nothing meaningful to write).
        """
        if not records:
            raise ValueError("write_parquet called with an empty records list")

        try:
            Config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
            output_path = (
                Config.PROCESSED_DATA_DIR / f"ingestion_{run_id}.parquet"
            )

            rows = []
            for record in records:
                row = record.model_dump()
                # Flatten the nested measurements list into a JSON string
                # so pandas/pyarrow can write a consistent flat schema.
                row["measurements"] = json.dumps(row["measurements"])
                rows.append(row)

            dataframe = pd.DataFrame(rows)
            dataframe.to_parquet(output_path, engine="pyarrow", index=False)

            logger.info(
                "Wrote %d profile record(s) to Parquet: %s",
                len(records),
                output_path,
            )
            return output_path

        except Exception as exc:  # noqa: BLE001 - file/serialization errors
            logger.error(
                "Failed to write Parquet file for run_id=%s", run_id, exc_info=True
            )
            raise OSError(f"Could not write Parquet for run_id={run_id}: {exc}") from exc


if __name__ == "__main__":
    # --- Self-test ---
    # Uses a stub ProfileRepository (no real DB connection) and a real
    # temp Parquet write so both methods are exercised standalone.
    import tempfile
    from datetime import datetime

    logger.info("Running ProfileLoader self-test")

    class _StubProfileRepository:
        """
        Minimal stand-in for database.repository.ProfileRepository that
        implements only insert_profile_record(), so this self-test never
        touches a real Postgres instance.
        """

        def __init__(self) -> None:
            self.inserted: list[ProfileRecord] = []

        def insert_profile_record(self, record: ProfileRecord) -> None:
            """Pretend to upsert a record; fail on a sentinel cycle_number."""
            if record.cycle_number == -1:
                raise RuntimeError("simulated insert failure")
            self.inserted.append(record)

    sample_records = [
        ProfileRecord(
            float_id="1901234",
            cycle_number=1,
            profile_date=datetime(2023, 1, 1, 6, 0, 0),
            latitude=15.0,
            longitude=85.0,
            ocean_region="Bay of Bengal",
            measurements=[
                {
                    "pressure_dbar": 5.0,
                    "depth_m": 5.0,
                    "temperature_c": 28.1,
                    "salinity_psu": 35.1,
                    "dissolved_oxygen": None,
                    "chlorophyll": None,
                    "ph": None,
                    "qc_flag": 1,
                }
            ],
        ),
        ProfileRecord(
            float_id="1901234",
            cycle_number=-1,  # sentinel: triggers the stub's simulated failure
            profile_date=datetime(2023, 2, 1, 6, 0, 0),
            latitude=15.0,
            longitude=85.0,
            ocean_region="Bay of Bengal",
            measurements=[],
        ),
    ]

    stub_repo = _StubProfileRepository()
    loader = ProfileLoader(repository=stub_repo)

    inserted = loader.bulk_insert(sample_records)
    assert inserted == 1, f"expected 1 successful insert, got {inserted}"
    assert len(stub_repo.inserted) == 1
    print(f"bulk_insert inserted {inserted} record(s)")  # noqa: T201

    with tempfile.TemporaryDirectory() as tmpdir:
        original_dir = Config.PROCESSED_DATA_DIR
        try:
            Config.PROCESSED_DATA_DIR = Path(tmpdir)  # type: ignore[misc]
            out_path = loader.write_parquet(sample_records, run_id="selftest001")
            assert out_path.exists()
            df_check = pd.read_parquet(out_path)
            assert len(df_check) == len(sample_records)
            print(f"write_parquet wrote {len(df_check)} row(s) to {out_path}")  # noqa: T201
        finally:
            Config.PROCESSED_DATA_DIR = original_dir  # type: ignore[misc]

    logger.info("ProfileLoader self-test passed")