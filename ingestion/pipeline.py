"""
Orchestration pipeline for OceanMind AI ingestion module.

Coordinates the full flow: parse -> qc_clean -> transform -> load
across every NetCDF (.nc) file in a target folder, producing a
summary of the run for logging/reporting purposes.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path

import xarray as xr

from config import Config
from shared.logger import get_logger
from shared.schemas import ProfileRecord
from ingestion.netcdf_parser import NetCDFParser
from ingestion.qc_cleaner import apply_qc_flags, handle_missing_bgc
from ingestion.transformer import to_profile_record
from ingestion.loader import ProfileLoader
from ingestion.exceptions import (
    MalformedNetCDFError,
    MissingVariableError,
    RegionAssignmentError,
)

logger = get_logger(__name__)


@dataclass
class IngestionSummary:
    """
    Summary of a single ingestion pipeline run.

    Attributes:
        run_id: Unique identifier for this ingestion run.
        files_processed: Number of .nc files successfully processed.
        profiles_inserted: Total number of profile records inserted/upserted.
        files_skipped: List of (filename, reason) tuples for files that
            failed to process and were skipped.
        parquet_path: Path to the Parquet mirror written for this run,
            or None if no records were produced.
    """

    run_id: str
    files_processed: int = 0
    profiles_inserted: int = 0
    files_skipped: list[tuple[str, str]] = field(default_factory=list)
    parquet_path: Path | None = None


class IngestionPipeline:
    """
    Orchestrates the ARGO NetCDF ingestion pipeline.

    Wires together parsing, QC cleaning, transformation, and loading
    for a batch of NetCDF files, producing an IngestionSummary.
    """

    def __init__(self) -> None:
        """Initialize the pipeline with its parser and loader dependencies."""
        self.parser = NetCDFParser()
        self.loader = ProfileLoader()

    def _process_single_file(self, nc_path: Path) -> list[ProfileRecord]:
        """
        Parse, clean, and transform a single NetCDF file into ProfileRecords.

        Args:
            nc_path: Path to the .nc file to process.

        Returns:
            List of ProfileRecord objects built from this file (one per
            profile/cycle found in the file).

        Raises:
            MalformedNetCDFError: If the file is structurally broken or
                cannot be opened/parsed.
            MissingVariableError: If a required (non-BGC) variable is
                absent from the dataset.
            RegionAssignmentError: If a profile's lat/lon cannot be
                mapped to a known ocean region.
        """
        records: list[ProfileRecord] = []
        try:
            with xr.open_dataset(nc_path) as ds:
                float_meta = self.parser.parse_float_metadata(ds)
                profile_meta_list = self.parser.parse_profile_metadata(ds)
                raw_measurements = self.parser.parse_measurements(ds)
        except MalformedNetCDFError:
            raise
        except MissingVariableError:
            raise
        except Exception as exc:
            logger.error("Failed to open/parse %s", nc_path, exc_info=True)
            raise MalformedNetCDFError(str(nc_path)) from exc

        try:
            cleaned = apply_qc_flags(raw_measurements)
            cleaned = handle_missing_bgc(cleaned)
        except Exception as exc:
            logger.error("QC cleaning failed for %s", nc_path, exc_info=True)
            raise MalformedNetCDFError(str(nc_path)) from exc

        for profile_meta in profile_meta_list:
            try:
                record = to_profile_record(
                    float_meta=float_meta,
                    profile_meta=profile_meta,
                    measurements=cleaned,
                )
                records.append(record)
            except RegionAssignmentError:
                raise
            except Exception as exc:
                logger.error(
                    "Failed to build ProfileRecord for cycle %s in %s",
                    profile_meta.get("cycle_number"),
                    nc_path,
                    exc_info=True,
                )
                raise MalformedNetCDFError(str(nc_path)) from exc

        return records

    def run(self, folder: Path) -> IngestionSummary:
        """
        Run the full ingestion pipeline over every .nc file in a folder.

        Args:
            folder: Directory containing raw ARGO NetCDF (.nc) files.

        Returns:
            IngestionSummary describing the outcome of the run.
        """
        run_id = uuid.uuid4().hex
        summary = IngestionSummary(run_id=run_id)

        try:
            nc_files = sorted(Path(folder).glob("*.nc"))
        except Exception:
            logger.error("Unable to list NetCDF files in %s", folder, exc_info=True)
            return summary

        if not nc_files:
            logger.warning("No .nc files found in %s", folder)
            return summary

        all_records: list[ProfileRecord] = []

        for nc_path in nc_files:
            try:
                records = self._process_single_file(nc_path)
                all_records.extend(records)
                summary.files_processed += 1
                logger.info(
                    "Processed %s: %d profile(s) extracted", nc_path.name, len(records)
                )
            except (MalformedNetCDFError, MissingVariableError, RegionAssignmentError) as exc:
                reason = f"{type(exc).__name__}: {exc}"
                summary.files_skipped.append((nc_path.name, reason))
                logger.error(
                    "Skipping %s due to %s", nc_path.name, reason, exc_info=True
                )
            except Exception as exc:
                reason = f"UnexpectedError: {exc}"
                summary.files_skipped.append((nc_path.name, reason))
                logger.error(
                    "Skipping %s due to unexpected error", nc_path.name, exc_info=True
                )

        if all_records:
            try:
                inserted = self.loader.bulk_insert(all_records)
                summary.profiles_inserted = inserted
                logger.info("Inserted/upserted %d profile record(s)", inserted)
            except Exception:
                logger.error("Bulk insert failed for run %s", run_id, exc_info=True)

            try:
                parquet_path = self.loader.write_parquet(all_records, run_id)
                summary.parquet_path = parquet_path
                logger.info("Wrote Parquet mirror to %s", parquet_path)
            except Exception:
                logger.error(
                    "Failed to write Parquet mirror for run %s", run_id, exc_info=True
                )
        else:
            logger.warning("No profile records produced for run %s", run_id)

        logger.info(
            "Ingestion run %s complete: %d file(s) processed, %d profile(s) inserted, "
            "%d file(s) skipped",
            run_id,
            summary.files_processed,
            summary.profiles_inserted,
            len(summary.files_skipped),
        )
        return summary


def run_ingestion(folder: Path) -> IngestionSummary:
    """
    Entry point: run the ingestion pipeline over all .nc files in a folder.

    Args:
        folder: Directory containing raw ARGO NetCDF (.nc) files.

    Returns:
        IngestionSummary describing files processed, profiles inserted,
        and any files skipped (with reasons).
    """
    pipeline = IngestionPipeline()
    return pipeline.run(Path(folder))


if __name__ == "__main__":
    # --- Self-test / demo ---
    target_folder = Config.RAW_NETCDF_DIR
    logger.info("Running ingestion pipeline self-test against %s", target_folder)
    result = run_ingestion(target_folder)
    print(  # noqa: T201 - demo-only summary output, not pipeline logging
        f"Run ID: {result.run_id}\n"
        f"Files processed: {result.files_processed}\n"
        f"Profiles inserted: {result.profiles_inserted}\n"
        f"Files skipped: {result.files_skipped}\n"
        f"Parquet path: {result.parquet_path}"
    )