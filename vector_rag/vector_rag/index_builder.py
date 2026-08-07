"""Builds and maintains the FAISS vector index over ARGO profile summaries."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import faiss
import pandas as pd

from config import Config
from shared.logger import get_logger
from shared.regions import REGION_NAMES
from vector_rag import index_store
from vector_rag.embedder import ProfileEmbedder, build_summary

logger = get_logger(__name__)

_DEFAULT_START_DATE = date(2000, 1, 1)


class IndexBuilderError(Exception):
    """Raised when index building or incremental updates fail."""


class VectorIndexBuilder:
    """Builds a FAISS index of embedded ARGO profile summaries.

    Supports two update modes:
      * build() / rebuild(): full rebuild from scratch, reading either
        Module 1's processed Parquet files or live rows from a
        ProfileRepository via Module 2's *actual locked API*
        (get_profiles_by_region + get_measurements_for_profile — there is
        no get_all_profiles() on that contract, so this module never
        assumes one).
      * add_incremental(): embeds and appends only new profile records to
        an already-built/loaded index, without recomputing everything.

    Hackathon decision: the demo pipeline runs a **full rebuild after
    each ingestion run** (rebuild()) — the dataset is small enough that
    this is fast and keeps the index trivially correct. add_incremental()
    is implemented and self-tested for a future live-ingestion scenario,
    but nothing in the demo path currently calls it.

    Source choice matters for profile_id (see _to_sidecar_entry):
      * Parquet source (Module 1's pre-DB-insert output) -> profile_id is
        a synthetic "{float_id}_{cycle_number}" string, since no Postgres
        row exists yet at that point.
      * ProfileRepository source (post-DB-insert, per Part 0's Integration
        Order: Ingestion -> Database -> Vector/RAG index) -> profile_id is
        the real profiles.id from Postgres, IF Module 2's
        get_profiles_by_region() dicts include that column. This module
        assumes they do (it's the table's PK, and the repository is
        documented to return plain dicts) -- flag it back to Module 2 if
        that assumption is wrong, since get_measurements_for_profile()
        needs it as the join key regardless of what vector_rag does with it.

    For the actual integrated demo, prefer building from the live
    ProfileRepository once Module 2 is merged, so profile_id lines up with
    real DB rows. The Parquet path remains available as a zero-DB-dependency
    fallback for solo development before Module 2 lands.

    Attributes:
        embedder: ProfileEmbedder used to turn summaries into vectors.
    """

    def __init__(self, embedder: ProfileEmbedder | None = None) -> None:
        """Initialise the builder.

        Args:
            embedder: Optional ProfileEmbedder instance; a new one is
                created from Config.EMBEDDING_MODEL if not given.
        """
        self.embedder: ProfileEmbedder = embedder or ProfileEmbedder()
        self._index: faiss.Index | None = None
        self._sidecar: list[dict] = []

    @property
    def index(self) -> faiss.Index | None:
        """The current in-memory FAISS index, or None if not built/loaded yet."""
        return self._index

    @property
    def sidecar(self) -> list[dict]:
        """The current in-memory sidecar list (profile_id/summary/metadata per vector)."""
        return self._sidecar

    def build(
        self,
        source: Path | Any,
        save: bool = True,
        start: date | None = None,
        end: date | None = None,
        region_names: list[str] | None = None,
    ) -> None:
        """Fully (re)build the index from a data source.

        Args:
            source: Either a Path to a directory of Module 1's processed
                Parquet files (normally Config.PROCESSED_DATA_DIR), or a
                Module 2 ProfileRepository instance exposing the locked
                `get_profiles_by_region(region, start, end) -> list[dict]`
                and `get_measurements_for_profile(profile_id) -> list[dict]`
                methods. Each resulting profile dict is expected to look
                like shared.schemas.ProfileRecord (float_id, cycle_number,
                profile_date, latitude, longitude, ocean_region,
                measurements).
            save: If True (default), persist the resulting index to
                Config.FAISS_INDEX_PATH via index_store.save_index().
                Set False to keep it in-memory only (used by tests).
            start: Only used for a ProfileRepository source — start of the
                date range passed to get_profiles_by_region(). Defaults to
                a wide floor (2000-01-01) so a full rebuild picks up
                everything ingested so far.
            end: Only used for a ProfileRepository source — end of the
                date range. Defaults to today.
            region_names: Only used for a ProfileRepository source — which
                regions to pull. Defaults to shared.regions.REGION_NAMES
                (the exact same list Module 1 assigns from and Module 6
                groups by).

        Raises:
            IndexBuilderError: If loading, embedding, or index construction fails.
        """
        try:
            profiles = self._load_profiles(source, start=start, end=end, region_names=region_names)
            if not profiles:
                logger.warning("No profiles found to index from source: %s", source)
                self._index = faiss.IndexFlatIP(self.embedder.dimension)
                self._sidecar = []
                if save:
                    index_store.save_index(self._index, self._sidecar, Config.FAISS_INDEX_PATH)
                return

            summaries: list[str] = []
            sidecar: list[dict] = []
            for profile in profiles:
                measurements = profile.get("measurements", [])
                summary = build_summary(profile, measurements)
                summaries.append(summary)
                sidecar.append(self._to_sidecar_entry(profile, summary))

            vectors = self.embedder.embed_batch(summaries)
            faiss.normalize_L2(vectors)

            index = faiss.IndexFlatIP(self.embedder.dimension)
            index.add(vectors)

            self._index = index
            self._sidecar = sidecar
            logger.info("Built FAISS index with %d profile vectors.", index.ntotal)

            if save:
                index_store.save_index(self._index, self._sidecar, Config.FAISS_INDEX_PATH)
        except IndexBuilderError:
            raise
        except Exception as exc:
            logger.error("Failed to build vector index from source: %s", source, exc_info=True)
            raise IndexBuilderError("Vector index build failed.") from exc

    def rebuild(self, source: Path | Any | None = None) -> None:
        """Full rebuild — the mode used for the hackathon demo, run once after
        each ingestion pass completes.

        Args:
            source: Defaults to Config.PROCESSED_DATA_DIR (Parquet, no DB
                dependency). Pass a live Module 2 ProfileRepository instead
                once it's merged, so this index's profile_id values match
                real Postgres rows.

        Raises:
            IndexBuilderError: If the rebuild fails.
        """
        self.build(source if source is not None else Config.PROCESSED_DATA_DIR, save=True)

    def add_incremental(self, new_records: list[dict], save: bool = True) -> None:
        """Embed and append new profile records to the existing index.

        Args:
            new_records: List of profile dicts (same shape as build()'s
                source rows) to add on top of the current index.
            save: If True (default), persist the updated index afterward.

        Raises:
            IndexBuilderError: If there is no existing index to append to,
                or embedding/appending fails.
        """
        if self._index is None:
            raise IndexBuilderError(
                "No existing index to add to; call build()/rebuild() or load() first."
            )
        if not new_records:
            logger.info("add_incremental called with no new records; nothing to do.")
            return
        try:
            summaries: list[str] = []
            new_sidecar: list[dict] = []
            for profile in new_records:
                measurements = profile.get("measurements", [])
                summary = build_summary(profile, measurements)
                summaries.append(summary)
                new_sidecar.append(self._to_sidecar_entry(profile, summary))

            vectors = self.embedder.embed_batch(summaries)
            faiss.normalize_L2(vectors)
            self._index.add(vectors)
            self._sidecar.extend(new_sidecar)
            logger.info(
                "Added %d profiles incrementally (index now has %d vectors).",
                len(new_records),
                self._index.ntotal,
            )
            if save:
                index_store.save_index(self._index, self._sidecar, Config.FAISS_INDEX_PATH)
        except Exception as exc:
            logger.error("Failed incremental add of %d records", len(new_records), exc_info=True)
            raise IndexBuilderError("Incremental index update failed.") from exc

    def load(self, path: Path | None = None) -> None:
        """Load a previously saved index + sidecar into memory.

        Args:
            path: Directory to load from; defaults to Config.FAISS_INDEX_PATH.

        Raises:
            IndexBuilderError: If loading fails.
        """
        try:
            self._index, self._sidecar = index_store.load_index(path or Config.FAISS_INDEX_PATH)
        except Exception as exc:
            logger.error("Failed to load index for builder", exc_info=True)
            raise IndexBuilderError("Could not load existing index.") from exc

    @staticmethod
    def _to_sidecar_entry(profile: dict, summary: str) -> dict:
        """Build one sidecar entry for a profile, keyed by a stable profile_id.

        Prefers the real Postgres profiles.id (present when the source was
        a ProfileRepository); falls back to a synthetic
        "{float_id}_{cycle_number}" composite key when the source was a
        pre-DB-insert Parquet row, which has no DB id yet.
        """
        profile_id = profile.get("id") or profile.get("profile_id") or (
            f"{profile.get('float_id', 'unknown')}_{profile.get('cycle_number', 0)}"
        )
        metadata = {
            "float_id": profile.get("float_id"),
            "ocean_region": profile.get("ocean_region"),
            "latitude": profile.get("latitude"),
            "longitude": profile.get("longitude"),
            "profile_date": str(profile.get("profile_date")),
        }
        return {"profile_id": str(profile_id), "summary_text": summary, "metadata": metadata}

    @staticmethod
    def _load_profiles(
        source: Path | Any,
        start: date | None = None,
        end: date | None = None,
        region_names: list[str] | None = None,
    ) -> list[dict]:
        """Normalise either a Parquet-directory Path or a ProfileRepository into profile dicts.

        Args:
            source: Path to a Config.PROCESSED_DATA_DIR-style Parquet
                directory, or a Module 2 ProfileRepository instance
                exposing get_profiles_by_region() and
                get_measurements_for_profile() (the actual locked API —
                there is no get_all_profiles()).
            start: Passed through to _load_from_repository.
            end: Passed through to _load_from_repository.
            region_names: Passed through to _load_from_repository.

        Returns:
            List of profile dicts, each with a 'measurements' list of dicts.

        Raises:
            IndexBuilderError: If the source is unreadable or of an
                unsupported type.
        """
        if isinstance(source, Path):
            return VectorIndexBuilder._load_from_parquet(source)
        if hasattr(source, "get_profiles_by_region") and hasattr(source, "get_measurements_for_profile"):
            return VectorIndexBuilder._load_from_repository(source, start, end, region_names)
        raise IndexBuilderError(
            f"Unsupported source type for VectorIndexBuilder: {type(source)!r}. "
            "Expected a Path (Parquet dir) or a ProfileRepository exposing "
            "get_profiles_by_region()/get_measurements_for_profile()."
        )

    @staticmethod
    def _load_from_repository(
        repo: Any,
        start: date | None,
        end: date | None,
        region_names: list[str] | None,
    ) -> list[dict]:
        """Read profiles + their measurements from Module 2's live ProfileRepository.

        Loops shared.regions.REGION_NAMES (or a caller-supplied subset),
        calling get_profiles_by_region() per region, then
        get_measurements_for_profile() per profile to attach depth-level
        readings. This is the module's only sanctioned way to pull live
        DB rows, matching Module 2's actual locked API exactly.

        Args:
            repo: A Module 2 ProfileRepository instance.
            start: Start of the date range (defaults to 2000-01-01).
            end: End of the date range (defaults to today).
            region_names: Regions to pull (defaults to REGION_NAMES).

        Returns:
            List of profile dicts, each with a 'measurements' list attached.

        Raises:
            IndexBuilderError: If any repository call fails.
        """
        effective_start = start or _DEFAULT_START_DATE
        effective_end = end or date.today()
        effective_regions = region_names or REGION_NAMES

        profiles: list[dict] = []
        for region in effective_regions:
            try:
                region_profiles = repo.get_profiles_by_region(region, effective_start, effective_end)
            except Exception as exc:
                logger.error("get_profiles_by_region(%s) failed", region, exc_info=True)
                raise IndexBuilderError(f"Failed to read profiles for region '{region}'.") from exc

            for profile in region_profiles:
                profile = dict(profile)
                profile_pk = profile.get("id")
                if profile_pk is None:
                    logger.warning(
                        "Profile row from get_profiles_by_region('%s') has no 'id' field; "
                        "falling back to a synthetic profile_id and skipping measurement lookup.",
                        region,
                    )
                    profile["measurements"] = []
                else:
                    try:
                        profile["measurements"] = repo.get_measurements_for_profile(profile_pk)
                    except Exception as exc:
                        logger.error(
                            "get_measurements_for_profile(%s) failed", profile_pk, exc_info=True
                        )
                        raise IndexBuilderError(
                            f"Failed to read measurements for profile id {profile_pk}."
                        ) from exc
                profiles.append(profile)

        return profiles

    @staticmethod
    def _load_from_parquet(folder: Path) -> list[dict]:
        """Read profile records out of Module 1's processed Parquet directory.

        Expects one or more .parquet files (Module 1 writes one per run as
        ingestion_{run_id}.parquet) with columns compatible with
        shared.schemas.ProfileRecord. A 'measurements' column, if present,
        may hold a JSON string or a native list; both are handled. These
        rows are pre-DB-insert, so they have no Postgres id yet — see
        _to_sidecar_entry for the resulting profile_id fallback.

        Args:
            folder: Directory containing .parquet files.

        Returns:
            List of profile dicts.

        Raises:
            IndexBuilderError: If reading fails.
        """
        if not folder.exists():
            logger.warning("Processed data folder does not exist: %s", folder)
            return []
        try:
            files = sorted(folder.glob("*.parquet"))
            if not files:
                logger.warning("No .parquet files found in %s", folder)
                return []
            frames = [pd.read_parquet(f) for f in files]
            df = pd.concat(frames, ignore_index=True)
            records = df.to_dict(orient="records")
            for record in records:
                measurements = record.get("measurements", [])
                if isinstance(measurements, str):
                    try:
                        record["measurements"] = json.loads(measurements)
                    except json.JSONDecodeError:
                        record["measurements"] = []
                elif measurements is None:
                    record["measurements"] = []
            return records
        except Exception as exc:
            logger.error("Failed to read Parquet profiles from %s", folder, exc_info=True)
            raise IndexBuilderError(f"Could not read processed profiles from {folder}") from exc


if __name__ == "__main__":
    # --- Self-test ---
    import tempfile

    logger.info("Running index_builder self-test with synthetic data...")
    builder = VectorIndexBuilder()

    synthetic_profiles = [
        {
            "float_id": "F001",
            "cycle_number": 1,
            "profile_date": "2023-01-01",
            "latitude": 15.0,
            "longitude": 65.0,
            "ocean_region": "Arabian Sea",
            "measurements": [{"temperature_c": 27.0, "salinity_psu": 35.5}],
        },
        {
            "float_id": "F002",
            "cycle_number": 1,
            "profile_date": "2023-01-02",
            "latitude": -30.0,
            "longitude": 60.0,
            "ocean_region": "Southern Indian Ocean",
            "measurements": [{"temperature_c": 8.0, "salinity_psu": 34.2}],
        },
    ]

    class _FakeProfileRepository:
        """Stand-in for Module 2's real, locked ProfileRepository API.

        Only implements get_profiles_by_region() / get_measurements_for_profile()
        -- exactly what this module actually calls -- used only in this self-test.
        """

        _BY_REGION = {
            "Arabian Sea": [
                {"id": 101, "float_id": "F001", "cycle_number": 1, "profile_date": "2023-01-01",
                 "latitude": 15.0, "longitude": 65.0, "ocean_region": "Arabian Sea"},
            ],
            "Southern Indian Ocean": [
                {"id": 102, "float_id": "F002", "cycle_number": 1, "profile_date": "2023-01-02",
                 "latitude": -30.0, "longitude": 60.0, "ocean_region": "Southern Indian Ocean"},
            ],
        }
        _MEASUREMENTS = {
            101: [{"temperature_c": 27.0, "salinity_psu": 35.5}],
            102: [{"temperature_c": 8.0, "salinity_psu": 34.2}],
        }

        def get_profiles_by_region(self, region: str, start: date, end: date) -> list[dict]:
            return self._BY_REGION.get(region, [])

        def get_measurements_for_profile(self, profile_id: int) -> list[dict]:
            return self._MEASUREMENTS.get(profile_id, [])

    with tempfile.TemporaryDirectory() as tmp:
        original_path = Config.FAISS_INDEX_PATH
        try:
            Config.FAISS_INDEX_PATH = Path(tmp)

            # Parquet-source path: empty dir -> 0 vectors, exercises the empty-source path
            builder.build(Path(tmp), save=False)
            assert builder.index is not None
            assert builder.index.ntotal == 0

            # Live-repository path using the real locked API shape
            builder.build(_FakeProfileRepository(), save=True)
            assert builder.index.ntotal == 2
            assert len(builder.sidecar) == 2
            assert builder.sidecar[0]["profile_id"] in {"101", "102"}  # real DB ids, not synthetic ones

            builder.add_incremental(
                [
                    {
                        "float_id": "F003",
                        "cycle_number": 1,
                        "profile_date": "2023-01-03",
                        "latitude": 10.0,
                        "longitude": 90.0,
                        "ocean_region": "Bay of Bengal",
                        "measurements": [{"temperature_c": 26.5, "salinity_psu": 33.9}],
                    }
                ]
            )
            assert builder.index.ntotal == 3
            logger.info("index_builder self-test passed.")
        finally:
            Config.FAISS_INDEX_PATH = original_path
