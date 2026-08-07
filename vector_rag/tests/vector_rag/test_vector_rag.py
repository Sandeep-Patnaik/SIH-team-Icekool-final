"""Standalone tests for the vector_rag module.

Module 2's ProfileRepository does not exist yet (and shouldn't be
imported even once it does) -- these tests use a small fake repository
class instead, so this suite runs independently of every other module.

Run with:
    pytest tests/vector_rag -v
"""
from __future__ import annotations

from pathlib import Path

import pytest

from config import Config
from vector_rag.embedder import EmbedderError, ProfileEmbedder, build_summary
from vector_rag.index_builder import IndexBuilderError, VectorIndexBuilder
from vector_rag.retriever import RagRetriever, RetrieverError

# 12 hand-written synthetic profiles: 6 warm Arabian Sea / Bay of Bengal,
# 6 cold Southern Indian Ocean, so retrieve() has an obvious semantically
# "closer" cluster to assert against.
SYNTHETIC_PROFILES: list[dict] = [
    {
        "float_id": "F001", "cycle_number": 1, "profile_date": "2023-01-01",
        "latitude": 15.0, "longitude": 65.0, "ocean_region": "Arabian Sea",
        "measurements": [{"temperature_c": 29.0, "salinity_psu": 35.5}],
    },
    {
        "float_id": "F002", "cycle_number": 1, "profile_date": "2023-01-02",
        "latitude": 16.5, "longitude": 66.5, "ocean_region": "Arabian Sea",
        "measurements": [{"temperature_c": 28.2, "salinity_psu": 35.8}],
    },
    {
        "float_id": "F003", "cycle_number": 1, "profile_date": "2023-01-03",
        "latitude": 12.0, "longitude": 70.0, "ocean_region": "Arabian Sea",
        "measurements": [{"temperature_c": 29.8, "salinity_psu": 36.0}],
    },
    {
        "float_id": "F004", "cycle_number": 1, "profile_date": "2023-01-04",
        "latitude": 10.0, "longitude": 85.0, "ocean_region": "Bay of Bengal",
        "measurements": [{"temperature_c": 27.5, "salinity_psu": 33.9}],
    },
    {
        "float_id": "F005", "cycle_number": 1, "profile_date": "2023-01-05",
        "latitude": 13.0, "longitude": 88.0, "ocean_region": "Bay of Bengal",
        "measurements": [{"temperature_c": 28.9, "salinity_psu": 34.1}],
    },
    {
        "float_id": "F006", "cycle_number": 1, "profile_date": "2023-01-06",
        "latitude": 18.0, "longitude": 72.0, "ocean_region": "Arabian Sea",
        "measurements": [{"temperature_c": 27.9, "salinity_psu": 35.6}],
    },
    {
        "float_id": "F007", "cycle_number": 1, "profile_date": "2023-01-07",
        "latitude": -35.0, "longitude": 60.0, "ocean_region": "Southern Indian Ocean",
        "measurements": [{"temperature_c": 6.0, "salinity_psu": 34.0}],
    },
    {
        "float_id": "F008", "cycle_number": 1, "profile_date": "2023-01-08",
        "latitude": -32.0, "longitude": 62.0, "ocean_region": "Southern Indian Ocean",
        "measurements": [{"temperature_c": 7.2, "salinity_psu": 34.1}],
    },
    {
        "float_id": "F009", "cycle_number": 1, "profile_date": "2023-01-09",
        "latitude": -30.0, "longitude": 65.0, "ocean_region": "Southern Indian Ocean",
        "measurements": [{"temperature_c": 8.5, "salinity_psu": 34.3}],
    },
    {
        "float_id": "F010", "cycle_number": 1, "profile_date": "2023-01-10",
        "latitude": -28.0, "longitude": 70.0, "ocean_region": "Southern Indian Ocean",
        "measurements": [{"temperature_c": 9.0, "salinity_psu": 34.4}],
    },
    {
        "float_id": "F011", "cycle_number": 1, "profile_date": "2023-01-11",
        "latitude": -25.0, "longitude": 75.0, "ocean_region": "Southern Indian Ocean",
        "measurements": [{"temperature_c": 10.1, "salinity_psu": 34.5}],
    },
    {
        "float_id": "F012", "cycle_number": 1, "profile_date": "2023-01-12",
        "latitude": -20.0, "longitude": 55.0, "ocean_region": "Southern Indian Ocean",
        "measurements": [{"temperature_c": 11.0, "salinity_psu": 34.6}],
    },
]


class FakeProfileRepository:
    """Stand-in for Module 2's real, locked ProfileRepository API.

    Only implements get_profiles_by_region() / get_measurements_for_profile()
    -- exactly the two methods vector_rag actually calls -- so these tests
    decouple from Module 2 while still exercising the real contract shape
    (there is no get_all_profiles() on the locked API).
    """

    def __init__(self, profiles: list[dict]) -> None:
        # Assign a synthetic DB id and index by region + by id, mirroring
        # what real rows out of Postgres would look like.
        self._by_id: dict[int, dict] = {}
        self._by_region: dict[str, list[dict]] = {}
        for i, profile in enumerate(profiles, start=1):
            row = dict(profile)
            row["id"] = i
            measurements = row.pop("measurements", [])
            self._by_id[i] = measurements
            self._by_region.setdefault(row["ocean_region"], []).append(row)

    def get_profiles_by_region(self, region: str, start, end) -> list[dict]:
        return list(self._by_region.get(region, []))

    def get_measurements_for_profile(self, profile_id: int) -> list[dict]:
        return list(self._by_id.get(profile_id, []))


@pytest.fixture(scope="module")
def embedder() -> ProfileEmbedder:
    """Shared embedder instance so the model only loads once for the whole test module."""
    return ProfileEmbedder()


@pytest.fixture()
def built_index_path(tmp_path: Path, embedder: ProfileEmbedder) -> Path:
    """Build a real FAISS index from SYNTHETIC_PROFILES into a temp directory and return it."""
    original_path = Config.FAISS_INDEX_PATH
    Config.FAISS_INDEX_PATH = tmp_path
    try:
        builder = VectorIndexBuilder(embedder=embedder)
        builder.build(FakeProfileRepository(SYNTHETIC_PROFILES), save=True)
        yield tmp_path
    finally:
        Config.FAISS_INDEX_PATH = original_path


# --- embedder.py -------------------------------------------------------


def test_build_summary_contains_region_and_values() -> None:
    summary = build_summary(SYNTHETIC_PROFILES[0], SYNTHETIC_PROFILES[0]["measurements"])
    assert "Arabian Sea" in summary
    assert "29.0" in summary


def test_build_summary_missing_field_raises() -> None:
    with pytest.raises(EmbedderError):
        build_summary({"longitude": 10.0}, [])


def test_embed_empty_text_raises(embedder: ProfileEmbedder) -> None:
    with pytest.raises(EmbedderError):
        embedder.embed("")


def test_embed_batch_shape(embedder: ProfileEmbedder) -> None:
    vectors = embedder.embed_batch(["profile one", "profile two", "profile three"])
    assert vectors.shape == (3, embedder.dimension)


# --- index_builder.py ----------------------------------------------------


def test_build_creates_index_with_expected_vector_count(built_index_path: Path, embedder: ProfileEmbedder) -> None:
    builder = VectorIndexBuilder(embedder=embedder)
    builder.load(built_index_path)
    assert builder.index.ntotal == len(SYNTHETIC_PROFILES)
    assert len(builder.sidecar) == len(SYNTHETIC_PROFILES)


def test_add_incremental_appends_without_full_rebuild(built_index_path: Path, embedder: ProfileEmbedder) -> None:
    builder = VectorIndexBuilder(embedder=embedder)
    builder.load(built_index_path)
    before = builder.index.ntotal

    new_profile = {
        "float_id": "F013", "cycle_number": 1, "profile_date": "2023-01-13",
        "latitude": 14.0, "longitude": 67.0, "ocean_region": "Arabian Sea",
        "measurements": [{"temperature_c": 29.5, "salinity_psu": 35.9}],
    }
    builder.add_incremental([new_profile], save=False)
    assert builder.index.ntotal == before + 1


def test_add_incremental_without_existing_index_raises(embedder: ProfileEmbedder) -> None:
    builder = VectorIndexBuilder(embedder=embedder)
    with pytest.raises(IndexBuilderError):
        builder.add_incremental([SYNTHETIC_PROFILES[0]])


def test_build_from_unsupported_source_raises(embedder: ProfileEmbedder) -> None:
    builder = VectorIndexBuilder(embedder=embedder)
    with pytest.raises(IndexBuilderError):
        builder.build(object())


# --- retriever.py ----------------------------------------------------------


def test_retrieve_warm_query_surfaces_arabian_sea_first(built_index_path: Path, embedder: ProfileEmbedder) -> None:
    retriever = RagRetriever(embedder=embedder, index_path=built_index_path)
    results = retriever.retrieve("warm high temperature Arabian Sea profile", k=3)

    assert len(results) == 3
    regions = {r["metadata"]["ocean_region"] for r in results}
    # The three nearest neighbours should all come from the warm cluster
    # (Arabian Sea / Bay of Bengal), not the cold Southern Indian Ocean one.
    assert "Southern Indian Ocean" not in regions


def test_retrieve_cold_query_surfaces_southern_ocean_first(built_index_path: Path, embedder: ProfileEmbedder) -> None:
    retriever = RagRetriever(embedder=embedder, index_path=built_index_path)
    results = retriever.retrieve("cold low temperature Southern Indian Ocean profile", k=3)

    assert len(results) == 3
    regions = {r["metadata"]["ocean_region"] for r in results}
    assert regions == {"Southern Indian Ocean"}


def test_retrieve_result_shape_matches_module4_contract(built_index_path: Path, embedder: ProfileEmbedder) -> None:
    retriever = RagRetriever(embedder=embedder, index_path=built_index_path)
    results = retriever.retrieve("Arabian Sea", k=1)

    assert len(results) == 1
    result = results[0]
    assert set(result.keys()) == {"profile_id", "summary_text", "similarity_score", "metadata"}
    assert isinstance(result["profile_id"], str)
    assert isinstance(result["summary_text"], str)
    assert isinstance(result["similarity_score"], float)
    assert isinstance(result["metadata"], dict)


def test_retrieve_empty_query_raises(built_index_path: Path, embedder: ProfileEmbedder) -> None:
    retriever = RagRetriever(embedder=embedder, index_path=built_index_path)
    with pytest.raises(RetrieverError):
        retriever.retrieve("   ")


def test_retriever_missing_index_raises(tmp_path: Path, embedder: ProfileEmbedder) -> None:
    empty_dir = tmp_path / "no_index_here"
    with pytest.raises(RetrieverError):
        RagRetriever(embedder=embedder, index_path=empty_dir)
