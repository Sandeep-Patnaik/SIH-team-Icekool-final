"""Retrieval-augmented generation retriever: serves top-k similar ARGO profiles for a query."""
from __future__ import annotations

from pathlib import Path

import faiss

from config import Config
from shared.logger import get_logger
from vector_rag import index_store
from vector_rag.embedder import ProfileEmbedder

logger = get_logger(__name__)


class RetrieverError(Exception):
    """Raised when retrieval fails (e.g. index not loaded, embedding failure)."""


class RagRetriever:
    """Retrieves the most semantically similar ARGO profiles for a natural-language query.

    This is the module boundary consumed by Module 4 (LLM Query Engine) —
    the shape of retrieve()'s return value must not change without
    flagging it to the team: each result dict has exactly the keys
    profile_id, summary_text, similarity_score, metadata.

    Attributes:
        embedder: ProfileEmbedder used to embed the incoming query.
    """

    def __init__(self, embedder: ProfileEmbedder | None = None, index_path: Path | None = None) -> None:
        """Load a previously built FAISS index + sidecar and prepare for retrieval.

        Args:
            embedder: Optional ProfileEmbedder; a new one is created if not given.
            index_path: Directory to load the index from; defaults to
                Config.FAISS_INDEX_PATH.

        Raises:
            RetrieverError: If no saved index can be found/loaded.
        """
        self.embedder: ProfileEmbedder = embedder or ProfileEmbedder()
        path = index_path or Config.FAISS_INDEX_PATH
        try:
            self._index, self._sidecar = index_store.load_index(path)
        except Exception as exc:
            logger.error("RagRetriever failed to load index from %s", path, exc_info=True)
            raise RetrieverError(f"Could not load FAISS index from {path}") from exc

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        """Retrieve the top-k most similar profiles to a natural-language query.

        Args:
            query: Free-text query, e.g. "warm profiles in the Arabian Sea".
            k: Number of results to return.

        Returns:
            List of up to k dicts, ordered most-to-least similar, each with keys:
              - profile_id (str)
              - summary_text (str)
              - similarity_score (float): cosine similarity in [-1, 1].
              - metadata (dict): float_id, ocean_region, latitude, longitude, profile_date.

        Raises:
            RetrieverError: If embedding the query or searching the index fails.
        """
        if not query or not query.strip():
            raise RetrieverError("Query text must not be empty.")
        if self._index.ntotal == 0:
            logger.warning("retrieve() called on an empty index.")
            return []
        try:
            query_vector = self.embedder.embed(query).reshape(1, -1)
            faiss.normalize_L2(query_vector)

            actual_k = min(k, self._index.ntotal)
            scores, indices = self._index.search(query_vector, actual_k)

            results: list[dict] = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                entry = self._sidecar[idx]
                results.append(
                    {
                        "profile_id": entry["profile_id"],
                        "summary_text": entry["summary_text"],
                        "similarity_score": float(score),
                        "metadata": entry["metadata"],
                    }
                )
            return results
        except Exception as exc:
            logger.error("Retrieval failed for query: %r", query, exc_info=True)
            raise RetrieverError("Retrieval failed.") from exc


if __name__ == "__main__":
    # --- Self-test ---
    import tempfile

    from vector_rag.index_builder import VectorIndexBuilder

    logger.info("Running retriever self-test with synthetic Arabian Sea vs Southern Ocean data...")

    class _FakeProfileRepository:
        """Stand-in for Module 2's real, locked ProfileRepository API.

        Only implements get_profiles_by_region() / get_measurements_for_profile()
        -- exactly what vector_rag actually calls -- used only in this self-test.
        """

        _BY_REGION = {
            "Arabian Sea": [
                {"id": 1, "float_id": "F001", "cycle_number": 1, "profile_date": "2023-01-01",
                 "latitude": 15.0, "longitude": 65.0, "ocean_region": "Arabian Sea"},
                {"id": 2, "float_id": "F002", "cycle_number": 1, "profile_date": "2023-01-02",
                 "latitude": 16.0, "longitude": 66.0, "ocean_region": "Arabian Sea"},
            ],
            "Southern Indian Ocean": [
                {"id": 3, "float_id": "F003", "cycle_number": 1, "profile_date": "2023-01-03",
                 "latitude": -35.0, "longitude": 60.0, "ocean_region": "Southern Indian Ocean"},
                {"id": 4, "float_id": "F004", "cycle_number": 1, "profile_date": "2023-01-04",
                 "latitude": -32.0, "longitude": 62.0, "ocean_region": "Southern Indian Ocean"},
            ],
        }
        _MEASUREMENTS = {
            1: [{"temperature_c": 29.0, "salinity_psu": 35.5}],
            2: [{"temperature_c": 28.5, "salinity_psu": 35.7}],
            3: [{"temperature_c": 6.0, "salinity_psu": 34.0}],
            4: [{"temperature_c": 7.2, "salinity_psu": 34.1}],
        }

        def get_profiles_by_region(self, region, start, end) -> list[dict]:
            return self._BY_REGION.get(region, [])

        def get_measurements_for_profile(self, profile_id: int) -> list[dict]:
            return self._MEASUREMENTS.get(profile_id, [])

    with tempfile.TemporaryDirectory() as tmp:
        original_path = Config.FAISS_INDEX_PATH
        try:
            Config.FAISS_INDEX_PATH = Path(tmp)
            builder = VectorIndexBuilder()
            builder.build(_FakeProfileRepository(), save=True, region_names=["Arabian Sea", "Southern Indian Ocean"])

            retriever = RagRetriever()
            results = retriever.retrieve("warm Arabian Sea profile", k=2)
            logger.info(
                "Top results: %s",
                [(r["profile_id"], r["metadata"]["ocean_region"], r["similarity_score"]) for r in results],
            )

            assert len(results) == 2
            assert all(r["metadata"]["ocean_region"] == "Arabian Sea" for r in results)
            logger.info("retriever self-test passed.")
        finally:
            Config.FAISS_INDEX_PATH = original_path
