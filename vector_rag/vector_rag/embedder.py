"""Embedding utilities for OceanMind AI's vector/RAG pipeline.

Wraps a sentence-transformers model to turn ARGO profile summaries into
dense vector embeddings, and provides a helper to build a consistent
human-readable summary string from a profile + its measurements. That
summary string is what gets embedded, and later shown back to the LLM
as retrieved context.
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from config import Config
from shared.logger import get_logger

logger = get_logger(__name__)


class EmbedderError(Exception):
    """Raised when loading the embedding model or embedding text fails."""


class ProfileEmbedder:
    """Wraps a sentence-transformers model (Config.EMBEDDING_MODEL) to embed text.

    Attributes:
        model_name: Name of the sentence-transformers model in use.
        dimension: Dimensionality of the embeddings produced by the model.
    """

    def __init__(self, model_name: str | None = None) -> None:
        """Load the sentence-transformers model.

        Args:
            model_name: Optional override; defaults to Config.EMBEDDING_MODEL.

        Raises:
            EmbedderError: If the model fails to load.
        """
        self.model_name: str = model_name or Config.EMBEDDING_MODEL
        try:
            self._model = SentenceTransformer(self.model_name)
            self.dimension: int = self._model.get_sentence_embedding_dimension()
            logger.info("Loaded embedding model '%s' (dim=%d)", self.model_name, self.dimension)
        except Exception as exc:
            logger.error("Failed to load embedding model '%s'", self.model_name, exc_info=True)
            raise EmbedderError(f"Could not load embedding model '{self.model_name}'") from exc

    def embed(self, text: str) -> np.ndarray:
        """Embed a single piece of text.

        Args:
            text: Text to embed.

        Returns:
            A 1-D float32 numpy array of length self.dimension.

        Raises:
            EmbedderError: If text is empty or encoding fails.
        """
        if not text or not text.strip():
            raise EmbedderError("Cannot embed empty text.")
        try:
            vector = self._model.encode(text, convert_to_numpy=True, normalize_embeddings=False)
            return vector.astype(np.float32)
        except Exception as exc:
            logger.error("Failed to embed text: %r", text[:80], exc_info=True)
            raise EmbedderError("Embedding failed for given text.") from exc

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts at once (more efficient than looping embed()).

        Args:
            texts: List of texts to embed.

        Returns:
            A 2-D float32 numpy array of shape (len(texts), self.dimension).

        Raises:
            EmbedderError: If the list is empty or encoding fails.
        """
        if not texts:
            raise EmbedderError("Cannot embed an empty list of texts.")
        try:
            vectors = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
            return vectors.astype(np.float32)
        except Exception as exc:
            logger.error("Failed to embed batch of %d texts", len(texts), exc_info=True)
            raise EmbedderError("Batch embedding failed.") from exc


def build_summary(profile: dict, measurements: list[dict]) -> str:
    """Build a human-readable summary string for one ARGO profile.

    Example output: "Profile at 12.30N 68.10E on 2023-03-04 in Arabian
    Sea, temp range 24.1-28.4C, salinity 35.1-36.0 PSU."

    Args:
        profile: Dict with at least latitude, longitude, profile_date,
            ocean_region keys (matches shared.schemas.ProfileRecord shape).
        measurements: List of measurement dicts with temperature_c /
            salinity_psu keys (may be empty).

    Returns:
        A one-line natural language summary of the profile.

    Raises:
        EmbedderError: If required profile fields are missing or invalid.
    """
    try:
        lat = float(profile["latitude"])
        lon = float(profile["longitude"])
        date_val = profile.get("profile_date", "unknown date")
        region = profile.get("ocean_region") or "Unclassified"

        lat_dir = "N" if lat >= 0 else "S"
        lon_dir = "E" if lon >= 0 else "W"

        temps = [m["temperature_c"] for m in measurements if m.get("temperature_c") is not None]
        sals = [m["salinity_psu"] for m in measurements if m.get("salinity_psu") is not None]

        temp_part = f"temp range {min(temps):.1f}-{max(temps):.1f}C" if temps else "no temperature data"
        sal_part = f"salinity {min(sals):.1f}-{max(sals):.1f} PSU" if sals else "no salinity data"

        return (
            f"Profile at {abs(lat):.2f}{lat_dir} {abs(lon):.2f}{lon_dir} on {date_val} "
            f"in {region}, {temp_part}, {sal_part}."
        )
    except (KeyError, TypeError, ValueError) as exc:
        logger.error("Failed to build summary for profile: %r", profile, exc_info=True)
        raise EmbedderError("Could not build profile summary; missing/invalid fields.") from exc


if __name__ == "__main__":
    # --- Self-test ---
    logger.info("Running embedder self-test...")
    sample_profile = {
        "latitude": 12.3,
        "longitude": 68.1,
        "profile_date": "2023-03-04",
        "ocean_region": "Arabian Sea",
    }
    sample_measurements = [
        {"temperature_c": 24.1, "salinity_psu": 35.1},
        {"temperature_c": 28.4, "salinity_psu": 36.0},
    ]
    summary = build_summary(sample_profile, sample_measurements)
    logger.info("Built summary: %s", summary)
    assert "Arabian Sea" in summary

    embedder = ProfileEmbedder()
    vec = embedder.embed(summary)
    logger.info("Embedded vector shape: %s, dtype: %s", vec.shape, vec.dtype)
    assert vec.shape[0] == embedder.dimension

    batch_vecs = embedder.embed_batch([summary, summary])
    assert batch_vecs.shape == (2, embedder.dimension)
    logger.info("Embedder self-test passed.")
