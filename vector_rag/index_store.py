"""Persistence helpers for the FAISS vector index and its profile_id sidecar.

FAISS indexes only store vectors + their positions, not any metadata, so
every save writes a companion JSON "sidecar" list where sidecar[i]
describes the profile that vector i belongs to. Positions must stay in
sync — this file is the only place that reads/writes both together.
"""
from __future__ import annotations

import json
from pathlib import Path

import faiss

from shared.logger import get_logger

logger = get_logger(__name__)

INDEX_FILENAME = "profiles.index"
SIDECAR_FILENAME = "profiles_sidecar.json"


class IndexStoreError(Exception):
    """Raised when saving or loading the FAISS index / sidecar fails."""


def save_index(index: faiss.Index, sidecar: list[dict], path: Path) -> None:
    """Save a FAISS index and its profile_id sidecar to disk.

    Args:
        index: The FAISS index to persist.
        sidecar: List of {profile_id, summary_text, metadata} dicts, one
            per vector, in the same order they were added to `index`.
        path: Directory to save into (created if missing). Typically
            Config.FAISS_INDEX_PATH.

    Raises:
        IndexStoreError: If writing to disk fails.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(path / INDEX_FILENAME))
        with open(path / SIDECAR_FILENAME, "w", encoding="utf-8") as f:
            json.dump(sidecar, f)
        logger.info("Saved FAISS index (%d vectors) to %s", index.ntotal, path)
    except Exception as exc:
        logger.error("Failed to save index to %s", path, exc_info=True)
        raise IndexStoreError(f"Could not save index to {path}") from exc


def load_index(path: Path) -> tuple[faiss.Index, list[dict]]:
    """Load a FAISS index and its profile_id sidecar from disk.

    Args:
        path: Directory previously written by save_index(). Typically
            Config.FAISS_INDEX_PATH.

    Returns:
        Tuple of (faiss index, sidecar list).

    Raises:
        IndexStoreError: If the index/sidecar files are missing or unreadable.
    """
    index_file = path / INDEX_FILENAME
    sidecar_file = path / SIDECAR_FILENAME
    if not index_file.exists() or not sidecar_file.exists():
        raise IndexStoreError(f"No saved index found at {path}")
    try:
        index = faiss.read_index(str(index_file))
        with open(sidecar_file, "r", encoding="utf-8") as f:
            sidecar = json.load(f)
        logger.info("Loaded FAISS index (%d vectors) from %s", index.ntotal, path)
        return index, sidecar
    except Exception as exc:
        logger.error("Failed to load index from %s", path, exc_info=True)
        raise IndexStoreError(f"Could not load index from {path}") from exc


if __name__ == "__main__":
    # --- Self-test ---
    import tempfile

    import numpy as np

    logger.info("Running index_store self-test...")
    dim = 8
    idx = faiss.IndexFlatIP(dim)
    vecs = np.random.rand(3, dim).astype("float32")
    faiss.normalize_L2(vecs)
    idx.add(vecs)
    fake_sidecar = [
        {"profile_id": f"p{i}", "summary_text": f"summary {i}", "metadata": {}} for i in range(3)
    ]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        save_index(idx, fake_sidecar, tmp_path)
        loaded_idx, loaded_sidecar = load_index(tmp_path)
        assert loaded_idx.ntotal == 3
        assert loaded_sidecar == fake_sidecar

        try:
            load_index(tmp_path / "does_not_exist")
            raise AssertionError("Expected IndexStoreError for missing path")
        except IndexStoreError:
            pass
    logger.info("index_store self-test passed.")
