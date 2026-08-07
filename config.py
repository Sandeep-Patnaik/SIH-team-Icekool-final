"""Central configuration for OceanMind AI, loaded from environment variables.

This file is owned collectively by the whole team (Part 0 of the prompt pack).
Nobody duplicates it inside their own module folder — everyone imports from here.
"""
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent


class Config:
    """Central configuration for OceanMind AI, loaded from environment variables."""

    # Part 0 canonical contract: required, no default. Do NOT reintroduce a
    # SQLite fallback here (e.g. os.environ.get("DATABASE_URL", "sqlite:///...")) —
    # a prior draft of this file silently deviated from the shared contract this
    # way, which would conflict when the 6 module folders are merged. Local/CI
    # runs must export DATABASE_URL explicitly (e.g. DATABASE_URL=sqlite:///./oceanmind_dev.db
    # in your local .env), not rely on a default baked in here.
    DATABASE_URL: str = os.environ["DATABASE_URL"]
    RAW_NETCDF_DIR: Path = BASE_DIR / "data" / "raw_netcdf"
    PROCESSED_DATA_DIR: Path = BASE_DIR / "data" / "processed"
    FAISS_INDEX_PATH: Path = BASE_DIR / "data" / "faiss_index"
    LLM_PROVIDER: str = os.environ.get("LLM_PROVIDER", "openai")  # openai | qwen | llama
    LLM_API_KEY: str = os.environ.get("LLM_API_KEY", "")
    EMBEDDING_MODEL: str = os.environ.get(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    MCP_SERVER_URL: str = os.environ.get("MCP_SERVER_URL", "")
    LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")
