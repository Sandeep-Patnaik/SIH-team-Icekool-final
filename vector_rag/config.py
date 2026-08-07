"""Central configuration for OceanMind AI, loaded from environment variables.

This file is the shared contract for all six modules — do not duplicate
it inside your own module folder. Every value comes from the .env file
(see .env.example for the full list of keys every module may need).
"""
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent


class Config:
    """Central configuration for OceanMind AI, loaded from environment variables."""

    DATABASE_URL: str = os.environ["DATABASE_URL"]
    RAW_NETCDF_DIR: Path = BASE_DIR / "data" / "raw_netcdf"
    PROCESSED_DATA_DIR: Path = BASE_DIR / "data" / "processed"
    FAISS_INDEX_PATH: Path = BASE_DIR / "data" / "faiss_index"
    LLM_PROVIDER: str = os.environ.get("LLM_PROVIDER", "openai")   # openai | qwen | llama
    LLM_API_KEY: str = os.environ.get("LLM_API_KEY", "")
    EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    MCP_SERVER_URL: str = os.environ.get("MCP_SERVER_URL", "")
    LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")
