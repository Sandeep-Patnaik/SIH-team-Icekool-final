# OceanMind AI — Module 3: Vector Store & RAG Pipeline (`vector_rag/`)

Owner: Person 3 (Vector Store & RAG Pipeline), branch `feature/vector-rag`.

This repo snapshot includes the **shared contract files** (`config.py`,
`shared/`) pasted in verbatim per the team spec, plus my module
(`vector_rag/`) and its tests. I only created files inside `vector_rag/`
and `tests/vector_rag/` — the rest is here so this folder runs standalone
without you having to hunt down the shared files separately.

## Module Overview

Turns ARGO profile data into embeddings, maintains a FAISS index, and
serves retrieval to Module 4 (LLM Query Engine). Pipeline:

```
Profile rows (Parquet from Module 1, or live DB rows from Module 2's
ProfileRepository)
  -> build_summary()            one-line NL summary per profile
  -> ProfileEmbedder.embed_batch()   sentence-transformers vectors
  -> FAISS IndexFlatIP (cosine similarity via L2-normalized vectors)
  -> index_store.save_index()   vectors + profile_id "sidecar" JSON
  -> RagRetriever.retrieve(query, k)  ->  Module 4
```

## Folder Structure

```
oceanmind-ai/
├── config.py                     # shared — pasted verbatim
├── .env.example                  # shared — pasted verbatim + this module's keys
├── shared/
│   ├── __init__.py
│   ├── schemas.py                 # shared — pasted verbatim
│   ├── regions.py                 # shared — pasted verbatim
│   └── logger.py                  # shared — pasted verbatim
├── vector_rag/                    # <-- this module
│   ├── __init__.py
│   ├── embedder.py
│   ├── index_builder.py
│   ├── index_store.py
│   └── retriever.py
├── data/
│   ├── raw_netcdf/
│   ├── processed/                 # Module 1 writes Parquet here
│   └── faiss_index/                # this module reads/writes here (Config.FAISS_INDEX_PATH)
├── tests/
│   └── vector_rag/
│       └── test_vector_rag.py
├── requirements.txt
└── README.md
```

## File Responsibilities

| File | Responsibility |
|---|---|
| `embedder.py` | `ProfileEmbedder` — loads `Config.EMBEDDING_MODEL` via sentence-transformers, `.embed()` / `.embed_batch()`. `build_summary(profile, measurements) -> str` — builds the exact string that gets embedded and later shown to the LLM as context. |
| `index_builder.py` | `VectorIndexBuilder` — `build(source)` (full build from a Parquet dir **or** a duck-typed `ProfileRepository`), `rebuild()` (alias: full rebuild from `Config.PROCESSED_DATA_DIR`, used by the demo), `add_incremental(new_records)` (append without recomputing everything), `load()`. |
| `index_store.py` | `save_index()` / `load_index()` — persists the FAISS index plus a JSON "sidecar" (profile_id/summary/metadata per vector, same order as the index) to `Config.FAISS_INDEX_PATH`. |
| `retriever.py` | `RagRetriever.retrieve(query, k=5) -> list[dict]` — the exact call Module 4 makes. Loads a saved index, embeds the query, does cosine-similarity search, returns results in the fixed shape below. |

## Dependencies (`requirements.txt`)

```
faiss-cpu
sentence-transformers
langchain
pandas
numpy
pyarrow
python-dotenv
pydantic
pytest
```

`langchain` is included per the fixed tech stack for Module 4's chain
composition; this module's own retrieval is implemented directly on top
of `faiss` rather than LangChain's `VectorStore` wrapper, so the returned
dict shape below stays exactly what's specified — not LangChain's
`Document` object. Flagging this now in case Module 4 assumed otherwise.

## Implementation Plan (as executed)

1. Paste shared `config.py` / `shared/*` verbatim (done here, so nobody
   is blocked reading this folder).
2. `embedder.py` — model wrapper + `build_summary()`.
3. `index_store.py` — save/load FAISS index + sidecar JSON.
4. `index_builder.py` — `build()` / `rebuild()` / `add_incremental()`,
   reading either Parquet or a duck-typed repository.
5. `retriever.py` — `RagRetriever.retrieve()` matching Module 4's contract.
6. Self-tests (`if __name__ == "__main__"`) in every file using synthetic
   data — no dependency on real ARGO files or a live Postgres.
7. `pytest` suite in `tests/vector_rag/` using a fake `ProfileRepository`
   so it runs standalone.

## Setup Instructions

```bash
# from the repo root
cp .env.example .env          # fill in DATABASE_URL / LLM_API_KEY / etc.
pip install -r requirements.txt

# run this module's self-tests (needs internet access to huggingface.co
# the first time, to download the sentence-transformers model)
python -m vector_rag.embedder
python -m vector_rag.index_store
python -m vector_rag.index_builder
python -m vector_rag.retriever

# run the pytest suite
pytest tests/vector_rag -v
```

> Note: `DATABASE_URL` is read eagerly by `config.py` (`os.environ["DATABASE_URL"]`)
> even though this module never touches Postgres directly — that's the
> shared contract, so `.env` needs *some* value in it even for a
> vector_rag-only checkout (a dummy `postgresql://localhost/placeholder`
> is fine until Module 2's real DB exists).

## Integration Notes

**Consumes:**
- `config.Config` — `EMBEDDING_MODEL`, `PROCESSED_DATA_DIR`, `FAISS_INDEX_PATH`.
- `shared.logger.get_logger` — for all logging.
- `shared.regions.REGION_NAMES` — to loop regions when building from a live
  `ProfileRepository` (same list Module 1 assigns from, Module 6 groups by).
- `shared.schemas.ProfileRecord` shape (informally — profile dicts passed
  in are expected to match its fields: `float_id`, `cycle_number`,
  `profile_date`, `latitude`, `longitude`, `ocean_region`, `measurements`).
- Two data sources for `build()`/`rebuild()`:
  1. **Module 1's processed Parquet files** at `Config.PROCESSED_DATA_DIR`
     (`ingestion_{run_id}.parquet`) — no DB dependency, works before
     Module 2 is merged.
  2. **Module 2's `ProfileRepository`** — consumed via its *actual locked
     API only*: `get_profiles_by_region(region, start, end)` looped over
     `REGION_NAMES`, then `get_measurements_for_profile(profile_id)` per
     row. (An earlier draft of this module assumed a `get_all_profiles()`
     method that isn't part of the locked contract — fixed; this module
     now calls only the methods Part 2 actually specifies.)

  **Flag for Module 2:** this assumes the dicts returned by
  `get_profiles_by_region()` include the `profiles.id` column (the
  table's PK) — it's needed both as the join key for
  `get_measurements_for_profile()` and as the `profile_id` this module
  hands back to Module 4. If that column isn't in the returned dict,
  please flag it back rather than us guessing.

  **profile_id caveat:** because Part 0's Integration Order builds the DB
  *before* the vector index, prefer passing the live `ProfileRepository`
  into `rebuild(source=...)` for the actual demo — then `profile_id` is
  the real Postgres integer id. If built from Parquet instead (pre-DB-insert,
  no id yet), `profile_id` falls back to a synthetic
  `"{float_id}_{cycle_number}"` string. That's fine as a stable FAISS key,
  but Module 4/6 shouldn't feed it straight into
  `get_measurements_for_profile(profile_id: int)` expecting a DB match —
  use `metadata.float_id` / `profile_date` / lat-lon for grounding instead
  when the index was Parquet-built.

**Produces for Module 4:**
- `RagRetriever.retrieve(query: str, k: int = 5) -> list[dict]`, each
  dict exactly:
  ```python
  {
      "profile_id": str,
      "summary_text": str,
      "similarity_score": float,   # cosine similarity, higher = more similar
      "metadata": {
          "float_id": str,
          "ocean_region": str,
          "latitude": float,
          "longitude": float,
          "profile_date": str,
      },
  }
  ```
  **Do not rename these keys without telling the team** — Module 4 calls
  this directly.

**Rebuild cadence (flag for whoever runs the demo / Module 6):**
Call `VectorIndexBuilder().rebuild(source=...)` **once, after each
Module 1 ingestion run finishes**. `rebuild()` defaults to
`Config.PROCESSED_DATA_DIR` (Parquet, no DB needed) if no `source` is
given — but for the actual integrated demo, pass the real
`ProfileRepository` instance instead (`rebuild(source=profile_repo)`) so
`profile_id` lines up with Postgres. `add_incremental()` exists and is
tested but is *not* wired into the demo — it's there for a future
live-ingestion scenario, not integration day.

**New shared field?** None added — no changes to `shared/schemas.py` were
needed for this module.

## Testing Instructions

```bash
pytest tests/vector_rag -v
```

The suite (`tests/vector_rag/test_vector_rag.py`) builds a real FAISS
index from 12 hand-written synthetic profile summaries (6 warm
Arabian Sea/Bay of Bengal, 6 cold Southern Indian Ocean) and asserts:
- a "warm ... Arabian Sea" query's top-3 results never include the cold
  cluster,
- a "cold ... Southern Indian Ocean" query's top-3 results are *only*
  the cold cluster,
- `retrieve()`'s return shape matches the Module 4 contract exactly,
- `add_incremental()` correctly appends without a full rebuild,
- error paths (empty query, missing index, unsupported build source,
  incremental add before any build) raise the right custom exceptions.

It mocks Module 2 with a small `FakeProfileRepository` class so it never
imports `database/` and runs fully standalone. The embedding model
downloads from huggingface.co on first run — no other network access is
required.
