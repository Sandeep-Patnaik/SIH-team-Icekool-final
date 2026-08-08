# OceanMind AI — Dashboard

Streamlit presentation tier for the OceanMind AI ocean-intelligence platform.

The backend — ingestion, PostgreSQL/SQLAlchemy, FAISS, RAG, the LLM query
engine and the intelligence engine — is **already complete and production-ready**.
This module is the only remaining piece. It consumes those modules through
ordinary Python imports and adds no business logic of its own.

---

## 1. Architecture — one process, no network

The dashboard runs **inside the same Python process as the backend**. There is no
REST layer, no HTTP hop and no serialisation boundary. Streamlit imports the
backend packages and calls them directly.

```
┌───────────────────────────────────────────────────────────────┐
│  python -m streamlit run dashboard/app.py     (one process)   │
│                                                               │
│   ┌─────────────────────────────────────────────────────┐     │
│   │  dashboard/                                         │     │
│   │  app.py → map_view · profile_plots · chat_panel     │     │
│   │           health_panel · export_utils               │     │
│   └───────────────────────┬─────────────────────────────┘     │
│                           │  plain Python calls               │
│                           ▼                                   │
│   ┌─────────────────────────────────────────────────────┐     │
│   │  OceanMind AI backend  (complete — do not modify)   │     │
│   │  database/ · vector_rag/ · llm_query_engine/        │     │
│   │  intelligence_engine/ · ingestion/ · shared/        │     │
│   └───────────────────────┬─────────────────────────────┘     │
└───────────────────────────┼───────────────────────────────────┘
                            ▼
                  PostgreSQL  ·  FAISS index  ·  LLM provider
```

**This is a deliberate design choice, not a shortcut.** A single process means:

- No API contract to keep in sync, and no duplicated request/response models.
- Full-fidelity objects — DataFrames and ORM instances arrive intact, never
  flattened through JSON.
- One command to demo, one place to debug, one dependency set.

The dashboard therefore needs **no API layer whatsoever** to reach live data.

---

## 2. Where this folder must live

`dashboard/` must sit at the backend root, as a sibling of `config.py` and
`main.py`, so that `import database`, `import llm_query_engine` and friends
resolve without any path manipulation:

```
oceanmind-ai/
├── ingestion/
├── database/
├── vector_rag/
├── llm_query_engine/
├── intelligence_engine/
├── shared/
├── data/
├── config.py
├── main.py
├── requirements.txt
└── dashboard/          ← here
```

If you keep `dashboard/` somewhere else during development, point `PYTHONPATH` at
the backend root instead:

```bash
# Windows PowerShell
$env:PYTHONPATH = "C:\path\to\oceanmind-ai"

# macOS / Linux
export PYTHONPATH=/path/to/oceanmind-ai
```

Verify the wiring before launching:

```bash
python -c "import database, llm_query_engine, intelligence_engine; print('backend reachable')"
```

---

## 3. Prerequisites and install

| Requirement | Notes |
|---|---|
| Python 3.11+ | Verified working on 3.12.3 |
| The OceanMind AI backend | Importable per §2 |
| PostgreSQL | As configured in the backend's `config.py` |

The backend's own `requirements.txt` is authoritative and out of scope for this
module. The dashboard adds these presentation-tier packages:

```bash
pip install streamlit plotly folium streamlit-folium pandas numpy xarray netCDF4
```

**Current environment status** (checked on this machine):

| Package | Status |
|---|---|
| `streamlit 1.55.0`, `plotly 6.6.0`, `pandas 2.2.3`, `numpy 2.2.1` | Installed |
| `folium`, `streamlit-folium` | **Missing** — the map tab cannot render |
| `xarray`, `netCDF4` | **Missing** — NetCDF export stays disabled |

Missing optional packages never crash the app. The NetCDF download renders as a
greyed-out button whose tooltip names the exact missing package.

> **Note:** these belong in the backend's existing `requirements.txt`, but that
> file is out of scope for this module. Please add them there — listed under
> Integration Notes (§9).

---

## 4. Running the dashboard

```bash
# from the backend root: oceanmind-ai/
streamlit run dashboard/app.py
```

Opens on <http://localhost:8501>.

Useful flags for a presentation machine:

```bash
streamlit run dashboard/app.py --server.port 8501 --server.headless true
```

That is the whole run procedure. One command, one process.

### Connected vs. demo mode

Every backend import is guarded, so the dashboard also runs standalone before the
backend is wired up. The sidebar always reports which mode is active:

- **`Backend: connected`** — real repositories, real query engine, real scores.
- **`Backend: offline (demo data)`** — signature-identical fallback stubs are
  serving synthetic ARGO profiles. Everything renders; **the numbers are not
  real**, so do not present them as findings.

Switching from demo to connected requires **no code change** — only a working
import path. That is the entire point of the stub design: each stub mirrors the
real signature exactly, so fixing the import is the whole migration.

---

## 5. Backend contracts consumed

The dashboard binds only to interfaces that already exist. It invents nothing.

| Dashboard module | Backend contract |
|---|---|
| `map_view.py` | `ProfileRepository.get_profiles_by_region()` |
| `map_view.py` | `ProfileRepository.get_profiles_near()` |
| `chat_panel.py` | `QueryEngine.answer(question)` |
| `health_panel.py` | `OceanHealthCalculator.compute(...)` |
| `health_panel.py` | `ReportGenerator` (existing implementation) |
| `profile_plots.py`, `export_utils.py`, `styles.py`, `utils.py` | none — pure presentation |

If a view appears to need something absent from this table, it is recorded under
Integration Notes (§9) rather than solved by touching backend code.

### Caching rules

Because the dashboard shares a process with the backend, Streamlit's rerun model
matters. Streamlit re-executes the whole script on every interaction, so
uncached work would reopen database connections and reload the FAISS index on
each click. Two rules keep this correct and fast:

| Object | Decorator | Why |
|---|---|---|
| Repositories, engines, DB sessions, FAISS index | `@st.cache_resource` | One shared instance per server; never re-created, never copied |
| Query results, computed DataFrames, health scores | `@st.cache_data` | Cached per argument set; returns a copy, so callers cannot corrupt the cache |

Never wrap a connection-holding object in `@st.cache_data` — it would attempt to
serialise a live connection.

---

## 6. Workflow

The four tabs form one analyst narrative:

```
🌍 Explore Ocean  →  🤖 AI Workspace  →  🌊 Ocean Health  →  📄 Reports
   map_view.py         chat_panel.py       health_panel.py     export_utils.py
   profile_plots.py
```

1. **Explore Ocean** — filter by region and date, locate ARGO floats on the
   Folium map, inspect trajectories, then read depth profiles and trends.
2. **AI Workspace** — ask questions in natural language via
   `QueryEngine.answer()`; review the generated SQL and the result table.
3. **Ocean Health** — a composite score with gauge, KPI cards, contributing
   factors and recommendations from the intelligence engine.
4. **Reports** — browse generated reports and export any dataset as CSV,
   NetCDF or ASCII.

Export is available from every tab, not only the last one — any table on screen
can be downloaded where it appears.

---

## 7. Folder layout and build status

```
dashboard/
├── __init__.py          Package marker, version constant
├── app.py               Entry point · sidebar nav · routing · layout
├── map_view.py          Folium map · markers · trajectories · filters
├── profile_plots.py     Pure Plotly figure builders
├── chat_panel.py        AI chat UI · history · SQL result table
├── health_panel.py      Gauge · KPI cards · recommendations
├── export_utils.py      CSV / NetCDF / ASCII export · download buttons
├── styles.py            Dark ocean theme · card CSS · typography
├── utils.py             Formatting · dates · KPI and colour helpers
├── assets/              Logo · icons · optional CSS
└── README.md            This file
```

**All modules are complete.** Verified by running the app through Streamlit's
`AppTest` harness: all four pages render with no exceptions, and the chat,
report-generation and filter flows were exercised end to end.

| File | Status |
|---|---|
| `__init__.py` | Complete |
| `app.py` | Complete — 4 pages route and render |
| `map_view.py` | Complete — degrades to `st.map` until `folium` is installed |
| `profile_plots.py` | Complete — 11 figure builders verified |
| `chat_panel.py` | Complete — history grows and clears correctly |
| `health_panel.py` | Complete — gauge, factors, recommendations, reports |
| `export_utils.py` | Complete — CSV/ASCII verified; NetCDF awaits `xarray` |
| `styles.py` | Complete |
| `utils.py` | Complete |
| `assets/logo.svg` | Complete |

Two paths remain unexercised on this machine because their dependencies are not
installed: the **Folium map** (needs `folium`, `streamlit-folium`) and the
**NetCDF encoder** (needs `xarray`, `netCDF4`). Both fail soft today — the map
falls back to `st.map`, and the NetCDF button renders disabled with a tooltip
naming the missing package. Install per §3 to enable and verify them.

---

## 8. Configuration

The dashboard introduces **no configuration of its own** and hardcodes no paths.
It inherits everything from the backend's existing `config.py` — database URL,
LLM credentials, data directories. Set those exactly as you would to run the
backend standalone, in the same shell you launch Streamlit from.

| Variable | Purpose |
|---|---|
| `PYTHONPATH` | Only if `dashboard/` is not at the backend root (§2) |
| *(backend's own variables)* | Read by `config.py`, unchanged |

Optional Streamlit server settings belong in `.streamlit/config.toml` at the
backend root — not in this module.

---

## 9. Integration Notes

Items that require backend or repository-level action. **None of these block the
dashboard**, which runs today in either connected or demo mode. Listed here
rather than solved by modifying backend interfaces.

1. **Presentation dependencies belong in `requirements.txt`.** Please add
   `streamlit`, `plotly`, `folium`, `streamlit-folium`, `xarray` and `netCDF4`
   to the existing file. That file is out of scope for this module, so it has
   not been edited.
2. **Confirm the real signatures.** The fallback stubs currently mirror
   `ProfileRepository.get_profiles_by_region()`, `.get_profiles_near()`,
   `QueryEngine.answer(question)` and `OceanHealthCalculator.compute(...)` as
   specified. Please confirm the exact parameter names, return types, and
   whether results arrive as `pandas.DataFrame` or as ORM objects — the stubs
   assume DataFrames, and that assumption shapes every plotting function.
3. **Float trajectory retrieval.** The map draws per-float trajectories, which
   needs "all profiles for one float, ordered by cycle". If an existing method
   covers this, please name it; no new repository method has been invented.
4. **Report artefact format.** `health_panel.py` will hand `ReportGenerator`
   output to `export_utils.render_binary_download()`. Please confirm whether the
   generator returns `bytes`, a file path, or writes to disk as a side effect.
5. **Thread safety.** Repositories will be held in `@st.cache_resource`, making
   them shared across Streamlit's script-runner threads. Please confirm the
   SQLAlchemy session strategy is safe for concurrent access, or advise whether
   a scoped session per rerun is preferred.

---

## 10. Scope boundary

Only `dashboard/` is generated. The backend modules — `ingestion/`, `database/`,
`vector_rag/`, `llm_query_engine/`, `intelligence_engine/`, `shared/`,
`config.py`, `main.py`, `requirements.txt` — are production-ready and are never
regenerated, redesigned, renamed, moved or duplicated.

No REST API, FastAPI service, Flask app or HTTP wrapper is part of this module.
Integration is by Python import only, as described in §1.

> **Directory note.** The working directory `Ishwa-SIH/` also contains an
> unrelated `oceanmind-frontend/` folder. It is **out of scope**: not part of
> this deliverable, not integrated with, and not modified. This module is the
> project's dashboard.

---

## 11. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Sidebar shows *"Backend: offline (demo data)"* | Backend not importable | Verify with the `python -c "import database…"` check in §2; place `dashboard/` at the backend root or set `PYTHONPATH` |
| `ModuleNotFoundError: streamlit_folium` | Map dependency missing | `pip install folium streamlit-folium` |
| NetCDF button greyed out | `xarray` / `netCDF4` missing | `pip install xarray netCDF4` — the tooltip names the missing package |
| Every click is slow; connections pile up | Missing cache decorators | Apply the §5 caching rules — `@st.cache_resource` for engines and repositories |
| `UnhashableParamError` from Streamlit | A repository was passed to `@st.cache_data` | Cache the *result*, not the object; hold the object in `@st.cache_resource` |
| Charts render but are empty | Filters exclude everything | Widen the region or date range; the export bar reports "no records match" for the same reason |
| Port 8501 already in use | Another Streamlit instance | `streamlit run dashboard/app.py --server.port 8502` |
| Database or LLM credential errors | Backend config not loaded | Set the backend's own environment variables in the shell that launches Streamlit (§8) |
