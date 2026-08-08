"""Natural-language AI workspace for the OceanMind AI dashboard.

Wraps the backend's LLM query engine in a chat interface: suggested prompts,
persistent conversation history, the generated SQL and the resulting table with
export controls attached.

Backend contract
----------------
``QueryEngine.answer(question)`` is the only interface consumed. Because the
concrete return type is owned by the backend, :func:`normalise_answer` adapts
whatever it produces -- dataclass, object, mapping, tuple or plain string --
into the :class:`AnswerView` this panel renders. That adapter is the single
place to touch if the backend's shape differs from the assumption.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Final, List, Mapping, Optional, Sequence

import pandas as pd
import streamlit as st

from dashboard.export_utils import render_export_bar
from dashboard.styles import OCEAN, section_header
from dashboard.utils import SESSION_CHAT, generate_demo_profiles

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Backend binding
# --------------------------------------------------------------------------- #

try:
    from database.repository import ProfileRepository  # type: ignore[import-not-found]
    from llm_query_engine.query_engine import QueryEngine as _BackendQueryEngine  # type: ignore[import-not-found]

    BACKEND_AVAILABLE: Final[bool] = True

    class QueryEngine(_BackendQueryEngine):  # type: ignore[no-redef]
        """Binds the backend's QueryEngine to the live database.

        The backend class defaults ``profile_repository`` to an in-memory
        ``MockProfileRepository`` (two hardcoded rows) so it's runnable
        standalone; here it's wired to the real, locked
        ``database.repository.ProfileRepository`` so ``.answer()`` executes
        against the actual ARGO dataset. ``rag_retriever`` is left on the
        backend's own default (its RAG pipeline module ships its own FAISS
        index lifecycle that's out of scope for this dashboard wiring).
        """

        def __init__(self) -> None:
            super().__init__(profile_repository=ProfileRepository())

except Exception:  # noqa: BLE001 - covers ImportError *and* a misconfigured
    # backend (e.g. DATABASE_URL not set, which raises KeyError at import
    # time from config.py) so the dashboard degrades to demo mode instead
    # of crashing.
    BACKEND_AVAILABLE = False

    class QueryEngine:  # type: ignore[no-redef]
        """Interface-compatible stub for the backend's LLM query engine.

        Mirrors ``QueryEngine.answer(question)`` exactly so that restoring the
        import is the only change needed for live answers. Responses are
        canned; no natural-language or SQL generation is reimplemented here.
        """

        def answer(self, question: str) -> Dict[str, Any]:
            """Return a canned response describing the demo dataset.

            Args:
                question: The user's natural-language question.

            Returns:
                A mapping with ``answer``, ``sql`` and ``dataframe`` keys,
                matching the shape this panel expects from the real engine.
            """
            frame = generate_demo_profiles(n_floats=6, n_levels=12, n_cycles=2)
            summary = (
                frame.groupby("float_id", as_index=False)
                .agg(
                    observations=("depth", "size"),
                    mean_temperature=("temperature", "mean"),
                    mean_salinity=("salinity", "mean"),
                    max_depth=("depth", "max"),
                )
                .round(3)
            )
            return {
                "answer": (
                    "Demo mode is active, so this is a canned response rather than a real "
                    f"analysis of your question: *{question}*\n\n"
                    f"The synthetic dataset holds {len(frame):,} observations across "
                    f"{frame['float_id'].nunique()} floats. Connect the backend to run "
                    "genuine natural-language queries against your ARGO database."
                ),
                "sql": (
                    "-- Demo mode: illustrative SQL, not executed\n"
                    "SELECT float_id,\n"
                    "       COUNT(*)          AS observations,\n"
                    "       AVG(temperature)  AS mean_temperature,\n"
                    "       AVG(salinity)     AS mean_salinity,\n"
                    "       MAX(depth)        AS max_depth\n"
                    "FROM   argo_profiles\n"
                    "GROUP  BY float_id\n"
                    "ORDER  BY observations DESC;"
                ),
                "dataframe": summary,
            }


SUGGESTED_PROMPTS: Final[Sequence[str]] = (
    "Which floats recorded the warmest surface temperatures this quarter?",
    "Show the average salinity profile in the Arabian Sea below 500 m.",
    "Compare temperature trends between the Bay of Bengal and the Arabian Sea.",
    "Which floats show signs of an expanding oxygen minimum zone?",
    "Summarise chlorophyll-a concentrations near the equator.",
    "List the floats with the deepest profiles in the last 90 days.",
)


# --------------------------------------------------------------------------- #
# Answer adaptation
# --------------------------------------------------------------------------- #


@dataclass
class AnswerView:
    """The renderable form of a query engine response.

    Attributes:
        text: The natural-language answer.
        sql: Generated SQL, when the engine exposes it.
        frame: Tabular results, when the engine returns rows.
        citations: Supporting source references from the RAG pipeline.
        raw: The untouched backend response, kept for debugging.
    """

    text: str
    sql: Optional[str] = None
    frame: Optional[pd.DataFrame] = None
    citations: List[str] = field(default_factory=list)
    raw: Any = None


def _coerce_frame(value: Any) -> Optional[pd.DataFrame]:
    """Best-effort conversion of a backend payload into a DataFrame.

    Args:
        value: Candidate tabular payload.

    Returns:
        A DataFrame, or ``None`` when the value is not table-like.
    """
    if value is None:
        return None
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, pd.Series):
        return value.to_frame()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        try:
            return pd.DataFrame(list(value))
        except (ValueError, TypeError):
            return None
    return None


def normalise_answer(raw: Any) -> AnswerView:
    """Adapt any ``QueryEngine.answer()`` return shape into an :class:`AnswerView`.

    The backend owns the concrete response type. Rather than assuming one, this
    reads the common shapes -- mapping, attribute-bearing object, ``(text, frame)``
    tuple, or plain string -- so the panel renders correctly either way.

    Args:
        raw: Whatever the query engine returned.

    Returns:
        A populated :class:`AnswerView`. Unrecognised payloads become their
        string representation rather than raising.
    """
    if isinstance(raw, AnswerView):
        return raw

    text: Optional[str] = None
    sql: Optional[str] = None
    frame: Optional[pd.DataFrame] = None
    citations: List[str] = []

    if isinstance(raw, Mapping):
        for key in ("answer", "text", "response", "message", "content", "summary", "summary_answer"):
            if raw.get(key):
                text = str(raw[key])
                break
        for key in ("sql", "query", "generated_sql"):
            if raw.get(key):
                sql = str(raw[key])
                break
        for key in ("dataframe", "data", "results", "rows", "table", "result_rows"):
            if key in raw:
                frame = _coerce_frame(raw[key])
                if frame is not None:
                    break
        for key in ("citations", "sources", "references", "rag_context_used"):
            if raw.get(key):
                citations = [str(item) for item in raw[key]]
                break

    elif isinstance(raw, tuple) and len(raw) >= 2:
        text = str(raw[0])
        frame = _coerce_frame(raw[1])

    elif hasattr(raw, "__dict__") or hasattr(raw, "answer"):
        # The real backend's QueryEngine.answer() returns a QueryResult
        # (pydantic model) with fields summary_answer / generated_sql /
        # result_rows / rag_context_used rather than answer/sql/dataframe,
        # so those exact names are checked here too.
        for key in ("answer", "text", "response", "message", "content", "summary", "summary_answer"):
            value = getattr(raw, key, None)
            if value and not callable(value):
                text = str(value)
                break
        for key in ("sql", "query", "generated_sql"):
            value = getattr(raw, key, None)
            if value and not callable(value):
                sql = str(value)
                break
        for key in ("dataframe", "data", "results", "rows", "table", "result_rows"):
            value = getattr(raw, key, None)
            if value is not None and not callable(value):
                frame = _coerce_frame(value)
                if frame is not None:
                    break
        for key in ("citations", "sources", "references", "rag_context_used"):
            value = getattr(raw, key, None)
            if value and not callable(value):
                citations = [str(item) for item in value]
                break

    if text is None:
        text = str(raw) if raw is not None else "The query engine returned no answer."

    return AnswerView(text=text, sql=sql, frame=frame, citations=citations, raw=raw)


# --------------------------------------------------------------------------- #
# Engine access
# --------------------------------------------------------------------------- #


@st.cache_resource(show_spinner=False)
def get_query_engine() -> QueryEngine:
    """Return the shared query engine instance.

    Cached as a resource so model clients and vector index handles are created
    once per server rather than on every rerun.

    Returns:
        The engine, real or stubbed depending on backend availability.
    """
    return QueryEngine()


def ask(question: str) -> AnswerView:
    """Send a question to the backend and adapt the response.

    Args:
        question: The user's natural-language question.

    Returns:
        A renderable :class:`AnswerView`. Backend failures are converted into
        an explanatory answer rather than propagating as an exception.
    """
    try:
        raw = get_query_engine().answer(question)
    except Exception as exc:  # noqa: BLE001 - surface engine failures in the UI
        logger.exception("QueryEngine.answer failed for question %r", question)
        return AnswerView(
            text=(
                "The query engine could not answer that question.\n\n"
                f"`{type(exc).__name__}: {exc}`\n\n"
                "Check the backend connection, database credentials and logs."
            )
        )
    return normalise_answer(raw)


# --------------------------------------------------------------------------- #
# History
# --------------------------------------------------------------------------- #


def _history() -> List[Dict[str, Any]]:
    """Return the mutable chat history held in session state."""
    st.session_state.setdefault(SESSION_CHAT, [])
    return st.session_state[SESSION_CHAT]


def record_exchange(question: str, view: AnswerView) -> None:
    """Append a question and its answer to the conversation history.

    Args:
        question: The user's question.
        view: The adapted engine response.
    """
    _history().append(
        {
            "question": question,
            "view": view,
            "asked_at": datetime.now(timezone.utc),
        }
    )


def clear_history() -> None:
    """Remove every exchange from the conversation history."""
    st.session_state[SESSION_CHAT] = []


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _render_answer(view: AnswerView, *, index: int) -> None:
    """Render one assistant answer with its SQL, table and exports.

    Args:
        view: The adapted response to display.
        index: Position in the history, used to namespace widget keys.
    """
    st.markdown(view.text)

    if view.sql:
        with st.expander("Generated SQL", expanded=False):
            st.code(view.sql, language="sql")

    if view.frame is not None and not view.frame.empty:
        st.dataframe(view.frame, width="stretch", height=min(360, 42 + 34 * len(view.frame)))
        st.caption(f"{len(view.frame):,} row(s) x {len(view.frame.columns)} column(s)")
        render_export_bar(
            view.frame,
            base_name="oceanmind_query_result",
            key_prefix=f"chat_export_{index}",
        )

    if view.citations:
        with st.expander(f"Sources ({len(view.citations)})", expanded=False):
            for citation in view.citations:
                st.markdown(f"- {citation}")


def render_suggested_prompts(*, key: str = "om_suggest") -> Optional[str]:
    """Render the suggested prompt chips.

    Args:
        key: Streamlit widget key namespace.

    Returns:
        The prompt the user clicked, or ``None``.
    """
    st.caption("Try one of these to get started:")
    chosen: Optional[str] = None
    for row_start in range(0, len(SUGGESTED_PROMPTS), 3):
        row = SUGGESTED_PROMPTS[row_start : row_start + 3]
        columns = st.columns(len(row))
        for offset, (column, prompt) in enumerate(zip(columns, row)):
            with column:
                if st.button(prompt, key=f"{key}_{row_start + offset}", width="stretch"):
                    chosen = prompt
    return chosen


def render_chat_panel() -> None:
    """Render the complete AI workspace tab.

    Composes the header, suggested prompts, conversation history and the chat
    input, dispatching each question to :func:`ask`.
    """
    section_header(
        "AI Workspace",
        "Ask questions in plain English. The query engine translates them into SQL, "
        "runs them against the ARGO database and explains the result.",
    )

    if not BACKEND_AVAILABLE:
        st.info(
            "Demo mode: the LLM query engine is not importable, so answers are canned. "
            "Connect the backend for genuine natural-language querying.",
            icon=":material/info:",
        )

    history = _history()

    toolbar = st.columns([3, 1])
    with toolbar[1]:
        if st.button("Clear conversation", width="stretch", disabled=not history):
            clear_history()
            st.rerun()

    pending: Optional[str] = None
    if not history:
        pending = render_suggested_prompts()

    for index, exchange in enumerate(history):
        with st.chat_message("user"):
            st.markdown(exchange["question"])
        with st.chat_message("assistant"):
            _render_answer(exchange["view"], index=index)

    typed = st.chat_input("Ask about ARGO floats, temperature, salinity, oxygen...")
    question = typed or pending

    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Consulting the query engine..."):
                view = ask(question)
            _render_answer(view, index=len(history))
        record_exchange(question, view)
        st.rerun()
