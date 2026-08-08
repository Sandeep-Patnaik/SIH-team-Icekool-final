"""LLM Query Engine module for OceanMind AI.

Translates natural-language questions into read-only SQL (nl_to_sql.py),
validates them (sql_guard.py), executes them via the database layer and
returns a QueryResult (query_engine.py).

    from llm_query_engine.query_engine import QueryEngine

Note: this __init__.py makes the folder importable as the ``llm_query_engine``
package so other modules (e.g. the dashboard) can import it with absolute,
package-qualified paths.
"""
