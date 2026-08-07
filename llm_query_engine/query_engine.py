import time
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from mcp_client import MCPClient
from nl_to_sql import NLToSQLTranslator
from sql_guard import validate_sql, UnsafeSQLError

logger = logging.getLogger("ocean_mind.query_engine")
logger.setLevel(logging.INFO)

class QueryResult(BaseModel):
    question: str
    generated_sql: str
    summary_answer: str
    result_rows: List[Dict[str, Any]]
    execution_time_ms: float
    rag_context_used: List[Dict[str, Any]]
    status: str
    error: Optional[str] = None


class MockRagRetriever:
    def retrieve(self, question: str) -> List[Dict[str, Any]]:
        return [
            {"title": "ARGO Float Salinity Norms", "content": "Normal ocean salinity ranges between 33 and 37 PSU."},
            {"title": "Ocean Regions", "content": "Profiles include ocean_region field such as 'North Atlantic', 'Pacific', 'Indian Ocean'."}
        ]


class MockProfileRepository:
    def run_raw_query(self, sql: str) -> List[Dict[str, Any]]:
        if "ERROR" in sql or "AMBIGUOUS" in sql:
            return []
        return [
            {"float_id": "ARGO_001", "temperature": 18.5, "salinity": 34.2, "ocean_region": "North Atlantic"},
            {"float_id": "ARGO_002", "temperature": 17.2, "salinity": 35.0, "ocean_region": "North Atlantic"}
        ]


class QueryEngine:
    def __init__(
        self,
        rag_retriever: Optional[Any] = None,
        translator: Optional[NLToSQLTranslator] = None,
        profile_repository: Optional[Any] = None,
        mcp_client: Optional[MCPClient] = None
    ):
        self.rag_retriever = rag_retriever or MockRagRetriever()
        self.translator = translator or NLToSQLTranslator(mcp_client=mcp_client)
        self.profile_repository = profile_repository or MockProfileRepository()

    def answer(self, question: str) -> QueryResult:
        start_time = time.time()
        rag_context = []
        generated_sql = ""
        
        if not question or not question.strip():
            execution_time = (time.time() - start_time) * 1000
            return QueryResult(
                question=question or "",
                generated_sql="",
                summary_answer="The question was empty or invalid.",
                result_rows=[],
                execution_time_ms=round(execution_time, 2),
                rag_context_used=[],
                status="NO_RESULTS",
                error=None
            )

        try:
            try:
                rag_context = self.rag_retriever.retrieve(question)
            except Exception as e:
                logger.warning(f"RAG retrieval warning: {e}")
                rag_context = []

            try:
                generated_sql = self.translator.translate(question, rag_context)
            except Exception as e:
                logger.error(f"LLM translation error: {e}")
                execution_time = (time.time() - start_time) * 1000
                return QueryResult(
                    question=question,
                    generated_sql="",
                    summary_answer="Failed to translate natural language question to SQL due to LLM error.",
                    result_rows=[],
                    execution_time_ms=round(execution_time, 2),
                    rag_context_used=rag_context,
                    status="LLM_ERROR",
                    error=str(e)
                )

            try:
                validated_sql = validate_sql(generated_sql)
            except UnsafeSQLError as se:
                logger.warning(f"SQL Guard rejected query: {generated_sql} | Reason: {se}")
                execution_time = (time.time() - start_time) * 1000
                return QueryResult(
                    question=question,
                    generated_sql=generated_sql,
                    summary_answer="The generated query violated security and safety rules.",
                    result_rows=[],
                    execution_time_ms=round(execution_time, 2),
                    rag_context_used=rag_context,
                    status="INVALID_SQL",
                    error=str(se)
                )
            except Exception as e:
                logger.error(f"Unexpected validation error: {e}")
                execution_time = (time.time() - start_time) * 1000
                return QueryResult(
                    question=question,
                    generated_sql=generated_sql,
                    summary_answer="An error occurred during query safety validation.",
                    result_rows=[],
                    execution_time_ms=round(execution_time, 2),
                    rag_context_used=rag_context,
                    status="INVALID_SQL",
                    error=str(e)
                )

            try:
                rows = self.profile_repository.run_raw_query(validated_sql)
            except Exception as de:
                logger.error(f"Database execution error: {de}")
                execution_time = (time.time() - start_time) * 1000
                return QueryResult(
                    question=question,
                    generated_sql=generated_sql,
                    summary_answer="An error occurred while executing the database query.",
                    result_rows=[],
                    execution_time_ms=round(execution_time, 2),
                    rag_context_used=rag_context,
                    status="DATABASE_ERROR",
                    error=str(de)
                )

            execution_time = (time.time() - start_time) * 1000

            if not rows:
                return QueryResult(
                    question=question,
                    generated_sql=generated_sql,
                    summary_answer="No matching records were found for your query in the ARGO dataset.",
                    result_rows=[],
                    execution_time_ms=round(execution_time, 2),
                    rag_context_used=rag_context,
                    status="NO_RESULTS",
                    error=None
                )

            summary = f"Successfully retrieved {len(rows)} record(s) from the oceanographic dataset answering your question."

            return QueryResult(
                question=question,
                generated_sql=generated_sql,
                summary_answer=summary,
                result_rows=rows,
                execution_time_ms=round(execution_time, 2),
                rag_context_used=rag_context,
                status="SUCCESS",
                error=None
            )

        except Exception as e:
            logger.error(f"Unexpected error in QueryEngine.answer: {e}")
            execution_time = (time.time() - start_time) * 1000
            return QueryResult(
                question=question,
                generated_sql=generated_sql,
                summary_answer="An unexpected error occurred while processing your request.",
                result_rows=[],
                execution_time_ms=round(execution_time, 2),
                rag_context_used=rag_context,
                status="LLM_ERROR",
                error=str(e)
            )