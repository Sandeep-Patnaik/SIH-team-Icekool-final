import logging
from typing import List, Dict, Any, Optional
from mcp_client import MCPClient
from prompt_templates import OCEANMIND_SYSTEM_PROMPT, build_nl_to_sql_prompt

logger = logging.getLogger("ocean_mind.nl_to_sql")
logger.setLevel(logging.INFO)

class NLToSQLTranslator:
    def __init__(self, mcp_client: Optional[MCPClient] = None, system_prompt: Optional[str] = None):
        self.mcp_client = mcp_client or MCPClient()
        self.system_prompt = system_prompt or OCEANMIND_SYSTEM_PROMPT

    def translate(self, question: str, rag_context: List[Dict[str, Any]]) -> str:
        if not question or not question.strip():
            logger.warning("Empty question received for translation.")
            return "SELECT 'EMPTY_QUESTION' AS error;"

        rag_context_str = "\n".join([f"- {item.get('content', str(item))}" for item in rag_context]) if rag_context else "None"
        formatted_system_prompt = self.system_prompt.format(rag_context_str=rag_context_str)
        
        user_prompt = build_nl_to_sql_prompt(question, rag_context)

        logger.info(f"Translating question: '{question}' with {len(rag_context)} RAG contexts.")
        raw_output = self.mcp_client.generate(prompt=user_prompt, system_prompt=formatted_system_prompt, temperature=0.0)

        cleaned_sql = self._clean_sql_output(raw_output)
        logger.info(f"Generated SQL: {cleaned_sql}")
        return cleaned_sql

    def _clean_sql_output(self, raw_output: str) -> str:
        output = raw_output.strip()
        if output.startswith("```sql"):
            output = output[6:]
        elif output.startswith("```"):
            output = output[3:]
        if output.endswith("```"):
            output = output[:-3]
        return output.strip()