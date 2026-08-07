import re
import logging
from typing import Set

logger = logging.getLogger("ocean_mind.sql_guard")
logger.setLevel(logging.INFO)

class UnsafeSQLError(Exception):
    """Raised when an SQL query violates security rules or contains unsafe operations."""
    pass

ALLOWED_TABLES: Set[str] = {"floats", "profiles", "measurements", "reports"}

FORBIDDEN_KEYWORDS: Set[str] = {
    "drop", "delete", "update", "insert", "alter", "create", 
    "truncate", "exec", "execute", "call", "merge", "copy", 
    "grant", "revoke", "xp_cmdshell", "union", "into", "load_file",
    "outfile", "dumpfile", "shutdown", "reconfigure"
}

def validate_sql(sql: str) -> str:
    if not sql or not isinstance(sql, str):
        raise UnsafeSQLError("SQL query is empty or not a string.")

    original_sql = sql.strip()
    
    if "--" in original_sql or "/*" in original_sql or "*/" in original_sql:
        raise UnsafeSQLError("SQL comments are strictly forbidden.")

    sql_no_strings = re.sub(r"'[^']*'", "''", original_sql)

    statements = [s.strip() for s in sql_no_strings.split(";") if s.strip()]
    if len(statements) > 1:
        raise UnsafeSQLError("Multiple SQL statements (stacked statements) are forbidden.")
    
    clean_sql = statements[0] if statements else sql_no_strings

    tokens = clean_sql.split()
    if not tokens or tokens[0].upper() != "SELECT":
        raise UnsafeSQLError("Query must be a single SELECT statement.")

    clean_upper = clean_sql.upper()

    words = re.findall(r'\b[A-Z_]+\b', clean_upper)
    for word in words:
        if word in FORBIDDEN_KEYWORDS:
            raise UnsafeSQLError(f"Forbidden keyword or operation detected: '{word}'")

    if "UNION" in words or "UNION ALL" in clean_upper:
        raise UnsafeSQLError("UNION operations/injections are forbidden.")

    table_pattern = re.compile(r'\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)', re.IGNORECASE)
    matched_tables = table_pattern.findall(clean_sql)

    if not matched_tables:
        if not (len(tokens) <= 4 and ("FROM" not in clean_upper)):
            raise UnsafeSQLError("Query does not specify any valid table source.")
    
    for tbl in matched_tables:
        tbl_lower = tbl.lower()
        if tbl_lower not in ALLOWED_TABLES:
            raise UnsafeSQLError(f"Access to unauthorized table '{tbl}' is forbidden. Allowed tables: {ALLOWED_TABLES}")

    logger.info("SQL successfully validated by SQLGuard.")
    return original_sql