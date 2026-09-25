import re
import pandas as pd
from typing import Dict, Any, List, Optional
from backend.app.recipes.base.recipe import BaseRecipe
from backend.app.core.logging import logger


class SQLStepRecipe(BaseRecipe):
    recipe_id = "sql_step"
    name = "Custom SQL Step"
    version = "1.0.0"
    category = "preprocessing"
    description = "Runs a user-written SQL query against the incoming dataframe (table name 'df') using DuckDB, enterprise visual-SQL style."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "title": "SQL Query",
                    "default": "SELECT * FROM df",
                    "description": "Standard SQL. Reference the incoming data as table 'df' (or 'data' / 'input_df')."
                },
                "query_templates": {
                    "type": "array",
                    "title": "Quick Templates",
                    "items": {"type": "object"},
                    "default": [
                        {"label": "Aggregate", "query": "SELECT category, SUM(amount) AS total FROM df GROUP BY category"},
                        {"label": "Sort", "query": "SELECT * FROM df ORDER BY column_name DESC"},
                        {"label": "Pivot", "query": "PIVOT df ON category USING SUM(amount)"},
                        {"label": "Unpivot", "query": "UNPIVOT df ON col1, col2 INTO NAME metric VALUE value"},
                        {"label": "Union", "query": "SELECT * FROM df WHERE year = 2023\nUNION\nSELECT * FROM df WHERE year = 2024"},
                        {"label": "CTE (WITH)", "query": "WITH filtered AS (\n    SELECT * FROM df WHERE amount > 0\n)\nSELECT category, AVG(amount) AS avg_amt FROM filtered GROUP BY category"}
                    ]
                }
            },
            "required": ["query"]
        }

    def _extract_first_token(self, query: str) -> str:
        # Strip single-line comments (-- ...), block comments (/* ... */), and whitespace
        cleaned = re.sub(r'--.*?(\n|$)', ' ', query or "")
        cleaned = re.sub(r'/\*.*?\*/', ' ', cleaned, flags=re.DOTALL).strip()
        return cleaned.lower().split(None, 1)[0] if cleaned else ""

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors = []
        q = (config.get("query") or "").strip()
        if not q:
            errors.append("SQL query is required.")
            return errors

        first_token = self._extract_first_token(q)
        if first_token not in ("select", "with", "pivot", "unpivot", "from"):
            errors.append("Only read-only SELECT, WITH, PIVOT, and UNPIVOT queries are permitted for safety.")

        # Safety check: block mutating DDL/DML statements
        cleaned = re.sub(r'--.*?(\n|$)', ' ', q)
        cleaned = re.sub(r'/\*.*?\*/', ' ', cleaned, flags=re.DOTALL)
        forbidden = ["drop ", "delete ", "insert ", "update ", "alter ", "create ", "truncate ", "attach ", "copy "]
        lower_q = cleaned.lower()
        for kw in forbidden:
            if re.search(rf"\b{kw.strip()}\b", lower_q):
                errors.append(f"Disallowed DDL/DML keyword '{kw.strip().upper()}' detected in SQL step.")
                break

        return errors

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        df: pd.DataFrame = inputs.get("dataframe")
        if df is None:
            raise ValueError("SQLStepRecipe expects 'dataframe' in inputs.")

        query = (config.get("query") or "SELECT * FROM df").strip()
        first_token = self._extract_first_token(query)
        if first_token not in ("select", "with", "pivot", "unpivot", "from"):
            raise ValueError(f"Only read-only SELECT, WITH, PIVOT, and UNPIVOT queries are permitted. Found: '{first_token}'.")

        # Check for forbidden mutations
        cleaned = re.sub(r'--.*?(\n|$)', ' ', query)
        cleaned = re.sub(r'/\*.*?\*/', ' ', cleaned, flags=re.DOTALL)
        forbidden = ["drop ", "delete ", "insert ", "update ", "alter ", "create ", "truncate ", "attach ", "copy "]
        lower_q = cleaned.lower()
        for kw in forbidden:
            if re.search(rf"\b{kw.strip()}\b", lower_q):
                raise ValueError(f"Disallowed DDL/DML keyword '{kw.strip().upper()}' detected in SQL query.")

        try:
            import duckdb
            try:
                con = duckdb.connect()
                # Register aliases for maximum developer usability
                con.register("df", df)
                con.register("data", df)
                con.register("input_df", df)
                result_df = con.execute(query).df()
            finally:
                try:
                    con.close()
                except Exception:
                    pass
        except ImportError:
            # Resilient fallback to Python's built-in sqlite3 in-memory engine
            import sqlite3
            conn = sqlite3.connect(":memory:")
            try:
                df.to_sql("df", conn, index=False, if_exists="replace")
                df.to_sql("data", conn, index=False, if_exists="replace")
                df.to_sql("input_df", conn, index=False, if_exists="replace")
                result_df = pd.read_sql_query(query, conn)
            except Exception as e:
                raise ValueError(f"SQL execution error: {str(e)} (Incoming table is available as 'df')") from e
            finally:
                conn.close()
        except Exception as e:
            raise ValueError(f"SQL execution error in DuckDB: {str(e)} (Incoming table is available as 'df')") from e

        return {
            "dataframe": result_df,
            "feature_names": list(result_df.columns),
            "output_summary": {"row_count": len(result_df), "columns": list(result_df.columns)}
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        q = config.get("query", "SELECT * FROM df")
        return f"import duckdb\ncon = duckdb.connect()\ncon.register('df', df)\nresult_df = con.execute(\"\"\"{q}\"\"\").df()\ncon.close()"