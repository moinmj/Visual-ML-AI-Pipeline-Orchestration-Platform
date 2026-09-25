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
                    "description": "Standard SQL. Reference the incoming data as table 'df'."
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
                        {"label": "Union", "query": "SELECT * FROM df WHERE year = 2023\nUNION\nSELECT * FROM df WHERE year = 2024"}
                    ]
                }
            },
            "required": ["query"]
        }

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors = []
        q = (config.get("query") or "").strip()
        if not q:
            errors.append("SQL query is required.")
        elif not q.lower().lstrip().split(None, 1)[0] in ("select", "with", "pivot", "unpivot", "from"):
            errors.append("Only read-only SELECT/WITH/PIVOT/UNPIVOT queries are permitted for safety.")
        return errors

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        df: pd.DataFrame = inputs.get("dataframe")
        if df is None:
            raise ValueError("SQLStepRecipe expects 'dataframe' in inputs.")

        query = (config.get("query") or "SELECT * FROM df").strip()
        if not query.lower().startswith("select"):
            raise ValueError("Only SELECT queries are permitted.")

        import duckdb
        result_df = duckdb.query_df(df, "df", query).to_df()

        return {
            "dataframe": result_df,
            "feature_names": list(result_df.columns),
            "output_summary": {"row_count": len(result_df), "columns": list(result_df.columns)}
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        q = config.get("query", "SELECT * FROM df")
        return f"import duckdb\nresult_df = duckdb.query_df(df, 'df', \"\"\"{q}\"\"\").to_df()"