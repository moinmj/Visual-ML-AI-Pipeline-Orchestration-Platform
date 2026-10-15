"""
Formula Calculator Recipe.
Sub-group: Custom Logic
Category: Data Preparation & Transformation (preprocessing)

Creates or updates columns using spreadsheet-style formulas with support for
arithmetic, conditional IF-logic, text manipulation, and statistical functions.
"""
import re
import ast
import operator
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class FormulaCalculatorRecipe(BaseRecipe):
    recipe_id = "formula_calculator"
    name = "Formula Calculator"
    version = "1.0.0"
    category = "preprocessing"
    group = "Data Preparation & Transformation"
    subgroup = "Custom Logic"
    description = "Creates or updates columns using spreadsheet-style formulas with support for arithmetic, IF conditions, and text/numeric transformations."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="input",
            label="Input Table",
            type="dataframe",
            required=True,
            max_connections=1,
            description="Dataset to apply formula calculations on"
        )
    ]

    outputs = [
        RecipePort(
            id="dataframe",
            label="Calculated Table",
            type="dataframe",
            description="Dataset augmented with new or updated calculated formula columns"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "target_column": {
                    "type": "string",
                    "title": "Target Column Name",
                    "description": "Name of the new or existing column to store the formula result.",
                    "default": "calculated_metric"
                },
                "formula": {
                    "type": "string",
                    "title": "Spreadsheet Formula",
                    "description": "Spreadsheet-style formula. Column names can be wrapped in brackets [Column Name]. Examples: '[Revenue] - [Cost]', 'IF([Age] >= 18, 1, 0)', 'UPPER([Category])', 'ROUND([Price] * 1.15, 2)'.",
                    "default": ""
                },
                "formulas": {
                    "type": "array",
                    "title": "Batch Formulas List",
                    "items": {
                        "type": "object",
                        "properties": {
                            "target_column": {"type": "string", "title": "Target Column"},
                            "formula": {"type": "string", "title": "Formula"}
                        },
                        "required": ["target_column", "formula"]
                    },
                    "default": [],
                    "description": "Optional list of multiple formulas to evaluate sequentially in one step."
                },
                "on_error": {
                    "type": "string",
                    "title": "Error Handling Action",
                    "enum": ["set_null", "raise_error", "keep_original"],
                    "default": "set_null",
                    "description": "Action when a formula evaluation fails (e.g. division by zero or invalid type)."
                }
            },
            "required": []
        }

    def _resolve_bracket_tokens(self, formula: str, df: pd.DataFrame) -> Tuple[str, Dict[str, str]]:
        """
        Replaces [Column Name] with safe internal variable identifiers.
        Returns transformed formula and mapping {var_name: actual_col_name}.
        """
        var_map: Dict[str, str] = {}
        counter = 0

        def replacer(match):
            nonlocal counter
            col_name = match.group(1).strip()
            var_name = f"__col_{counter}__"
            var_map[var_name] = col_name
            counter += 1
            return var_name

        pattern = r"\[([^\]]+)\]"
        transformed = re.sub(pattern, replacer, formula)
        return transformed, var_map

    def _evaluate_single_formula(self, df: pd.DataFrame, formula: str, on_error: str) -> pd.Series:
        """Evaluates an individual spreadsheet formula against df."""
        if not formula or not formula.strip():
            return pd.Series([None] * len(df), index=df.index)

        clean_formula = formula.strip()
        if clean_formula.startswith("="):
            clean_formula = clean_formula[1:].strip()

        transformed_expr, var_map = self._resolve_bracket_tokens(clean_formula, df)

        # Build execution scope
        scope: Dict[str, Any] = {
            "np": np,
            "pd": pd,
            "IF": lambda cond, t, f: np.where(cond, t, f),
            "if_else": lambda cond, t, f: np.where(cond, t, f),
            "ROUND": lambda val, n=2: np.round(pd.to_numeric(val, errors="coerce"), n),
            "ABS": lambda val: np.abs(pd.to_numeric(val, errors="coerce")),
            "SQRT": lambda val: np.sqrt(pd.to_numeric(val, errors="coerce")),
            "LOG": lambda val: np.log(pd.to_numeric(val, errors="coerce")),
            "LN": lambda val: np.log(pd.to_numeric(val, errors="coerce")),
            "POWER": lambda base, exp: pd.to_numeric(base, errors="coerce") ** exp,
            "UPPER": lambda s: pd.Series(s, index=df.index).astype(str).str.upper(),
            "LOWER": lambda s: pd.Series(s, index=df.index).astype(str).str.lower(),
            "TRIM": lambda s: pd.Series(s, index=df.index).astype(str).str.strip(),
            "LEN": lambda s: pd.Series(s, index=df.index).astype(str).str.len(),
            "COALESCE": lambda *args: pd.concat([pd.Series(a, index=df.index) for a in args], axis=1).bfill(axis=1).iloc[:, 0],
            "CONCAT": lambda *args: pd.Series("", index=df.index).str.cat([pd.Series(a, index=df.index).fillna("").astype(str) for a in args]),
            "CONCATENATE": lambda *args: pd.Series("", index=df.index).str.cat([pd.Series(a, index=df.index).fillna("").astype(str) for a in args]),
            "MAX": lambda a, b: np.maximum(a, b),
            "MIN": lambda a, b: np.minimum(a, b),
        }

        # Inject column series from var_map
        for var_name, real_col in var_map.items():
            if real_col in df.columns:
                scope[var_name] = df[real_col]
            else:
                if on_error == "raise_error":
                    raise KeyError(f"Column '{real_col}' referenced in formula not found in dataset columns: {list(df.columns)}")
                scope[var_name] = pd.Series([np.nan] * len(df), index=df.index)

        # Also inject columns directly if they are valid python identifiers
        for col in df.columns:
            if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", str(col)) and col not in scope:
                scope[col] = df[col]

        # First attempt: eval with full vectorized scope
        try:
            res = eval(transformed_expr, {"__builtins__": {}}, scope)
            if isinstance(res, pd.Series):
                return res
            elif isinstance(res, np.ndarray):
                return pd.Series(res, index=df.index)
            else:
                # Scalar result (e.g. 42 or 'constant')
                return pd.Series([res] * len(df), index=df.index)
        except Exception as e:
            # Second attempt: row-wise evaluation for complex string or branching expressions
            try:
                def row_eval(row):
                    row_scope = dict(scope)
                    for var_name, real_col in var_map.items():
                        row_scope[var_name] = row.get(real_col)
                    for col in df.columns:
                        if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", str(col)):
                            row_scope[col] = row.get(col)
                    row_scope["IF"] = lambda c, t, f: t if c else f
                    row_scope["ROUND"] = lambda v, n=2: round(float(v), n) if pd.notna(v) else None
                    row_scope["ABS"] = lambda v: abs(float(v)) if pd.notna(v) else None
                    row_scope["UPPER"] = lambda s: str(s).upper() if pd.notna(s) else ""
                    row_scope["LOWER"] = lambda s: str(s).lower() if pd.notna(s) else ""
                    row_scope["TRIM"] = lambda s: str(s).strip() if pd.notna(s) else ""
                    row_scope["LEN"] = lambda s: len(str(s)) if pd.notna(s) else 0
                    row_scope["CONCAT"] = lambda *args: "".join(str(a) for a in args if pd.notna(a))
                    return eval(transformed_expr, {"__builtins__": {}}, row_scope)

                return df.apply(row_eval, axis=1)
            except Exception as e2:
                if on_error == "raise_error":
                    raise ValueError(f"Failed to evaluate formula '{formula}': {str(e2)}")
                return pd.Series([np.nan] * len(df), index=df.index)

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        df = inputs.get("dataframe") if "dataframe" in inputs else inputs.get("input")
        if df is None:
            df = inputs.get("df")

        if df is None:
            empty_df = pd.DataFrame()
            return {
                "dataframe": empty_df,
                "metrics": {"formulas_evaluated": 0, "rows_processed": 0},
                "output_summary": {"title": "Formula Calculator", "message": "No input table provided.", "metrics": {}}
            }

        df = df.copy()
        if len(df) == 0:
            return {
                "dataframe": df,
                "metrics": {"formulas_evaluated": 0, "rows_processed": 0},
                "output_summary": {"title": "Formula Calculator", "message": "Input table is empty.", "metrics": {}}
            }

        target_column = config.get("target_column")
        formula = config.get("formula")
        formulas_list = config.get("formulas", [])
        on_error = config.get("on_error", "set_null")

        batch_tasks: List[Dict[str, str]] = []
        if target_column and formula:
            batch_tasks.append({"target_column": target_column, "formula": formula})

        for item in formulas_list:
            t = item.get("target_column")
            f = item.get("formula")
            if t and f:
                batch_tasks.append({"target_column": t, "formula": f})

        computed_cols = []
        for task in batch_tasks:
            col_name = task["target_column"]
            expr = task["formula"]
            orig_col = df[col_name].copy() if col_name in df.columns else None

            res_series = self._evaluate_single_formula(df, expr, on_error)

            if on_error == "keep_original" and orig_col is not None:
                df[col_name] = res_series.combine_first(orig_col)
            else:
                df[col_name] = res_series
            computed_cols.append(col_name)

        output_summary = {
            "title": "Formula Calculator",
            "message": f"Successfully calculated {len(computed_cols)} column(s): {', '.join(computed_cols) if computed_cols else 'None'}.",
            "metrics": {
                "formulas_evaluated": len(computed_cols),
                "columns_computed": computed_cols,
                "rows_processed": len(df)
            }
        }

        return {
            "dataframe": df,
            "metrics": output_summary["metrics"],
            "output_summary": output_summary
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        target_column = repr(config.get("target_column", "calculated_metric"))
        formula = repr(config.get("formula", ""))
        return (
            f"# Recipe: {self.name}\n"
            f"# ID: {self.recipe_id}\n"
            f"# Formula: {formula} -> {target_column}\n"
            f"# Use formula_calculator to evaluate spreadsheet expressions\n"
        )
