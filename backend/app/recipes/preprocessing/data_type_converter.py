import pandas as pd
from typing import Dict, Any, List, Optional
from backend.app.recipes.base.recipe import BaseRecipe


class DataTypeConverterRecipe(BaseRecipe):
    recipe_id = "data_type_converter"
    name = "Data Type Converter & Cast"
    version = "1.0.0"
    category = "preprocessing"
    description = "Explicitly converts and casts column data types (numeric, integer, string, categorical, datetime, boolean)."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "conversions": {
                    "type": "object",
                    "title": "Column Type Mappings",
                    "description": "Key-value dictionary mapping column names to target types: 'numeric', 'integer', 'string', 'categorical', 'datetime', 'boolean'. Example: {\"age\": \"integer\", \"income\": \"numeric\", \"signup_date\": \"datetime\"}",
                    "default": {}
                },
                "datetime_format": {
                    "type": "string",
                    "title": "Datetime Format",
                    "description": "Optional explicit datetime parsing format (e.g. '%Y-%m-%d' or '%d/%m/%Y'). Leave empty for automatic inference.",
                    "default": ""
                },
                "errors": {
                    "type": "string",
                    "title": "Invalid Value Handling",
                    "enum": ["coerce", "ignore", "raise"],
                    "default": "coerce",
                    "description": "'coerce' replaces unconvertible values with NaN; 'ignore' leaves invalid values untouched; 'raise' halts execution on conversion failure."
                }
            }
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        df: pd.DataFrame = inputs.get("dataframe")
        if df is None:
            if context and isinstance(context, dict) and "dataframe" in context:
                df = context["dataframe"]
            else:
                raise ValueError("DataTypeConverterRecipe expects 'dataframe' in inputs.")

        df_out = df.copy()
        conversions = config.get("conversions", {})
        dt_format = config.get("datetime_format", "") or None
        errors = config.get("errors", "coerce")

        # Support conversions as dict or list of dicts [{"column": "x", "target_type": "int"}]
        rule_map = {}
        if isinstance(conversions, dict):
            rule_map = conversions
        elif isinstance(conversions, (list, tuple)):
            for item in conversions:
                if isinstance(item, dict) and "column" in item and "target_type" in item:
                    rule_map[item["column"]] = item["target_type"]

        for col_name, target_type in rule_map.items():
            if col_name not in df_out.columns:
                continue

            target = str(target_type).strip().lower()
            series = df_out[col_name]

            try:
                if target in ["numeric", "float", "number"]:
                    df_out[col_name] = pd.to_numeric(series, errors=errors)

                elif target in ["integer", "int"]:
                    num_s = pd.to_numeric(series, errors="coerce")
                    df_out[col_name] = num_s.fillna(0).astype(int)

                elif target in ["string", "text", "str"]:
                    df_out[col_name] = series.astype(str)

                elif target in ["categorical", "category"]:
                    df_out[col_name] = series.astype("category")

                elif target in ["datetime", "timestamp", "date"]:
                    if dt_format:
                        df_out[col_name] = pd.to_datetime(series, format=dt_format, errors=errors)
                    else:
                        df_out[col_name] = pd.to_datetime(series, errors=errors)

                elif target in ["boolean", "bool"]:
                    bool_map = {
                        "true": True, "1": True, "yes": True, "t": True, "y": True,
                        "false": False, "0": False, "no": False, "f": False, "n": False
                    }
                    if pd.api.types.is_string_dtype(series) or series.dtype == object:
                        df_out[col_name] = series.astype(str).str.strip().str.lower().map(bool_map).fillna(False)
                    else:
                        df_out[col_name] = series.astype(bool)

            except Exception as e:
                if errors == "raise":
                    raise ValueError(f"Failed to convert column '{col_name}' to '{target_type}': {str(e)}")

        return {"dataframe": df_out}

    def to_code(self, config: Dict[str, Any]) -> str:
        conversions = config.get("conversions", {})
        lines = ["# Data Type Conversions", f"conversions = {conversions}"]
        lines.append("for col, target in conversions.items():")
        lines.append("    if col in df.columns:")
        lines.append("        if target in ['numeric', 'float']:")
        lines.append("            df[col] = pd.to_numeric(df[col], errors='coerce')")
        lines.append("        elif target in ['integer', 'int']:")
        lines.append("            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)")
        lines.append("        elif target in ['string', 'text']:")
        lines.append("            df[col] = df[col].astype(str)")
        lines.append("        elif target in ['datetime', 'date']:")
        lines.append("            df[col] = pd.to_datetime(df[col], errors='coerce')")
        lines.append("        elif target in ['categorical', 'category']:")
        lines.append("            df[col] = df[col].astype('category')")
        lines.append("        elif target in ['boolean', 'bool']:")
        lines.append("            df[col] = df[col].astype(bool)")
        return "\n".join(lines)
