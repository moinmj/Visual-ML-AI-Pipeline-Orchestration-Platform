import re
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from backend.app.recipes.base.recipe import BaseRecipe


def normalize_datetime_format(fmt: Optional[str]) -> Tuple[Optional[str], bool]:
    """
    Translates common human-readable date formats (e.g. 'DD-MM-YYYY') into
    Python strftime format strings ('%d-%m-%Y') and flags dayfirst.
    """
    if not fmt:
        return None, False
    s = str(fmt).strip()
    if "(" in s and ")" in s:
        match = re.search(r"\((%[^)]+)\)", s)
        if match:
            s = match.group(1)

    mapping = {
        "dd-mm-yyyy": ("%d-%m-%Y", True),
        "dd/mm/yyyy": ("%d/%m/%Y", True),
        "dd.mm.yyyy": ("%d.%m.%Y", True),
        "yyyy-mm-dd": ("%Y-%m-%d", False),
        "yyyy/mm/dd": ("%Y/%m/%d", False),
        "yyyy.mm.dd": ("%Y.%m.%d", False),
        "mm-dd-yyyy": ("%m-%d-%Y", False),
        "mm/dd/yyyy": ("%m/%d/%Y", False),
        "%d-%m-%y": ("%d-%m-%Y", True),
        "%d/%m/%y": ("%d/%m/%Y", True),
        "%d-%m-%Y": ("%d-%m-%Y", True),
        "%d/%m/%Y": ("%d/%m/%Y", True),
        "%Y-%m-%d": ("%Y-%m-%d", False),
        "%Y/%m/%d": ("%Y/%m/%d", False),
        "%m-%d-%Y": ("%m-%d-%Y", False),
        "%m/%d/%Y": ("%m/%d/%Y", False),
    }
    lower_s = s.lower()
    if lower_s in mapping:
        return mapping[lower_s]
    is_dayfirst = "%d" in s and s.find("%d") < s.find("%m") if ("%d" in s and "%m" in s) else False
    return s, is_dayfirst


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
                    "description": "Key-value dictionary mapping column names to target types: 'numeric', 'integer', 'string', 'categorical', 'datetime', 'boolean'. Example: {\"Store\": \"categorical\", \"Date\": \"datetime\", \"Holiday_Flag\": \"boolean\"}",
                    "default": {}
                },
                "datetime_format": {
                    "type": "string",
                    "title": "Datetime Format",
                    "description": "Explicit datetime format like 'DD-MM-YYYY' (%d-%m-%Y) or 'YYYY-MM-DD' (%Y-%m-%d). Leave empty for auto-inference.",
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

        # Support conversions as dict, JSON string, or list of dicts [{"column": "x", "target_type": "int"}]
        rule_map = {}
        if isinstance(conversions, str):
            c_str = conversions.strip()
            if c_str in ["[object Object]", "object Object", ""]:
                conversions = {}
            else:
                try:
                    import json
                    conversions = json.loads(c_str)
                except Exception:
                    conversions = {}
        if isinstance(conversions, dict):
            rule_map = dict(conversions)
        elif isinstance(conversions, (list, tuple)):
            for item in conversions:
                if isinstance(item, dict) and "column" in item and "target_type" in item:
                    rule_map[item["column"]] = item["target_type"]

        # Auto-recovery: If no explicit datetime rule exists in rule_map, detect temporal columns automatically
        has_datetime_rule = any(str(v).lower() in ["datetime", "date", "timestamp"] for v in rule_map.values())
        if not has_datetime_rule:
            for col in df_out.columns:
                if any(kw in col.lower() for kw in ["date", "time", "timestamp", "ds", "period"]):
                    if not pd.api.types.is_datetime64_any_dtype(df_out[col]):
                        rule_map[col] = "datetime"

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
                    # Convert IDs and numbers to string-backed categories so models don't treat them as ordinal numbers
                    if pd.api.types.is_numeric_dtype(series):
                        df_out[col_name] = series.astype(str).astype("category")
                    else:
                        df_out[col_name] = series.astype("category")

                elif target in ["datetime", "timestamp", "date"]:
                    fmt_to_use, is_dayfirst = normalize_datetime_format(dt_format)
                    if fmt_to_use:
                        try:
                            df_out[col_name] = pd.to_datetime(series, format=fmt_to_use, errors=errors)
                        except Exception:
                            df_out[col_name] = pd.to_datetime(series, dayfirst=is_dayfirst, errors=errors)
                    else:
                        df_out[col_name] = pd.to_datetime(series, errors=errors)

                elif target in ["boolean", "bool"]:
                    bool_map = {
                        "true": True, "1": True, "yes": True, "t": True, "y": True, 1: True, 1.0: True,
                        "false": False, "0": False, "no": False, "f": False, "n": False, 0: False, 0.0: False
                    }
                    if pd.api.types.is_numeric_dtype(series):
                        df_out[col_name] = series.map({1: True, 0: False, 1.0: True, 0.0: False}).fillna(series.astype(bool)).astype(bool)
                    elif pd.api.types.is_string_dtype(series) or series.dtype == object:
                        df_out[col_name] = series.astype(str).str.strip().str.lower().map(bool_map).fillna(False).astype(bool)
                    else:
                        df_out[col_name] = series.astype(bool)

            except Exception as e:
                if errors == "raise":
                    raise ValueError(f"Failed to convert column '{col_name}' to '{target_type}': {str(e)}")

        return {"dataframe": df_out}

    def to_code(self, config: Dict[str, Any]) -> str:
        conversions = config.get("conversions", {})
        dt_format = config.get("datetime_format", "")
        lines = [
            "# Data Type Conversions & Casting",
            f"conversions = {conversions}",
            f"datetime_format = '{dt_format}'",
            "for col, target in conversions.items():",
            "    if col in df.columns:",
            "        if target in ['numeric', 'float']:",
            "            df[col] = pd.to_numeric(df[col], errors='coerce')",
            "        elif target in ['integer', 'int']:",
            "            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)",
            "        elif target in ['categorical', 'category']:",
            "            df[col] = df[col].astype(str).astype('category')",
            "        elif target in ['datetime', 'date']:",
            "            df[col] = pd.to_datetime(df[col], format='%d-%m-%Y' if 'DD-MM' in datetime_format else None, errors='coerce')",
            "        elif target in ['boolean', 'bool']:",
            "            df[col] = df[col].map({1: True, 0: False, '1': True, '0': False, 'true': True, 'false': False}).fillna(False).astype(bool)",
            "        elif target in ['string', 'text']:",
            "            df[col] = df[col].astype(str)"
        ]
        return "\n".join(lines)
