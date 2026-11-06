import csv
import io
import json
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from backend.app.core.exceptions import ValidationException
from backend.app.core.logging import logger
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class ManualDataEntryRecipe(BaseRecipe):
    recipe_id = "manual_data_entry"
    name = "Manual Data Entry"
    version = "1.0.0"
    category = "ingestion"
    description = (
        "Type or paste rows directly into an editable table or text area "
        "(JSON records, CSV text, or 2D grid) to use as pipeline input."
    )
    input_types = []
    output_types = ["dataframe"]
    inputs = []
    outputs = [
        RecipePort(
            id="output",
            label="Output Table",
            type="dataframe",
            description="Primary dataset generated from manual data entry",
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "data_format": {
                    "type": "string",
                    "title": "Data Format",
                    "enum": ["json_records", "csv_text", "grid_rows"],
                    "default": "json_records",
                    "input_type": "select",
                    "description": (
                        "Format of the input data: 'json_records' (array of objects), "
                        "'csv_text' (delimited text), or 'grid_rows' (2D array with headers)."
                    ),
                },
                "data": {
                    "type": "string",
                    "title": "Data / Content",
                    "default": "",
                    "input_type": "textarea",
                    "description": (
                        "Data payload: raw CSV text, JSON array string (e.g. [{\"col1\": \"val1\"}]), "
                        "or list of objects."
                    ),
                },
                "delimiter": {
                    "type": "string",
                    "title": "Delimiter",
                    "default": ",",
                    "enum": [",", "\t", ";", "|"],
                    "input_type": "select",
                    "description": "Character separating columns when data_format is 'csv_text'.",
                },
                "has_header": {
                    "type": "boolean",
                    "title": "First Row Is Header",
                    "default": True,
                    "input_type": "checkbox",
                    "description": "Whether the CSV text or 2D grid array includes a header row with column names.",
                },
                "infer_types": {
                    "type": "boolean",
                    "title": "Infer Types",
                    "default": True,
                    "input_type": "checkbox",
                    "description": "Automatically infer numeric, boolean, and datetime data types.",
                },
                "column_types": {
                    "type": "object",
                    "title": "Explicit Column Types",
                    "default": {},
                    "input_type": "key_value",
                    "description": (
                        "Optional mapping of column names to target types "
                        "('int', 'float', 'str', 'bool', 'datetime')."
                    ),
                },
                "strip_whitespace": {
                    "type": "boolean",
                    "title": "Strip Whitespace",
                    "default": True,
                    "input_type": "checkbox",
                    "description": "Strip leading and trailing whitespace from string values and column headers.",
                },
                "null_values": {
                    "type": "array",
                    "items": {"type": "string"},
                    "title": "Null Value Tokens",
                    "default": ["", "null", "NULL", "None", "NaN", "nan", "NA", "N/A"],
                    "input_type": "tags",
                    "description": "List of string tokens to treat as null / missing values.",
                },
                "coerce_errors": {
                    "type": "boolean",
                    "title": "Coerce Type Errors",
                    "default": True,
                    "input_type": "checkbox",
                    "description": "If True, type conversion errors are converted to null/NaN; if False, an error is raised.",
                },
            },
            "required": [],
        }

    def execute(
        self,
        inputs: Dict[str, Any],
        config: Dict[str, Any],
        context: Optional[Any] = None,
    ) -> Dict[str, Any]:
        # 1. Passthrough if DataFrame is already provided
        if "dataframe" in inputs and isinstance(inputs["dataframe"], pd.DataFrame):
            df = inputs["dataframe"]
            return {
                "dataframe": df,
                "output": df,
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns),
            }
        if "dataframe" in config and isinstance(config["dataframe"], pd.DataFrame):
            df = config["dataframe"]
            return {
                "dataframe": df,
                "output": df,
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns),
            }

        # 2. Extract configuration
        data_format = config.get("data_format", "json_records")
        raw_data = config.get("data")
        if raw_data is None:
            raw_data = (
                config.get("dataframe_records")
                or config.get("records")
                or config.get("rows")
                or config.get("csv_text")
                or ""
            )

        delimiter = config.get("delimiter", ",") or ","
        has_header = config.get("has_header", True)
        infer_types = config.get("infer_types", True)
        column_types = config.get("column_types", {}) or {}
        strip_whitespace = config.get("strip_whitespace", True)
        null_tokens = config.get(
            "null_values", ["", "null", "NULL", "None", "NaN", "nan", "NA", "N/A"]
        )
        null_tokens_set = set(null_tokens) if isinstance(null_tokens, list) else set()
        coerce_errors = config.get("coerce_errors", True)

        # 3. Parse input into DataFrame
        df = self._parse_data(
            raw_data=raw_data,
            data_format=data_format,
            delimiter=delimiter,
            has_header=has_header,
        )

        # 4. Handle empty DataFrame edge case
        if df.empty and not df.columns.any() and column_types:
            df = pd.DataFrame(columns=list(column_types.keys()))

        if df.empty:
            return {
                "dataframe": df,
                "output": df,
                "row_count": 0,
                "column_count": len(df.columns),
                "columns": list(df.columns),
            }

        # 5. Clean column names
        if strip_whitespace:
            df.columns = [str(c).strip() for c in df.columns]

        # 6. Apply null token replacement and string stripping
        for col in df.columns:
            if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
                if strip_whitespace:
                    df[col] = df[col].apply(
                        lambda x: x.strip() if isinstance(x, str) else x
                    )
                if null_tokens_set:
                    df[col] = df[col].apply(
                        lambda x: np.nan if (isinstance(x, str) and x in null_tokens_set) or x is None else x
                    )

        # 7. Explicit column casting
        for col, target_type in column_types.items():
            if col not in df.columns:
                continue
            t_str = str(target_type).strip().lower()
            try:
                if t_str in ("int", "integer", "int64", "int32"):
                    df[col] = pd.to_numeric(
                        df[col], errors="coerce" if coerce_errors else "raise"
                    ).astype("Int64")
                elif t_str in ("float", "float64", "number", "double"):
                    df[col] = pd.to_numeric(
                        df[col], errors="coerce" if coerce_errors else "raise"
                    ).astype("float64")
                elif t_str in ("bool", "boolean"):
                    df[col] = self._cast_boolean(df[col], coerce_errors=coerce_errors)
                elif t_str in ("datetime", "date", "timestamp"):
                    df[col] = pd.to_datetime(
                        df[col], errors="coerce" if coerce_errors else "raise"
                    )
                elif t_str in ("str", "string", "text"):
                    df[col] = df[col].astype("string")
            except Exception as e:
                raise ValidationException(
                    f"Failed to cast column '{col}' to type '{target_type}': {str(e)}"
                )

        # 8. Auto type inference for remaining columns
        if infer_types:
            for col in df.columns:
                if col in column_types:
                    continue
                # Try numeric conversion
                if df[col].dtype == object:
                    try:
                        converted_numeric = pd.to_numeric(df[col])
                        df[col] = converted_numeric
                        continue
                    except (ValueError, TypeError):
                        pass
                    # Try boolean conversion if all non-null values look like booleans
                    non_nulls = df[col].dropna()
                    if len(non_nulls) > 0 and all(
                        str(x).strip().lower() in ("true", "false", "1", "0", "yes", "no")
                        for x in non_nulls
                    ):
                        try:
                            df[col] = self._cast_boolean(df[col], coerce_errors=True)
                            continue
                        except Exception:
                            pass
                    # Try datetime conversion if values match ISO dates
                    if len(non_nulls) > 0 and all(
                        isinstance(x, str) and len(x) >= 8 and any(sep in x for sep in ("-", "/", ":"))
                        for x in non_nulls
                    ):
                        try:
                            converted_dt = pd.to_datetime(df[col], errors="coerce")
                            if converted_dt.notna().sum() == len(non_nulls):
                                df[col] = converted_dt
                        except Exception:
                            pass

        return {
            "dataframe": df,
            "output": df,
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
        }

    def _parse_data(
        self,
        raw_data: Any,
        data_format: str,
        delimiter: str,
        has_header: bool,
    ) -> pd.DataFrame:
        """Parses raw input payload according to the chosen data format."""
        if raw_data is None:
            return pd.DataFrame()

        # If already a DataFrame
        if isinstance(raw_data, pd.DataFrame):
            return raw_data.copy()

        # If list or dict structure directly passed
        if isinstance(raw_data, list):
            if not raw_data:
                return pd.DataFrame()
            first = raw_data[0]
            if isinstance(first, dict):
                return pd.DataFrame(raw_data)
            if isinstance(first, (list, tuple)):
                return self._parse_grid_rows(raw_data, has_header)
            return pd.DataFrame({"value": raw_data})

        if isinstance(raw_data, dict):
            if not raw_data:
                return pd.DataFrame()
            # If dict of lists
            if any(isinstance(v, list) for v in raw_data.values()):
                return pd.DataFrame(raw_data)
            return pd.DataFrame([raw_data])

        # If raw_data is a string
        if isinstance(raw_data, str):
            text = raw_data.strip()
            if not text:
                return pd.DataFrame()

            fmt = (data_format or "json_records").lower()

            if fmt == "json_records":
                return self._parse_json_text(text, delimiter, has_header)
            elif fmt == "csv_text":
                return self._parse_csv_text(text, delimiter, has_header)
            elif fmt == "grid_rows":
                return self._parse_grid_text(text, delimiter, has_header)
            else:
                # Default fallback: try json first, then csv
                try:
                    return self._parse_json_text(text, delimiter, has_header)
                except Exception:
                    return self._parse_csv_text(text, delimiter, has_header)

        # Fallback for unexpected data types
        try:
            return pd.DataFrame(raw_data)
        except Exception as e:
            raise ValidationException(f"Unsupported manual data payload type ({type(raw_data)}): {str(e)}")

    def _parse_json_text(self, text: str, delimiter: str, has_header: bool) -> pd.DataFrame:
        """Parse JSON text into DataFrame, with fallback to CSV if user pasted CSV into json mode."""
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                if not parsed:
                    return pd.DataFrame()
                if isinstance(parsed[0], dict):
                    return pd.DataFrame(parsed)
                if isinstance(parsed[0], (list, tuple)):
                    return self._parse_grid_rows(parsed, has_header)
                return pd.DataFrame({"value": parsed})
            elif isinstance(parsed, dict):
                if any(isinstance(v, list) for v in parsed.values()):
                    return pd.DataFrame(parsed)
                return pd.DataFrame([parsed])
            else:
                return pd.DataFrame({"value": [parsed]})
        except json.JSONDecodeError as json_err:
            # Check if user accidentally pasted CSV text into json_records mode
            if not text.startswith(("{", "[")) and (delimiter in text or "\n" in text):
                logger.info("JSON parse failed; attempting CSV fallback on manual entry text.")
                try:
                    return self._parse_csv_text(text, delimiter, has_header)
                except Exception:
                    pass
            raise ValidationException(f"Invalid JSON format for Manual Data Entry: {json_err.msg} at line {json_err.lineno}")

    def _parse_csv_text(self, text: str, delimiter: str, has_header: bool) -> pd.DataFrame:
        """Parses CSV/delimited text into DataFrame handling irregularities."""
        try:
            df = pd.read_csv(
                io.StringIO(text),
                sep=delimiter,
                header=0 if has_header else None,
                skipinitialspace=True,
            )
            if not has_header and not df.empty:
                df.columns = [f"column_{i+1}" for i in range(len(df.columns))]
            return df
        except Exception:
            # Fallback to csv.reader with row padding if ragged lines exist
            reader = csv.reader(io.StringIO(text), delimiter=delimiter)
            rows = [row for row in reader if row]
            if not rows:
                return pd.DataFrame()
            return self._parse_grid_rows(rows, has_header)

    def _parse_grid_text(self, text: str, delimiter: str, has_header: bool) -> pd.DataFrame:
        """Parses 2D grid text (either JSON array of arrays or delimited rows)."""
        if text.startswith("["):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return self._parse_grid_rows(parsed, has_header)
            except Exception:
                pass
        return self._parse_csv_text(text, delimiter, has_header)

    def _parse_grid_rows(self, rows: List[Any], has_header: bool) -> pd.DataFrame:
        """Parses a 2D list of rows into a DataFrame, padding jagged rows."""
        if not rows:
            return pd.DataFrame()

        # Find maximum row length
        max_cols = max(len(r) if isinstance(r, (list, tuple)) else 1 for r in rows)
        if max_cols == 0:
            return pd.DataFrame()

        normalized_rows: List[List[Any]] = []
        for r in rows:
            if isinstance(r, (list, tuple)):
                row_list = list(r)
                if len(row_list) < max_cols:
                    row_list.extend([None] * (max_cols - len(row_list)))
                normalized_rows.append(row_list[:max_cols])
            else:
                normalized_rows.append([r] + [None] * (max_cols - 1))

        if has_header and len(normalized_rows) > 0:
            headers = [
                str(h).strip() if h is not None and str(h).strip() else f"column_{i+1}"
                for i, h in enumerate(normalized_rows[0])
            ]
            data_rows = normalized_rows[1:]
            return pd.DataFrame(data_rows, columns=headers)
        else:
            headers = [f"column_{i+1}" for i in range(max_cols)]
            return pd.DataFrame(normalized_rows, columns=headers)

    def _cast_boolean(self, series: pd.Series, coerce_errors: bool = True) -> pd.Series:
        """Helper to cast series to nullable boolean dtype."""
        truthy = {"true", "1", "yes", "t", "y"}
        falsy = {"false", "0", "no", "f", "n"}

        def map_bool(val: Any) -> Any:
            if val is None or pd.isna(val):
                return pd.NA
            if isinstance(val, bool):
                return val
            if isinstance(val, (int, float)):
                if val == 1:
                    return True
                if val == 0:
                    return False
            s = str(val).strip().lower()
            if s in truthy:
                return True
            if s in falsy:
                return False
            if coerce_errors:
                return pd.NA
            raise ValueError(f"Cannot cast '{val}' to boolean")

        mapped = series.apply(map_bool)
        return mapped.astype("boolean")
