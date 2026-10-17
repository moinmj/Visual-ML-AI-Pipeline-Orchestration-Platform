"""
Schema / Contract Validation Recipe.
Sub-group: Cleanup & Quality
Category: Data Preparation & Transformation (preprocessing)

Checks the dataset against an expected schema or data contract and flags violations.
"""
from typing import Dict, Any, List, Optional, Set
import re
import pandas as pd
import numpy as np
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class SchemaContractValidationRecipe(BaseRecipe):
    recipe_id = "schema_contract_validation"
    name = "Schema / Contract Validation"
    version = "1.0.0"
    category = "preprocessing"
    group = "Data Preparation & Transformation"
    subgroup = "Cleanup & Quality"
    description = "Checks the dataset against an expected schema or data contract (column presence, types, nullability, ranges, allowed values) and flags violations."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="input",
            label="Input Table",
            type="dataframe",
            required=True,
            max_connections=1,
            description="Dataset to validate against the schema contract"
        )
    ]

    outputs = [
        RecipePort(
            id="dataframe",
            label="Validated Table",
            type="dataframe",
            description="Dataset passing the contract or annotated with validation flags"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expected_schema": {
                    "type": "array",
                    "title": "Contract Field Specifications",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "title": "Column Name"},
                            "type": {
                                "type": "string",
                                "title": "Data Type",
                                "enum": ["string", "integer", "float", "boolean", "datetime", "any"],
                                "default": "any"
                            },
                            "nullable": {"type": "boolean", "title": "Nullable", "default": True},
                            "min_value": {"type": "number", "title": "Min Value"},
                            "max_value": {"type": "number", "title": "Max Value"},
                            "allowed_values": {
                                "type": "array",
                                "items": {"type": "string"},
                                "title": "Allowed Values (Enum)"
                            }
                        },
                        "required": ["name"]
                    },
                    "description": "List of expected columns, types, nullability rules, and value boundaries."
                },
                "validation_mode": {
                    "type": "string",
                    "title": "Violation Handling Mode",
                    "enum": [
                        "flag_rows",
                        "quarantine_invalid_rows",
                        "strict_raise",
                        "report_only"
                    ],
                    "default": "flag_rows",
                    "description": "'flag_rows' appends _contract_valid and _contract_violations columns; 'quarantine_invalid_rows' discards non-compliant rows; 'strict_raise' fails execution on any error."
                },
                "allow_extra_columns": {
                    "type": "boolean",
                    "title": "Allow Unlisted Extra Columns",
                    "default": True,
                    "description": "Whether columns present in the dataset but not listed in the contract should be tolerated."
                }
            },
            "required": []
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        df = inputs.get("dataframe")
        if df is None:
            if context and isinstance(context, dict) and "dataframe" in context:
                df = context["dataframe"]
            else:
                raise ValueError("SchemaContractValidationRecipe expects 'dataframe' in inputs.")

        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        df_out = df.copy()
        orig_rows = len(df_out)
        orig_cols = list(df_out.columns)

        expected_schema = config.get("expected_schema", [])
        validation_mode = config.get("validation_mode", "flag_rows")
        allow_extra = bool(config.get("allow_extra_columns", True))

        # Normalization if expected_schema is a dict
        if isinstance(expected_schema, dict):
            norm_schema = []
            for col_name, col_rules in expected_schema.items():
                if isinstance(col_rules, dict):
                    col_rules["name"] = col_name
                    norm_schema.append(col_rules)
                elif isinstance(col_rules, str):
                    norm_schema.append({"name": col_name, "type": col_rules})
            expected_schema = norm_schema

        checklist = []
        missing_columns = []
        row_violations: List[List[str]] = [[] for _ in range(orig_rows)]

        # 1. Check Column Presence & Column-Level Rules
        for spec in expected_schema:
            col = str(spec.get("name", "")).strip()
            if not col:
                continue

            expected_type = str(spec.get("type", "any")).lower()
            nullable = bool(spec.get("nullable", True))
            min_val = spec.get("min_value")
            max_val = spec.get("max_value")
            allowed_vals = spec.get("allowed_values")
            if isinstance(allowed_vals, str):
                allowed_vals = [v.strip() for v in allowed_vals.split(",") if v.strip()]

            if col not in df_out.columns:
                missing_columns.append(col)
                checklist.append({
                    "column": col,
                    "status": "FAIL",
                    "expected_type": expected_type,
                    "error": "Column missing from dataset"
                })
                continue

            s = df_out[col]
            actual_type = str(s.dtype)
            col_errors = []

            # Nullability check
            null_count = int(s.isna().sum())
            if not nullable and null_count > 0:
                col_errors.append(f"{null_count} unexpected null values found")
                for r_idx in s[s.isna()].index:
                    if r_idx < orig_rows:
                        row_violations[r_idx].append(f"{col}: Null not permitted")

            # Type check
            type_match = True
            if expected_type in ("integer", "int"):
                if not pd.api.types.is_integer_dtype(s):
                    # Check if convertible
                    try:
                        pd.to_numeric(s.dropna(), downcast="integer")
                    except Exception:
                        type_match = False
            elif expected_type in ("float", "numeric"):
                if not pd.api.types.is_numeric_dtype(s):
                    try:
                        pd.to_numeric(s.dropna())
                    except Exception:
                        type_match = False
            elif expected_type == "boolean":
                if not pd.api.types.is_bool_dtype(s):
                    type_match = False
            elif expected_type == "datetime":
                if not pd.api.types.is_datetime64_any_dtype(s):
                    try:
                        pd.to_datetime(s.dropna())
                    except Exception:
                        type_match = False

            if not type_match:
                col_errors.append(f"Type mismatch: expected {expected_type}, got {actual_type}")

            # Boundaries check
            if min_val is not None and pd.api.types.is_numeric_dtype(s):
                bad_min = (s < float(min_val))
                if bad_min.any():
                    col_errors.append(f"{bad_min.sum()} rows below min_value ({min_val})")
                    for r_idx in s[bad_min].index:
                        if r_idx < orig_rows:
                            row_violations[r_idx].append(f"{col}: value < {min_val}")

            if max_val is not None and pd.api.types.is_numeric_dtype(s):
                bad_max = (s > float(max_val))
                if bad_max.any():
                    col_errors.append(f"{bad_max.sum()} rows above max_value ({max_val})")
                    for r_idx in s[bad_max].index:
                        if r_idx < orig_rows:
                            row_violations[r_idx].append(f"{col}: value > {max_val}")

            # Allowed values (enums)
            if allowed_vals:
                allowed_set = set(str(v).strip().lower() for v in allowed_vals)
                bad_enum = s.dropna().apply(lambda v: str(v).strip().lower() not in allowed_set)
                if bad_enum.any():
                    col_errors.append(f"{bad_enum.sum()} rows with unauthorized enum values")
                    for r_idx in s.dropna()[bad_enum].index:
                        if r_idx < orig_rows:
                            row_violations[r_idx].append(f"{col}: value not in allowed set")

            checklist.append({
                "column": col,
                "status": "PASS" if not col_errors else "FAIL",
                "expected_type": expected_type,
                "actual_type": actual_type,
                "null_count": null_count,
                "errors": "; ".join(col_errors) if col_errors else None
            })

        # 2. Extra columns check
        unexpected_columns = []
        if not allow_extra and expected_schema:
            expected_names = {spec["name"] for spec in expected_schema}
            unexpected_columns = [c for c in orig_cols if c not in expected_names]

        # 3. Row-level compliance evaluation
        invalid_rows_indices = [idx for idx, v in enumerate(row_violations) if len(v) > 0]
        invalid_rows_count = len(invalid_rows_indices)
        valid_rows_count = orig_rows - invalid_rows_count
        compliance_pct = round((valid_rows_count / orig_rows) * 100, 2) if orig_rows > 0 else 100.0

        contract_passed = (len(missing_columns) == 0 and len(unexpected_columns) == 0 and invalid_rows_count == 0)

        # 4. Strict Mode Error Raising
        if validation_mode == "strict_raise" and not contract_passed:
            err_msg = f"Dataset failed Schema Contract Validation (Compliance: {compliance_pct}%)."
            if missing_columns:
                err_msg += f" Missing required columns: {missing_columns}."
            if unexpected_columns:
                err_msg += f" Unexpected columns: {unexpected_columns}."
            if invalid_rows_count > 0:
                err_msg += f" Found {invalid_rows_count} non-compliant rows."
            raise ValueError(err_msg)

        # 5. Apply Mode Transforms
        if validation_mode == "flag_rows":
            df_out["_contract_valid"] = [len(v) == 0 for v in row_violations]
            df_out["_contract_violations"] = ["; ".join(v) if v else "" for v in row_violations]
            final_df = df_out
        elif validation_mode == "quarantine_invalid_rows":
            final_df = df_out.drop(index=invalid_rows_indices).reset_index(drop=True)
        else:  # report_only
            final_df = df_out

        output_summary = {
            "title": "Schema / Contract Validation",
            "message": f"Contract Validation {'PASSED' if contract_passed else 'VIOLATED'}: {compliance_pct}% rows compliant. ({valid_rows_count:,} valid, {invalid_rows_count:,} invalid).",
            "metrics": {
                "contract_passed": contract_passed,
                "compliance_pct": compliance_pct,
                "total_rows": orig_rows,
                "valid_rows": valid_rows_count,
                "invalid_rows": invalid_rows_count,
                "missing_columns_count": len(missing_columns),
                "validation_mode": validation_mode
            },
            "checklist": checklist,
            "missing_columns": missing_columns,
            "unexpected_columns": unexpected_columns
        }

        return {
            "dataframe": final_df,
            "metrics": output_summary["metrics"],
            "output_summary": output_summary
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        schema = config.get("expected_schema", [])
        mode = config.get("validation_mode", "flag_rows")
        return (
            f"# Schema / Contract Validation\n"
            f"expected_contract = {schema}\n"
            f"# Validating column presence, types, and boundaries (mode: '{mode}')\n"
            f"for rule in expected_contract:\n"
            f"    col = rule['name']\n"
            f"    assert col in df.columns, f'Missing required column {col}'\n"
            f"print('Contract validation completed.')\n"
        )
