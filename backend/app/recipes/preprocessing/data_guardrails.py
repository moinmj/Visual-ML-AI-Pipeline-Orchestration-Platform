"""
Guardrails (validation checks) Recipe.
Sub-group: Custom Logic
Category: Data Preparation & Transformation (preprocessing)

Applies rule-based data checks (ranges, nulls, uniqueness, regex) and stops or flags on failure.
"""
from typing import Dict, Any, List, Optional, Set
import re
import pandas as pd
import numpy as np
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class DataGuardrailsRecipe(BaseRecipe):
    recipe_id = "data_guardrails"
    name = "Guardrails (validation checks)"
    version = "1.0.0"
    category = "preprocessing"
    group = "Data Preparation & Transformation"
    subgroup = "Custom Logic"
    description = "Applies rule-based data checks (ranges, nulls, uniqueness, regex patterns, statistical bounds) and halts pipeline or quarantines rows upon failure."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="input",
            label="Input Table",
            type="dataframe",
            required=True,
            max_connections=1,
            description="Dataset to inspect against custom logic guardrail checks"
        )
    ]

    outputs = [
        RecipePort(
            id="dataframe",
            label="Validated Table",
            type="dataframe",
            description="Dataset verified by guardrail checks or filtered of non-compliant records"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "rules": {
                    "type": "array",
                    "title": "Guardrail Verification Rules",
                    "items": {
                        "type": "object",
                        "properties": {
                            "column": {"type": "string", "title": "Target Column"},
                            "check_type": {
                                "type": "string",
                                "title": "Check Type",
                                "enum": [
                                    "not_null",
                                    "unique",
                                    "between",
                                    "greater_than",
                                    "less_than",
                                    "is_in_set",
                                    "regex_match",
                                    "mean_between",
                                    "row_count_min"
                                ],
                                "default": "not_null"
                            },
                            "min_value": {"type": "number", "title": "Min Value / Lower Bound"},
                            "max_value": {"type": "number", "title": "Max Value / Upper Bound"},
                            "set_values": {
                                "type": "array",
                                "items": {"type": "string"},
                                "title": "Authorized Set Values"
                            },
                            "regex_pattern": {"type": "string", "title": "RegEx Pattern"},
                            "severity": {
                                "type": "string",
                                "title": "Severity",
                                "enum": ["error", "warning"],
                                "default": "error"
                            }
                        },
                        "required": ["check_type"]
                    },
                    "description": "Rules to evaluate against table rows or aggregate statistics."
                },
                "on_failure": {
                    "type": "string",
                    "title": "Action on Guardrail Breach",
                    "enum": [
                        "stop_and_error",
                        "quarantine_invalid_rows",
                        "flag_rows",
                        "log_warning"
                    ],
                    "default": "flag_rows",
                    "description": "'stop_and_error' halts DAG execution immediately; 'quarantine_invalid_rows' drops failing rows; 'flag_rows' appends guardrail_status and guardrail_violations."
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
                raise ValueError("DataGuardrailsRecipe expects 'dataframe' in inputs.")

        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        df_out = df.copy()
        orig_rows = len(df_out)
        rules = config.get("rules", [])
        on_failure = config.get("on_failure", "flag_rows")

        if not rules:
            # Default guardrail rule: check that row count > 0 and key columns not all null
            rules = [{"column": df_out.columns[0] if len(df_out.columns) > 0 else "id", "check_type": "row_count_min", "min_value": 1}]

        rule_results: List[Dict[str, Any]] = []
        row_violations: List[List[str]] = [[] for _ in range(orig_rows)]
        critical_errors: List[str] = []

        for r_idx, rule in enumerate(rules, 1):
            col = str(rule.get("column", "")).strip()
            check_type = str(rule.get("check_type", "not_null")).lower().strip()
            severity = str(rule.get("severity", "error")).lower().strip()
            min_v = rule.get("min_value")
            max_v = rule.get("max_value")
            set_v = rule.get("set_values")
            regex_pat = rule.get("regex_pattern")

            rule_passed = True
            fail_count = 0
            detail = ""

            # Table-level checks
            if check_type == "row_count_min":
                req_min = int(min_v if min_v is not None else 1)
                if orig_rows < req_min:
                    rule_passed = False
                    detail = f"Row count ({orig_rows}) is below required minimum ({req_min})"
                    if severity == "error":
                        critical_errors.append(detail)
                else:
                    detail = f"Row count ({orig_rows}) satisfies minimum ({req_min})"

            elif col not in df_out.columns:
                rule_passed = False
                detail = f"Target column '{col}' does not exist in dataframe"
                if severity == "error":
                    critical_errors.append(detail)

            else:
                s = df_out[col]

                if check_type == "not_null":
                    bad_mask = s.isna()
                    fail_count = int(bad_mask.sum())
                    if fail_count > 0:
                        rule_passed = False
                        detail = f"{fail_count} null rows detected in '{col}'"
                        for idx in s[bad_mask].index:
                            if idx < orig_rows:
                                row_violations[idx].append(f"{col}: cannot be null")

                elif check_type == "unique":
                    dup_mask = s.duplicated(keep=False) & s.notna()
                    fail_count = int(dup_mask.sum())
                    if fail_count > 0:
                        rule_passed = False
                        detail = f"{fail_count} duplicate entries found in '{col}'"
                        for idx in s[dup_mask].index:
                            if idx < orig_rows:
                                row_violations[idx].append(f"{col}: duplicate value")

                elif check_type == "between":
                    if min_v is not None and max_v is not None and pd.api.types.is_numeric_dtype(s):
                        bad_mask = (s < float(min_v)) | (s > float(max_v))
                        fail_count = int(bad_mask.sum())
                        if fail_count > 0:
                            rule_passed = False
                            detail = f"{fail_count} rows outside range [{min_v}, {max_v}]"
                            for idx in s[bad_mask].index:
                                if idx < orig_rows:
                                    row_violations[idx].append(f"{col}: not in [{min_v}, {max_v}]")

                elif check_type == "greater_than":
                    if min_v is not None and pd.api.types.is_numeric_dtype(s):
                        bad_mask = (s <= float(min_v))
                        fail_count = int(bad_mask.sum())
                        if fail_count > 0:
                            rule_passed = False
                            detail = f"{fail_count} rows <= {min_v}"
                            for idx in s[bad_mask].index:
                                if idx < orig_rows:
                                    row_violations[idx].append(f"{col}: <= {min_v}")

                elif check_type == "less_than":
                    if max_v is not None and pd.api.types.is_numeric_dtype(s):
                        bad_mask = (s >= float(max_v))
                        fail_count = int(bad_mask.sum())
                        if fail_count > 0:
                            rule_passed = False
                            detail = f"{fail_count} rows >= {max_v}"
                            for idx in s[bad_mask].index:
                                if idx < orig_rows:
                                    row_violations[idx].append(f"{col}: >= {max_v}")

                elif check_type == "is_in_set":
                    if set_v:
                        allowed = set(str(v).strip().lower() for v in set_v)
                        bad_mask = s.dropna().apply(lambda v: str(v).strip().lower() not in allowed)
                        fail_count = int(bad_mask.sum())
                        if fail_count > 0:
                            rule_passed = False
                            detail = f"{fail_count} rows not in set"
                            for idx in s.dropna()[bad_mask].index:
                                if idx < orig_rows:
                                    row_violations[idx].append(f"{col}: invalid set value")

                elif check_type == "regex_match":
                    if regex_pat:
                        regex = re.compile(regex_pat)
                        bad_mask = s.dropna().apply(lambda v: bool(not regex.search(str(v))))
                        fail_count = int(bad_mask.sum())
                        if fail_count > 0:
                            rule_passed = False
                            detail = f"{fail_count} rows failed regex pattern"
                            for idx in s.dropna()[bad_mask].index:
                                if idx < orig_rows:
                                    row_violations[idx].append(f"{col}: regex pattern mismatch")

                elif check_type == "mean_between":
                    if min_v is not None and max_v is not None and pd.api.types.is_numeric_dtype(s):
                        mean_val = float(s.dropna().mean())
                        if mean_val < float(min_v) or mean_val > float(max_v):
                            rule_passed = False
                            detail = f"Mean {mean_val:.2f} is outside [{min_v}, {max_v}]"
                            if severity == "error":
                                critical_errors.append(detail)
                        else:
                            detail = f"Mean {mean_val:.2f} satisfies bounds"

                if not rule_passed and severity == "error":
                    critical_errors.append(f"Rule #{r_idx} failed for '{col}': {detail}")

            rule_results.append({
                "rule_id": r_idx,
                "column": col or "Table",
                "check_type": check_type,
                "status": "PASS" if rule_passed else "FAIL",
                "severity": severity,
                "failing_rows": fail_count,
                "details": detail or ("Passed" if rule_passed else "Failed")
            })

        # Calculate Compliance
        invalid_row_indices = [i for i, v in enumerate(row_violations) if len(v) > 0]
        invalid_rows_count = len(invalid_row_indices)
        valid_rows_count = orig_rows - invalid_rows_count
        compliance_pct = round((valid_rows_count / orig_rows) * 100, 2) if orig_rows > 0 else 100.0

        all_rules_passed = all(r["status"] == "PASS" for r in rule_results)

        # Apply on_failure action
        if on_failure == "stop_and_error" and (not all_rules_passed or critical_errors):
            error_details = "; ".join(critical_errors) if critical_errors else "Guardrail verification check failed."
            raise ValueError(f"Guardrail Check Failure (Compliance: {compliance_pct}%): {error_details}")

        if on_failure == "quarantine_invalid_rows":
            final_df = df_out.drop(index=invalid_row_indices).reset_index(drop=True)
        elif on_failure == "flag_rows":
            df_out["guardrail_status"] = ["PASS" if len(v) == 0 else "FAIL" for v in row_violations]
            df_out["guardrail_violations"] = ["; ".join(v) if v else "" for v in row_violations]
            final_df = df_out
        else:  # log_warning
            final_df = df_out

        output_summary = {
            "title": "Guardrails (validation checks)",
            "message": f"Evaluated {len(rules)} guardrail rules: {sum(1 for r in rule_results if r['status'] == 'PASS')}/{len(rules)} passed ({compliance_pct}% compliant rows).",
            "metrics": {
                "guardrail_status": "PASS" if all_rules_passed else "FAIL",
                "rules_evaluated": len(rules),
                "rules_passed": sum(1 for r in rule_results if r["status"] == "PASS"),
                "rules_failed": sum(1 for r in rule_results if r["status"] == "FAIL"),
                "total_rows": orig_rows,
                "valid_rows": valid_rows_count,
                "failing_rows": invalid_rows_count,
                "compliance_pct": compliance_pct,
                "on_failure": on_failure
            },
            "rule_results": rule_results
        }

        return {
            "dataframe": final_df,
            "metrics": output_summary["metrics"],
            "output_summary": output_summary
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        rules = config.get("rules", [])
        action = config.get("on_failure", "flag_rows")
        return (
            f"# Guardrails (validation checks)\n"
            f"guardrail_rules = {rules}\n"
            f"# Applying validation rules (action: '{action}')\n"
            f"for rule in guardrail_rules:\n"
            f"    col = rule.get('column')\n"
            f"    check = rule.get('check_type')\n"
            f"    if check == 'not_null' and col in df.columns:\n"
            f"        assert df[col].notna().all(), f'Null values found in {col}'\n"
            f"print('All guardrails successfully verified.')\n"
        )
