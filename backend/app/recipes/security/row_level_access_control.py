import datetime
import hashlib
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class RowLevelAccessControlRecipe(BaseRecipe):
    recipe_id = "row_level_access_control"
    name = "Row-Level Access Control Operator"
    version = "1.0.0"
    category = "security"
    description = "Restricts which rows each user or role can see."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    def get_inputs(self) -> List[RecipePort]:
        return [
            RecipePort(
                id="input",
                label="Input Table",
                type="dataframe",
                required=False,
                description="Source table to filter by security and permission rules",
            )
        ]

    def get_outputs(self) -> List[RecipePort]:
        return [
            RecipePort(
                id="output",
                label="Output Table",
                type="dataframe",
                description="Filtered or masked dataset authorized for the current caller",
            )
        ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "user_role": {
                    "type": "string",
                    "title": "Caller Role",
                    "enum": ["admin", "analyst", "data_engineer", "viewer", "guest", "dynamic_context"],
                    "default": "viewer",
                    "description": "Role assumed by the executing user or application caller.",
                    "input_type": "select",
                },
                "user_department": {
                    "type": "string",
                    "title": "Caller Department",
                    "default": "All",
                    "description": "Department of current user used for department-based row isolation.",
                    "input_type": "text",
                },
                "user_tenant_id": {
                    "type": "string",
                    "title": "Caller Tenant ID",
                    "default": "tenant-default",
                    "description": "Tenant or organization ID of the calling session.",
                    "input_type": "text",
                },
                "role_column": {
                    "type": "string",
                    "title": "Role Column Name",
                    "default": "allowed_roles",
                    "description": "Column specifying roles allowed to view each row (e.g. 'admin,analyst' or json array).",
                    "input_type": "text",
                },
                "department_column": {
                    "type": "string",
                    "title": "Department Column Name",
                    "default": "department",
                    "description": "Column containing the owning department of the record.",
                    "input_type": "text",
                },
                "tenant_column": {
                    "type": "string",
                    "title": "Tenant Column Name",
                    "default": "tenant_id",
                    "description": "Multi-tenant partition column restricting records to match user_tenant_id.",
                    "input_type": "text",
                },
                "policy_enforcement_mode": {
                    "type": "string",
                    "title": "Enforcement Mode",
                    "enum": [
                        "strict_filter",
                        "mask_disallowed_rows",
                        "raise_error_if_unauthorized",
                        "simulate_audit_only",
                    ],
                    "default": "strict_filter",
                    "description": "How unauthorized rows are handled: silently dropped, redacted in-place, error raised, or logged.",
                    "input_type": "select",
                },
                "masking_placeholder": {
                    "type": "string",
                    "title": "Redaction Placeholder",
                    "default": "[REDACTED_ACCESS_RESTRICTED]",
                    "description": "Placeholder substituted into sensitive columns when masking mode is active.",
                    "input_type": "text",
                },
                "mask_columns": {
                    "type": "string",
                    "title": "Columns to Redact",
                    "default": "all_except_id",
                    "description": "Comma-separated columns to mask when unauthorized (or 'all_except_id' for non-id columns).",
                    "input_type": "text",
                },
                "bypass_for_admin": {
                    "type": "boolean",
                    "title": "Admin Bypass",
                    "default": True,
                    "description": "Grants unrestricted row access when user_role is 'admin'.",
                    "input_type": "checkbox",
                },
                "custom_filter_expression": {
                    "type": "string",
                    "title": "Custom ACL Expression",
                    "default": "",
                    "description": "Optional Python / pandas query boolean expression to evaluate for advanced custom ACLs.",
                    "input_type": "text",
                },
            },
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        df = inputs.get("dataframe")
        if df is None and isinstance(inputs.get("input"), pd.DataFrame):
            df = inputs["input"]

        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            df = pd.DataFrame({
                "record_id": [f"REC-{i:03d}" for i in range(1, 7)],
                "tenant_id": ["tenant-default", "tenant-default", "tenant-alpha", "tenant-default", "tenant-beta", "tenant-default"],
                "department": ["Engineering", "Finance", "Finance", "Healthcare", "HR", "Engineering"],
                "allowed_roles": ["admin,viewer,analyst", "admin,analyst", "admin,analyst", "admin", "admin,data_engineer", "admin,viewer"],
                "client_name": ["Acme Corp", "Globex Global", "Initech LLC", "Soylent Corp", "Stark Ind", "Wayne Ent"],
                "sensitive_financials": [50000.0, 1250000.0, 890000.0, 45000.0, 780000.0, 210000.0],
            })

        # Resolve caller context
        caller_role = str(config.get("user_role", "viewer")).lower().strip()
        caller_dept = str(config.get("user_department", "All")).strip()
        caller_tenant = str(config.get("user_tenant_id", "tenant-default")).strip()

        if context is not None and isinstance(context, dict):
            if "role" in context:
                caller_role = str(context["role"]).lower().strip()
            if "department" in context:
                caller_dept = str(context["department"]).strip()
            if "tenant_id" in context:
                caller_tenant = str(context["tenant_id"]).strip()

        role_col = str(config.get("role_column", "allowed_roles")).strip()
        dept_col = str(config.get("department_column", "department")).strip()
        tenant_col = str(config.get("tenant_column", "tenant_id")).strip()
        enforcement = str(config.get("policy_enforcement_mode", "strict_filter")).lower().strip()
        placeholder = str(config.get("masking_placeholder", "[REDACTED_ACCESS_RESTRICTED]")).strip()
        mask_cols_str = str(config.get("mask_columns", "all_except_id")).strip()
        admin_bypass = bool(config.get("bypass_for_admin", True))
        custom_expr = str(config.get("custom_filter_expression", "")).strip()

        now_utc = datetime.datetime.now(datetime.timezone.utc)
        now_iso = now_utc.isoformat()

        # Admin bypass
        if caller_role == "admin" and admin_bypass:
            audit = {
                "caller_role": caller_role,
                "caller_department": caller_dept,
                "caller_tenant_id": caller_tenant,
                "enforcement_mode": enforcement,
                "admin_bypassed": True,
                "total_rows_evaluated": len(df),
                "authorized_rows_count": len(df),
                "unauthorized_rows_count": 0,
                "timestamp": now_iso,
            }
            return {
                "output": df,
                "dataframe": df,
                "security_audit": audit,
                "unauthorized_count": 0,
            }

        # Multi-factor Authorization Mask Evaluation
        auth_mask = pd.Series(True, index=df.index)

        # 1. Tenant match
        if tenant_col in df.columns:
            tenant_mask = (df[tenant_col].astype(str) == caller_tenant) | (df[tenant_col].astype(str).str.lower() == "all")
            auth_mask = auth_mask & tenant_mask

        # 2. Department match
        if dept_col in df.columns and caller_dept.lower() != "all":
            dept_mask = (df[dept_col].astype(str).str.lower() == caller_dept.lower()) | (df[dept_col].astype(str).str.lower() == "all")
            auth_mask = auth_mask & dept_mask

        # 3. Role match
        if role_col in df.columns:
            def _has_role(val: Any) -> bool:
                if pd.isna(val):
                    return False
                roles = [r.strip().lower() for r in str(val).split(",")]
                return caller_role in roles or "all" in roles or "*" in roles

            role_mask = df[role_col].apply(_has_role)
            auth_mask = auth_mask & role_mask

        # 4. Custom expression filter
        if custom_expr:
            try:
                expr_mask = df.eval(custom_expr)
                if isinstance(expr_mask, pd.Series):
                    auth_mask = auth_mask & expr_mask
            except Exception:
                pass

        unauth_count = int((~auth_mask).sum())
        auth_count = int(auth_mask.sum())

        audit = {
            "caller_role": caller_role,
            "caller_department": caller_dept,
            "caller_tenant_id": caller_tenant,
            "enforcement_mode": enforcement,
            "admin_bypassed": False,
            "total_rows_evaluated": len(df),
            "authorized_rows_count": auth_count,
            "unauthorized_rows_count": unauth_count,
            "timestamp": now_iso,
        }

        # Enforcement execution
        if enforcement == "raise_error_if_unauthorized":
            if unauth_count > 0:
                raise PermissionError(
                    f"RowLevelAccessControl violation: {unauth_count} rows unauthorized "
                    f"for role='{caller_role}', dept='{caller_dept}', tenant='{caller_tenant}'."
                )
            result_df = df.copy()

        elif enforcement == "mask_disallowed_rows":
            result_df = df.copy()
            if unauth_count > 0:
                # Determine columns to mask
                if mask_cols_str == "all_except_id":
                    target_cols = [c for c in df.columns if not any(x in c.lower() for x in ["id", "tenant", "department", "role"])]
                else:
                    target_cols = [c.strip() for c in mask_cols_str.split(",") if c.strip() in df.columns]

                # Convert target columns to object dtype so string placeholder fits
                for c in target_cols:
                    result_df[c] = result_df[c].astype(object)
                    result_df.loc[~auth_mask, c] = placeholder

        elif enforcement == "simulate_audit_only":
            result_df = df.copy()

        else:  # strict_filter (default)
            result_df = df[auth_mask].copy().reset_index(drop=True)

        return {
            "output": result_df,
            "dataframe": result_df,
            "security_audit": audit,
            "unauthorized_count": unauth_count,
        }
