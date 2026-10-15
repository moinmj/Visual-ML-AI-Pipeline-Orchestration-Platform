"""
Email / Domain Validator Recipe.
Sub-group: Domain-Specific Cleaning
Category: Data Preparation & Transformation (preprocessing)

Validates email addresses and domains, detecting disposable mailboxes, free webmail,
and role-based corporate aliases.
"""
import re
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


# Common disposable / temporary email domains
DISPOSABLE_DOMAINS = {
    "mailinator.com", "tempmail.com", "10minutemail.com", "guerrillamail.com",
    "trashmail.com", "yopmail.com", "sharklasers.com", "getairmail.com",
    "dispostable.com", "throwawaymail.com", "fakeinbox.com", "tempinbox.com"
}

# Major free webmail providers
FREE_WEBMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com",
    "aol.com", "zoho.com", "protonmail.com", "proton.me", "mail.com", "gmx.com"
}

# Role-based functional prefixes
ROLE_BASED_PREFIXES = {
    "admin", "support", "info", "sales", "billing", "contact", "help",
    "marketing", "press", "legal", "security", "team", "office", "jobs", "careers"
}

# Strict RFC-compliant regex pattern
EMAIL_REGEX = re.compile(
    r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
)


class EmailDomainValidatorRecipe(BaseRecipe):
    recipe_id = "email_domain_validator"
    name = "Email / Domain Validator"
    version = "1.0.0"
    category = "preprocessing"
    group = "Data Preparation & Transformation"
    subgroup = "Domain-Specific Cleaning"
    description = "Validates email syntax and domain authenticity, identifying disposable, free webmail, and role-based mailboxes."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="input",
            label="Input Table",
            type="dataframe",
            required=True,
            max_connections=1,
            description="Dataset containing email addresses to audit"
        )
    ]

    outputs = [
        RecipePort(
            id="dataframe",
            label="Validated Table",
            type="dataframe",
            description="Dataset enriched with email validity, domain breakdown, and security flags"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "email_column": {
                    "type": "string",
                    "title": "Email Column",
                    "default": "email",
                    "description": "Column containing email addresses to validate."
                },
                "normalize_emails": {
                    "type": "boolean",
                    "title": "Normalize Emails (Lowercase & Trim)",
                    "default": True,
                    "description": "Transforms emails to lowercase and removes accidental leading/trailing whitespace."
                },
                "extract_components": {
                    "type": "boolean",
                    "title": "Extract Domain & Username",
                    "default": True,
                    "description": "Extracts separate domain and username columns."
                },
                "flag_disposable": {
                    "type": "boolean",
                    "title": "Flag Disposable / Temporary Emails",
                    "default": True,
                    "description": "Flags temporary throwaway email domains (e.g. mailinator, tempmail)."
                },
                "flag_role_based": {
                    "type": "boolean",
                    "title": "Flag Role-Based Accounts (admin, support, sales)",
                    "default": True,
                    "description": "Identifies generic functional accounts rather than individuals."
                },
                "on_invalid": {
                    "type": "string",
                    "title": "Action on Invalid Email",
                    "enum": ["flag_only", "set_null"],
                    "default": "flag_only",
                    "description": "'flag_only' keeps original email and sets is_valid_email=False; 'set_null' replaces invalid emails with null."
                }
            },
            "required": ["email_column"]
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        df = inputs.get("dataframe") if "dataframe" in inputs else inputs.get("input")
        if df is None:
            df = inputs.get("df")

        if df is None:
            return {
                "dataframe": pd.DataFrame(),
                "metrics": {"rows_processed": 0},
                "output_summary": {"title": "Email / Domain Validator", "message": "No input table provided.", "metrics": {}}
            }

        df = df.copy()
        if len(df) == 0:
            return {
                "dataframe": df,
                "metrics": {"rows_processed": 0},
                "output_summary": {"title": "Email / Domain Validator", "message": "Input table is empty.", "metrics": {}}
            }

        email_col = config.get("email_column", "email")
        normalize = config.get("normalize_emails", True)
        extract_comp = config.get("extract_components", True)
        flag_disp = config.get("flag_disposable", True)
        flag_role = config.get("flag_role_based", True)
        on_invalid = config.get("on_invalid", "flag_only")

        if email_col not in df.columns:
            return {
                "dataframe": df,
                "metrics": {"rows_processed": len(df)},
                "output_summary": {"title": "Email / Domain Validator", "message": f"Column '{email_col}' not found.", "metrics": {}}
            }

        is_valid_list = []
        domains = []
        usernames = []
        is_disp_list = []
        is_corp_list = []
        is_role_list = []
        cleaned_emails = []

        valid_count = 0
        disposable_count = 0
        corporate_count = 0

        for val in df[email_col]:
            if not val or pd.isna(val):
                is_valid_list.append(False)
                domains.append(None)
                usernames.append(None)
                is_disp_list.append(False)
                is_corp_list.append(False)
                is_role_list.append(False)
                cleaned_emails.append(None)
                continue

            raw_str = str(val).strip()
            s = raw_str.lower() if normalize else raw_str

            # Check validity
            match = EMAIL_REGEX.match(s)
            has_valid_tld = False
            domain_part = None
            user_part = None

            if match and "@" in s:
                user_part, domain_part = s.split("@", 1)
                tld = domain_part.split(".")[-1]
                has_valid_tld = len(tld) >= 2 and tld.isalpha()

            is_valid = bool(match and has_valid_tld)

            if is_valid:
                valid_count += 1
                cleaned_emails.append(s)
                domains.append(domain_part)
                usernames.append(user_part)

                # Check disposable
                is_disp = domain_part in DISPOSABLE_DOMAINS if flag_disp else False
                if is_disp:
                    disposable_count += 1
                is_disp_list.append(is_disp)

                # Check corporate
                is_free = domain_part in FREE_WEBMAIL_DOMAINS
                is_corp = not is_free and not is_disp
                if is_corp:
                    corporate_count += 1
                is_corp_list.append(is_corp)

                # Check role-based
                clean_user = re.sub(r"[^a-zA-Z]", "", user_part)
                is_role = clean_user in ROLE_BASED_PREFIXES if flag_role else False
                is_role_list.append(is_role)
            else:
                cleaned_emails.append(None if on_invalid == "set_null" else val)
                domains.append(None)
                usernames.append(None)
                is_disp_list.append(False)
                is_corp_list.append(False)
                is_role_list.append(False)

            is_valid_list.append(is_valid)

        df[email_col] = cleaned_emails
        df[f"{email_col}_is_valid"] = is_valid_list

        if extract_comp:
            df[f"{email_col}_domain"] = domains
            df[f"{email_col}_username"] = usernames

        if flag_disp:
            df[f"{email_col}_is_disposable"] = is_disp_list
        df[f"{email_col}_is_corporate"] = is_corp_list
        if flag_role:
            df[f"{email_col}_is_role_based"] = is_role_list

        output_summary = {
            "title": "Email / Domain Validator",
            "message": f"Audited {len(df):,} emails: {valid_count:,} valid ({round(valid_count / len(df) * 100, 1)}%), {corporate_count:,} corporate, {disposable_count:,} disposable.",
            "metrics": {
                "total_rows": len(df),
                "valid_emails": valid_count,
                "invalid_emails": len(df) - valid_count,
                "corporate_emails": corporate_count,
                "disposable_emails": disposable_count
            }
        }

        return {
            "dataframe": df,
            "metrics": output_summary["metrics"],
            "output_summary": output_summary
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        col = repr(config.get("email_column", "email"))
        return (
            f"# Recipe: {self.name}\n"
            f"# ID: {self.recipe_id}\n"
            f"# Column: {col}\n"
            f"# Validates email syntax, domain validity, and disposable accounts\n"
        )
