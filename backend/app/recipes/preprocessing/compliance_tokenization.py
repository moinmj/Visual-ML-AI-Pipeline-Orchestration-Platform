"""
Compliance Tokenization (PCI/PII Masking) Recipe.
Sub-group: Domain-Specific Cleaning
Category: Data Preparation & Transformation (preprocessing)

Replaces credit card numbers, SSNs, personal identifiers, and sensitive data with
cryptographic HMAC-SHA256 tokens, UUID surrogates, or PCI-DSS/HIPAA compliant masks.
"""
import hashlib
import hmac
import re
import uuid
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class ComplianceTokenizationRecipe(BaseRecipe):
    recipe_id = "compliance_tokenization"
    name = "Compliance Tokenization (PCI/PII masking)"
    version = "1.0.0"
    category = "preprocessing"
    group = "Data Preparation & Transformation"
    subgroup = "Domain-Specific Cleaning"
    description = "Replaces card numbers and personal data with reversible or irreversible tokens, masks, or UUID surrogates."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="input",
            label="Input Table",
            type="dataframe",
            required=True,
            max_connections=1,
            description="Dataset containing sensitive PCI/PII data to mask or tokenize"
        )
    ]

    outputs = [
        RecipePort(
            id="dataframe",
            label="Tokenized Table",
            type="dataframe",
            description="Sanitized dataset with compliant tokens and masked attributes"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "columns": {
                    "type": "array",
                    "title": "Columns to Tokenize",
                    "items": {"type": "string"},
                    "default": [],
                    "description": "Columns containing sensitive credit cards, SSNs, emails, or personal identifiers."
                },
                "strategy": {
                    "type": "string",
                    "title": "Tokenization Strategy",
                    "enum": ["pci_credit_card", "ssn_mask", "hmac_sha256", "surrogate_uuid", "email_mask", "redact"],
                    "default": "pci_credit_card",
                    "description": "'pci_credit_card' (****-****-****-1234); 'ssn_mask' (***-**-1234); 'hmac_sha256' cryptographic token; 'surrogate_uuid' pseudonym; 'email_mask' (j***e@domain.com); 'redact' ([REDACTED])."
                },
                "salt": {
                    "type": "string",
                    "title": "HMAC Salt / Secret Key",
                    "default": "secret_salt",
                    "description": "Salt key for deterministic pseudonymization (ensures identical raw values map to identical tokens across pipelines)."
                },
                "output_mode": {
                    "type": "string",
                    "title": "Output Mode",
                    "enum": ["in_place", "new_column"],
                    "default": "in_place",
                    "description": "Replace in-place or generate new columns with a suffix."
                },
                "new_column_suffix": {
                    "type": "string",
                    "title": "New Column Suffix",
                    "default": "_tokenized",
                    "description": "Suffix for new tokenized columns."
                }
            },
            "required": ["columns", "strategy"]
        }

    def _tokenize_val(self, val: Any, strategy: str, salt: str) -> Optional[str]:
        if pd.isna(val) or val is None or str(val).strip() == "":
            return None

        s = str(val).strip()

        if strategy == "pci_credit_card":
            # Strip spaces and dashes
            digits = re.sub(r"\D", "", s)
            if len(digits) >= 4:
                last4 = digits[-4:]
                return f"****-****-****-{last4}"
            return "****"

        elif strategy == "ssn_mask":
            digits = re.sub(r"\D", "", s)
            if len(digits) >= 4:
                last4 = digits[-4:]
                return f"***-**-{last4}"
            return "***-**-****"

        elif strategy == "hmac_sha256":
            token = hmac.new(salt.encode("utf-8"), s.encode("utf-8"), hashlib.sha256).hexdigest()
            return f"tok_{token[:16]}"

        elif strategy == "surrogate_uuid":
            # Deterministic UUIDv5 using URL namespace and salted input
            salted_str = f"{salt}:{s}"
            return str(uuid.uuid5(uuid.NAMESPACE_DNS, salted_str))

        elif strategy == "email_mask":
            if "@" in s:
                user, domain = s.split("@", 1)
                if len(user) <= 2:
                    masked_user = user[0] + "*"
                else:
                    masked_user = user[0] + ("*" * (len(user) - 2)) + user[-1]
                return f"{masked_user}@{domain}"
            return "[EMAIL_REDACTED]"

        elif strategy == "redact":
            return "[REDACTED]"

        return s

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        df = inputs.get("dataframe") if "dataframe" in inputs else inputs.get("input")
        if df is None:
            df = inputs.get("df")

        if df is None:
            return {
                "dataframe": pd.DataFrame(),
                "metrics": {"tokens_generated": 0, "rows_processed": 0},
                "output_summary": {"title": "Compliance Tokenization", "message": "No input table provided.", "metrics": {}}
            }

        df = df.copy()
        if len(df) == 0:
            return {
                "dataframe": df,
                "metrics": {"tokens_generated": 0, "rows_processed": 0},
                "output_summary": {"title": "Compliance Tokenization", "message": "Input table is empty.", "metrics": {}}
            }

        cols_config = config.get("columns", [])
        strategy = config.get("strategy", "pci_credit_card")
        salt = config.get("salt", "secret_salt")
        output_mode = config.get("output_mode", "in_place")
        suffix = config.get("new_column_suffix", "_tokenized")

        target_cols = [c for c in cols_config if c in df.columns]
        if not target_cols:
            # Auto-detect sensitive columns
            for c in df.columns:
                c_low = str(c).lower()
                if any(k in c_low for k in ["card", "ssn", "credit", "cvv", "pan", "account", "tax_id"]):
                    target_cols.append(c)

        if not target_cols:
            return {
                "dataframe": df,
                "metrics": {"tokens_generated": 0, "rows_processed": len(df)},
                "output_summary": {"title": "Compliance Tokenization", "message": "No sensitive columns found.", "metrics": {}}
            }

        tokens_count = 0
        for col in target_cols:
            dest_col = col if output_mode == "in_place" else f"{col}{suffix}"
            tokenized_series = df[col].apply(lambda x: self._tokenize_val(x, strategy, salt))
            tokens_count += int(tokenized_series.notna().sum())
            df[dest_col] = tokenized_series

        output_summary = {
            "title": "Compliance Tokenization (PCI/PII)",
            "message": f"Tokenized {len(target_cols)} column(s) ({', '.join(target_cols)}) using strategy '{strategy}'.",
            "metrics": {
                "strategy": strategy,
                "columns_tokenized": target_cols,
                "tokens_generated": tokens_count,
                "compliance_standards": ["PCI-DSS", "GDPR", "HIPAA", "CCPA"],
                "rows_processed": len(df)
            }
        }

        return {
            "dataframe": df,
            "metrics": output_summary["metrics"],
            "output_summary": output_summary
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        strat = repr(config.get("strategy", "pci_credit_card"))
        cols = repr(config.get("columns", []))
        return (
            f"# Recipe: {self.name}\n"
            f"# ID: {self.recipe_id}\n"
            f"# Strategy: {strat}, Columns: {cols}\n"
            f"# Tokenizes sensitive PCI/PII data\n"
        )
