"""
Phone Number Validator / Formatter Recipe.
Sub-group: Domain-Specific Cleaning
Category: Data Preparation & Transformation (preprocessing)

Validates international and domestic phone numbers and standardizes them into E.164,
national, international, or digits-only formats.
"""
import re
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


# Standard country calling code mappings
COUNTRY_CODES = {
    "US": "1",
    "CA": "1",
    "UK": "44",
    "GB": "44",
    "IN": "91",
    "DE": "49",
    "FR": "33",
    "AU": "61",
    "JP": "81",
    "SG": "65",
    "AE": "971",
    "BR": "55",
    "MX": "52"
}


class PhoneNumberValidatorRecipe(BaseRecipe):
    recipe_id = "phone_number_validator"
    name = "Phone Number Validator / Formatter"
    version = "1.0.0"
    category = "preprocessing"
    group = "Data Preparation & Transformation"
    subgroup = "Domain-Specific Cleaning"
    description = "Validates telephone numbers and formats them into standard E.164, national, international, or numeric representations."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="input",
            label="Input Table",
            type="dataframe",
            required=True,
            max_connections=1,
            description="Dataset containing phone numbers to validate and standardize"
        )
    ]

    outputs = [
        RecipePort(
            id="dataframe",
            label="Standardized Table",
            type="dataframe",
            description="Dataset with validated phone numbers and standard formatting"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "phone_column": {
                    "type": "string",
                    "title": "Phone Number Column",
                    "default": "phone",
                    "description": "Column containing raw phone numbers."
                },
                "default_country": {
                    "type": "string",
                    "title": "Default Country Code",
                    "enum": ["US", "CA", "UK", "IN", "DE", "FR", "AU", "JP", "SG", "AE", "BR", "MX"],
                    "default": "US",
                    "description": "Country assumption for phone numbers without a leading '+'."
                },
                "target_format": {
                    "type": "string",
                    "title": "Target Phone Format",
                    "enum": ["e164", "national", "international", "digits_only"],
                    "default": "e164",
                    "description": "'e164' (+15551234567); 'national' ((555) 123-4567); 'international' (+1 555-123-4567); 'digits_only' (15551234567)."
                },
                "output_column": {
                    "type": "string",
                    "title": "Output Phone Column",
                    "default": "formatted_phone",
                    "description": "Column name for formatted phone numbers. If blank, updates in-place."
                },
                "on_invalid": {
                    "type": "string",
                    "title": "Action on Invalid Phone Number",
                    "enum": ["flag_only", "set_null"],
                    "default": "flag_only",
                    "description": "'flag_only' keeps raw string and sets is_valid_phone=False; 'set_null' replaces invalid phone numbers with null."
                }
            },
            "required": ["phone_column"]
        }

    def _normalize_phone(self, raw_val: Any, default_country: str, target_fmt: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Returns (is_valid, formatted_phone, country_code).
        """
        if not raw_val or pd.isna(raw_val):
            return False, None, None

        s = str(raw_val).strip()
        has_plus = s.startswith("+")

        # Strip all non-digits
        digits = re.sub(r"\D", "", s)
        if not digits or len(digits) < 7 or len(digits) > 15:
            return False, None, None

        default_cc = COUNTRY_CODES.get(default_country.upper(), "1")

        country_code = None
        national_number = None

        if has_plus:
            # Check for matching country code prefixes
            matched_cc = None
            for cc in sorted(set(COUNTRY_CODES.values()), key=len, reverse=True):
                if digits.startswith(cc) and len(digits) > len(cc):
                    matched_cc = cc
                    break
            if matched_cc:
                country_code = matched_cc
                national_number = digits[len(matched_cc):]
            else:
                country_code = default_cc
                national_number = digits
        else:
            if len(digits) == 10 and default_cc == "1":
                country_code = "1"
                national_number = digits
            elif len(digits) == 11 and digits.startswith("1") and default_cc == "1":
                country_code = "1"
                national_number = digits[1:]
            elif len(digits) == 10 and default_cc == "91":
                country_code = "91"
                national_number = digits
            else:
                country_code = default_cc
                national_number = digits

        # Basic validity check on national number length (7 to 12 digits)
        if not (7 <= len(national_number) <= 12):
            return False, None, None

        # Build formatted output
        full_digits = f"{country_code}{national_number}"

        if target_fmt == "digits_only":
            formatted = full_digits
        elif target_fmt == "e164":
            formatted = f"+{full_digits}"
        elif target_fmt == "national":
            if len(national_number) == 10:
                formatted = f"({national_number[:3]}) {national_number[3:6]}-{national_number[6:]}"
            elif len(national_number) == 7:
                formatted = f"{national_number[:3]}-{national_number[3:]}"
            else:
                formatted = national_number
        elif target_fmt == "international":
            if len(national_number) == 10:
                formatted = f"+{country_code} {national_number[:3]}-{national_number[3:6]}-{national_number[6:]}"
            else:
                formatted = f"+{country_code} {national_number}"
        else:
            formatted = f"+{full_digits}"

        return True, formatted, f"+{country_code}"

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        df = inputs.get("dataframe") if "dataframe" in inputs else inputs.get("input")
        if df is None:
            df = inputs.get("df")

        if df is None:
            return {
                "dataframe": pd.DataFrame(),
                "metrics": {"rows_processed": 0},
                "output_summary": {"title": "Phone Number Validator", "message": "No input table provided.", "metrics": {}}
            }

        df = df.copy()
        if len(df) == 0:
            return {
                "dataframe": df,
                "metrics": {"rows_processed": 0},
                "output_summary": {"title": "Phone Number Validator", "message": "Input table is empty.", "metrics": {}}
            }

        phone_col = config.get("phone_column", "phone")
        default_country = config.get("default_country", "US")
        target_fmt = config.get("target_format", "e164")
        output_col = config.get("output_column") or phone_col
        on_invalid = config.get("on_invalid", "flag_only")

        if phone_col not in df.columns:
            return {
                "dataframe": df,
                "metrics": {"rows_processed": len(df)},
                "output_summary": {"title": "Phone Number Validator", "message": f"Column '{phone_col}' not found.", "metrics": {}}
            }

        val_flags = []
        formatted_list = []
        country_codes = []

        valid_count = 0

        for val in df[phone_col]:
            is_valid, fmt, cc = self._normalize_phone(val, default_country, target_fmt)
            val_flags.append(is_valid)
            country_codes.append(cc)

            if is_valid:
                formatted_list.append(fmt)
                valid_count += 1
            else:
                if on_invalid == "set_null":
                    formatted_list.append(None)
                else:
                    formatted_list.append(val)

        df[output_col] = formatted_list
        df[f"{phone_col}_is_valid"] = val_flags
        df[f"{phone_col}_country_code"] = country_codes

        output_summary = {
            "title": "Phone Number Validator & Formatter",
            "message": f"Validated {len(df):,} phone numbers ({valid_count:,} valid, {len(df) - valid_count:,} invalid) formatted as '{target_fmt}'.",
            "metrics": {
                "total_rows": len(df),
                "valid_phones_count": valid_count,
                "invalid_phones_count": len(df) - valid_count,
                "valid_rate": f"{round((valid_count / len(df)) * 100, 1)}%",
                "target_format": target_fmt
            }
        }

        return {
            "dataframe": df,
            "metrics": output_summary["metrics"],
            "output_summary": output_summary
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        col = repr(config.get("phone_column", "phone"))
        fmt = repr(config.get("target_format", "e164"))
        return (
            f"# Recipe: {self.name}\n"
            f"# ID: {self.recipe_id}\n"
            f"# Column: {col}, Format: {fmt}\n"
            f"# Standardizes phone numbers to standard representation\n"
        )
