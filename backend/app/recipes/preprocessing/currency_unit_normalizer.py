"""
Currency / Unit Normalizer Recipe.
Sub-group: Domain-Specific Cleaning
Category: Data Preparation & Transformation (preprocessing)

Converts amounts to a single currency or unit of measure across weight, length, volume,
temperature, speed, and digital storage.
"""
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


# Standard benchmark FX rates relative to USD (1 USD = X Currency)
DEFAULT_FX_RATES_TO_USD = {
    "USD": 1.0,
    "EUR": 1.085,     # 1 EUR = 1.085 USD
    "GBP": 1.282,     # 1 GBP = 1.282 USD
    "CAD": 0.735,     # 1 CAD = 0.735 USD
    "AUD": 0.655,     # 1 AUD = 0.655 USD
    "JPY": 0.00665,   # 1 JPY = 0.00665 USD
    "CHF": 1.112,     # 1 CHF = 1.112 USD
    "INR": 0.01195,   # 1 INR = 0.01195 USD
    "CNY": 0.138,     # 1 CNY = 0.138 USD
    "SGD": 0.745,     # 1 SGD = 0.745 USD
    "NZD": 0.605,     # 1 NZD = 0.605 USD
    "BRL": 0.185,     # 1 BRL = 0.185 USD
    "MXN": 0.052,     # 1 MXN = 0.052 USD
}

# Unit conversion factors relative to base metric unit
UNIT_CONVERSIONS = {
    # Weight (base: kg)
    "weight": {
        "kg": 1.0,
        "g": 0.001,
        "mg": 0.000001,
        "lbs": 0.45359237,
        "lb": 0.45359237,
        "oz": 0.0283495,
        "ton": 907.18474,       # US short ton
        "metric_ton": 1000.0,
    },
    # Length (base: meter)
    "length": {
        "m": 1.0,
        "km": 1000.0,
        "cm": 0.01,
        "mm": 0.001,
        "miles": 1609.344,
        "mile": 1609.344,
        "ft": 0.3048,
        "feet": 0.3048,
        "in": 0.0254,
        "inches": 0.0254,
        "yd": 0.9144,
        "yards": 0.9144,
    },
    # Volume (base: liter)
    "volume": {
        "l": 1.0,
        "liters": 1.0,
        "ml": 0.001,
        "gallons": 3.78541,
        "gal": 3.78541,
        "quarts": 0.946353,
        "fl_oz": 0.0295735,
        "cubic_meters": 1000.0,
        "m3": 1000.0,
    },
    # Digital Storage (base: bytes)
    "data_size": {
        "bytes": 1.0,
        "b": 1.0,
        "kb": 1024.0,
        "mb": 1024.0 ** 2,
        "gb": 1024.0 ** 3,
        "tb": 1024.0 ** 4,
    },
    # Speed (base: km/h)
    "speed": {
        "km/h": 1.0,
        "kmh": 1.0,
        "mph": 1.609344,
        "m/s": 3.6,
        "knots": 1.852,
    }
}


class CurrencyUnitNormalizerRecipe(BaseRecipe):
    recipe_id = "currency_unit_normalizer"
    name = "Currency / Unit Normalizer"
    version = "1.0.0"
    category = "preprocessing"
    group = "Data Preparation & Transformation"
    subgroup = "Domain-Specific Cleaning"
    description = "Converts amounts to a single currency or unit of measure across financial, physical, and digital dimensions."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="input",
            label="Input Table",
            type="dataframe",
            required=True,
            max_connections=1,
            description="Dataset containing multi-currency or multi-unit measurements"
        )
    ]

    outputs = [
        RecipePort(
            id="dataframe",
            label="Normalized Table",
            type="dataframe",
            description="Dataset with amounts normalized to target currency or standard unit"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "normalization_type": {
                    "type": "string",
                    "title": "Normalization Type",
                    "enum": ["currency", "unit"],
                    "default": "currency",
                    "description": "'currency' converts financial amounts; 'unit' normalizes physical/metric units of measure."
                },
                "amount_column": {
                    "type": "string",
                    "title": "Amount Column",
                    "default": "amount",
                    "description": "Column containing the numeric amounts or measurements to convert."
                },
                "source_unit_column": {
                    "type": "string",
                    "title": "Source Unit / Currency Column",
                    "default": "",
                    "description": "Optional column containing row-level currency codes (e.g. 'USD', 'EUR') or unit symbols (e.g. 'kg', 'lbs')."
                },
                "source_unit": {
                    "type": "string",
                    "title": "Static Source Unit / Currency",
                    "default": "USD",
                    "description": "Default source currency or unit when not provided row-wise in a column."
                },
                "target_unit": {
                    "type": "string",
                    "title": "Target Currency / Unit",
                    "default": "USD",
                    "description": "Target currency code (e.g. 'USD', 'EUR', 'GBP') or target unit (e.g. 'kg', 'km', 'liters', 'mb')."
                },
                "dimension": {
                    "type": "string",
                    "title": "Physical Dimension (for unit mode)",
                    "enum": ["weight", "length", "volume", "data_size", "speed", "temperature"],
                    "default": "weight",
                    "description": "Measurement dimension when normalization_type is 'unit'."
                },
                "custom_exchange_rates": {
                    "type": "object",
                    "title": "Custom Exchange Rates (to USD)",
                    "default": {},
                    "additionalProperties": {"type": "number"},
                    "description": "Optional custom FX rates mapping (e.g. {'EUR': 1.10, 'GBP': 1.30})."
                },
                "output_column": {
                    "type": "string",
                    "title": "Output Column Name",
                    "default": "amount_normalized",
                    "description": "Name of output column. If blank, updates in-place."
                }
            },
            "required": ["amount_column"]
        }

    def _convert_temperature(self, val: float, src: str, tgt: str) -> float:
        src = src.upper().strip()
        tgt = tgt.upper().strip()
        # Convert to Celsius first
        if src in ["C", "CELSIUS"]:
            c = val
        elif src in ["F", "FAHRENHEIT"]:
            c = (val - 32.0) * 5.0 / 9.0
        elif src in ["K", "KELVIN"]:
            c = val - 273.15
        else:
            c = val

        # Convert Celsius to target
        if tgt in ["C", "CELSIUS"]:
            return c
        elif tgt in ["F", "FAHRENHEIT"]:
            return (c * 9.0 / 5.0) + 32.0
        elif tgt in ["K", "KELVIN"]:
            return c + 273.15
        return c

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        df = inputs.get("dataframe") if "dataframe" in inputs else inputs.get("input")
        if df is None:
            df = inputs.get("df")

        if df is None:
            return {
                "dataframe": pd.DataFrame(),
                "metrics": {"rows_processed": 0},
                "output_summary": {"title": "Currency / Unit Normalizer", "message": "No input table provided.", "metrics": {}}
            }

        df = df.copy()
        if len(df) == 0:
            return {
                "dataframe": df,
                "metrics": {"rows_processed": 0},
                "output_summary": {"title": "Currency / Unit Normalizer", "message": "Input table is empty.", "metrics": {}}
            }

        norm_type = config.get("normalization_type", "currency")
        amount_col = config.get("amount_column", "amount")
        src_unit_col = config.get("source_unit_column")
        static_src_unit = config.get("source_unit", "USD")
        target_unit = config.get("target_unit", "USD").strip()
        dimension = config.get("dimension", "weight")
        custom_fx = config.get("custom_exchange_rates", {})
        output_col = config.get("output_column") or f"{amount_col}_{target_unit.lower()}"

        if amount_col not in df.columns:
            return {
                "dataframe": df,
                "metrics": {"rows_processed": len(df)},
                "output_summary": {"title": "Currency / Unit Normalizer", "message": f"Amount column '{amount_col}' not found.", "metrics": {}}
            }

        amounts = pd.to_numeric(df[amount_col], errors="coerce").fillna(0.0)

        # 1. CURRENCY NORMALIZATION
        if norm_type == "currency":
            fx_rates = dict(DEFAULT_FX_RATES_TO_USD)
            if custom_fx:
                for k, v in custom_fx.items():
                    fx_rates[str(k).upper()] = float(v)

            target_curr = target_unit.upper()
            target_rate_usd = fx_rates.get(target_curr, 1.0)

            normalized_amounts = []
            for i in range(len(df)):
                val = amounts.iloc[i]
                if src_unit_col and src_unit_col in df.columns:
                    src_c = str(df[src_unit_col].iloc[i]).upper().strip()
                else:
                    src_c = static_src_unit.upper().strip()

                src_rate_usd = fx_rates.get(src_c, 1.0)
                # Convert to USD then to target currency
                usd_val = val * src_rate_usd
                target_val = usd_val / target_rate_usd
                normalized_amounts.append(round(float(target_val), 4))

            df[output_col] = normalized_amounts

        # 2. UNIT OF MEASURE NORMALIZATION
        else:
            if dimension == "temperature":
                normalized_amounts = []
                for i in range(len(df)):
                    val = amounts.iloc[i]
                    if src_unit_col and src_unit_col in df.columns:
                        src_u = str(df[src_unit_col].iloc[i]).strip()
                    else:
                        src_u = static_src_unit.strip()
                    res_temp = self._convert_temperature(val, src_u, target_unit)
                    normalized_amounts.append(round(float(res_temp), 2))
                df[output_col] = normalized_amounts
            else:
                dim_table = UNIT_CONVERSIONS.get(dimension, UNIT_CONVERSIONS["weight"])
                target_factor = dim_table.get(target_unit.lower(), 1.0)

                normalized_amounts = []
                for i in range(len(df)):
                    val = amounts.iloc[i]
                    if src_unit_col and src_unit_col in df.columns:
                        src_u = str(df[src_unit_col].iloc[i]).lower().strip()
                    else:
                        src_u = static_src_unit.lower().strip()

                    src_factor = dim_table.get(src_u, 1.0)
                    # Convert to base then to target
                    base_val = val * src_factor
                    target_val = base_val / target_factor
                    normalized_amounts.append(round(float(target_val), 4))

                df[output_col] = normalized_amounts

        output_summary = {
            "title": "Currency / Unit Normalizer",
            "message": f"Normalized column '{amount_col}' to target '{target_unit}' in mode '{norm_type}'.",
            "metrics": {
                "normalization_type": norm_type,
                "target_unit": target_unit,
                "output_column": output_col,
                "rows_processed": len(df)
            }
        }

        return {
            "dataframe": df,
            "metrics": output_summary["metrics"],
            "output_summary": output_summary
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        t_unit = repr(config.get("target_unit", "USD"))
        n_type = repr(config.get("normalization_type", "currency"))
        return (
            f"# Recipe: {self.name}\n"
            f"# ID: {self.recipe_id}\n"
            f"# Type: {n_type}, Target: {t_unit}\n"
            f"# Normalizes amounts to single currency or unit\n"
        )
