"""
Data Anonymization (Differential Privacy) Recipe.
Sub-group: Domain-Specific Cleaning
Category: Data Preparation & Transformation (preprocessing)

Adds mathematically calibrated Laplace or Gaussian noise to numeric features,
providing formal differential privacy guarantees (epsilon-DP) while preserving aggregate statistical utility.
"""
import math
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class DifferentialPrivacyAnonymizerRecipe(BaseRecipe):
    recipe_id = "data_anonymization_differential_privacy"
    name = "Data Anonymization (Differential Privacy)"
    version = "1.0.0"
    category = "preprocessing"
    group = "Data Preparation & Transformation"
    subgroup = "Domain-Specific Cleaning"
    description = "Adds calibrated noise to protect individual privacy while keeping aggregate accuracy."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="input",
            label="Input Table",
            type="dataframe",
            required=True,
            max_connections=1,
            description="Dataset containing sensitive numeric fields to privatize"
        )
    ]

    outputs = [
        RecipePort(
            id="dataframe",
            label="Privatized Table",
            type="dataframe",
            description="Dataset with differential privacy calibrated noise applied"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "columns": {
                    "type": "array",
                    "title": "Columns to Privatize",
                    "items": {"type": "string"},
                    "default": [],
                    "description": "Numeric columns containing sensitive data to perturb (e.g. ['salary', 'net_worth', 'medical_score'])."
                },
                "epsilon": {
                    "type": "number",
                    "title": "Privacy Budget (Epsilon > 0)",
                    "default": 1.0,
                    "description": "Epsilon parameter. Smaller values (0.1 - 0.5) provide stronger privacy; larger values (1.0 - 5.0) preserve higher numerical accuracy."
                },
                "mechanism": {
                    "type": "string",
                    "title": "Noise Mechanism",
                    "enum": ["laplace", "gaussian"],
                    "default": "laplace",
                    "description": "'laplace' satisfies pure epsilon-DP; 'gaussian' satisfies (epsilon, delta)-DP."
                },
                "delta": {
                    "type": "number",
                    "title": "Delta Parameter (Gaussian only)",
                    "default": 1e-5,
                    "description": "Relaxation parameter for Gaussian mechanism."
                },
                "clip_negative": {
                    "type": "boolean",
                    "title": "Clip Negative Values (>= 0)",
                    "default": True,
                    "description": "Clamps negative perturbed values to zero for inherently non-negative metrics."
                },
                "output_mode": {
                    "type": "string",
                    "title": "Output Mode",
                    "enum": ["in_place", "new_column"],
                    "default": "in_place",
                    "description": "Whether to replace columns in-place or generate new columns with a suffix."
                },
                "new_column_suffix": {
                    "type": "string",
                    "title": "New Column Suffix",
                    "default": "_dp_noisy",
                    "description": "Suffix for newly generated columns."
                },
                "random_state": {
                    "type": "integer",
                    "title": "Random Seed",
                    "default": 42,
                    "description": "Seed for deterministic testability and reproducibility."
                }
            },
            "required": ["columns"]
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        df = inputs.get("dataframe") if "dataframe" in inputs else inputs.get("input")
        if df is None:
            df = inputs.get("df")

        if df is None:
            return {
                "dataframe": pd.DataFrame(),
                "metrics": {"columns_privatized": 0, "rows_processed": 0},
                "output_summary": {"title": "Data Anonymization", "message": "No input table provided.", "metrics": {}}
            }

        df = df.copy()
        if len(df) == 0:
            return {
                "dataframe": df,
                "metrics": {"columns_privatized": 0, "rows_processed": 0},
                "output_summary": {"title": "Data Anonymization", "message": "Input table is empty.", "metrics": {}}
            }

        cols_config = config.get("columns", [])
        epsilon = max(0.001, float(config.get("epsilon", 1.0)))
        mechanism = config.get("mechanism", "laplace")
        delta = max(1e-9, float(config.get("delta", 1e-5)))
        clip_neg = config.get("clip_negative", True)
        output_mode = config.get("output_mode", "in_place")
        suffix = config.get("new_column_suffix", "_dp_noisy")
        seed = config.get("random_state", 42)

        rng = np.random.RandomState(seed)

        target_cols = [c for c in cols_config if c in df.columns]
        if not target_cols:
            # Fallback to numeric columns
            target_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]

        if not target_cols:
            return {
                "dataframe": df,
                "metrics": {"columns_privatized": 0, "rows_processed": len(df)},
                "output_summary": {"title": "Data Anonymization", "message": "No numeric columns found to privatize.", "metrics": {}}
            }

        col_deviations = {}
        for col in target_cols:
            dest_col = col if output_mode == "in_place" else f"{col}{suffix}"
            s = pd.to_numeric(df[col], errors="coerce").fillna(0.0).values

            # Sensitivity estimate: Delta f = (max - min) / sqrt(N)
            min_val = np.min(s)
            max_val = np.max(s)
            data_range = max_val - min_val
            n_records = len(s)
            sensitivity = max(1.0, data_range / math.sqrt(max(1, n_records)))

            # Noise generation
            if mechanism == "gaussian":
                sigma = (sensitivity / epsilon) * math.sqrt(2.0 * math.log(1.25 / delta))
                noise = rng.normal(loc=0.0, scale=sigma, size=len(s))
            else:  # laplace
                b = sensitivity / epsilon
                noise = rng.laplace(loc=0.0, scale=b, size=len(s))

            noisy_s = s + noise
            if clip_neg and min_val >= 0:
                noisy_s = np.maximum(0.0, noisy_s)

            df[dest_col] = np.round(noisy_s, 3)
            mad = float(round(np.mean(np.abs(noisy_s - s)), 3))
            col_deviations[col] = mad

        output_summary = {
            "title": "Differential Privacy Anonymizer",
            "message": f"Privatized {len(target_cols)} column(s) using {mechanism.capitalize()} mechanism with epsilon={epsilon}.",
            "metrics": {
                "epsilon": epsilon,
                "mechanism": mechanism,
                "columns_privatized": target_cols,
                "mean_absolute_noise": col_deviations,
                "rows_processed": len(df)
            }
        }

        return {
            "dataframe": df,
            "metrics": output_summary["metrics"],
            "output_summary": output_summary
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        eps = repr(config.get("epsilon", 1.0))
        mech = repr(config.get("mechanism", "laplace"))
        cols = repr(config.get("columns", []))
        return (
            f"# Recipe: {self.name}\n"
            f"# ID: {self.recipe_id}\n"
            f"# Mechanism: {mech}, Epsilon: {eps}, Columns: {cols}\n"
            f"# Applies differential privacy noise to sensitive fields\n"
        )
