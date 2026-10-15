"""
Sampling Operator (random/stratified/systematic) Recipe.
Sub-group: Domain-Specific Cleaning
Category: Data Preparation & Transformation (preprocessing)

Extracts representative sub-samples of large datasets using simple random,
class-balanced stratified, or systematic interval sampling methodologies.
"""
import math
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class SamplingOperatorRecipe(BaseRecipe):
    recipe_id = "sampling_operator"
    name = "Sampling Operator (random/stratified/systematic)"
    version = "1.0.0"
    category = "preprocessing"
    group = "Data Preparation & Transformation"
    subgroup = "Domain-Specific Cleaning"
    description = "Takes a random, stratified or systematic sample of rows."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="input",
            label="Input Table",
            type="dataframe",
            required=True,
            max_connections=1,
            description="Dataset to sample from"
        )
    ]

    outputs = [
        RecipePort(
            id="dataframe",
            label="Sampled Table",
            type="dataframe",
            description="Sampled subset of records"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "sampling_method": {
                    "type": "string",
                    "title": "Sampling Methodology",
                    "enum": ["random", "stratified", "systematic"],
                    "default": "random",
                    "description": "'random' selects uniformly at random; 'stratified' preserves subgroup proportions; 'systematic' selects every k-th row."
                },
                "sample_mode": {
                    "type": "string",
                    "title": "Sample Size Mode",
                    "enum": ["fraction", "count"],
                    "default": "fraction",
                    "description": "'fraction' selects percentage of rows (e.g. 0.25 = 25%); 'count' selects fixed number of rows (e.g. 500)."
                },
                "sample_fraction": {
                    "type": "number",
                    "title": "Sample Fraction (0.0 - 1.0)",
                    "default": 0.2,
                    "description": "Proportion of dataset to include when sample_mode is 'fraction'."
                },
                "sample_count": {
                    "type": "integer",
                    "title": "Sample Row Count",
                    "default": 1000,
                    "description": "Target number of rows when sample_mode is 'count'."
                },
                "strata_column": {
                    "type": "string",
                    "title": "Strata Column (for Stratified)",
                    "default": "",
                    "description": "Categorical column to stratify on (e.g. 'label', 'churn', 'segment')."
                },
                "replace": {
                    "type": "boolean",
                    "title": "Sample with Replacement",
                    "default": False,
                    "description": "Whether records can be selected more than once (bootstrapping)."
                },
                "random_state": {
                    "type": "integer",
                    "title": "Random Seed",
                    "default": 42,
                    "description": "Seed for deterministic reproducibility."
                }
            },
            "required": ["sampling_method"]
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        df = inputs.get("dataframe") if "dataframe" in inputs else inputs.get("input")
        if df is None:
            df = inputs.get("df")

        if df is None:
            return {
                "dataframe": pd.DataFrame(),
                "metrics": {"original_rows": 0, "sampled_rows": 0},
                "output_summary": {"title": "Sampling Operator", "message": "No input table provided.", "metrics": {}}
            }

        df = df.copy()
        total_rows = len(df)
        if total_rows == 0:
            return {
                "dataframe": df,
                "metrics": {"original_rows": 0, "sampled_rows": 0},
                "output_summary": {"title": "Sampling Operator", "message": "Input table is empty.", "metrics": {}}
            }

        method = config.get("sampling_method", "random")
        mode = config.get("sample_mode", "fraction")
        fraction = float(config.get("sample_fraction", 0.2))
        target_count = int(config.get("sample_count", 1000))
        strata_col = config.get("strata_column")
        replace = config.get("replace", False)
        seed = config.get("random_state", 42)

        rng = np.random.RandomState(seed)

        # Compute target n
        if mode == "fraction":
            target_n = max(1, min(total_rows, int(total_rows * fraction)))
        else:
            target_n = max(1, min(total_rows, target_count))

        # 1. RANDOM SAMPLING
        if method == "random":
            if replace:
                sampled_df = df.sample(n=target_n, replace=True, random_state=seed)
            else:
                sampled_df = df.sample(n=target_n, replace=False, random_state=seed)

        # 2. STRATIFIED SAMPLING
        elif method == "stratified":
            if not strata_col or strata_col not in df.columns:
                # Fallback to random if strata col missing
                sampled_df = df.sample(n=target_n, replace=replace, random_state=seed)
            else:
                # Stratified sample proportional to group sizes by index
                frac = target_n / total_rows
                sampled_indices = []
                for _, grp in df.groupby(strata_col):
                    n_grp_samples = max(1, round(len(grp) * frac))
                    n_grp_samples = min(len(grp), n_grp_samples) if not replace else n_grp_samples
                    if n_grp_samples > 0:
                        grp_idx = grp.sample(n=n_grp_samples, replace=replace, random_state=seed).index
                        sampled_indices.extend(grp_idx)

                sampled_df = df.loc[sampled_indices] if sampled_indices else df.sample(n=min(len(df), target_n), random_state=seed)

        # 3. SYSTEMATIC SAMPLING
        else:
            # Select every k-th record starting from random offset
            k = max(1, total_rows // target_n)
            start_offset = rng.randint(0, max(1, k))
            indices = list(range(start_offset, total_rows, k))[:target_n]
            sampled_df = df.iloc[indices]

        sampled_df = sampled_df.reset_index(drop=True)

        output_summary = {
            "title": "Sampling Operator",
            "message": f"Sampled {len(sampled_df):,} records ({round(len(sampled_df) / total_rows * 100, 1)}%) from {total_rows:,} rows via '{method}' sampling.",
            "metrics": {
                "sampling_method": method,
                "original_rows": total_rows,
                "sampled_rows": len(sampled_df),
                "sampling_ratio": f"{round(len(sampled_df) / total_rows * 100, 2)}%",
                "strata_column": strata_col if method == "stratified" else None
            }
        }

        return {
            "dataframe": sampled_df,
            "metrics": output_summary["metrics"],
            "output_summary": output_summary
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        method = repr(config.get("sampling_method", "random"))
        mode = repr(config.get("sample_mode", "fraction"))
        return (
            f"# Recipe: {self.name}\n"
            f"# ID: {self.recipe_id}\n"
            f"# Method: {method}, Mode: {mode}\n"
            f"# Samples dataset using specified strategy\n"
        )
