"""
Reusable Template / Macro Block Library Recipe (Canvas Tools -> Versioning & Reuse).

Saves groups of nodes as reusable, modular macro templates (e.g. Standard Scaler + Imputer + OneHot,
or Model Tournament + Calibration + Explainability) and instantiates them into other pipelines.
"""

import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

try:
    import pandas as pd
except ImportError:
    pd = None

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class TemplateMacroLibraryRecipe(BaseRecipe):
    recipe_id = "template_macro_library"
    name = "Reusable Template / Macro Block Library"
    version = "1.0.0"
    category = "canvas"
    group = "Canvas Tools"
    subgroup = "Versioning & Reuse"
    description = "Saves groups of nodes as reusable blocks and inserts them into other pipelines."
    input_types = ["dataframe"]
    output_types = ["dataframe", "report"]

    inputs: List[RecipePort] = [
        RecipePort(
            id="input",
            label="Input Dataset (Optional)",
            type="dataframe",
            required=False,
            max_connections=1,
            description="Upstream dataset to pass through the macro blueprint."
        )
    ]

    outputs: List[RecipePort] = [
        RecipePort(
            id="output",
            label="Passthrough Table",
            type="dataframe",
            description="Forwarded dataset passed untouched to downstream nodes."
        ),
        RecipePort(
            id="report",
            label="Macro Specification",
            type="report",
            description="Template architectural spec and parameterized sub-DAG definition."
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "template_name": {
                    "type": "string",
                    "title": "Macro Template Name",
                    "input_type": "text",
                    "default": "Standard Tabular Preprocessing Macro",
                    "description": "Unique title identifying this reusable sub-DAG block in team library."
                },
                "category": {
                    "type": "string",
                    "title": "Template Domain",
                    "input_type": "dropdown",
                    "enum": ["Preprocessing", "Feature Engineering", "Modeling", "Evaluation", "Custom"],
                    "default": "Preprocessing",
                    "description": "Functional domain categorization for catalog filtering."
                },
                "description": {
                    "type": "string",
                    "title": "Macro Block Description",
                    "input_type": "textarea",
                    "default": "Modular bundle containing missing value imputation, standard scaling, and one-hot encoding.",
                    "description": "Detailed explanation of what this macro encapsulates."
                },
                "target_pipeline_type": {
                    "type": "string",
                    "title": "Target Problem Type",
                    "input_type": "dropdown",
                    "enum": ["tabular_classification", "tabular_regression", "time_series", "nlp", "generic"],
                    "default": "tabular_classification",
                    "description": "Machine learning problem type this macro template is architected for."
                },
                "parameter_overrides": {
                    "type": "string",
                    "title": "Parameter Overrides (JSON)",
                    "input_type": "textarea",
                    "default": "{\"missing_value_imputer\": {\"strategy\": \"median\"}, \"feature_scaler\": {\"method\": \"standard\"}}",
                    "description": "JSON object providing default parameter values for nodes inside this macro."
                },
                "is_public": {
                    "type": "boolean",
                    "title": "Share with Organization",
                    "input_type": "checkbox",
                    "default": True,
                    "description": "Make template accessible across all workspace tenants and projects."
                }
            }
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        upstream_df = inputs.get("dataframe") if "dataframe" in inputs else inputs.get("input")

        # Edge Case 1: Parameter extraction with defaults
        name = str(config.get("template_name", "Standard Tabular Preprocessing Macro")).strip() or "Macro Template"
        category = str(config.get("category", "Preprocessing"))
        description = str(config.get("description", "")).strip()
        pipeline_type = str(config.get("target_pipeline_type", "tabular_classification"))
        is_public = bool(config.get("is_public", True))

        # Edge Case 2: Parse parameter overrides JSON safely
        raw_overrides = config.get("parameter_overrides", "{}")
        overrides: Dict[str, Any] = {}
        json_error = None
        if isinstance(raw_overrides, dict):
            overrides = raw_overrides
        elif isinstance(raw_overrides, str):
            clean_str = raw_overrides.strip()
            if clean_str:
                try:
                    overrides = json.loads(clean_str)
                    if not isinstance(overrides, dict):
                        overrides = {}
                        json_error = "Parsed JSON was not a dictionary object."
                except Exception as ex:
                    json_error = f"Malformed JSON: {str(ex)}"
                    overrides = {}

        # Preset template sub-DAG blueprints
        PRESET_BLUEPRINTS = {
            "tabular_classification": [
                {"step": 1, "recipe_id": "missing_value_imputer", "name": "Missing Value Imputer", "default_config": {"strategy": "median"}},
                {"step": 2, "recipe_id": "feature_scaler", "name": "Feature Scaler", "default_config": {"method": "standard"}},
                {"step": 3, "recipe_id": "categorical_encoder", "name": "Categorical Encoder", "default_config": {"method": "one_hot"}},
                {"step": 4, "recipe_id": "train_test_split", "name": "Train/Test Split", "default_config": {"test_size": 0.2}},
                {"step": 5, "recipe_id": "xgboost_trainer", "name": "XGBoost Classifier", "default_config": {"task_type": "classification", "n_estimators": 100}},
                {"step": 6, "recipe_id": "model_evaluator", "name": "Model Performance Evaluator", "default_config": {}}
            ],
            "tabular_regression": [
                {"step": 1, "recipe_id": "missing_value_imputer", "name": "Missing Value Imputer", "default_config": {"strategy": "mean"}},
                {"step": 2, "recipe_id": "feature_scaler", "name": "Feature Scaler", "default_config": {"method": "robust"}},
                {"step": 3, "recipe_id": "train_test_split", "name": "Train/Test Split", "default_config": {"test_size": 0.2}},
                {"step": 4, "recipe_id": "random_forest_trainer", "name": "Random Forest Regressor", "default_config": {"task_type": "regression"}},
                {"step": 5, "recipe_id": "model_evaluator", "name": "Regression Evaluator", "default_config": {}}
            ],
            "time_series": [
                {"step": 1, "recipe_id": "data_type_converter", "name": "Date Type Parser", "default_config": {}},
                {"step": 2, "recipe_id": "time_series_split", "name": "Time-Series Splitter", "default_config": {}},
                {"step": 3, "recipe_id": "lightgbm_trainer", "name": "LightGBM Regressor", "default_config": {"task_type": "regression"}},
                {"step": 4, "recipe_id": "model_evaluator", "name": "Forecast Evaluator", "default_config": {}}
            ]
        }

        base_nodes = PRESET_BLUEPRINTS.get(pipeline_type, PRESET_BLUEPRINTS["tabular_classification"])

        # Apply parameter overrides to template nodes
        for node in base_nodes:
            rid = node["recipe_id"]
            if rid in overrides and isinstance(overrides[rid], dict):
                node["default_config"].update(overrides[rid])

        now_iso = datetime.now(timezone.utc).isoformat()
        md_report = (
            f"### 🧩 Macro Template: **{name}**\n"
            f"**Domain:** `{category}` | **Target Pipeline:** `{pipeline_type}` | **Public:** `{is_public}`\n\n"
            f"{description or 'No description provided.'}\n\n"
            f"#### 🏗️ Instantiated Sub-DAG Nodes ({len(base_nodes)} steps):\n"
            + "\n".join([f"- **Step {n['step']}:** `{n['recipe_id']}` ({n['name']}) ➔ config: `{n['default_config']}`" for n in base_nodes])
        )

        if json_error:
            md_report += f"\n\n> ⚠️ *Parameter Override Notice:* {json_error} (Used default template parameters instead)."

        result: Dict[str, Any] = {
            "template_name": name,
            "category": category,
            "pipeline_type": pipeline_type,
            "sub_dag_nodes": base_nodes,
            "parameter_overrides": overrides,
            "is_public": is_public,
            "metrics": {
                "nodes_in_macro": len(base_nodes),
                "overrides_applied": len(overrides),
                "has_parameter_error": int(json_error is not None)
            },
            "report": {
                "title": f"Macro: {name}",
                "markdown": md_report,
                "timestamp": now_iso
            },
            "output_summary": {
                "status": "MACRO_COMPILED",
                "template": name,
                "nodes_count": len(base_nodes),
                "domain": category
            }
        }

        if upstream_df is not None:
            result["dataframe"] = upstream_df
            result["output"] = upstream_df

        return result
