import os
import sys
import json
import datetime
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort
from backend.app.recipes.training.encoder_utils import pass_through_metadata


class ModelCardGeneratorRecipe(BaseRecipe):
    """
    Model Card Generator Recipe.
    Generates standardized, regulatory-grade Model Cards following the Mitchell et al. (2019) /
    Google / HuggingFace Model Cards standard.
    Documents model details, intended domain, training data distribution, evaluation benchmarks,
    fairness audit summary, ethical considerations, environmental footprint, and limitations.
    Outputs executive Markdown Model Cards and structured JSON metadata.
    """

    recipe_id = "model_card_generator"
    name = "Model Card Generator (Mitchell et al. Standard)"
    version = "1.0.0"
    category = "governance"
    group = "Evaluation & Governance"
    subgroup = "Fairness & Compliance"
    description = (
        "Generates standardized Model Cards documenting architecture, intended use, data distribution, "
        "performance benchmarks, fairness audits, ethical considerations, and limitations."
    )
    input_types = ["model", "dataframe", "metrics"]
    output_types = ["report", "metrics", "dataframe"]

    inputs: List[RecipePort] = [
        RecipePort(
            id="model",
            label="Trained Model (Optional)",
            type="model",
            required=False,
            max_connections=1,
            description="Trained model artifact to inspect parameters and architecture."
        ),
        RecipePort(
            id="data",
            label="Dataset / Test Split (Optional)",
            type="dataframe",
            required=False,
            max_connections=1,
            description="Dataset to document features, data distribution, and sample sizes."
        ),
        RecipePort(
            id="metrics",
            label="Evaluation / Fairness Metrics (Optional)",
            type="metrics",
            required=False,
            max_connections=1,
            description="Metrics dictionary from model evaluator or fairness audit."
        )
    ]

    outputs: List[RecipePort] = [
        RecipePort(
            id="report",
            label="Executive Markdown Model Card",
            type="report",
            description="Standardized Markdown Model Card document."
        ),
        RecipePort(
            id="model_card_json",
            label="Structured Model Card JSON",
            type="metrics",
            description="Machine-readable Model Card dictionary conforming to governance specifications."
        ),
        RecipePort(
            id="summary_table",
            label="Model Card Summary Table",
            type="dataframe",
            description="Tabular overview of model card metadata and key metrics."
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "model_name": {
                    "type": "string",
                    "title": "Model Name",
                    "input_type": "text",
                    "default": "Enterprise Predictive Model",
                    "description": "Descriptive name of the model."
                },
                "model_version": {
                    "type": "string",
                    "title": "Model Version",
                    "input_type": "text",
                    "default": "1.0.0",
                    "description": "Release version identifier (e.g., 1.0.0, v2-rc1)."
                },
                "model_overview": {
                    "type": "string",
                    "title": "Model Description / Overview",
                    "input_type": "textarea",
                    "default": "Machine learning model trained for automated decisioning, ranking, or risk assessment.",
                    "description": "High-level summary of model function and architecture."
                },
                "intended_use": {
                    "type": "string",
                    "title": "Intended Use & Domain",
                    "input_type": "textarea",
                    "default": "Intended for production automated inference and batch decision scoring within validated domain distributions.",
                    "description": "Primary intended applications and user workflows."
                },
                "out_of_scope_use": {
                    "type": "string",
                    "title": "Out-of-Scope & Prohibited Use Cases",
                    "input_type": "textarea",
                    "default": "Not designed for autonomous life-critical applications or unvalidated demographic subgroups without human oversight.",
                    "description": "Applications for which the model is not intended or certified."
                },
                "target_users": {
                    "type": "string",
                    "title": "Target Users & Stakeholders",
                    "input_type": "text",
                    "default": "Data Science Practitioners, Operations Teams, and Automated Decision Services",
                    "description": "Who will interact with or rely on this model."
                },
                "ethical_considerations": {
                    "type": "string",
                    "title": "Ethical & Fairness Considerations",
                    "input_type": "textarea",
                    "default": "Audited for demographic parity, disparate impact, and protected attribute equity under EEOC guidelines.",
                    "description": "Key ethical trade-offs, fairness assessments, and mitigation controls."
                },
                "limitations": {
                    "type": "string",
                    "title": "Model Limitations & Failure Modes",
                    "input_type": "textarea",
                    "default": "Performance may degrade under severe covariate drift, unexpected null spikes, or extreme out-of-distribution values.",
                    "description": "Known edge cases, limitations, and vulnerability conditions."
                },
                "environmental_impact": {
                    "type": "string",
                    "title": "Environmental Impact & Carbon Footprint",
                    "input_type": "text",
                    "default": "Trained on cloud CPU/GPU instances with energy-efficient hardware acceleration.",
                    "description": "Compute and environmental resource consumption estimates."
                }
            },
            "required": []
        }

    def _extract_model_specs(self, model: Any) -> Dict[str, Any]:
        if model is None:
            return {"architecture": "Custom / Black-Box Estimator", "parameters": {}}

        cls_name = model.__class__.__name__
        mod_name = model.__class__.__module__

        params: Dict[str, Any] = {}
        if hasattr(model, "get_params"):
            try:
                for k, v in model.get_params(deep=False).items():
                    if isinstance(v, (int, float, str, bool, type(None))):
                        params[k] = v
                    else:
                        params[k] = str(v)
            except Exception:
                pass

        return {
            "model_class": cls_name,
            "model_module": mod_name,
            "architecture": f"{cls_name} ({mod_name})",
            "parameters": params
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        # 1. Resolve Inputs
        model = inputs.get("model")
        df = inputs.get("data")
        if df is None:
            df = inputs.get("dataframe")
        if df is None:
            df = inputs.get("test_data")
        if df is None and isinstance(context, dict):
            df = context.get("dataframe")

        metrics_input = inputs.get("metrics")
        if metrics_input is None and isinstance(context, dict):
            metrics_input = context.get("metrics")

        # 2. Extract Configuration
        model_name = config.get("model_name", "Enterprise Predictive Model")
        model_version = config.get("model_version", "1.0.0")
        model_overview = config.get("model_overview", "Machine learning model trained for automated decisioning.")
        intended_use = config.get("intended_use", "Intended for production automated inference and batch decision scoring.")
        out_of_scope = config.get("out_of_scope_use", "Not designed for autonomous life-critical applications.")
        target_users = config.get("target_users", "Data Scientists, Risk Analysts, and Automated Decision Services")
        ethical_cons = config.get("ethical_considerations", "Audited for demographic parity and disparate impact under EEOC guidelines.")
        limitations = config.get("limitations", "Performance may degrade under severe covariate drift or unseen distributions.")
        env_impact = config.get("environmental_impact", "Trained on cloud CPU/GPU instances with low carbon footprint.")

        timestamp_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        # 3. Model Architecture Details
        model_specs = self._extract_model_specs(model)

        # 4. Training / Evaluation Data Specs
        data_specs: Dict[str, Any] = {}
        if isinstance(df, pd.DataFrame) and not df.empty:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
            data_specs = {
                "total_samples": int(len(df)),
                "total_features": int(len(df.columns)),
                "numeric_features_count": len(num_cols),
                "categorical_features_count": len(cat_cols),
                "feature_names": list(df.columns)
            }
        else:
            data_specs = {"total_samples": "N/A", "total_features": "N/A", "status": "No dataset attached"}

        # 5. Extract & Format Quantitative Benchmarks
        benchmarks: Dict[str, Any] = {}
        if isinstance(metrics_input, dict):
            for k, v in metrics_input.items():
                if isinstance(v, (int, float, str, bool)):
                    benchmarks[k] = v
                elif isinstance(v, dict):
                    # Flatten top-level items
                    for sub_k, sub_v in v.items():
                        if isinstance(sub_v, (int, float, str, bool)):
                            benchmarks[f"{k}_{sub_k}"] = sub_v

        # 6. Assemble Machine-Readable Model Card JSON
        model_card_dict = {
            "schema_version": "1.0.0",
            "model_details": {
                "name": model_name,
                "version": model_version,
                "created_at": timestamp_str,
                "architecture": model_specs.get("architecture", "Unknown"),
                "parameters": model_specs.get("parameters", {}),
                "overview": model_overview
            },
            "intended_use": {
                "primary_uses": intended_use,
                "out_of_scope_uses": out_of_scope,
                "target_users": target_users
            },
            "data_factors": data_specs,
            "quantitative_analyses": {
                "performance_benchmarks": benchmarks
            },
            "ethical_considerations": ethical_cons,
            "limitations": limitations,
            "environmental_impact": env_impact
        }

        # 7. Generate Executive Markdown Model Card Document
        report_lines = [
            f"# === Model Card: {model_name} (v{model_version}) ===",
            f"**Generated**: `{timestamp_str}` | **Standard**: Mitchell et al. / HuggingFace Model Cards",
            "",
            "## 1. Model Details",
            f"- **Model Architecture**: `{model_specs.get('architecture', 'Custom Estimator')}`",
            f"- **Model Version**: `{model_version}`",
            f"- **Overview**: {model_overview}",
            "",
            "## 2. Intended Use & Domain",
            f"- **Primary Intended Use**: {intended_use}",
            f"- **Target Users**: {target_users}",
            f"- **Out-of-Scope / Prohibited Applications**: {out_of_scope}",
            "",
            "## 3. Data Factors & Distribution",
        ]

        if isinstance(df, pd.DataFrame) and not df.empty:
            report_lines.append(f"- **Sample Size (N)**: `{data_specs['total_samples']:,}` rows")
            report_lines.append(f"- **Total Features**: `{data_specs['total_features']}` ({data_specs['numeric_features_count']} numeric, {data_specs['categorical_features_count']} categorical)")
            report_lines.append(f"- **Feature Schema**: `{', '.join(data_specs['feature_names'][:12])}{'...' if len(data_specs['feature_names']) > 12 else ''}`")
        else:
            report_lines.append("- *(No training/evaluation dataset was connected to this card generation node)*")

        report_lines.append("")
        report_lines.append("## 4. Quantitative Evaluation Benchmarks")
        if benchmarks:
            report_lines.append("| Metric Name | Evaluated Value |")
            report_lines.append("| :--- | :--- |")
            for k, v in list(benchmarks.items())[:15]:
                v_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                report_lines.append(f"| `{k}` | **`{v_str}`** |")
        else:
            report_lines.append("- *(No quantitative evaluation metrics dictionary was passed into this node)*")

        report_lines.append("")
        report_lines.append("## 5. Ethical Considerations & Algorithmic Fairness")
        report_lines.append(f"{ethical_cons}")
        report_lines.append("")
        report_lines.append("## 6. Limitations & Caveats")
        report_lines.append(f"> [!NOTE]\n> {limitations}")
        report_lines.append("")
        report_lines.append("## 7. Environmental Impact & Sustainability")
        report_lines.append(f"- **Compute Profile**: {env_impact}")
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("*(Reference: Mitchell, M., et al. 'Model Cards for Model Reporting', FAT* 2019.)*")

        report_markdown = "\n".join(report_lines)

        # 8. Summary DataFrame
        summary_rows = [
            {"Property": "Model Name", "Value": model_name},
            {"Property": "Version", "Value": model_version},
            {"Property": "Architecture", "Value": model_specs.get("architecture", "Custom")},
            {"Property": "Dataset Rows", "Value": str(data_specs.get("total_samples", "N/A"))},
            {"Property": "Dataset Features", "Value": str(data_specs.get("total_features", "N/A"))},
            {"Property": "Target Users", "Value": target_users},
            {"Property": "Benchmark Metrics Count", "Value": str(len(benchmarks))},
            {"Property": "Created At", "Value": timestamp_str}
        ]
        summary_df = pd.DataFrame(summary_rows)

        output: Dict[str, Any] = {
            "report": report_markdown,
            "model_card_json": model_card_dict,
            "metrics": model_card_dict,
            "summary_table": summary_df,
            "dataframe": summary_df
        }

        return pass_through_metadata(inputs, output, context)

    def to_code(self, config: Dict[str, Any]) -> str:
        name = config.get("model_name", "Production Model")
        ver = config.get("model_version", "1.0.0")
        return (
            "import json\n\n"
            f"# Generate standardized Model Card for '{name}' (v{ver})\n"
            "card = {\n"
            f"    'model_name': '{name}',\n"
            f"    'version': '{ver}',\n"
            "    'architecture': str(type(model)),\n"
            "    'n_samples': len(df) if 'df' in locals() else 'N/A'\n"
            "}\n"
            "print('Model Card generated successfully.')"
        )
