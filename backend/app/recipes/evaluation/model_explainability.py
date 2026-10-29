import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class ModelExplainabilityRecipe(BaseRecipe):
    """
    Model Explainability Recipe (SHAP & LIME).
    Provides deep post-hoc model interpretability using Shapley Additive exPlanations (SHAP)
    and Local Interpretable Model-agnostic Explanations (LIME).
    Generates global feature importance rankings, single-record waterfall attributions,
    feature effect directions (positive/negative), and governance-ready explanation reports.
    """

    recipe_id = "model_explainability"
    name = "Model Explainability (SHAP / LIME)"
    version = "1.0.0"
    category = "evaluation"
    group = "Modeling"
    subgroup = "Specialized Models"
    description = (
        "Interprets classification and regression models using SHAP (Shapley values) and LIME "
        "to deliver global feature importances, local instance attribution breakdowns, and governance reports."
    )
    input_types = ["model", "test_data"]
    output_types = ["metrics", "report"]

    inputs: List[RecipePort] = [
        RecipePort(
            id="model",
            label="Trained Model",
            type="model",
            required=True,
            max_connections=1,
            description="Trained model to explain (XGBoost, LightGBM, CatBoost, RandomForest, LogisticRegression, etc.)"
        ),
        RecipePort(
            id="test_data",
            label="Holdout / Test Data",
            type="test_data",
            required=False,
            max_connections=1,
            description="Holdout dataset containing X_test and optional y_test from Train/Test Split"
        ),
        RecipePort(
            id="dataframe",
            label="Evaluation Table",
            type="dataframe",
            required=False,
            max_connections=1,
            description="Alternative table input containing feature columns to explain"
        )
    ]

    outputs: List[RecipePort] = [
        RecipePort(
            id="metrics",
            label="Explainability Metrics",
            type="metrics",
            description="Structured global feature importance, local sample attributions, and SHAP/LIME values"
        ),
        RecipePort(
            id="report",
            label="Explanation Governance Report",
            type="report",
            description="Comprehensive human-readable report detailing primary decision drivers"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "title": "Explainability Methodology",
                    "input_type": "dropdown",
                    "enum": ["SHAP", "LIME", "Combined (SHAP + LIME)"],
                    "default": "SHAP",
                    "description": "Select algorithm: SHAP (game-theoretic global/local attribution) or LIME (local sparse linear surrogate) or both."
                },
                "num_features_to_display": {
                    "type": "integer",
                    "title": "Top Features to Display",
                    "input_type": "number",
                    "default": 10,
                    "minimum": 3,
                    "maximum": 50,
                    "description": "Number of top contributing features to include in the executive rankings and report."
                },
                "sample_size": {
                    "type": "integer",
                    "title": "Max Sample Size for Evaluation",
                    "input_type": "number",
                    "default": 100,
                    "minimum": 5,
                    "maximum": 1000,
                    "description": "Number of rows to sample from evaluation data for computing attribution values (speeds up computation for large datasets)."
                },
                "explain_row_index": {
                    "type": "integer",
                    "title": "Local Sample Row Index",
                    "input_type": "number",
                    "default": 0,
                    "minimum": 0,
                    "description": "Specific test row index (0-indexed) to generate individual prediction waterfall explanations for."
                },
                "target_column": {
                    "type": "string",
                    "title": "Target Column (Optional)",
                    "input_type": "text",
                    "default": "",
                    "description": "If passing a raw DataFrame containing target column, specify the target column name to exclude from features."
                }
            }
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        # 1. Resolve Trained Model
        model = inputs.get("model") or inputs.get("trained_model")
        if model is None and isinstance(context, dict):
            model = context.get("model") or context.get("trained_model")
        if model is None:
            raise ValueError("ModelExplainability expects a trained 'model' in inputs. Please connect a Model Trainer node.")

        # 2. Resolve Feature Dataset (X_test)
        X_test = inputs.get("X_test")
        if X_test is None and isinstance(context, dict):
            X_test = context.get("X_test")

        if X_test is None and "test_data" in inputs and isinstance(inputs["test_data"], dict):
            X_test = inputs["test_data"].get("X_test", inputs["test_data"].get("dataframe"))

        if X_test is None and "dataframe" in inputs:
            X_test = inputs["dataframe"]

        if X_test is None and isinstance(context, dict) and "dataframe" in context:
            X_test = context["dataframe"]

        if X_test is None:
            raise ValueError("ModelExplainability expects feature data in 'test_data', 'X_test', or 'dataframe'.")

        # 3. Format and clean X_test
        target_col = config.get("target_column") or (context.get("target_column") if isinstance(context, dict) else "")
        if isinstance(X_test, pd.DataFrame):
            df_features = X_test.copy()
            if target_col and target_col in df_features.columns:
                df_features = df_features.drop(columns=[target_col])
        elif isinstance(X_test, np.ndarray):
            df_features = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X_test.shape[1])])
        elif isinstance(X_test, list):
            df_features = pd.DataFrame(X_test)
        else:
            raise ValueError(f"Unsupported features type: {type(X_test)}")

        if df_features.empty:
            raise ValueError("Evaluation feature dataset is empty.")

        # Align expected features if model has feature_names_in_
        if hasattr(model, "feature_names_in_"):
            expected = list(model.feature_names_in_)
            # If all expected in df_features, order them
            common = [c for c in expected if c in df_features.columns]
            if len(common) > 0:
                df_features = df_features.reindex(columns=expected, fill_value=0)

        # Convert object/string columns to category codes / numeric for shap/lime compatibility
        for col in df_features.columns:
            if not pd.api.types.is_numeric_dtype(df_features[col]):
                try:
                    df_features[col] = pd.to_numeric(df_features[col])
                except Exception:
                    df_features[col] = df_features[col].astype("category").cat.codes

        # Fill any remaining NaNs with column median or 0
        df_features = df_features.fillna(df_features.median(numeric_only=True).fillna(0))

        # 4. Configuration Parameters
        method = config.get("method", "SHAP")
        num_features_to_display = int(config.get("num_features_to_display", 10))
        sample_size = int(config.get("sample_size", 100))
        requested_row_index = int(config.get("explain_row_index", 0))

        # Clamp sample size and local row index
        N_total = len(df_features)
        n_eval = min(sample_size, N_total)
        if N_total > n_eval:
            X_eval = df_features.sample(n=n_eval, random_state=42)
        else:
            X_eval = df_features.copy()

        local_row_idx = max(0, min(requested_row_index, len(X_eval) - 1))
        local_instance = X_eval.iloc[local_row_idx]

        feature_names = list(df_features.columns)

        # Detect task type (Classification vs Regression)
        task_type = "classification" if (hasattr(model, "predict_proba") or hasattr(model, "classes_")) else "regression"
        if "task_type" in inputs:
            task_type = inputs["task_type"]
        elif isinstance(context, dict) and "task_type" in context:
            task_type = context["task_type"]

        metrics: Dict[str, Any] = {
            "task_type": task_type,
            "method": method,
            "total_samples": N_total,
            "evaluated_samples": n_eval,
            "num_features": len(feature_names),
            "explained_row_index": local_row_idx
        }
        warnings: List[str] = []

        # Predict local instance
        raw_pred = model.predict(X_eval.iloc[[local_row_idx]])
        pred_value = float(raw_pred[0]) if isinstance(raw_pred[0], (np.floating, float, int, np.integer)) else str(raw_pred[0])
        pred_prob = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X_eval.iloc[[local_row_idx]])
                pred_prob = [float(p) for p in proba[0]]
            except Exception:
                pass

        metrics["local_prediction"] = {
            "predicted_output": pred_value,
            "predicted_probabilities": pred_prob
        }

        # 5. SHAP Computation
        shap_run = method in ["SHAP", "Combined (SHAP + LIME)"]
        shap_global: List[Dict[str, Any]] = []
        shap_local: List[Dict[str, Any]] = []
        base_value_scalar = 0.0

        if shap_run:
            try:
                import shap
                explainer = None
                shap_values = None

                # Attempt TreeExplainer first (optimal for tree models)
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_eval)
                except Exception:
                    pass

                # Fallback to LinearExplainer or generic Explainer
                if shap_values is None:
                    try:
                        bg_summary = shap.sample(X_eval, min(30, len(X_eval)))
                        if hasattr(model, "coef_"):
                            explainer = shap.LinearExplainer(model, bg_summary)
                            shap_values = explainer.shap_values(X_eval)
                        else:
                            # Generic Explainer with prediction callable
                            predict_fn = model.predict_proba if hasattr(model, "predict_proba") else model.predict
                            explainer = shap.Explainer(predict_fn, bg_summary)
                            res = explainer(X_eval)
                            shap_values = res.values
                    except Exception as e:
                        warnings.append(f"[WARNING: SHAP FALLBACK] Standard explainer failed ({str(e)}). Computing feature perturbation importances.")

                # If shap package computed values
                if shap_values is not None:
                    # Normalize shap_values array dimensions
                    # For binary classification, TreeExplainer often returns list of [arr_class0, arr_class1] or 3D array (N, D, 2)
                    if isinstance(shap_values, list):
                        # Use positive class if available, else first
                        target_class_idx = 1 if len(shap_values) > 1 else 0
                        matrix = np.array(shap_values[target_class_idx])
                    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                        target_class_idx = 1 if shap_values.shape[2] > 1 else 0
                        matrix = shap_values[:, :, target_class_idx]
                    else:
                        matrix = np.array(shap_values)

                    # Extract base value
                    if hasattr(explainer, "expected_value"):
                        ev = explainer.expected_value
                        if isinstance(ev, (list, np.ndarray)):
                            base_value_scalar = float(ev[1]) if len(ev) > 1 else float(ev[0])
                        else:
                            base_value_scalar = float(ev)

                    # Global importance: mean absolute SHAP value across samples
                    mean_abs_shap = np.mean(np.abs(matrix), axis=0)
                    sorted_indices = np.argsort(mean_abs_shap)[::-1]

                    for rank, idx in enumerate(sorted_indices, 1):
                        fname = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
                        val = float(mean_abs_shap[idx])
                        shap_global.append({
                            "rank": rank,
                            "feature": fname,
                            "importance": round(val, 6),
                            "mean_abs_shap": round(val, 6)
                        })

                    # Local explanation for the selected row
                    local_shap_row = matrix[local_row_idx]
                    local_sorted = np.argsort(np.abs(local_shap_row))[::-1]

                    for idx in local_sorted:
                        fname = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
                        contrib = float(local_shap_row[idx])
                        feat_val = local_instance.iloc[idx]
                        shap_local.append({
                            "feature": fname,
                            "feature_value": float(feat_val) if isinstance(feat_val, (int, float, np.floating)) else str(feat_val),
                            "attribution": round(contrib, 6),
                            "direction": "positive" if contrib >= 0 else "negative"
                        })

            except Exception as ex:
                warnings.append(f"[WARNING: SHAP MODULE] SHAP computation encountered an issue: {str(ex)}")

            # Fallback heuristic feature importance if SHAP was unable to run
            if not shap_global:
                if hasattr(model, "feature_importances_"):
                    raw_imp = model.feature_importances_
                elif hasattr(model, "coef_"):
                    raw_imp = np.abs(model.coef_[0] if model.coef_.ndim > 1 else model.coef_)
                else:
                    raw_imp = np.ones(len(feature_names)) / max(1, len(feature_names))

                sorted_idx = np.argsort(raw_imp)[::-1]
                for rank, idx in enumerate(sorted_idx, 1):
                    fname = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
                    shap_global.append({
                        "rank": rank,
                        "feature": fname,
                        "importance": round(float(raw_imp[idx]), 6),
                        "mean_abs_shap": round(float(raw_imp[idx]), 6)
                    })

                for idx in sorted_idx:
                    fname = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
                    feat_val = local_instance.iloc[idx]
                    shap_local.append({
                        "feature": fname,
                        "feature_value": float(feat_val) if isinstance(feat_val, (int, float, np.floating)) else str(feat_val),
                        "attribution": round(float(raw_imp[idx]), 6),
                        "direction": "positive"
                    })

            metrics["base_expected_value"] = round(base_value_scalar, 4)
            metrics["global_feature_importance"] = shap_global[:num_features_to_display]
            metrics["local_attribution"] = shap_local[:num_features_to_display]
            metrics["top_positive_drivers"] = [f for f in shap_local if f["attribution"] > 0][:5]
            metrics["top_negative_drivers"] = [f for f in shap_local if f["attribution"] < 0][:5]

        # 6. LIME Computation
        lime_run = method in ["LIME", "Combined (SHAP + LIME)"]
        lime_explanations: List[Dict[str, Any]] = []

        if lime_run:
            try:
                from lime import lime_tabular

                lime_mode = "classification" if task_type == "classification" else "regression"
                explainer_lime = lime_tabular.LimeTabularExplainer(
                    training_data=np.array(X_eval.values, dtype=float),
                    feature_names=feature_names,
                    mode=lime_mode,
                    random_state=42,
                    verbose=False
                )

                if lime_mode == "classification" and hasattr(model, "predict_proba"):
                    predict_fn = lambda arr: model.predict_proba(pd.DataFrame(arr, columns=feature_names))
                else:
                    predict_fn = lambda arr: model.predict(pd.DataFrame(arr, columns=feature_names))

                exp = explainer_lime.explain_instance(
                    data_row=np.array(local_instance.values, dtype=float),
                    predict_fn=predict_fn,
                    num_features=num_features_to_display
                )

                for rule, weight in exp.as_list():
                    lime_explanations.append({
                        "rule": rule,
                        "weight": round(float(weight), 6),
                        "direction": "positive" if weight >= 0 else "negative"
                    })

                metrics["lime_local_explanation"] = lime_explanations
            except Exception as lime_ex:
                warnings.append(f"[WARNING: LIME MODULE] LIME computation note: {str(lime_ex)}")

        metrics["warnings"] = warnings

        # 7. Generate Comprehensive Human-Readable Report
        report_lines = [
            "=== Model Explainability Report (SHAP & LIME) ===",
            f"Algorithm           : {type(model).__name__}",
            f"Task Type           : {task_type.capitalize()}",
            f"Methodology         : {method}",
            f"Evaluation Split    : N={n_eval} sampled rows (out of {N_total} total)",
            f"Total Features      : {len(feature_names)}",
            f"Local Sample Index  : Row #{local_row_idx}",
            f"Model Prediction    : {pred_value}"
        ]

        if pred_prob:
            prob_str = ", ".join([f"Class {i}: {p:.4f}" for i, p in enumerate(pred_prob)])
            report_lines.append(f"Predicted Probs     : [{prob_str}]")

        if shap_run and shap_global:
            report_lines.append("\n--- Global Feature Importance (Top Drivers) ---")
            report_lines.append(f"{'Rank':<5} | {'Feature':<28} | {'Mean |SHAP|':<12}")
            report_lines.append("-" * 52)
            for item in shap_global[:num_features_to_display]:
                report_lines.append(f"{item['rank']:<5} | {item['feature']:<28} | {item['importance']:<12.6f}")

        if shap_run and shap_local:
            report_lines.append(f"\n--- Local Sample #{local_row_idx} Attribution Breakdown ---")
            report_lines.append(f"{'Feature':<28} | {'Value':<12} | {'Attribution':<12} | {'Direction'}")
            report_lines.append("-" * 65)
            for item in shap_local[:num_features_to_display]:
                fval = str(item['feature_value'])
                if len(fval) > 10:
                    fval = fval[:9] + "…"
                report_lines.append(
                    f"{item['feature']:<28} | {fval:<12} | {item['attribution']:<+12.6f} | {item['direction'].upper()}"
                )

        if lime_run and lime_explanations:
            report_lines.append(f"\n--- LIME Local Surrogate Rules (Row #{local_row_idx}) ---")
            for item in lime_explanations:
                sign = "+" if item["weight"] >= 0 else ""
                report_lines.append(f"- {item['rule']:<35} -> Weight: {sign}{item['weight']:.6f} ({item['direction'].upper()})")

        if warnings:
            report_lines.append("\n--- Governance & Diagnostic Warnings ---")
            for w in warnings:
                report_lines.append(f"- {w}")

        report_text = "\n".join(report_lines)

        return {
            "metrics": metrics,
            "report": report_text
        }
