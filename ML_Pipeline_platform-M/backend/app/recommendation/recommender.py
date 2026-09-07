import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from backend.app.profiling.profiler import DataProfiler


class AIRecommender:
    """
    Analyzes dataset profile characteristics and user intent to detect problem types
    and rank optimal preprocessing recipes and ML models (Section 8 of spec).
    """

    @classmethod
    def recommend_pipeline(
        cls,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        task_type: Optional[str] = None
    ) -> Dict[str, Any]:
        profile = DataProfiler.profile_dataframe(df)
        columns = profile.get("columns", {})
        row_count = profile.get("row_count", 0)
        missing_cells = profile.get("total_missing_cells", 0)

        date_cols = [c for c, m in columns.items() if m.get("inferred_type") == "datetime"]
        cat_cols = [c for c, m in columns.items() if m.get("inferred_type") == "categorical"]
        num_cols = [c for c, m in columns.items() if m.get("inferred_type") == "numeric"]

        # 1. Determine Target Column
        if target_column and target_column in df.columns:
            selected_target = target_column
        else:
            # Pick logical target (non-date, prefer last column or columns named target/churn/sales/price)
            candidates = [c for c in df.columns if c not in date_cols]
            named_candidates = [c for c in candidates if any(k in c.lower() for k in ["target", "churn", "survived", "label", "price", "sales", "revenue"])]
            selected_target = named_candidates[0] if named_candidates else (candidates[-1] if candidates else list(df.columns)[-1])

        # 2. Determine / Infer Task Type
        if task_type in ["classification", "regression", "time_series_forecasting", "anomaly_detection"]:
            detected_task = task_type
            explanation = f"User specified problem task as **{task_type.replace('_', ' ').title()}** on target `{selected_target}`."
        else:
            if date_cols and len(num_cols) >= 1 and (selected_target in num_cols):
                detected_task = "time_series_forecasting"
                explanation = f"Detected chronological time-series with timestamp `{date_cols[0]}` and numeric metric `{selected_target}`."
            else:
                target_meta = columns.get(selected_target, {})
                inferred = target_meta.get("inferred_type")
                nunique = target_meta.get("unique_count", df[selected_target].nunique() if selected_target in df.columns else 0)
                series = df[selected_target].dropna() if selected_target in df.columns else pd.Series()

                if inferred in ["categorical", "text", "boolean"] or nunique == 2:
                    detected_task = "classification"
                    explanation = f"Detected discrete classification on `{selected_target}` ({nunique} unique classes)."
                elif inferred == "numeric":
                    # Check if integer classification vs continuous regression
                    is_float_continuous = any(series % 1 != 0) if not series.empty and pd.api.types.is_numeric_dtype(series) else False
                    if is_float_continuous or nunique > 20:
                        detected_task = "regression"
                        explanation = f"Detected continuous numerical regression on target `{selected_target}`."
                    elif nunique <= 10 and nunique < (row_count * 0.1):
                        detected_task = "classification"
                        explanation = f"Detected multi-class classification on discrete target `{selected_target}` ({nunique} distinct categories)."
                    else:
                        detected_task = "regression"
                        explanation = f"Detected numeric regression on target `{selected_target}`."
                else:
                    detected_task = "anomaly_detection"
                    explanation = "Unlabeled or high-dimensional continuous feature space suitable for anomaly detection."

        # 3. Recommended Preprocessing Chain
        cleaning_steps = []
        if missing_cells > 0:
            cleaning_steps.append({
                "recipe_id": "missing_value_imputer",
                "name": "🧹 Missing Value Imputer",
                "recipe_name": "🧹 Missing Value Imputer",
                "config": {"strategy": "median"},
                "reason": f"Dataset contains {missing_cells} missing cells requiring imputation."
            })

        # Exclude target from encoding/scaling lists
        feature_cats = [c for c in cat_cols if c != selected_target]
        feature_nums = [c for c in num_cols if c != selected_target]

        if feature_cats:
            cleaning_steps.append({
                "recipe_id": "categorical_encoder",
                "name": "🔤 Categorical One-Hot Encoder",
                "recipe_name": "🔤 Categorical One-Hot Encoder",
                "config": {"method": "one_hot"},
                "reason": f"Found {len(feature_cats)} categorical features ({', '.join(feature_cats[:3])}) requiring numerical encoding."
            })

        if feature_nums:
            cleaning_steps.append({
                "recipe_id": "feature_scaler",
                "name": "⚖️ Feature Scaler",
                "recipe_name": "⚖️ Feature Scaler",
                "config": {"method": "standard"},
                "reason": "Standardizing variance across numerical features for model stability."
            })

        # 4. Recommended Algorithm Rankings
        recommended_models = []
        if detected_task in ["classification", "regression"]:
            recommended_models.append({
                "recipe_id": "xgboost_trainer",
                "name": f"⚡ XGBoost {detected_task.title()}",
                "score": 9.8,
                "tier": "Tier-1 Gold Standard",
                "reason": "Highest accuracy and regularized gradient boosting for tabular datasets."
            })
            recommended_models.append({
                "recipe_id": "lightgbm_trainer",
                "name": f"🚀 LightGBM {detected_task.title()}",
                "score": 9.5,
                "tier": "Tier-1 High Speed",
                "reason": "Optimal for ultra-fast training with histogram-based leaf growth."
            })
            if feature_cats:
                recommended_models.append({
                    "recipe_id": "catboost_trainer",
                    "name": f"🐱 CatBoost {detected_task.title()}",
                    "score": 9.4,
                    "tier": "Tier-1 Categorical",
                    "reason": "Native handling of high-cardinality categorical features without one-hot expansion."
                })
            else:
                recommended_models.append({
                    "recipe_id": "random_forest_trainer",
                    "name": "🌲 Random Forest",
                    "score": 8.5,
                    "tier": "Tier-1 Ensemble",
                    "reason": "Robust non-linear bagging baseline."
                })

        elif detected_task == "time_series_forecasting":
            recommended_models.append({
                "recipe_id": "prophet_forecaster",
                "name": "🔮 Meta Prophet",
                "score": 9.5,
                "tier": "Tier-1 Business Standard",
                "reason": "Decomposes trend, weekly/yearly seasonality, and handles irregular intervals with prediction bands."
            })
            recommended_models.append({
                "recipe_id": "arima_forecaster",
                "name": "📊 ARIMA / SARIMAX",
                "score": 8.5,
                "tier": "Tier-1 Statistical",
                "reason": "Rigorous classical statistical baseline with lag and error differencing."
            })

        else: # Anomaly Detection
            recommended_models.append({
                "recipe_id": "isolation_forest",
                "name": "🌲 Isolation Forest",
                "score": 9.6,
                "tier": "Tier-1 Outlier Standard",
                "reason": "Linear-time unsupervised isolation partitioning that scales to high dimensions."
            })
            recommended_models.append({
                "recipe_id": "statistical_guardrail",
                "name": "🛡️ Statistical Guardrail",
                "score": 8.0,
                "tier": "Tier-1 ELT Filter",
                "reason": "Z-Score / IQR standard-deviation thresholding for data quality filtering."
            })

        rec_result = {
            "task_type": detected_task,
            "explanation": explanation,
            "target_column": selected_target,
            "profile_summary": {
                "rows": row_count,
                "columns": len(columns),
                "missing_cells": missing_cells,
                "categorical_columns": len(cat_cols),
                "numeric_columns": len(num_cols),
                "date_columns": len(date_cols)
            },
            "preprocessing_recommendations": cleaning_steps,
            "model_rankings": recommended_models
        }

        rec_result["recommended_dag"] = cls.build_recommended_dag(
            rec_result,
            df=df,
            target_column=selected_target,
            date_column=date_cols[0] if date_cols else None
        )
        return rec_result

    @classmethod
    def build_recommended_dag(
        cls,
        recommendation: Dict[str, Any],
        df: Optional[pd.DataFrame] = None,
        target_column: Optional[str] = None,
        date_column: Optional[str] = None,
        dataset_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Synthesizes a visual, ready-to-render DAG (nodes, edges, layout coordinates,
        and node configs) from an AI recommendation analysis.
        """
        task = recommendation.get("task_type", "classification")
        target_col = target_column or recommendation.get("target_column")
        nodes = []
        edges = []
        node_configs = {}

        # 1. Ingestion Node
        csv_config = {}
        if dataset_id:
            csv_config["dataset_id"] = dataset_id

        nodes.append({
            "id": "node_csv",
            "recipe_id": "csv_loader",
            "label": "📄 Data Ingestion",
            "position": {"x": 40, "y": 100},
            "config": csv_config
        })
        node_configs["node_csv"] = {
            "recipe_id": "csv_loader",
            "label": "Dataset Ingestion",
            "config": csv_config
        }

        prev_node_id = "node_csv"
        cur_x = 280

        if task == "time_series_forecasting":
            # Time-Series Pipeline
            # 2. Imputer
            nodes.append({
                "id": "node_impute",
                "recipe_id": "missing_value_imputer",
                "label": "🧹 Time Imputer (ffill)",
                "position": {"x": cur_x, "y": 100},
                "config": {"strategy": "ffill"}
            })
            node_configs["node_impute"] = {
                "recipe_id": "missing_value_imputer",
                "label": "Imputer",
                "config": {"strategy": "ffill"}
            }
            edges.append({"id": "e_csv_impute", "source": "node_csv", "target": "node_impute", "animated": True})
            prev_node_id = "node_impute"
            cur_x += 280

            # 3. Prophet Forecaster
            p_config = {
                "date_column": date_column or "Date",
                "target_column": target_col or "Value",
                "horizon_periods": 30
            }
            nodes.append({
                "id": "node_prophet",
                "recipe_id": "prophet_forecaster",
                "label": "🔮 Prophet Forecaster",
                "position": {"x": cur_x, "y": 100},
                "config": p_config
            })
            node_configs["node_prophet"] = {
                "recipe_id": "prophet_forecaster",
                "label": "Prophet Forecaster",
                "config": p_config
            }
            edges.append({"id": "e_impute_prophet", "source": prev_node_id, "target": "node_prophet", "animated": True})

        elif task == "anomaly_detection":
            # Anomaly Detection Pipeline
            # 2. Imputer
            nodes.append({
                "id": "node_impute",
                "recipe_id": "missing_value_imputer",
                "label": "🧹 Imputer (Median)",
                "position": {"x": cur_x, "y": 100},
                "config": {"strategy": "median"}
            })
            node_configs["node_impute"] = {
                "recipe_id": "missing_value_imputer",
                "label": "Imputer",
                "config": {"strategy": "median"}
            }
            edges.append({"id": "e_csv_impute", "source": "node_csv", "target": "node_impute", "animated": True})
            prev_node_id = "node_impute"
            cur_x += 280

            # 3. Isolation Forest
            nodes.append({
                "id": "node_iso",
                "recipe_id": "isolation_forest",
                "label": "🌲 Isolation Forest",
                "position": {"x": cur_x, "y": 100},
                "config": {"contamination": 0.05, "n_estimators": 100}
            })
            node_configs["node_iso"] = {
                "recipe_id": "isolation_forest",
                "label": "Isolation Forest",
                "config": {"contamination": 0.05, "n_estimators": 100}
            }
            edges.append({"id": "e_impute_iso", "source": prev_node_id, "target": "node_iso", "animated": True})

        else:
            # Classification / Regression Pipeline
            # 2. Add recommended preprocessing steps
            pre_steps = recommendation.get("preprocessing_recommendations", [])
            for idx, step in enumerate(pre_steps):
                step_id = f"node_prep_{idx+1}"
                r_id = step["recipe_id"]
                step_label = step.get("name", r_id.replace("_", " ").title())
                step_cfg = step.get("config", {})

                nodes.append({
                    "id": step_id,
                    "recipe_id": r_id,
                    "label": step_label,
                    "position": {"x": cur_x, "y": 100},
                    "config": step_cfg
                })
                node_configs[step_id] = {
                    "recipe_id": r_id,
                    "label": step_label,
                    "config": step_cfg
                }
                edges.append({
                    "id": f"e_{prev_node_id}_{step_id}",
                    "source": prev_node_id,
                    "target": step_id,
                    "animated": True
                })
                prev_node_id = step_id
                cur_x += 240

            # 3. Train/Test Splitter
            split_id = "node_split"
            split_cfg = {"target_column": target_col or "target", "test_size": 0.2}
            nodes.append({
                "id": split_id,
                "recipe_id": "train_test_split",
                "label": "✂️ Train/Test Split",
                "position": {"x": cur_x, "y": 100},
                "config": split_cfg
            })
            node_configs[split_id] = {
                "recipe_id": "train_test_split",
                "label": "Splitter",
                "config": split_cfg
            }
            edges.append({
                "id": f"e_{prev_node_id}_{split_id}",
                "source": prev_node_id,
                "target": split_id,
                "animated": True
            })
            cur_x += 240

            # 4. Top Ranked Model Trainer
            top_model = recommendation.get("model_rankings", [{}])[0]
            model_recipe = top_model.get("recipe_id", "xgboost_trainer")
            model_name = top_model.get("name", "XGBoost Trainer")
            model_id = "node_model"
            model_cfg = {
                "task_type": "regression" if task == "regression" else "classification",
                "n_estimators": 100,
                "max_depth": 6
            }
            nodes.append({
                "id": model_id,
                "recipe_id": model_recipe,
                "label": f"⚡ {model_name}",
                "position": {"x": cur_x, "y": 50},
                "config": model_cfg
            })
            node_configs[model_id] = {
                "recipe_id": model_recipe,
                "label": model_name,
                "config": model_cfg
            }
            edges.append({
                "id": f"e_{split_id}_{model_id}",
                "source": split_id,
                "target": model_id,
                "animated": True
            })
            cur_x += 240

            # 5. Model Evaluator
            eval_id = "node_eval"
            eval_recipe = "regression_evaluator" if task == "regression" else "classification_evaluator"
            nodes.append({
                "id": eval_id,
                "recipe_id": eval_recipe,
                "label": "🎯 Model Evaluator",
                "position": {"x": cur_x, "y": 100},
                "config": {"report_type": "Comprehensive"}
            })
            node_configs[eval_id] = {
                "recipe_id": eval_recipe,
                "label": "Evaluator",
                "config": {"report_type": "Comprehensive"}
            }
            edges.append({
                "id": f"e_{split_id}_{eval_id}",
                "source": split_id,
                "target": eval_id,
                "animated": True
            })
            edges.append({
                "id": f"e_{model_id}_{eval_id}",
                "source": model_id,
                "target": eval_id,
                "animated": True
            })

        return {
            "nodes": nodes,
            "edges": edges,
            "node_configs": node_configs
        }
