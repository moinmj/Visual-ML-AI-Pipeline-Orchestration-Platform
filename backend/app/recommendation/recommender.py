import pandas as pd
from typing import Dict, Any, List
from backend.app.profiling.profiler import DataProfiler


class AIRecommender:
    """
    Analyzes dataset profile characteristics to detect problem types
    and rank optimal preprocessing recipes and ML models (Section 8 of spec).
    """

    @classmethod
    def recommend_pipeline(cls, df: pd.DataFrame) -> Dict[str, Any]:
        profile = DataProfiler.profile_dataframe(df)
        columns = profile.get("columns", {})
        row_count = profile.get("row_count", 0)
        missing_cells = profile.get("total_missing_cells", 0)

        # 1. Detect Problem Paradigm
        date_cols = [c for c, m in columns.items() if m.get("inferred_type") == "datetime"]
        cat_cols = [c for c, m in columns.items() if m.get("inferred_type") == "categorical"]
        num_cols = [c for c, m in columns.items() if m.get("inferred_type") == "numeric"]

        task_type = "classification"
        explanation = "Detected tabular classification based on discrete target features."
        suggested_target = None

        if date_cols and len(num_cols) >= 1:
            task_type = "time_series_forecasting"
            suggested_target = num_cols[-1]
            explanation = f"Detected chronological time-series with date column `{date_cols[0]}` and numeric metric `{suggested_target}`."
        else:
            # Check last column
            last_col = list(columns.keys())[-1]
            last_meta = columns[last_col]
            suggested_target = last_col

            if last_meta.get("unique_count", 0) <= 15:
                task_type = "classification"
                explanation = f"Detected classification with discrete target column `{last_col}` ({last_meta.get('unique_count')} classes)."
            elif last_meta.get("inferred_type") == "numeric":
                task_type = "regression"
                explanation = f"Detected continuous regression with numerical target `{last_col}`."
            else:
                task_type = "anomaly_detection"
                explanation = "Unlabeled or continuous feature space suitable for unsupervised anomaly isolation."

        # 2. Recommended Preprocessing Chain
        cleaning_steps = []
        if missing_cells > 0:
            cleaning_steps.append({
                "recipe_id": "missing_value_imputer",
                "name": "🧹 Missing Value Imputer",
                "config": {"strategy": "median"},
                "reason": f"Dataset contains {missing_cells} missing cells requiring imputation."
            })

        if cat_cols:
            cleaning_steps.append({
                "recipe_id": "categorical_encoder",
                "name": "🔤 Categorical One-Hot Encoder",
                "config": {"method": "one_hot"},
                "reason": f"Found {len(cat_cols)} categorical columns ({', '.join(cat_cols[:3])}) requiring numerical encoding."
            })

        if num_cols:
            cleaning_steps.append({
                "recipe_id": "feature_scaler",
                "name": "⚖️ Feature Scaler",
                "config": {"method": "standard"},
                "reason": "Standardizing variance across numerical features for model stability."
            })

        # 3. Recommended Algorithm Rankings
        recommended_models = []
        if task_type in ["classification", "regression"]:
            recommended_models.append({
                "recipe_id": "xgboost_trainer",
                "name": "⚡ XGBoost",
                "score": 9.8,
                "tier": "Tier-1 Gold Standard",
                "reason": "Highest accuracy and regularized gradient boosting for tabular datasets."
            })
            recommended_models.append({
                "recipe_id": "lightgbm_trainer",
                "name": "🚀 LightGBM",
                "score": 9.5,
                "tier": "Tier-1 High Speed",
                "reason": "Optimal for ultra-fast training with histogram-based leaf growth."
            })
            if cat_cols:
                recommended_models.append({
                    "recipe_id": "catboost_trainer",
                    "name": "🐱 CatBoost",
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
                    "reason": "Robust out-of-the-box non-linear bagging baseline."
                })

        elif task_type == "time_series_forecasting":
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

        return {
            "task_type": task_type,
            "explanation": explanation,
            "target_column": suggested_target,
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
