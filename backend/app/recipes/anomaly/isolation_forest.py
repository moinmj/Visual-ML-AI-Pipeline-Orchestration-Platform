import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from sklearn.ensemble import IsolationForest
from backend.app.recipes.base.recipe import BaseRecipe


class IsolationForestRecipe(BaseRecipe):
    recipe_id = "isolation_forest"
    name = "Isolation Forest Anomaly Detector"
    version = "1.0.0"
    category = "anomaly"
    description = "Unsupervised tree-based anomaly detection that isolates outliers based on partition path length (Tier-1 Enterprise Standard)."
    input_types = ["dataframe"]
    output_types = ["dataframe", "model", "metrics"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "contamination": {
                    "type": "number",
                    "title": "Expected Outlier Ratio (Contamination)",
                    "default": 0.05,
                    "minimum": 0.01,
                    "maximum": 0.5,
                    "description": "The proportion of outliers in the data set (e.g. 0.05 = 5% anomalies)."
                },
                "n_estimators": {
                    "type": "integer",
                    "title": "Number of Isolation Trees",
                    "default": 100,
                    "minimum": 10,
                    "maximum": 500
                },
                "random_state": {
                    "type": "integer",
                    "title": "Random Seed",
                    "default": 42
                }
            }
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        df: pd.DataFrame = inputs.get("dataframe")
        if df is None:
            if context and isinstance(context, dict) and "dataframe" in context:
                df = context["dataframe"]
            else:
                raise ValueError("IsolationForest expects 'dataframe' in inputs.")

        df_out = df.copy()
        
        # Edge Case 1: Handle string and categorical columns via auto one-hot encoding
        numeric_df = df_out.copy()
        non_numeric = [c for c in numeric_df.columns if not pd.api.types.is_numeric_dtype(numeric_df[c])]
        if non_numeric:
            numeric_df = pd.get_dummies(numeric_df, columns=non_numeric, drop_first=True, dtype=float)

        # Edge Case 2: Handle missing values (NaNs) via median imputation so IsolationForest doesn't crash
        numeric_df = numeric_df.fillna(numeric_df.median(numeric_only=True)).fillna(0)

        contamination = float(config.get("contamination", 0.05))
        n_estimators = int(config.get("n_estimators", 100))
        random_state = int(config.get("random_state", 42))

        # Train Isolation Forest
        iso_forest = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )

        # Predictions: -1 for outlier, 1 for inlier
        raw_preds = iso_forest.fit_predict(numeric_df)
        
        # Continuous Anomaly Score (lower/negative score = more anomalous)
        raw_scores = iso_forest.decision_function(numeric_df)
        
        # Normalize score to 0.0 - 1.0 (where 1.0 is most anomalous)
        score_min, score_max = raw_scores.min(), raw_scores.max()
        if score_max > score_min:
            normalized_scores = 1.0 - ((raw_scores - score_min) / (score_max - score_min))
        else:
            normalized_scores = np.zeros(len(raw_scores))

        # Add flags and scores to output DataFrame
        df_out["is_anomaly"] = np.where(raw_preds == -1, 1, 0)
        df_out["anomaly_score"] = np.round(normalized_scores, 4)

        total_records = len(df_out)
        anomaly_count = int(df_out["is_anomaly"].sum())
        anomaly_pct = float(round((anomaly_count / total_records) * 100, 2)) if total_records > 0 else 0.0

        metrics = {
            "task_type": "anomaly_detection",
            "algorithm": "Isolation Forest",
            "total_records": total_records,
            "anomaly_count": anomaly_count,
            "normal_count": total_records - anomaly_count,
            "anomaly_percentage": anomaly_pct,
            "contamination_threshold": contamination,
            "top_anomalies_indices": df_out.sort_values(by="anomaly_score", ascending=False).head(10).index.tolist()
        }

        return {
            "dataframe": df_out,
            "model": iso_forest,
            "metrics": metrics,
            "anomaly_summary": metrics
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        contam = config.get("contamination", 0.05)
        n_est = config.get("n_estimators", 100)
        return f"from sklearn.ensemble import IsolationForest\n\niso = IsolationForest(contamination={contam}, n_estimators={n_est}, random_state=42)\nis_anomaly = (iso.fit_predict(df) == -1).astype(int)"
