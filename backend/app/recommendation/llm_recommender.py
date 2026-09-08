import json
import logging
from typing import Dict, Any, Optional, List
import httpx
import pandas as pd

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.recipes.base.registry import recipe_registry
from backend.app.recommendation.recommender import AIRecommender


SYSTEM_PROMPT = """You are an expert AI Machine Learning Solutions Architect.
Your role is to analyze a user's natural language goal and dataset schema, and generate an optimal, production-ready visual ML DAG (Directed Acyclic Graph) workflow.

AVAILABLE RECIPES IN THE PLATFORM (You MUST ONLY use these exact recipe_id values):
- Ingestion:
  * "csv_loader": Ingests tabular dataset. config: {"dataset_id": "..."}
- Preprocessing:
  * "missing_value_imputer": Handles NaNs. config: {"strategy": "median"|"mean"|"most_frequent"|"constant"|"ffill"|"bfill"}
  * "categorical_encoder": Encodes text/categorical features. config: {"method": "one_hot"|"label"}
  * "feature_scaler": Normalizes numeric features. config: {"method": "standard"|"minmax"|"robust"}
  * "text_preprocessor": Cleans unstructured text. config: {"remove_stopwords": true, "lowercase": true}
  * "text_vectorizer": Vectorizes text. config: {"method": "tfidf"|"count"}
- Splitting:
  * "train_test_split": Splits train/test sets. config: {"target_column": "<col_name>", "test_size": 0.2}
- Model Training (Supervised):
  * "xgboost_trainer": config: {"task_type": "classification"|"regression", "n_estimators": 100, "max_depth": 6}
  * "lightgbm_trainer": config: {"task_type": "classification"|"regression", "n_estimators": 100, "max_depth": 6}
  * "catboost_trainer": config: {"task_type": "classification"|"regression", "iterations": 100}
  * "random_forest_trainer": config: {"task_type": "classification"|"regression", "n_estimators": 100, "max_depth": 6}
  * "logistic_regression_trainer": config: {"C": 1.0, "penalty": "l2"}
- Time Series Forecasting:
  * "prophet_forecaster": config: {"date_column": "<date_col>", "target_column": "<target_col>", "horizon_periods": 30}
  * "arima_forecaster": config: {"date_column": "<date_col>", "target_column": "<target_col>", "horizon_periods": 30}
- Anomaly Detection:
  * "isolation_forest": config: {"contamination": 0.05, "n_estimators": 100}
- Evaluation (CRITICAL: Always use "model_evaluator" for all supervised model evaluation):
  * "model_evaluator": Evaluates classification, regression, and ranking. config: {"report_type": "Comprehensive"}
- Governance & Tracking:
  * "model_governance_card": Generates model card. config: {"author": "AI Architect", "version": "1.0.0"}
  * "mlflow_tracker": Logs run metrics. config: {"experiment_name": "Pipeline Run"}

RULES FOR DAG CONSTRUCTION:
1. Always start with "csv_loader" (id: "node_csv", position x=40, y=100).
2. For Supervised tasks (classification/regression):
   - Ingestion -> Imputer (if nulls) -> Categorical Encoder (if text/cats) -> Scaler (if requested/numeric) -> Train/Test Split -> Model Trainer.
   - Both Train/Test Split (node_split) AND Model Trainer (node_model) MUST connect to Model Evaluator (node_eval).
   - "model_evaluator" receives X_test, y_test from "node_split", and trained model from "node_model".
3. Layout coordinates: space nodes along x-axis with delta x ≈ 240px.
4. Output MUST be valid JSON with this exact structure:
{
  "task_type": "classification" | "regression" | "time_series_forecasting" | "anomaly_detection",
  "target_column": "string",
  "explanation": "Concise architectural explanation of why this pipeline was chosen for the user's objective",
  "recommended_dag": {
    "nodes": [
      {"id": "node_csv", "recipe_id": "csv_loader", "label": "📄 Data Ingestion", "position": {"x": 40, "y": 100}, "config": {}},
      ...
    ],
    "edges": [
      {"id": "e1", "source": "node_csv", "target": "node_impute", "animated": true},
      ...
    ]
  }
}
"""


class LLMRecommender:
    """
    Groq LLM-Powered Machine Learning Pipeline Architect.
    Interprets natural language queries, analyzes tabular dataset schemas,
    and synthesizes optimized, executable visual DAG pipelines.
    """

    @classmethod
    async def recommend_pipeline_async(
        cls,
        df: pd.DataFrame,
        query: Optional[str] = None,
        target_column: Optional[str] = None,
        time_column: Optional[str] = None,
        task_type: Optional[str] = None,
        preset: str = "balanced",
        dataset_name: str = "Dataset"
    ) -> Dict[str, Any]:
        """
        Synthesize visual ML DAG using Groq LLM if API key is available,
        with seamless fallback to deterministic heuristic recommender.
        """
        api_key = settings.GROQ_API_KEY
        if not api_key:
            logger.info("GROQ_API_KEY not configured. Falling back to heuristic AIRecommender.")
            return AIRecommender.recommend_pipeline(df, target_column=target_column, task_type=task_type)

        try:
            # Construct dataset summary
            columns_info = []
            for col in df.columns:
                dtype = str(df[col].dtype)
                nulls = int(df[col].isnull().sum())
                unique = int(df[col].nunique())
                sample_vals = df[col].dropna().head(3).tolist()
                columns_info.append(f"- {col} (dtype: {dtype}, nulls: {nulls}, unique: {unique}, samples: {sample_vals})")

            dataset_summary = (
                f"Dataset Name: {dataset_name}\n"
                f"Rows: {len(df)}, Columns: {len(df.columns)}\n"
                f"Columns:\n" + "\n".join(columns_info)
            )

            user_prompt = (
                f"USER QUERY / OBJECTIVE:\n{query or 'Build the best end-to-end machine learning pipeline for this dataset.'}\n\n"
                f"DATASET SCHEMA & PROFILE:\n{dataset_summary}\n\n"
                f"CONSTRAINTS & HINTS:\n"
                f"- Target column override: {target_column or 'Auto-detect from query/schema'}\n"
                f"- Task type override: {task_type or 'Auto-detect'}\n"
                f"- Optimization preset: {preset}\n\n"
                f"Synthesize the visual ML DAG and return valid JSON only."
            )

            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": settings.GROQ_MODEL,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt}
                        ],
                        "response_format": {"type": "json_object"},
                        "temperature": 0.2
                    }
                )

            if response.status_code != 200:
                logger.warning(f"Groq API returned status {response.status_code}: {response.text}. Using fallback.")
                return AIRecommender.recommend_pipeline(df, target_column=target_column, task_type=task_type)

            resp_json = response.json()
            content = resp_json["choices"][0]["message"]["content"]
            result = json.loads(content)

            # Validate and format result
            return cls._validate_and_enrich_dag(result, df)

        except Exception as e:
            logger.error(f"Error during LLM pipeline recommendation: {str(e)}. Falling back to heuristic recommender.", exc_info=True)
            return AIRecommender.recommend_pipeline(df, target_column=target_column, task_type=task_type)

    @classmethod
    def _validate_and_enrich_dag(cls, result: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validates node recipe IDs against recipe_registry, fixes coordinates,
        and constructs node_configs map.
        """
        dag = result.get("recommended_dag", {})
        nodes = dag.get("nodes", [])
        edges = dag.get("edges", [])

        # Validate each recipe_id against the registry
        valid_nodes = []
        node_configs = {}

        for n in nodes:
            r_id = n.get("recipe_id", "")
            # If alias or known typo, resolve it
            if r_id in ["classification_evaluator", "regression_evaluator"]:
                r_id = "model_evaluator"
                n["recipe_id"] = r_id

            if not recipe_registry.has(r_id):
                # Fallback to closest or imputer/scaler
                logger.warning(f"Unknown recipe '{r_id}' recommended by LLM. Substituting fallback.")
                if "eval" in r_id:
                    r_id = "model_evaluator"
                elif "imput" in r_id:
                    r_id = "missing_value_imputer"
                elif "scale" in r_id:
                    r_id = "feature_scaler"
                elif "split" in r_id:
                    r_id = "train_test_split"
                else:
                    r_id = "csv_loader"
                n["recipe_id"] = r_id

            node_id = n.get("id") or f"node_{len(valid_nodes)}"
            n["id"] = node_id
            if "position" not in n:
                n["position"] = {"x": 40 + len(valid_nodes) * 240, "y": 100}

            valid_nodes.append(n)
            node_configs[node_id] = {
                "recipe_id": r_id,
                "label": n.get("label", r_id),
                "config": n.get("config", {})
            }

        dag["nodes"] = valid_nodes
        dag["edges"] = edges
        dag["node_configs"] = node_configs

        return {
            "task_type": result.get("task_type", "classification"),
            "target_column": result.get("target_column"),
            "explanation": result.get("explanation", "Custom AI-architected pipeline generated by Groq LLM."),
            "preprocessing_recommendations": result.get("preprocessing_recommendations", []),
            "model_rankings": result.get("model_rankings", []),
            "recommended_dag": dag,
            "llm_generated": True
        }
