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
  * "column_selector": Keeps or drops selected columns/IDs. config: {"columns": ["col1"], "mode": "include"|"exclude"}
  * "data_type_converter": Casts feature types safely (numeric, datetime, category, string). config: {"conversions": {"col_a": "datetime"}}
  * "missing_value_imputer": Handles NaNs. config: {"strategy": "median"|"mean"|"most_frequent"|"constant"|"ffill"|"bfill"}
  * "outlier_handler": Detects and caps or removes extreme values. config: {"method": "iqr"|"zscore", "action": "clip"|"remove", "threshold": 1.5}
  * "categorical_encoder": Encodes text/categorical features. config: {"method": "one_hot"|"label"}
  * "feature_scaler": Normalizes numeric features. config: {"method": "standard"|"minmax"|"robust"}
  * "feature_selector": Selects top-k most predictive features. config: {"method": "kbest"|"variance"|"correlation", "k": 10}
  * "text_preprocessor": Cleans unstructured text. config: {"remove_stopwords": true, "lowercase": true}
  * "text_vectorizer": Vectorizes text. config: {"method": "tfidf"|"count"}
- Splitting:
  * "train_test_split": Standard random train/test split. config: {"target_column": "<col_name>", "test_size": 0.2, "time_series_mode": false}
  * "stratified_split": Preserves exact class ratio for classification datasets (best for imbalanced targets). config: {"target_column": "<col_name>", "test_size": 0.2}
  * "time_series_split": Strict chronological split for temporal datasets to prevent future data leakage. config: {"target_column": "<col_name>", "date_column": "<date_col>", "test_size": 0.2, "gap": 0}
  * "walk_forward_split": Sliding/expanding rolling window split for financial & time-series backtesting. config: {"target_column": "<col_name>", "train_window_size": 100, "test_window_size": 20, "expanding": false}
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
2. For Classification, strongly prefer "stratified_split" to preserve class distribution across train/test sets.
3. For Time Series or datasets with temporal date/year features:
   - Prefer "time_series_split" or "walk_forward_split" for regression/GBDT models to prevent future data leakage.
   - Or connect directly to "prophet_forecaster" / "arima_forecaster" if forecasting is requested.
4. For Supervised tasks (classification/regression):
   - Ingestion -> Preprocessing -> Splitter -> Model Trainer.
   - Both the Splitter node (node_split) AND the Model Trainer node (node_model) MUST connect to Model Evaluator (node_eval).
   - "model_evaluator" receives X_test, y_test from "node_split", and trained model from "node_model".
4. Layout coordinates: space nodes along x-axis with delta x ≈ 240px.
5. Output MUST be valid JSON with this exact structure:
{
  "task_type": "classification" | "regression" | "time_series_forecasting" | "anomaly_detection",
  "target_column": "string",
  "explanation": "Concise architectural explanation of why this pipeline was chosen for the user's objective",
  "recommended_dag": {
    "nodes": [
      {"id": "node_csv", "recipe_id": "csv_loader", "label": "Data Ingestion", "position": {"x": 40, "y": 100}, "config": {}},
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

    @staticmethod
    def _heuristic_fallback(df: pd.DataFrame, target_column: Optional[str], task_type: Optional[str], reason: str) -> Dict[str, Any]:
        """
        Routes to the deterministic heuristic recommender and stamps the result with
        WHY the LLM path wasn't used, so callers (and the UI) never have to guess whether
        a recommendation was genuinely reasoned by the LLM or produced by the rule-based
        fallback. `_heuristic_recommend_pipeline` already sets llm_generated=False /
        recommendation_source="heuristic_fallback"; this only adds the specific reason.
        """
        result = AIRecommender._heuristic_recommend_pipeline(df, target_column=target_column, task_type=task_type)
        result["fallback_reason"] = reason
        return result

    @classmethod
    def recommend_pipeline(
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
        Synchronous wrapper for LLM pipeline recommendation.
        """
        import asyncio
        import concurrent.futures

        def _run_sync():
            return asyncio.run(
                cls.recommend_pipeline_async(
                    df=df, query=query, target_column=target_column,
                    time_column=time_column, task_type=task_type,
                    preset=preset, dataset_name=dataset_name
                )
            )

        try:
            return asyncio.run(
                cls.recommend_pipeline_async(
                    df=df, query=query, target_column=target_column,
                    time_column=time_column, task_type=task_type,
                    preset=preset, dataset_name=dataset_name
                )
            )
        except Exception:
            try:
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(_run_sync)
                    return future.result()
            except Exception as e:
                logger.warning(f"Sync wrapper for LLMRecommender failed: {str(e)}")
                return cls._heuristic_fallback(df, target_column, task_type, reason=f"Sync execution wrapper failed: {str(e)}")

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
        # Determine active LLM provider (Gemini, OpenAI, or Groq)
        groq_key = getattr(settings, "GROQ_API_KEY", None)
        gemini_key = getattr(settings, "GEMINI_API_KEY", None)
        openai_key = getattr(settings, "OPENAI_API_KEY", None)

        # If target_column is passed as a secondary trailing column (e.g. Unemployment) but the dataset
        # contains a primary metric like Weekly_Sales, prioritize the primary metric unless explicitly locked.
        primary_metrics = [c for c in df.columns if any(kw in c.lower() for kw in ["weekly_sales", "sales", "revenue", "demand", "price", "amount"])]
        if target_column and target_column.lower() in ["unemployment", "cpi", "fuel_price", "temperature", "store"] and primary_metrics:
            if not query or any(kw in query.lower() for kw in ["sale", "revenue", "demand", "weekly", "predict", "forecast"]):
                target_column = primary_metrics[0]

        if gemini_key:
            endpoint = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
            api_key = gemini_key
            models_to_try = ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"]
        elif openai_key:
            endpoint = "https://api.openai.com/v1/chat/completions"
            api_key = openai_key
            models_to_try = ["gpt-4o-mini", "gpt-4o"]
        elif groq_key:
            endpoint = "https://api.groq.com/openai/v1/chat/completions"
            api_key = groq_key
            models_to_try = [m for m in [settings.GROQ_MODEL, "openai/gpt-oss-20b", "openai/gpt-oss-120b", "groq/compound", "qwen/qwen3.8-27b"] if m]
        else:
            logger.info("No LLM API Key (GROQ, GEMINI, or OPENAI) configured. Falling back to heuristic AIRecommender.")
            return cls._heuristic_fallback(df, target_column, task_type, reason="No LLM API key configured (GROQ_API_KEY / GEMINI_API_KEY / OPENAI_API_KEY all unset).")

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

            # Deduplicate preserving order
            models_to_try = list(dict.fromkeys(models_to_try))

            response = None
            async with httpx.AsyncClient(timeout=30.0) as client:
                for model_id in models_to_try:
                    if not model_id:
                        continue
                    try:
                        response = await client.post(
                            endpoint,
                            headers={"Authorization": f"Bearer {api_key}"},
                            json={
                                "model": model_id,
                                "messages": [
                                    {"role": "system", "content": SYSTEM_PROMPT},
                                    {"role": "user", "content": user_prompt}
                                ],
                                "response_format": {"type": "json_object"},
                                "temperature": 0.0
                            }
                        )
                        if response.status_code == 200:
                            break
                        elif response.status_code == 429:
                            import asyncio as a_io
                            await a_io.sleep(1.0)
                        else:
                            logger.warning(f"LLM model {model_id} returned status {response.status_code}: {response.text[:200]}")
                    except Exception as e:
                        logger.warning(f"LLM model {model_id} via {endpoint} failed: {str(e)}")

            if not response or response.status_code != 200:
                status_note = str(response.status_code) if response else "no response (all models/timeouts exhausted)"
                logger.warning(f"LLM API returned status {status_note}. Using fallback.")
                return cls._heuristic_fallback(df, target_column, task_type, reason=f"LLM API call failed (HTTP status: {status_note}).")

            resp_json = response.json()
            content = resp_json["choices"][0]["message"]["content"]
            result = json.loads(content)

            # Validate and format result
            return cls._validate_and_enrich_dag(result, df)

        except Exception as e:
            logger.error(f"Error during LLM pipeline recommendation: {str(e)}. Falling back to heuristic recommender.", exc_info=True)
            return cls._heuristic_fallback(df, target_column, task_type, reason=f"LLM recommendation raised an exception: {str(e)}")

    @classmethod
    def _validate_and_enrich_dag(cls, result: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validates node recipe IDs against recipe_registry, fixes coordinates,
        and constructs node_configs map.
        """
        dag = result.get("recommended_dag", {})
        nodes = dag.get("nodes", [])
        edges = dag.get("edges", [])

        # Infer temporal column from DataFrame if available
        temporal_cols = [
            c for c in df.columns
            if any(kw in c.lower() for kw in ["date", "time", "year", "ds", "period", "month", "week"])
        ]
        detected_date_col = temporal_cols[0] if temporal_cols else None

        target_col = result.get("target_column") or target_column
        task_type = result.get("task_type") or task_type

        # For datasets with a clear date/time column, auto-enforce time_series_forecasting if task is ambiguous or classification
        if detected_date_col and (not task_type or task_type in ["classification", "auto", "Auto-Detect Task Type", ""]):
            task_type = "time_series_forecasting"
            result["task_type"] = task_type

        # Filter and validate each node
        valid_nodes = []
        node_configs = {}

        for n in nodes:
            if not isinstance(n, dict):
                continue

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
                elif "strat" in r_id:
                    r_id = "stratified_split"
                elif "time" in r_id and "split" in r_id:
                    r_id = "time_series_split"
                elif "walk" in r_id or "forward" in r_id:
                    r_id = "walk_forward_split"
                elif "split" in r_id:
                    r_id = "train_test_split"
                elif "outlier" in r_id:
                    r_id = "outlier_handler"
                elif "col" in r_id and "select" in r_id:
                    r_id = "column_selector"
                else:
                    r_id = "csv_loader"
                n["recipe_id"] = r_id

            node_id = n.get("id") or f"node_{len(valid_nodes)}"
            n["id"] = node_id
            if "position" not in n:
                n["position"] = {"x": 40 + len(valid_nodes) * 240, "y": 100}

            node_cfg = dict(n.get("config", {}))

            # Enrich specific recipe node configs with dataset context
            if r_id == "feature_scaler":
                if target_col:
                    node_cfg["target_column"] = target_col
                node_cfg["exclude_target"] = True
                if "method" not in node_cfg:
                    node_cfg["method"] = "standard"
                if detected_date_col and node_cfg.get("columns"):
                    if isinstance(node_cfg["columns"], list):
                        node_cfg["columns"] = [c for c in node_cfg["columns"] if c.lower().strip() != detected_date_col.lower().strip()]

            elif r_id == "data_type_converter":
                if not node_cfg.get("conversions") and detected_date_col:
                    node_cfg["conversions"] = {detected_date_col: "datetime"}
                elif isinstance(node_cfg.get("conversions"), str) and node_cfg["conversions"].strip() in ["[object Object]", "object Object", ""]:
                    if detected_date_col:
                        node_cfg["conversions"] = {detected_date_col: "datetime"}

            elif r_id in ["train_test_split", "stratified_split", "time_series_split", "walk_forward_split"]:
                if target_col and not node_cfg.get("target_column"):
                    node_cfg["target_column"] = target_col
                if r_id == "time_series_split" and detected_date_col and not node_cfg.get("date_column"):
                    node_cfg["date_column"] = detected_date_col
                if "test_size" not in node_cfg:
                    node_cfg["test_size"] = 0.2

            elif r_id in ["prophet_forecaster", "arima_forecaster"]:
                if target_col and not node_cfg.get("target_column"):
                    node_cfg["target_column"] = target_col
                if detected_date_col and not node_cfg.get("date_column"):
                    node_cfg["date_column"] = detected_date_col
                if "horizon_periods" not in node_cfg:
                    node_cfg["horizon_periods"] = 30

            elif r_id in ["xgboost_trainer", "lightgbm_trainer", "catboost_trainer", "random_forest_trainer"]:
                if "task_type" not in node_cfg:
                    node_cfg["task_type"] = task_type if task_type in ["classification", "regression"] else "regression"

            elif r_id == "model_evaluator":
                if "report_type" not in node_cfg:
                    node_cfg["report_type"] = "Comprehensive"

            n["config"] = node_cfg
            valid_nodes.append(n)
            node_configs[node_id] = {
                "recipe_id": r_id,
                "label": n.get("label", r_id),
                "config": node_cfg
            }

        # Filter edges to only include valid node connections
        valid_node_ids = {n["id"] for n in valid_nodes}
        valid_edges = []
        for e in edges:
            if isinstance(e, dict) and e.get("source") in valid_node_ids and e.get("target") in valid_node_ids:
                valid_edges.append(e)

        # AUTO-WIRE GUARANTEE: If edges are empty or incomplete, build complete topological edges connecting all nodes sequentially
        existing_targets = {e["target"] for e in valid_edges}
        needs_autowire = len(valid_edges) == 0 or (len(valid_nodes) >= 2 and any(valid_nodes[i]["id"] not in existing_targets for i in range(1, len(valid_nodes))))

        if needs_autowire and len(valid_nodes) >= 2:
            auto_edges = []
            split_node_id = None
            eval_node_id = None

            for i in range(len(valid_nodes) - 1):
                src_id = valid_nodes[i]["id"]
                tgt_id = valid_nodes[i+1]["id"]
                auto_edges.append({
                    "id": f"e_{src_id}_{tgt_id}",
                    "source": src_id,
                    "target": tgt_id,
                    "animated": True
                })

                r_src = valid_nodes[i]["recipe_id"]
                r_tgt = valid_nodes[i+1]["recipe_id"]
                if "split" in r_src:
                    split_node_id = src_id
                if r_tgt == "model_evaluator":
                    eval_node_id = tgt_id

            if "split" in valid_nodes[-1]["recipe_id"]:
                split_node_id = valid_nodes[-1]["id"]
            if valid_nodes[-1]["recipe_id"] == "model_evaluator":
                eval_node_id = valid_nodes[-1]["id"]

            if split_node_id and eval_node_id and split_node_id != eval_node_id:
                if not any(e["source"] == split_node_id and e["target"] == eval_node_id for e in auto_edges):
                    auto_edges.append({
                        "id": f"e_{split_node_id}_{eval_node_id}",
                        "source": split_node_id,
                        "target": eval_node_id,
                        "animated": True
                    })
            valid_edges = auto_edges

        from backend.app.recommendation.autowire_utils import ensure_semantic_edge_handles
        valid_edges = ensure_semantic_edge_handles(valid_nodes, valid_edges)

        dag["nodes"] = valid_nodes
        dag["edges"] = valid_edges
        dag["node_configs"] = node_configs

        pre_recs = result.get("preprocessing_recommendations", [])
        if not pre_recs:
            pre_recs = []
            for n in valid_nodes:
                r_id = n["recipe_id"]
                if r_id in ["missing_value_imputer", "categorical_encoder", "feature_scaler", "data_type_converter", "outlier_handler", "column_selector"]:
                    pre_recs.append({
                        "recipe_id": r_id,
                        "name": n.get("label", r_id),
                        "recipe_name": n.get("label", r_id),
                        "config": n.get("config", {}),
                        "reason": f"Recommended preprocessing step {r_id}."
                    })
            if not pre_recs and isinstance(df, pd.DataFrame):
                from backend.app.recommendation.recommender import AIRecommender
                heur_res = AIRecommender._heuristic_recommend_pipeline(df, target_column=target_col, task_type=task_type)
                pre_recs = heur_res.get("preprocessing_recommendations", [])

        m_rankings = result.get("model_rankings", [])
        if not m_rankings:
            if task_type == "time_series_forecasting":
                m_rankings = [{"recipe_id": "prophet_forecaster", "name": "Prophet Forecaster", "tier": "Primary Model", "reason": "Recommended for time-series forecasting."}]
            elif task_type == "regression":
                m_rankings = [{"recipe_id": "xgboost_trainer", "name": "XGBoost Regressor", "tier": "Primary Model", "reason": "Recommended for continuous regression."}]
            elif task_type == "classification":
                m_rankings = [{"recipe_id": "xgboost_trainer", "name": "XGBoost Classifier", "tier": "Primary Model", "reason": "Recommended for discrete classification."}]
            else:
                m_rankings = [{"recipe_id": "isolation_forest", "name": "Isolation Forest", "tier": "Primary Model", "reason": "Recommended for anomaly detection."}]

        return {
            "task_type": task_type,
            "target_column": target_col,
            "explanation": result.get("explanation", "Custom AI-architected pipeline generated by Groq LLM."),
            "preprocessing_recommendations": pre_recs,
            "model_rankings": m_rankings,
            "recommended_dag": dag,
            "llm_generated": True,
            "recommendation_source": "llm"
        }