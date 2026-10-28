from typing import Dict, List, Optional, Tuple
from backend.app.recipes.base.recipe import BaseRecipe, RecipeMetadata
from backend.app.core.exceptions import NotFoundException, ValidationException
from backend.app.core.logging import logger


_RECIPE_GROUPS: Dict[str, Tuple[str, str]] = {
    "webhook_trigger": ("Ingestion & Orchestration", "Triggers"),
    "cron_trigger": ("Ingestion & Orchestration", "Triggers"),
    "email_trigger_imap": ("Ingestion & Orchestration", "Triggers"),
    "csv_loader": ("Ingestion & Orchestration", "Load"),
    "export_writer": ("Ingestion & Orchestration", "Output"),
    "dataset_join": ("Data Preparation & Transformation", "Combine/Structure"),
    "column_selector": ("Data Preparation & Transformation", "Combine/Structure"),
    "data_type_converter": ("Data Preparation & Transformation", "Cleanup & Quality"),
    "duplicate_remover": ("Data Preparation & Transformation", "Cleanup & Quality"),
    "category_sanitizer": ("Data Preparation & Transformation", "Cleanup & Quality"),
    "missing_value_imputer": ("Data Preparation & Transformation", "Cleanup & Quality"),
    "outlier_handler": ("Data Preparation & Transformation", "Cleanup & Quality"),
    "correlation_filter": ("Data Preparation & Transformation", "Feature Engineering"),
    "variance_filter": ("Data Preparation & Transformation", "Feature Engineering"),
    "feature_selector": ("Data Preparation & Transformation", "Feature Engineering"),
    "feature_scaler": ("Data Preparation & Transformation", "Feature Engineering"),
    "categorical_encoder": ("Data Preparation & Transformation", "Feature Engineering"),
    "class_imbalance_resampler": ("Data Preparation & Transformation", "Feature Engineering"),
    "text_preprocessor": ("Data Preparation & Transformation", "NLP/Text"),
    "text_vectorizer": ("Data Preparation & Transformation", "NLP/Text"),
    "sentiment_analysis": ("Data Preparation & Transformation", "NLP/Text"),
    "text_translation": ("Data Preparation & Transformation", "NLP/Text"),
    "translation": ("Data Preparation & Transformation", "NLP/Text"),
    "text_summarization": ("Data Preparation & Transformation", "NLP/Text"),
    "summarization": ("Data Preparation & Transformation", "NLP/Text"),
    "pii_entity_masking": ("Data Preparation & Transformation", "NLP/Text"),
    "document_parsing": ("Data Preparation & Transformation", "NLP/Text"),
    "train_test_split": ("Data Preparation & Transformation", "Splitting"),
    "stratified_split": ("Data Preparation & Transformation", "Splitting"),
    "time_series_split": ("Data Preparation & Transformation", "Splitting"),
    "walk_forward_split": ("Data Preparation & Transformation", "Splitting"),
    "xgboost_trainer": ("Modeling", "Supervised — Tree Ensembles"),
    "random_forest_trainer": ("Modeling", "Supervised — Tree Ensembles"),
    "lightgbm_trainer": ("Modeling", "Supervised — Tree Ensembles"),
    "catboost_trainer": ("Modeling", "Supervised — Tree Ensembles"),
    "model_calibration": ("Modeling", "Deep Learning / AutoML"),
    "ensemble_stacker": ("Modeling", "Deep Learning / AutoML"),
    "hyperparameter_tuner": ("Modeling", "Deep Learning / AutoML"),
    "automl_search": ("Modeling", "Deep Learning / AutoML"),
    "linear_trainer": ("Modeling", "Supervised — Linear"),
    "lag_feature_engineering": ("Modeling", "Forecasting"),
    "prophet_forecaster": ("Modeling", "Forecasting"),
    "arima_forecaster": ("Modeling", "Forecasting"),
    "isolation_forest": ("Modeling", "Anomaly Detection"),
    "statistical_guardrail": ("Modeling", "Anomaly Detection"),
    "model_evaluator": ("Evaluation & Governance", "Evaluation"),
    "ab_testing_framework": ("Evaluation & Governance", "Evaluation"),
    "champion_challenger": ("Evaluation & Governance", "Evaluation"),
    "model_explainability": ("Modeling", "Specialized Models"),
    "shap_explainer": ("Modeling", "Specialized Models"),
    "object_detection": ("Modeling", "Computer Vision & Audio"),
    "image_classifier": ("Modeling", "Computer Vision & Audio"),
    "ocr": ("Modeling", "Computer Vision & Audio"),
    "image_embedding_generator": ("Modeling", "Computer Vision & Audio"),
    "video_object_tracking": ("Modeling", "Computer Vision & Audio"),
    "speech_to_text": ("Modeling", "Computer Vision & Audio"),
    "text_to_speech": ("Modeling", "Computer Vision & Audio"),
    "mlflow_tracker": ("Evaluation & Governance", "Tracking & Registry"),

    # Flow Control
    "if_condition": ("Flow Control", "Routing & Branching"),
    "row_filter": ("Flow Control", "Filter"),
    "switch": ("Flow Control", "Routing & Branching"),
    "merge": ("Flow Control", "Combine"),
    "loop": ("Flow Control", "Iteration"),
    "delay": ("Flow Control", "Timing"),
    "sql_step": ("Data Preparation & Transformation", "Custom Logic"),
    "custom_python_step": ("Data Preparation & Transformation", "Custom Logic"),
    "limit": ("Data Preparation & Transformation", "Combine/Structure"),
    "sort": ("Data Preparation & Transformation", "Combine/Structure"),
    "aggregate": ("Data Preparation & Transformation", "Combine/Structure"),
    "combine": ("Data Preparation & Transformation", "Combine/Structure"),
    "pivot": ("Data Preparation & Transformation", "Combine/Structure"),

    # Apps / Integrations
    "slack": ("Apps & Integrations", "Communication"),
    "discord": ("Apps & Integrations", "Communication"),
    "telegram": ("Apps & Integrations", "Communication"),
    "gmail": ("Apps & Integrations", "Email & Notification"),
    "google_sheets": ("Apps & Integrations", "Google Workspace"),
    "google_drive": ("Apps & Integrations", "Google Workspace"),
    "openweathermap": ("Apps & Integrations", "External APIs"),
    "stock_market_api": ("Apps & Integrations", "Market & Reference Data"),
    "currency_exchange_api": ("Apps & Integrations", "Market & Reference Data"),
    "news_api": ("Apps & Integrations", "Market & Reference Data"),
    "rss_news_reader": ("Apps & Integrations", "Market & Reference Data"),
    "open_data_api": ("Apps & Integrations", "Market & Reference Data"),
    "twitter_connector": ("Apps & Integrations", "Social & Comms"),
    "reddit_connector": ("Apps & Integrations", "Social & Comms"),
    "linkedin_connector": ("Apps & Integrations", "Social & Comms"),

    # Unsupervised & Dimensionality Reduction
    "clustering": ("Modeling", "Unsupervised & Dimensionality Reduction"),
    "pca": ("Modeling", "Unsupervised & Dimensionality Reduction"),
    "umap": ("Modeling", "Unsupervised & Dimensionality Reduction"),
    "tsne": ("Modeling", "Unsupervised & Dimensionality Reduction"),
    "association_rule_mining": ("Modeling", "Unsupervised & Dimensionality Reduction"),

    # Deployment & Monitoring
    "model_serving_endpoint": ("Evaluation & Governance", "Deployment & Monitoring"),
    "model_drift_monitor": ("Evaluation & Governance", "Deployment & Monitoring"),
    "data_drift_detector": ("Evaluation & Governance", "Deployment & Monitoring"),
    "rollback_trigger": ("Evaluation & Governance", "Deployment & Monitoring"),
    "cost_latency_monitor": ("Evaluation & Governance", "Deployment & Monitoring"),
    "cost_attribution_report": ("Evaluation & Governance", "Deployment & Monitoring"),

    # Agents & LLM Orchestration
    "prompt_template_manager": ("Agents & LLM Orchestration", "Prompt & Memory Management"),
    "conversation_memory": ("Agents & LLM Orchestration", "Prompt & Memory Management"),
    "structured_output_enforcer": ("Agents & LLM Orchestration", "Prompt & Memory Management"),
    "embedding_generator": ("Agents & LLM Orchestration", "RAG & Retrieval"),
    "vector_db_indexer": ("Agents & LLM Orchestration", "RAG & Retrieval"),
    "rag_pipeline": ("Agents & LLM Orchestration", "RAG & Retrieval"),
    "agent_guardrail": ("Agents & LLM Orchestration", "Governance & Ops"),
    "cost_aware_model_router": ("Agents & LLM Orchestration", "Governance & Ops"),
    "llm_evaluation": ("Agents & LLM Orchestration", "Governance & Ops"),
    "fine_tuning_trigger": ("Agents & LLM Orchestration", "Governance & Ops"),
    "llm_agent": ("Agents & LLM Orchestration", "Core Agents"),
    "multi_agent_orchestrator": ("Agents & LLM Orchestration", "Core Agents"),
    "agent_handoff_router": ("Agents & LLM Orchestration", "Core Agents"),
    "web_browsing_agent": ("Agents & LLM Orchestration", "Core Agents"),
    "document_qa_agent": ("Agents & LLM Orchestration", "Core Agents"),
    "multimodal_agent": ("Agents & LLM Orchestration", "Core Agents"),
    "code_execution_sandbox": ("Agents & LLM Orchestration", "Core Agents"),
    "chatbot_endpoint": ("Agents & LLM Orchestration", "Core Agents"),
}




class RecipeRegistry:
    def __init__(self):
        self._recipes: Dict[str, BaseRecipe] = {}
        self._aliases: Dict[str, str] = {
            "translation": "text_translation",
            "summarization": "text_summarization",
            "classification_evaluator": "model_evaluator",
            "regression_evaluator": "model_evaluator",
            "model_governance_card": "mlflow_tracker",
            "governance_card": "mlflow_tracker",
            "if_else": "if_condition",
            "filter": "row_filter",
            "wait": "delay",
            "email": "gmail",
            "weather": "openweathermap",
            "union": "merge",
            "imap": "email_trigger_imap",
            "imap_trigger": "email_trigger_imap",
            "email_trigger": "email_trigger_imap",
            "drift_monitoring_operator": "model_drift_monitor",
            "drift_operator": "model_drift_monitor",
            "model_drift_operator": "model_drift_monitor",
            "drift_monitor": "model_drift_monitor",
            "performance_drift_operator": "model_drift_monitor",
            "model_serving": "model_serving_endpoint",
            "serving_endpoint": "model_serving_endpoint",
            "input_drift_detector": "data_drift_detector",
            "data_drift": "data_drift_detector",
            "cost_latency": "cost_latency_monitor",
            "latency_monitor": "cost_latency_monitor",
            "cost_attribution": "cost_attribution_report",
            "finops_report": "cost_attribution_report",
            "ab_testing": "ab_testing_framework",
            "ab_test": "ab_testing_framework",
            "experiment_framework": "ab_testing_framework",
            "champion_challenger_comparison": "champion_challenger",
            "champion_vs_challenger": "champion_challenger",
            "forex": "currency_exchange_api",
            "exchange_rate": "currency_exchange_api",
            "fx": "currency_exchange_api",
            "news": "news_api",
            "newsapi": "news_api",
            "news_feed": "news_api",
            "headlines": "news_api",
            "open_data": "open_data_api",
            "government_data": "open_data_api",
            "gov_data": "open_data_api",
            "public_data": "open_data_api",
            "data_gov": "open_data_api",
            "twitter": "twitter_connector",
            "x_connector": "twitter_connector",
            "x": "twitter_connector",
            "reddit": "reddit_connector",
            "linkedin": "linkedin_connector",
            "embeddings": "embedding_generator",
            "vector_embedder": "embedding_generator",
            "vector_search": "vector_db_indexer",
            "vector_index": "vector_db_indexer",
            "rag": "rag_pipeline",
            "retrieval_pipeline": "rag_pipeline",
            "guardrail": "agent_guardrail",
            "moderation": "agent_guardrail",
            "model_router": "cost_aware_model_router",
            "llm_eval": "llm_evaluation",
            "groundedness_scorer": "llm_evaluation",
            "fine_tuning": "fine_tuning_trigger",
            "fine_tune": "fine_tuning_trigger",
            "agent": "llm_agent",
            "react_agent": "llm_agent",
            "orchestrator": "multi_agent_orchestrator",
            "multi_agent": "multi_agent_orchestrator",
            "handoff_router": "agent_handoff_router",
            "agent_handoff": "agent_handoff_router",
            "web_agent": "web_browsing_agent",
            "web_browse": "web_browsing_agent",
            "doc_qa": "document_qa_agent",
            "multimodal": "multimodal_agent",
            "code_sandbox": "code_execution_sandbox",
            "python_sandbox": "code_execution_sandbox",
            "chatbot": "chatbot_endpoint",
            "shap": "model_explainability",
            "lime": "model_explainability",
            "explainability": "model_explainability",
            "feature_attribution": "model_explainability",
            "shap_explainer": "model_explainability",
            "calibration": "model_calibration",
            "platt_scaling": "model_calibration",
            "isotonic_regression": "model_calibration",
            "calibrator": "model_calibration",
            "stacking": "ensemble_stacker",
            "stacker": "ensemble_stacker",
            "stacking_classifier": "ensemble_stacker",
            "stacking_regressor": "ensemble_stacker",
            "ensemble_stacking": "ensemble_stacker",
            "optuna": "hyperparameter_tuner",
            "hyperparameter_tuning": "hyperparameter_tuner",
            "bayesian_optimization": "hyperparameter_tuner",
            "grid_search": "hyperparameter_tuner",
            "random_search": "hyperparameter_tuner",
            "tuner": "hyperparameter_tuner",
            "automl": "automl_search",
            "model_tournament": "automl_search",
            "algorithm_selection": "automl_search",
            "auto_ml": "automl_search",
            "automl_classifier": "automl_search",
            "automl_regressor": "automl_search",
        }



    def register(self, recipe: BaseRecipe):
        self._recipes[recipe.recipe_id] = recipe
        logger.debug(f"Registered recipe: {recipe.recipe_id} ({recipe.name})")

    def register_alias(self, alias: str, target_id: str):
        self._aliases[alias] = target_id
        logger.debug(f"Registered recipe alias: '{alias}' -> '{target_id}'")

    def _to_metadata_with_group(self, recipe: BaseRecipe) -> RecipeMetadata:
        meta = recipe.to_metadata()
        group, subgroup = _RECIPE_GROUPS.get(recipe.recipe_id, (None, None))
        meta.group = group
        meta.subgroup = subgroup
        return meta

    def _planned(self) -> Dict[str, RecipeMetadata]:
        """Planned recipes that are not implemented yet (a registered class always wins)."""
        from backend.app.recipes.base.planned_recipes import PLANNED_RECIPES
        return {rid: meta for rid, meta in PLANNED_RECIPES.items() if rid not in self._recipes}

    def get(self, recipe_id: str) -> BaseRecipe:
        """
        Returns an implemented (enabled) recipe. Everything that runs, tests,
        validates or auto-wires recipes goes through here, so planned recipes are
        refused in one place with a clear message.
        """
        resolved_id = self._aliases.get(recipe_id, recipe_id)
        if resolved_id not in self._recipes:
            planned = self._planned().get(resolved_id)
            if planned:
                raise ValidationException(
                    f"Recipe '{planned.name}' ({resolved_id}) is not available yet (coming soon). "
                    "Remove it from the workflow to run or test the pipeline."
                )
            raise NotFoundException("Recipe", recipe_id)
        return self._recipes[resolved_id]

    def has(self, recipe_id: str) -> bool:
        """True only for implemented (enabled) recipes."""
        resolved_id = self._aliases.get(recipe_id, recipe_id)
        return resolved_id in self._recipes

    def get_metadata(self, recipe_id: str) -> RecipeMetadata:
        """Metadata (with group/subgroup and 'enabled') for an enabled OR planned recipe."""
        resolved_id = self._aliases.get(recipe_id, recipe_id)
        if resolved_id in self._recipes:
            return self._to_metadata_with_group(self._recipes[resolved_id])
        planned = self._planned().get(resolved_id)
        if planned:
            return planned
        raise NotFoundException("Recipe", recipe_id)

    def list_all(self, category: Optional[str] = None, enabled: Optional[bool] = None) -> List[RecipeMetadata]:
        """
        All recipes for the catalog: implemented ones ("enabled": true) followed by
        planned ones ("enabled": false). Optional filters: category, enabled.
        """
        metas = [self._to_metadata_with_group(r) for r in self._recipes.values()]
        metas += list(self._planned().values())
        if category:
            metas = [m for m in metas if m.category.lower() == category.lower()]
        if enabled is not None:
            metas = [m for m in metas if m.enabled == enabled]
        return metas

    def get_categories(self) -> List[str]:
        return sorted({m.category for m in self.list_all()})


recipe_registry = RecipeRegistry()