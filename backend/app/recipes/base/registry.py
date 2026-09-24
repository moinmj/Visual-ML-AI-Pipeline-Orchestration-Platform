from typing import Dict, List, Optional, Tuple
from backend.app.recipes.base.recipe import BaseRecipe, RecipeMetadata
from backend.app.core.exceptions import NotFoundException
from backend.app.core.logging import logger


_RECIPE_GROUPS: Dict[str, Tuple[str, str]] = {
    "webhook_trigger": ("Ingestion & Orchestration", "Triggers"),
    "cron_trigger": ("Ingestion & Orchestration", "Triggers"),
    "csv_loader": ("Ingestion & Orchestration", "Load"),
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
    "train_test_split": ("Data Preparation & Transformation", "Splitting"),
    "stratified_split": ("Data Preparation & Transformation", "Splitting"),
    "time_series_split": ("Data Preparation & Transformation", "Splitting"),
    "walk_forward_split": ("Data Preparation & Transformation", "Splitting"),
    "xgboost_trainer": ("Modeling", "Supervised — Tree Ensembles"),
    "random_forest_trainer": ("Modeling", "Supervised — Tree Ensembles"),
    "lightgbm_trainer": ("Modeling", "Supervised — Tree Ensembles"),
    "catboost_trainer": ("Modeling", "Supervised — Tree Ensembles"),
    "linear_trainer": ("Modeling", "Supervised — Linear"),
    "lag_feature_engineering": ("Modeling", "Forecasting"),
    "prophet_forecaster": ("Modeling", "Forecasting"),
    "arima_forecaster": ("Modeling", "Forecasting"),
    "isolation_forest": ("Modeling", "Anomaly Detection"),
    "statistical_guardrail": ("Modeling", "Anomaly Detection"),
    "model_evaluator": ("Evaluation & Governance", "Evaluation"),
    "mlflow_tracker": ("Evaluation & Governance", "Tracking & Registry"),

    # Flow Control
    "if_condition": ("Flow Control", "Routing & Branching"),
    "row_filter": ("Flow Control", "Filter"),
    "switch": ("Flow Control", "Routing & Branching"),
    "merge": ("Flow Control", "Combine"),
    "loop": ("Flow Control", "Iteration"),
    "delay": ("Flow Control", "Timing"),

    # Apps / Integrations
    "slack": ("Apps & Integrations", "Communication"),
    "discord": ("Apps & Integrations", "Communication"),
    "telegram": ("Apps & Integrations", "Communication"),
    "gmail": ("Apps & Integrations", "Email & Notification"),
    "google_sheets": ("Apps & Integrations", "Google Workspace"),
    "google_drive": ("Apps & Integrations", "Google Workspace"),
    "openweathermap": ("Apps & Integrations", "External APIs"),
}


class RecipeRegistry:
    def __init__(self):
        self._recipes: Dict[str, BaseRecipe] = {}
        self._aliases: Dict[str, str] = {
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

    def get(self, recipe_id: str) -> BaseRecipe:
        resolved_id = self._aliases.get(recipe_id, recipe_id)
        if resolved_id not in self._recipes:
            raise NotFoundException("Recipe", recipe_id)
        return self._recipes[resolved_id]

    def has(self, recipe_id: str) -> bool:
        resolved_id = self._aliases.get(recipe_id, recipe_id)
        return resolved_id in self._recipes

    def list_all(self, category: Optional[str] = None) -> List[RecipeMetadata]:
        recipes = list(self._recipes.values())
        if category:
            recipes = [r for r in recipes if r.category.lower() == category.lower()]
        return [self._to_metadata_with_group(r) for r in recipes]

    def get_categories(self) -> List[str]:
        return sorted(list(set(r.category for r in self._recipes.values())))


recipe_registry = RecipeRegistry()