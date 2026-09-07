from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.ingestion.csv_loader import CSVLoaderRecipe
from backend.app.recipes.preprocessing.missing_values import MissingValueImputerRecipe
from backend.app.recipes.preprocessing.scaling import FeatureScalerRecipe
from backend.app.recipes.preprocessing.encoding import CategoricalEncoderRecipe
from backend.app.recipes.preprocessing.duplicates import (
    DuplicateRemoverRecipe,
    CategorySanitizerRecipe,
    CorrelationFilterRecipe,
    VarianceFilterRecipe
)
from backend.app.recipes.splitting.train_test_split import TrainTestSplitRecipe
from backend.app.recipes.training.xgboost_trainer import XGBoostTrainerRecipe
from backend.app.recipes.training.random_forest_trainer import RandomForestTrainerRecipe
from backend.app.recipes.training.logistic_regression_trainer import LogisticRegressionTrainerRecipe
from backend.app.recipes.training.lightgbm_trainer import LightGBMTrainerRecipe
from backend.app.recipes.training.catboost_trainer import CatBoostTrainerRecipe
from backend.app.recipes.evaluation.model_evaluator import ModelEvaluatorRecipe
from backend.app.recipes.anomaly.isolation_forest import IsolationForestRecipe
from backend.app.recipes.anomaly.statistical_guardrail import StatisticalGuardrailRecipe
from backend.app.recipes.forecasting.lag_features import LagFeatureEngineeringRecipe
from backend.app.recipes.forecasting.prophet_forecaster import ProphetForecasterRecipe
from backend.app.recipes.forecasting.arima_forecaster import ARIMAForecasterRecipe
from backend.app.recipes.governance.mlflow_tracker import MLflowTrackerRecipe
from backend.app.recipes.triggers.webhook_trigger import WebhookTriggerRecipe
from backend.app.recipes.triggers.cron_trigger import CronScheduleTriggerRecipe


def register_all_recipes():
    recipe_registry.register(CSVLoaderRecipe())
    recipe_registry.register(WebhookTriggerRecipe())
    recipe_registry.register(CronScheduleTriggerRecipe())
    recipe_registry.register(DuplicateRemoverRecipe())
    recipe_registry.register(CategorySanitizerRecipe())
    recipe_registry.register(CorrelationFilterRecipe())
    recipe_registry.register(VarianceFilterRecipe())
    recipe_registry.register(MissingValueImputerRecipe())
    recipe_registry.register(FeatureScalerRecipe())
    recipe_registry.register(CategoricalEncoderRecipe())
    recipe_registry.register(TrainTestSplitRecipe())
    recipe_registry.register(XGBoostTrainerRecipe())
    recipe_registry.register(RandomForestTrainerRecipe())
    recipe_registry.register(LogisticRegressionTrainerRecipe())
    recipe_registry.register(LightGBMTrainerRecipe())
    recipe_registry.register(CatBoostTrainerRecipe())
    recipe_registry.register(ModelEvaluatorRecipe())
    recipe_registry.register(IsolationForestRecipe())
    recipe_registry.register(StatisticalGuardrailRecipe())
    recipe_registry.register(LagFeatureEngineeringRecipe())
    recipe_registry.register(ProphetForecasterRecipe())
    recipe_registry.register(ARIMAForecasterRecipe())
    recipe_registry.register(MLflowTrackerRecipe())


# Automatically register upon module load
register_all_recipes()
