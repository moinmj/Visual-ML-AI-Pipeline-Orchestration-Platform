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
from backend.app.recipes.preprocessing.class_imbalance import ClassImbalanceResamplerRecipe
from backend.app.recipes.nlp.text_preprocessor import TextPreprocessorRecipe
from backend.app.recipes.nlp.text_vectorizer import TextVectorizerRecipe
from backend.app.recipes.splitting.train_test_split import TrainTestSplitRecipe
from backend.app.recipes.splitting.stratified_split import StratifiedSplitRecipe
from backend.app.recipes.splitting.time_series_split import TimeSeriesSplitRecipe
from backend.app.recipes.splitting.walk_forward_split import WalkForwardSplitRecipe
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
from backend.app.recipes.preprocessing.column_selector import ColumnSelectorRecipe
from backend.app.recipes.preprocessing.data_type_converter import DataTypeConverterRecipe
from backend.app.recipes.preprocessing.outlier_handler import OutlierHandlerRecipe
from backend.app.recipes.preprocessing.feature_selector import FeatureSelectorRecipe
from backend.app.recipes.preprocessing.dataset_join import DatasetJoinRecipe
from backend.app.recipes.governance.mlflow_tracker import MLflowTrackerRecipe
from backend.app.recipes.triggers.webhook_trigger import WebhookTriggerRecipe
from backend.app.recipes.triggers.cron_trigger import CronScheduleTriggerRecipe

# Flow Control Recipes
from backend.app.recipes.flow_control.if_condition import IfConditionRecipe
from backend.app.recipes.flow_control.row_filter import RowFilterRecipe
from backend.app.recipes.flow_control.switch_node import SwitchRecipe
from backend.app.recipes.flow_control.merge_node import MergeDatasetsRecipe
from backend.app.recipes.flow_control.loop_node import LoopBatchRecipe
from backend.app.recipes.flow_control.delay_node import DelayRecipe

# Apps & Integrations Recipes
from backend.app.recipes.integrations.slack_notifier import SlackRecipe
from backend.app.recipes.integrations.discord_notifier import DiscordRecipe
from backend.app.recipes.integrations.telegram_notifier import TelegramRecipe
from backend.app.recipes.integrations.gmail_notifier import GmailRecipe
from backend.app.recipes.integrations.google_sheets import GoogleSheetsRecipe
from backend.app.recipes.integrations.google_drive import GoogleDriveRecipe
from backend.app.recipes.integrations.openweathermap import OpenWeatherMapRecipe


def register_all_recipes():
    recipe_registry.register(CSVLoaderRecipe())
    recipe_registry.register(DatasetJoinRecipe())
    recipe_registry.register(WebhookTriggerRecipe())
    recipe_registry.register(CronScheduleTriggerRecipe())
    recipe_registry.register(ColumnSelectorRecipe())
    recipe_registry.register(DataTypeConverterRecipe())
    recipe_registry.register(DuplicateRemoverRecipe())
    recipe_registry.register(CategorySanitizerRecipe())
    recipe_registry.register(CorrelationFilterRecipe())
    recipe_registry.register(VarianceFilterRecipe())
    recipe_registry.register(MissingValueImputerRecipe())
    recipe_registry.register(OutlierHandlerRecipe())
    recipe_registry.register(FeatureSelectorRecipe())
    recipe_registry.register(FeatureScalerRecipe())
    recipe_registry.register(CategoricalEncoderRecipe())
    recipe_registry.register(ClassImbalanceResamplerRecipe())
    recipe_registry.register(TextPreprocessorRecipe())
    recipe_registry.register(TextVectorizerRecipe())
    recipe_registry.register(TrainTestSplitRecipe())
    recipe_registry.register(StratifiedSplitRecipe())
    recipe_registry.register(TimeSeriesSplitRecipe())
    recipe_registry.register(WalkForwardSplitRecipe())
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

    # Flow Control
    recipe_registry.register(IfConditionRecipe())
    recipe_registry.register(RowFilterRecipe())
    recipe_registry.register(SwitchRecipe())
    recipe_registry.register(MergeDatasetsRecipe())
    recipe_registry.register(LoopBatchRecipe())
    recipe_registry.register(DelayRecipe())

    # Apps & Integrations
    recipe_registry.register(SlackRecipe())
    recipe_registry.register(DiscordRecipe())
    recipe_registry.register(TelegramRecipe())
    recipe_registry.register(GmailRecipe())
    recipe_registry.register(GoogleSheetsRecipe())
    recipe_registry.register(GoogleDriveRecipe())
    recipe_registry.register(OpenWeatherMapRecipe())


# Automatically register upon module load
register_all_recipes()
