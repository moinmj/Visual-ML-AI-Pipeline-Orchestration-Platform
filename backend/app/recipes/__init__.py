from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.ingestion.csv_loader import CSVLoaderRecipe
from backend.app.recipes.ingestion.export_writer import ExportWriterRecipe
from backend.app.recipes.preprocessing.custom_python_step import CustomPythonStepRecipe
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
from backend.app.recipes.nlp.sentiment_analysis import SentimentAnalysisRecipe
from backend.app.recipes.nlp.translation import TranslationRecipe
from backend.app.recipes.nlp.summarization import SummarizationRecipe
from backend.app.recipes.nlp.pii_masking import PIIEntityMaskingRecipe
from backend.app.recipes.nlp.document_parsing import DocumentParsingRecipe
from backend.app.recipes.splitting.train_test_split import TrainTestSplitRecipe
from backend.app.recipes.splitting.stratified_split import StratifiedSplitRecipe
from backend.app.recipes.splitting.time_series_split import TimeSeriesSplitRecipe
from backend.app.recipes.splitting.walk_forward_split import WalkForwardSplitRecipe
from backend.app.recipes.training.xgboost_trainer import XGBoostTrainerRecipe
from backend.app.recipes.training.random_forest_trainer import RandomForestTrainerRecipe
from backend.app.recipes.training.logistic_regression_trainer import LogisticRegressionTrainerRecipe
from backend.app.recipes.training.lightgbm_trainer import LightGBMTrainerRecipe
from backend.app.recipes.training.catboost_trainer import CatBoostTrainerRecipe
from backend.app.recipes.training.model_calibration import ModelCalibrationRecipe
from backend.app.recipes.training.ensemble_stacker import EnsembleStackerRecipe
from backend.app.recipes.training.hyperparameter_tuner import HyperparameterTunerRecipe
from backend.app.recipes.training.automl_search import AutoMLSearchRecipe
from backend.app.recipes.training.neural_network_trainer import NeuralNetworkTrainerRecipe
from backend.app.recipes.training.survival_model import SurvivalModelRecipe
from backend.app.recipes.training.recommendation_engine import RecommendationEngineRecipe
from backend.app.recipes.training.next_best_offer import NextBestOfferRecipe
from backend.app.recipes.evaluation.model_evaluator import ModelEvaluatorRecipe
from backend.app.recipes.evaluation.ab_testing_framework import ABTestingFrameworkRecipe
from backend.app.recipes.evaluation.champion_challenger import ChampionChallengerRecipe
from backend.app.recipes.evaluation.model_explainability import ModelExplainabilityRecipe
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
from backend.app.recipes.preprocessing.select_rename_reorder import SelectRenameReorderRecipe
from backend.app.recipes.preprocessing.fuzzy_deduplication import FuzzyMatchDeduplicationRecipe
from backend.app.recipes.preprocessing.semantic_deduplication import SemanticDeduplicationRecipe
from backend.app.recipes.preprocessing.data_profiling import DataProfilingReportRecipe
from backend.app.recipes.preprocessing.schema_validation import SchemaContractValidationRecipe
from backend.app.recipes.preprocessing.data_guardrails import DataGuardrailsRecipe
from backend.app.recipes.preprocessing.formula_calculator import FormulaCalculatorRecipe
from backend.app.recipes.preprocessing.replace_value import ReplaceValueRecipe
from backend.app.recipes.preprocessing.regex_replace_extract import RegexReplaceExtractRecipe
from backend.app.recipes.preprocessing.parse_date import ParseDateRecipe
from backend.app.recipes.preprocessing.geocode_distance import GeocodeDistanceCalculatorRecipe
from backend.app.recipes.preprocessing.timezone_converter import TimezoneConverterRecipe
from backend.app.recipes.preprocessing.business_day_holiday import BusinessDayHolidayCalculatorRecipe
from backend.app.recipes.preprocessing.currency_unit_normalizer import CurrencyUnitNormalizerRecipe
from backend.app.recipes.preprocessing.address_parser_standardizer import AddressParserStandardizerRecipe
from backend.app.recipes.preprocessing.phone_number_validator import PhoneNumberValidatorRecipe
from backend.app.recipes.preprocessing.email_domain_validator import EmailDomainValidatorRecipe
from backend.app.recipes.preprocessing.json_xml_flattener import JsonXmlFlattenerNesterRecipe
from backend.app.recipes.preprocessing.hierarchical_tree_flattener import HierarchicalTreeFlattenerRecipe
from backend.app.recipes.preprocessing.cross_tab_generator import CrossTabGeneratorRecipe
from backend.app.recipes.preprocessing.date_range_expander import DateRangeExpanderRecipe
from backend.app.recipes.preprocessing.differential_privacy import DifferentialPrivacyAnonymizerRecipe
from backend.app.recipes.preprocessing.compliance_tokenization import ComplianceTokenizationRecipe
from backend.app.recipes.preprocessing.sampling_operator import SamplingOperatorRecipe
from backend.app.recipes.governance.mlflow_tracker import MLflowTrackerRecipe
from backend.app.recipes.triggers.webhook_trigger import WebhookTriggerRecipe
from backend.app.recipes.triggers.cron_trigger import CronScheduleTriggerRecipe
from backend.app.recipes.triggers.email_trigger_imap import EmailTriggerIMAPRecipe

# Flow Control Recipes
from backend.app.recipes.flow_control.if_condition import IfConditionRecipe
from backend.app.recipes.flow_control.row_filter import RowFilterRecipe
from backend.app.recipes.flow_control.switch_node import SwitchRecipe
from backend.app.recipes.flow_control.merge_node import MergeDatasetsRecipe
from backend.app.recipes.flow_control.loop_node import LoopBatchRecipe
from backend.app.recipes.flow_control.delay_node import DelayRecipe
from backend.app.recipes.preprocessing.sql_step import SQLStepRecipe
from backend.app.recipes.preprocessing.limit import LimitRecipe
from backend.app.recipes.preprocessing.sort_rows import SortRecipe
from backend.app.recipes.preprocessing.aggregate import AggregateRecipe
from backend.app.recipes.preprocessing.combine import CombineRecipe
from backend.app.recipes.preprocessing.pivot import PivotRecipe

# Apps & Integrations Recipes
from backend.app.recipes.integrations.slack_notifier import SlackRecipe
from backend.app.recipes.integrations.discord_notifier import DiscordRecipe
from backend.app.recipes.integrations.telegram_notifier import TelegramRecipe
from backend.app.recipes.integrations.gmail_notifier import GmailRecipe
from backend.app.recipes.integrations.google_sheets import GoogleSheetsRecipe
from backend.app.recipes.integrations.google_drive import GoogleDriveRecipe
from backend.app.recipes.integrations.openweathermap import OpenWeatherMapRecipe
from backend.app.recipes.integrations.stock_market_api import StockMarketApiRecipe
from backend.app.recipes.integrations.currency_exchange_api import CurrencyExchangeApiRecipe
from backend.app.recipes.integrations.news_api import NewsApiRecipe
from backend.app.recipes.integrations.rss_news_reader import RssNewsReaderRecipe
from backend.app.recipes.integrations.open_data_api import OpenDataApiRecipe
from backend.app.recipes.integrations.twitter_connector import TwitterConnectorRecipe
from backend.app.recipes.integrations.reddit_connector import RedditConnectorRecipe
from backend.app.recipes.integrations.linkedin_connector import LinkedInConnectorRecipe

# Unsupervised & Dimensionality Reduction Recipes
from backend.app.recipes.unsupervised.clustering import ClusteringRecipe
from backend.app.recipes.unsupervised.pca import PCARecipe
from backend.app.recipes.unsupervised.umap import UMAPRecipe
from backend.app.recipes.unsupervised.tsne import TSNERecipe
from backend.app.recipes.unsupervised.association_rules import AssociationRuleMiningRecipe

# Deployment & Monitoring Recipes
from backend.app.recipes.monitoring import (
    ModelServingEndpointRecipe,
    ModelDriftMonitorRecipe,
    DataDriftDetectorRecipe,
    RollbackTriggerRecipe,
    CostLatencyMonitorRecipe,
    CostAttributionReportRecipe,
)

# Computer vision recipes
from backend.app.recipes.vision.object_detection import ObjectDetectionRecipe
from backend.app.recipes.vision.image_classifier import ImageClassifierRecipe
from backend.app.recipes.vision.ocr import OCRRecipe
from backend.app.recipes.vision.image_embedding import ImageEmbeddingRecipe
from backend.app.recipes.vision.video_tracking import VideoObjectTrackingRecipe
from backend.app.recipes.vision.speech_to_text import SpeechToTextRecipe
from backend.app.recipes.vision.text_to_speech import TextToSpeechRecipe

# Agents & LLM Orchestration recipes
from backend.app.recipes.agents.prompt_template_manager import PromptTemplateManagerRecipe
from backend.app.recipes.agents.conversation_memory import ConversationMemoryRecipe
from backend.app.recipes.agents.structured_output_enforcer import StructuredOutputEnforcerRecipe
from backend.app.recipes.agents.embedding_generator import EmbeddingGeneratorRecipe
from backend.app.recipes.agents.vector_db_indexer import VectorDbIndexerRecipe
from backend.app.recipes.agents.rag_pipeline import RagPipelineRecipe
from backend.app.recipes.agents.agent_guardrail import AgentGuardrailRecipe
from backend.app.recipes.agents.cost_aware_model_router import CostAwareModelRouterRecipe
from backend.app.recipes.agents.llm_evaluation import LlmEvaluationRecipe
from backend.app.recipes.agents.fine_tuning_trigger import FineTuningTriggerRecipe
from backend.app.recipes.agents.llm_agent import LlmAgentRecipe
from backend.app.recipes.agents.multi_agent_orchestrator import MultiAgentOrchestratorRecipe
from backend.app.recipes.agents.agent_handoff_router import AgentHandoffRouterRecipe
from backend.app.recipes.agents.web_browsing_agent import WebBrowsingAgentRecipe
from backend.app.recipes.agents.document_qa_agent import DocumentQaAgentRecipe
from backend.app.recipes.agents.multimodal_agent import MultimodalAgentRecipe
from backend.app.recipes.agents.code_execution_sandbox import CodeExecutionSandboxRecipe
from backend.app.recipes.agents.chatbot_endpoint import ChatbotEndpointRecipe



def register_all_recipes():
    recipe_registry.register(CSVLoaderRecipe())
    recipe_registry.register(ExportWriterRecipe())
    recipe_registry.register(DatasetJoinRecipe())
    recipe_registry.register(WebhookTriggerRecipe())
    recipe_registry.register(CronScheduleTriggerRecipe())
    recipe_registry.register(EmailTriggerIMAPRecipe())
    recipe_registry.register(ColumnSelectorRecipe())
    recipe_registry.register(DataTypeConverterRecipe())
    recipe_registry.register(DuplicateRemoverRecipe())
    recipe_registry.register(CategorySanitizerRecipe())
    recipe_registry.register(SelectRenameReorderRecipe())
    recipe_registry.register(FuzzyMatchDeduplicationRecipe())
    recipe_registry.register(SemanticDeduplicationRecipe())
    recipe_registry.register(DataProfilingReportRecipe())
    recipe_registry.register(SchemaContractValidationRecipe())
    recipe_registry.register(DataGuardrailsRecipe())
    recipe_registry.register(FormulaCalculatorRecipe())
    recipe_registry.register(ReplaceValueRecipe())
    recipe_registry.register(RegexReplaceExtractRecipe())
    recipe_registry.register(ParseDateRecipe())
    recipe_registry.register(GeocodeDistanceCalculatorRecipe())
    recipe_registry.register(TimezoneConverterRecipe())
    recipe_registry.register(BusinessDayHolidayCalculatorRecipe())
    recipe_registry.register(CurrencyUnitNormalizerRecipe())
    recipe_registry.register(AddressParserStandardizerRecipe())
    recipe_registry.register(PhoneNumberValidatorRecipe())
    recipe_registry.register(EmailDomainValidatorRecipe())
    recipe_registry.register(JsonXmlFlattenerNesterRecipe())
    recipe_registry.register(HierarchicalTreeFlattenerRecipe())
    recipe_registry.register(CrossTabGeneratorRecipe())
    recipe_registry.register(DateRangeExpanderRecipe())
    recipe_registry.register(DifferentialPrivacyAnonymizerRecipe())
    recipe_registry.register(ComplianceTokenizationRecipe())
    recipe_registry.register(SamplingOperatorRecipe())
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
    recipe_registry.register(SentimentAnalysisRecipe())
    recipe_registry.register(TranslationRecipe())
    recipe_registry.register(SummarizationRecipe())
    recipe_registry.register(PIIEntityMaskingRecipe())
    recipe_registry.register(DocumentParsingRecipe())
    recipe_registry.register(TrainTestSplitRecipe())
    recipe_registry.register(StratifiedSplitRecipe())
    recipe_registry.register(TimeSeriesSplitRecipe())
    recipe_registry.register(WalkForwardSplitRecipe())
    recipe_registry.register(XGBoostTrainerRecipe())
    recipe_registry.register(RandomForestTrainerRecipe())
    recipe_registry.register(LogisticRegressionTrainerRecipe())
    recipe_registry.register(LightGBMTrainerRecipe())
    recipe_registry.register(CatBoostTrainerRecipe())
    recipe_registry.register(ModelCalibrationRecipe())
    recipe_registry.register(EnsembleStackerRecipe())
    recipe_registry.register(HyperparameterTunerRecipe())
    recipe_registry.register(AutoMLSearchRecipe())
    recipe_registry.register(NeuralNetworkTrainerRecipe())
    recipe_registry.register(SurvivalModelRecipe())
    recipe_registry.register(RecommendationEngineRecipe())
    recipe_registry.register(NextBestOfferRecipe())
    recipe_registry.register(ModelEvaluatorRecipe())
    recipe_registry.register(ABTestingFrameworkRecipe())
    recipe_registry.register(ChampionChallengerRecipe())
    recipe_registry.register(ModelExplainabilityRecipe())
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
    recipe_registry.register(SQLStepRecipe())
    recipe_registry.register(CustomPythonStepRecipe())
    recipe_registry.register(LimitRecipe())
    recipe_registry.register(SortRecipe())
    recipe_registry.register(AggregateRecipe())
    recipe_registry.register(CombineRecipe())
    recipe_registry.register(PivotRecipe())

    # Apps & Integrations
    recipe_registry.register(SlackRecipe())
    recipe_registry.register(DiscordRecipe())
    recipe_registry.register(TelegramRecipe())
    recipe_registry.register(GmailRecipe())
    recipe_registry.register(GoogleSheetsRecipe())
    recipe_registry.register(GoogleDriveRecipe())
    recipe_registry.register(OpenWeatherMapRecipe())
    recipe_registry.register(StockMarketApiRecipe())
    recipe_registry.register(CurrencyExchangeApiRecipe())
    recipe_registry.register(NewsApiRecipe())
    recipe_registry.register(RssNewsReaderRecipe())
    recipe_registry.register(OpenDataApiRecipe())
    recipe_registry.register(TwitterConnectorRecipe())
    recipe_registry.register(RedditConnectorRecipe())
    recipe_registry.register(LinkedInConnectorRecipe())

    # Unsupervised & Dimensionality Reduction
    recipe_registry.register(ClusteringRecipe())
    recipe_registry.register(PCARecipe())
    recipe_registry.register(UMAPRecipe())
    recipe_registry.register(TSNERecipe())
    recipe_registry.register(AssociationRuleMiningRecipe())

    # Deployment & Monitoring
    recipe_registry.register(ModelServingEndpointRecipe())
    recipe_registry.register(ModelDriftMonitorRecipe())
    recipe_registry.register(DataDriftDetectorRecipe())
    recipe_registry.register(RollbackTriggerRecipe())
    recipe_registry.register(CostLatencyMonitorRecipe())
    recipe_registry.register(CostAttributionReportRecipe())

    # Computer Vision
    recipe_registry.register(ObjectDetectionRecipe())
    recipe_registry.register(ImageClassifierRecipe())
    recipe_registry.register(OCRRecipe())
    recipe_registry.register(ImageEmbeddingRecipe())
    recipe_registry.register(VideoObjectTrackingRecipe())
    recipe_registry.register(SpeechToTextRecipe())
    recipe_registry.register(TextToSpeechRecipe())

    # Agents & LLM Orchestration
    recipe_registry.register(PromptTemplateManagerRecipe())
    recipe_registry.register(ConversationMemoryRecipe())
    recipe_registry.register(StructuredOutputEnforcerRecipe())
    recipe_registry.register(EmbeddingGeneratorRecipe())
    recipe_registry.register(VectorDbIndexerRecipe())
    recipe_registry.register(RagPipelineRecipe())
    recipe_registry.register(AgentGuardrailRecipe())
    recipe_registry.register(CostAwareModelRouterRecipe())
    recipe_registry.register(LlmEvaluationRecipe())
    recipe_registry.register(FineTuningTriggerRecipe())
    recipe_registry.register(LlmAgentRecipe())
    recipe_registry.register(MultiAgentOrchestratorRecipe())
    recipe_registry.register(AgentHandoffRouterRecipe())
    recipe_registry.register(WebBrowsingAgentRecipe())
    recipe_registry.register(DocumentQaAgentRecipe())
    recipe_registry.register(MultimodalAgentRecipe())
    recipe_registry.register(CodeExecutionSandboxRecipe())
    recipe_registry.register(ChatbotEndpointRecipe())




# Automatically register upon module load
register_all_recipes()