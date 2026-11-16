from typing import Dict, Any, Optional, Tuple

# Official Recipe Icon Registry: maps recipe_id -> official icon metadata
OFFICIAL_RECIPE_ICONS: Dict[str, Dict[str, Any]] = {
    # ── Cloud Data Warehouses & Lakes ──
    "snowflake_connector": {
        "icon": "snowflake",
        "svg": "snowflake.svg",
        "color": "#29B5E8",
        "label": "Snowflake",
    },
    "bigquery_connector": {
        "icon": "google-bigquery",
        "svg": "bigquery.svg",
        "color": "#4285F4",
        "label": "Google BigQuery",
    },
    "s3_data_lake_connector": {
        "icon": "amazon-s3",
        "svg": "aws-s3.svg",
        "color": "#569A31",
        "label": "AWS S3 Data Lake",
    },
    "azure_data_lake_connector": {
        "icon": "azure",
        "svg": "azure.svg",
        "color": "#0089D6",
        "label": "Azure ADLS Gen2",
    },
    "gcs_connector": {
        "icon": "google-cloud-storage",
        "svg": "gcs.svg",
        "color": "#4285F4",
        "label": "Google Cloud Storage",
    },
    "databricks_connector": {
        "icon": "databricks",
        "svg": "databricks.svg",
        "color": "#FF3621",
        "label": "Databricks Lakehouse",
    },
    "iceberg_lakehouse_connector": {
        "icon": "apache-iceberg",
        "svg": "bigquery.svg",
        "color": "#0087BE",
        "label": "Apache Iceberg",
    },
    "delta_lake_connector": {
        "icon": "delta-lake",
        "svg": "databricks.svg",
        "color": "#001E2B",
        "label": "Delta Lake",
    },

    # ── Relational & NoSQL Databases ──
    "postgresql_connector": {
        "icon": "postgresql",
        "svg": "postgresql.svg",
        "color": "#336791",
        "label": "PostgreSQL",
    },
    "mysql_connector": {
        "icon": "mysql",
        "svg": "mysql.svg",
        "color": "#00758F",
        "label": "MySQL",
    },
    "oracle_db_connector": {
        "icon": "oracle",
        "svg": "default.svg",
        "color": "#F80000",
        "label": "Oracle Database",
    },
    "sqlserver_connector": {
        "icon": "microsoft-sql-server",
        "svg": "azure.svg",
        "color": "#CC292B",
        "label": "Microsoft SQL Server",
    },
    "mongodb_connector": {
        "icon": "mongodb",
        "svg": "mongodb.svg",
        "color": "#47A248",
        "label": "MongoDB",
    },
    "redis_connector": {
        "icon": "redis",
        "svg": "redis.svg",
        "color": "#DC382D",
        "label": "Redis",
    },
    "elasticsearch_connector": {
        "icon": "elasticsearch",
        "svg": "elasticsearch.svg",
        "color": "#005571",
        "label": "Elasticsearch",
    },
    "clickhouse_connector": {
        "icon": "clickhouse",
        "svg": "clickhouse.svg",
        "color": "#FFCC01",
        "label": "ClickHouse",
    },
    "sqlite_connector": {
        "icon": "sqlite",
        "svg": "postgresql.svg",
        "color": "#003B57",
        "label": "SQLite",
    },

    # ── Streaming & Event Ingestion / CDC ──
    "kafka_consumer": {
        "icon": "apache-kafka",
        "svg": "kafka.svg",
        "color": "#231F20",
        "label": "Apache Kafka Consumer",
    },
    "kafka_publisher": {
        "icon": "apache-kafka",
        "svg": "kafka.svg",
        "color": "#231F20",
        "label": "Apache Kafka Publisher",
    },
    "rabbitmq_consumer": {
        "icon": "rabbitmq",
        "svg": "rabbitmq.svg",
        "color": "#FF6600",
        "label": "RabbitMQ Consumer",
    },
    "event_bus_publisher": {
        "icon": "event-bus",
        "svg": "api.svg",
        "color": "#6366F1",
        "label": "Event Bus Publisher",
    },
    "cdc_ingestion": {
        "icon": "cdc",
        "svg": "kafka.svg",
        "color": "#0284C7",
        "label": "Change Data Capture (CDC)",
    },
    "websocket_listener": {
        "icon": "websocket",
        "svg": "api.svg",
        "color": "#8B5CF6",
        "label": "WebSocket Listener",
    },

    # ── File & API Ingestion ──
    "csv_loader": {
        "icon": "csv",
        "svg": "csv.svg",
        "color": "#107C41",
        "label": "CSV File Loader",
    },
    "parquet_loader": {
        "icon": "parquet",
        "svg": "csv.svg",
        "color": "#2563EB",
        "label": "Apache Parquet Loader",
    },
    "json_loader": {
        "icon": "json",
        "svg": "api.svg",
        "color": "#F59E0B",
        "label": "JSON Document Loader",
    },
    "excel_loader": {
        "icon": "excel",
        "svg": "csv.svg",
        "color": "#107C41",
        "label": "Excel Loader",
    },
    "export_writer": {
        "icon": "export",
        "svg": "csv.svg",
        "color": "#0D9488",
        "label": "Data Export Writer",
    },
    "manual_data_entry": {
        "icon": "manual-entry",
        "svg": "csv.svg",
        "color": "#64748B",
        "label": "Manual Data Entry",
    },
    "api_response_endpoint": {
        "icon": "api",
        "svg": "api.svg",
        "color": "#6366F1",
        "label": "API Response Endpoint",
    },

    # ── Triggers & Orchestration ──
    "cron_trigger": {
        "icon": "clock",
        "svg": "clock.svg",
        "color": "#8B5CF6",
        "label": "Cron Schedule Trigger",
    },
    "webhook_trigger": {
        "icon": "webhook",
        "svg": "webhook.svg",
        "color": "#F43F5E",
        "label": "HTTP Webhook Trigger",
    },
    "email_trigger_imap": {
        "icon": "email",
        "svg": "api.svg",
        "color": "#0EA5E9",
        "label": "Email IMAP Trigger",
    },

    # ── Supervised & Unsupervised ML Trainers ──
    "xgboost_trainer": {
        "icon": "xgboost",
        "svg": "xgboost.svg",
        "color": "#FF6F00",
        "label": "XGBoost Classifier / Regressor",
    },
    "lightgbm_trainer": {
        "icon": "lightgbm",
        "svg": "lightgbm.svg",
        "color": "#02569B",
        "label": "LightGBM Trainer",
    },
    "catboost_trainer": {
        "icon": "catboost",
        "svg": "catboost.svg",
        "color": "#FFCC00",
        "label": "CatBoost Trainer",
    },
    "random_forest_trainer": {
        "icon": "scikit-learn",
        "svg": "scikit-learn.svg",
        "color": "#F7931E",
        "label": "Random Forest",
    },
    "linear_trainer": {
        "icon": "scikit-learn",
        "svg": "scikit-learn.svg",
        "color": "#2563EB",
        "label": "Linear Regression",
    },
    "logistic_regression_trainer": {
        "icon": "scikit-learn",
        "svg": "scikit-learn.svg",
        "color": "#3B82F6",
        "label": "Logistic Regression",
    },
    "neural_network_trainer": {
        "icon": "pytorch",
        "svg": "pytorch.svg",
        "color": "#EE4C2C",
        "label": "Neural Network (PyTorch)",
    },
    "automl_search": {
        "icon": "automl",
        "svg": "target.svg",
        "color": "#8B5CF6",
        "label": "AutoML Search",
    },
    "hyperparameter_tuner": {
        "icon": "tuning",
        "svg": "target.svg",
        "color": "#EC4899",
        "label": "Hyperparameter Tuner",
    },
    "ensemble_stacker": {
        "icon": "stack",
        "svg": "scikit-learn.svg",
        "color": "#10B981",
        "label": "Ensemble Stacker",
    },
    "model_calibration": {
        "icon": "calibration",
        "svg": "target.svg",
        "color": "#06B6D4",
        "label": "Probability Calibration",
    },

    # ── Forecasting & Time Series ──
    "prophet_forecaster": {
        "icon": "meta-prophet",
        "svg": "clock.svg",
        "color": "#1877F2",
        "label": "Prophet Forecaster",
    },
    "arima_forecaster": {
        "icon": "arima",
        "svg": "clock.svg",
        "color": "#6366F1",
        "label": "ARIMA / SARIMAX Forecaster",
    },
    "lag_feature_engineering": {
        "icon": "time-series",
        "svg": "clock.svg",
        "color": "#0284C7",
        "label": "Lag Feature Generator",
    },

    # ── Anomaly Detection & Guardrails ──
    "isolation_forest": {
        "icon": "isolation-forest",
        "svg": "scikit-learn.svg",
        "color": "#059669",
        "label": "Isolation Forest Anomaly Guard",
    },
    "statistical_guardrail": {
        "icon": "guardrail",
        "svg": "target.svg",
        "color": "#DC2626",
        "label": "Statistical Guardrail",
    },

    # ── Evaluation & Governance ──
    "model_evaluator": {
        "icon": "target",
        "svg": "target.svg",
        "color": "#E11D48",
        "label": "Model Evaluator & Metrics",
    },
    "ab_testing_framework": {
        "icon": "ab-testing",
        "svg": "target.svg",
        "color": "#7C3AED",
        "label": "A/B Testing Framework",
    },
    "bayesian_ab_estimator": {
        "icon": "bayesian",
        "svg": "target.svg",
        "color": "#4F46E5",
        "label": "Bayesian A/B Estimator",
    },
    "champion_challenger": {
        "icon": "champion-challenger",
        "svg": "target.svg",
        "color": "#EA580C",
        "label": "Champion-Challenger Router",
    },
    "bias_fairness_audit": {
        "icon": "fairness",
        "svg": "target.svg",
        "color": "#0D9488",
        "label": "Bias & Fairness Audit",
    },
    "mlflow_tracker": {
        "icon": "mlflow",
        "svg": "mlflow.svg",
        "color": "#0194E2",
        "label": "MLflow Experiment Tracking",
    },
    "shap_explainer": {
        "icon": "shap",
        "svg": "sparkles.svg",
        "color": "#DB2777",
        "label": "SHAP Explainability",
    },
    "model_explainability": {
        "icon": "explainability",
        "svg": "sparkles.svg",
        "color": "#DB2777",
        "label": "Model Explainability",
    },

    # ── Data Preparation & Splitting ──
    "train_test_split": {
        "icon": "scissors",
        "svg": "scissors.svg",
        "color": "#0D9488",
        "label": "Train/Test Split",
    },
    "stratified_split": {
        "icon": "scissors",
        "svg": "scissors.svg",
        "color": "#0D9488",
        "label": "Stratified Split",
    },
    "time_series_split": {
        "icon": "clock",
        "svg": "clock.svg",
        "color": "#0284C7",
        "label": "Time Series Split",
    },
    "walk_forward_split": {
        "icon": "clock",
        "svg": "clock.svg",
        "color": "#0284C7",
        "label": "Walk-Forward Split",
    },
    "missing_value_imputer": {
        "icon": "imputer",
        "svg": "sparkles.svg",
        "color": "#8B5CF6",
        "label": "Missing Value Imputer",
    },
    "outlier_handler": {
        "icon": "outlier",
        "svg": "sparkles.svg",
        "color": "#DC2626",
        "label": "Outlier Handler",
    },
    "duplicate_remover": {
        "icon": "dedup",
        "svg": "sparkles.svg",
        "color": "#6366F1",
        "label": "Duplicate Remover",
    },
    "feature_scaler": {
        "icon": "scaler",
        "svg": "sparkles.svg",
        "color": "#10B981",
        "label": "Feature Scaler",
    },
    "categorical_encoder": {
        "icon": "encoder",
        "svg": "sparkles.svg",
        "color": "#F59E0B",
        "label": "Categorical Encoder",
    },
    "feature_selector": {
        "icon": "selector",
        "svg": "sparkles.svg",
        "color": "#EC4899",
        "label": "Feature Selector",
    },
    "correlation_filter": {
        "icon": "correlation",
        "svg": "target.svg",
        "color": "#3B82F6",
        "label": "Correlation Filter",
    },
    "variance_filter": {
        "icon": "variance",
        "svg": "target.svg",
        "color": "#F97316",
        "label": "Variance Filter",
    },
    "column_selector": {
        "icon": "columns",
        "svg": "csv.svg",
        "color": "#475569",
        "label": "Column Selector",
    },
    "data_type_converter": {
        "icon": "convert",
        "svg": "sparkles.svg",
        "color": "#6366F1",
        "label": "Data Type Converter",
    },
    "dataset_join": {
        "icon": "join",
        "svg": "csv.svg",
        "color": "#0284C7",
        "label": "Dataset Join / Merge",
    },
    # ── Flow Control & Sub-Workflows ──
    "add_workflow": {
        "icon": "sub-workflow",
        "svg": "api.svg",
        "color": "#6366F1",
        "label": "Sub-Workflow Runner",
    },
    "workflow_start": {
        "icon": "workflow-start",
        "svg": "clock.svg",
        "color": "#10B981",
        "label": "Workflow Start Anchor",
    },
    "workflow_end": {
        "icon": "workflow-end",
        "svg": "target.svg",
        "color": "#EF4444",
        "label": "Workflow End Anchor",
    },
}

# Category fallback icon mapping
CATEGORY_FALLBACK_ICONS: Dict[str, Tuple[str, str, str]] = {
    "ingestion": ("database", "csv.svg", "#0284C7"),
    "triggers": ("zap", "webhook.svg", "#F43F5E"),
    "flow_control": ("sub-workflow", "api.svg", "#6366F1"),
    "preprocessing": ("sliders", "sparkles.svg", "#8B5CF6"),
    "feature_engineering": ("sparkles", "sparkles.svg", "#EC4899"),
    "splitting": ("scissors", "scissors.svg", "#0D9488"),
    "training": ("cpu", "scikit-learn.svg", "#7C3AED"),
    "evaluation": ("bar-chart", "target.svg", "#E11D48"),
    "anomaly": ("shield-alert", "target.svg", "#DC2626"),
    "forecasting": ("trending-up", "clock.svg", "#0284C7"),
}


def resolve_recipe_icon(
    recipe_id: str,
    category: Optional[str] = None,
    existing_icon: Optional[str] = None,
    existing_icon_url: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Resolves the official icon slug and icon_url for a recipe.
    Priority order:
    1. Explicit class icon and icon_url
    2. Registry entry in OFFICIAL_RECIPE_ICONS
    3. Category fallback
    4. Default generic icon
    """
    if existing_icon and existing_icon_url:
        return existing_icon, existing_icon_url

    if existing_icon:
        return existing_icon, f"/static/icons/recipes/{existing_icon}.svg"

    info = OFFICIAL_RECIPE_ICONS.get(recipe_id)
    if info:
        icon_slug = info["icon"]
        svg_file = info.get("svg", f"{icon_slug}.svg")
        return icon_slug, f"/static/icons/recipes/{svg_file}"

    if category:
        cat_info = CATEGORY_FALLBACK_ICONS.get(category.lower())
        if cat_info:
            icon_slug, svg_file, _ = cat_info
            return icon_slug, f"/static/icons/recipes/{svg_file}"

    return "default", "/static/icons/recipes/default.svg"
