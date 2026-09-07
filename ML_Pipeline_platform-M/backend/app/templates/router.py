from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

router = APIRouter(prefix="/templates", tags=["Pipeline Templates & Blueprints"])


TEMPLATES_CATALOG = {
    "ml_supervised": {
        "id": "ml_supervised",
        "name": "Supervised ML Classification / Regression",
        "category": "Machine Learning",
        "description": "End-to-end supervised learning blueprint with median imputation, standard scaling, train/test splitting, XGBoost boosting, and evaluation reports.",
        "icon": "⚡",
        "node_count": 6,
        "dag": {
            "nodes": [
                {"id": "node_csv", "recipe_id": "csv_loader", "label": "📄 Data Ingestion", "position": {"x": 40, "y": 100}, "config": {}},
                {"id": "node_impute", "recipe_id": "missing_value_imputer", "label": "🧹 Imputer (Median)", "position": {"x": 280, "y": 100}, "config": {"strategy": "median"}},
                {"id": "node_scale", "recipe_id": "feature_scaler", "label": "⚖️ Feature Scaler", "position": {"x": 520, "y": 100}, "config": {"method": "standard"}},
                {"id": "node_split", "recipe_id": "train_test_split", "label": "✂️ Train/Test Split", "position": {"x": 760, "y": 100}, "config": {"target_column": "target", "test_size": 0.2}},
                {"id": "node_xgb", "recipe_id": "xgboost_trainer", "label": "⚡ XGBoost Classifier", "position": {"x": 1000, "y": 50}, "config": {"task_type": "classification", "n_estimators": 100, "max_depth": 6}},
                {"id": "node_eval", "recipe_id": "classification_evaluator", "label": "🎯 Model Evaluator", "position": {"x": 1240, "y": 100}, "config": {"report_type": "Comprehensive"}}
            ],
            "edges": [
                {"id": "e1", "source": "node_csv", "target": "node_impute", "animated": True},
                {"id": "e2", "source": "node_impute", "target": "node_scale", "animated": True},
                {"id": "e3", "source": "node_scale", "target": "node_split", "animated": True},
                {"id": "e4", "source": "node_split", "target": "node_xgb", "animated": True},
                {"id": "e5", "source": "node_split", "target": "node_eval", "animated": True},
                {"id": "e6", "source": "node_xgb", "target": "node_eval", "animated": True}
            ],
            "node_configs": {
                "node_csv": {"recipe_id": "csv_loader", "label": "Dataset Ingestion", "config": {}},
                "node_impute": {"recipe_id": "missing_value_imputer", "label": "Imputer", "config": {"strategy": "median"}},
                "node_scale": {"recipe_id": "feature_scaler", "label": "Scaler", "config": {"method": "standard"}},
                "node_split": {"recipe_id": "train_test_split", "label": "Splitter", "config": {"target_column": "target", "test_size": 0.2}},
                "node_xgb": {"recipe_id": "xgboost_trainer", "label": "XGBoost", "config": {"task_type": "classification", "n_estimators": 100, "max_depth": 6}},
                "node_eval": {"recipe_id": "classification_evaluator", "label": "Evaluator", "config": {"report_type": "Comprehensive"}}
            }
        }
    },
    "time_series_forecasting": {
        "id": "time_series_forecasting",
        "name": "Time-Series Seasonality & Trend Forecasting",
        "category": "Time-Series",
        "description": "Chronological time-series forecasting blueprint using forward-fill imputation and Meta Prophet with 30-period future confidence intervals.",
        "icon": "🔮",
        "node_count": 3,
        "dag": {
            "nodes": [
                {"id": "node_csv", "recipe_id": "csv_loader", "label": "📄 Time-Series Data", "position": {"x": 40, "y": 100}, "config": {}},
                {"id": "node_impute", "recipe_id": "missing_value_imputer", "label": "🧹 Time Imputer (ffill)", "position": {"x": 300, "y": 100}, "config": {"strategy": "ffill"}},
                {"id": "node_prophet", "recipe_id": "prophet_forecaster", "label": "🔮 Prophet Forecaster", "position": {"x": 560, "y": 100}, "config": {"date_column": "Date", "target_column": "Sales", "horizon_periods": 30}}
            ],
            "edges": [
                {"id": "e1", "source": "node_csv", "target": "node_impute", "animated": True},
                {"id": "e2", "source": "node_impute", "target": "node_prophet", "animated": True}
            ],
            "node_configs": {
                "node_csv": {"recipe_id": "csv_loader", "label": "Dataset Ingestion", "config": {}},
                "node_impute": {"recipe_id": "missing_value_imputer", "label": "Imputer", "config": {"strategy": "ffill"}},
                "node_prophet": {"recipe_id": "prophet_forecaster", "label": "Prophet Forecaster", "config": {"date_column": "Date", "target_column": "Sales", "horizon_periods": 30}}
            }
        }
    },
    "anomaly_detection": {
        "id": "anomaly_detection",
        "name": "Unsupervised Anomaly & Outlier Detection",
        "category": "Anomaly Detection",
        "description": "Unsupervised outlier isolation blueprint leveraging Scikit-Learn Isolation Forest with multi-dimensional partition clustering.",
        "icon": "🚨",
        "node_count": 3,
        "dag": {
            "nodes": [
                {"id": "node_csv", "recipe_id": "csv_loader", "label": "📄 Transaction Ingestion", "position": {"x": 40, "y": 100}, "config": {}},
                {"id": "node_impute", "recipe_id": "missing_value_imputer", "label": "🧹 Imputer (Median)", "position": {"x": 300, "y": 100}, "config": {"strategy": "median"}},
                {"id": "node_iso", "recipe_id": "isolation_forest", "label": "🌲 Isolation Forest", "position": {"x": 560, "y": 100}, "config": {"contamination": 0.05, "n_estimators": 100}}
            ],
            "edges": [
                {"id": "e1", "source": "node_csv", "target": "node_impute", "animated": True},
                {"id": "e2", "source": "node_impute", "target": "node_iso", "animated": True}
            ],
            "node_configs": {
                "node_csv": {"recipe_id": "csv_loader", "label": "Dataset Ingestion", "config": {}},
                "node_impute": {"recipe_id": "missing_value_imputer", "label": "Imputer", "config": {"strategy": "median"}},
                "node_iso": {"recipe_id": "isolation_forest", "label": "Isolation Forest", "config": {"contamination": 0.05, "n_estimators": 100}}
            }
        }
    },
    "enterprise_governance": {
        "id": "enterprise_governance",
        "name": "Enterprise CatBoost with Model Governance Card",
        "category": "Governance & Compliance",
        "description": "Full audit-ready pipeline with categorical encoding, CatBoost classifier, comprehensive metric diagnostics, and exportable Model Governance Card.",
        "icon": "🛡️",
        "node_count": 7,
        "dag": {
            "nodes": [
                {"id": "node_csv", "recipe_id": "csv_loader", "label": "📄 Enterprise Data", "position": {"x": 40, "y": 100}, "config": {}},
                {"id": "node_impute", "recipe_id": "missing_value_imputer", "label": "🧹 Imputer", "position": {"x": 260, "y": 100}, "config": {"strategy": "median"}},
                {"id": "node_encode", "recipe_id": "categorical_encoder", "label": "🔤 Cat Encoder", "position": {"x": 480, "y": 100}, "config": {"method": "one_hot"}},
                {"id": "node_split", "recipe_id": "train_test_split", "label": "✂️ Splitter", "position": {"x": 700, "y": 100}, "config": {"target_column": "target", "test_size": 0.2}},
                {"id": "node_cb", "recipe_id": "catboost_trainer", "label": "🐱 CatBoost Classifier", "position": {"x": 920, "y": 50}, "config": {"task_type": "classification", "iterations": 100}},
                {"id": "node_eval", "recipe_id": "classification_evaluator", "label": "🎯 Evaluator", "position": {"x": 1140, "y": 100}, "config": {"report_type": "Comprehensive"}},
                {"id": "node_gov", "recipe_id": "model_governance_card", "label": "🛡️ Governance Card", "position": {"x": 1360, "y": 100}, "config": {"author": "AI Engineering Team", "organization": "Enterprise AI", "version": "1.0.0"}}
            ],
            "edges": [
                {"id": "e1", "source": "node_csv", "target": "node_impute", "animated": True},
                {"id": "e2", "source": "node_impute", "target": "node_encode", "animated": True},
                {"id": "e3", "source": "node_encode", "target": "node_split", "animated": True},
                {"id": "e4", "source": "node_split", "target": "node_cb", "animated": True},
                {"id": "e5", "source": "node_split", "target": "node_eval", "animated": True},
                {"id": "e6", "source": "node_cb", "target": "node_eval", "animated": True},
                {"id": "e7", "source": "node_eval", "target": "node_gov", "animated": True}
            ],
            "node_configs": {
                "node_csv": {"recipe_id": "csv_loader", "label": "Data Ingestion", "config": {}},
                "node_impute": {"recipe_id": "missing_value_imputer", "label": "Imputer", "config": {"strategy": "median"}},
                "node_encode": {"recipe_id": "categorical_encoder", "label": "Encoder", "config": {"method": "one_hot"}},
                "node_split": {"recipe_id": "train_test_split", "label": "Splitter", "config": {"target_column": "target", "test_size": 0.2}},
                "node_cb": {"recipe_id": "catboost_trainer", "label": "CatBoost", "config": {"task_type": "classification", "iterations": 100}},
                "node_eval": {"recipe_id": "classification_evaluator", "label": "Evaluator", "config": {"report_type": "Comprehensive"}},
                "node_gov": {"recipe_id": "model_governance_card", "label": "Governance Card", "config": {"author": "AI Engineering Team", "organization": "Enterprise AI", "version": "1.0.0"}}
            }
        }
    }
}


@router.get("/")
async def list_templates() -> List[Dict[str, Any]]:
    """
    List all available pre-configured visual pipeline blueprints.
    """
    return [
        {
            "id": t["id"],
            "name": t["name"],
            "category": t["category"],
            "description": t["description"],
            "icon": t["icon"],
            "node_count": t["node_count"]
        }
        for t in TEMPLATES_CATALOG.values()
    ]


@router.get("/{template_id}")
async def get_template(template_id: str) -> Dict[str, Any]:
    """
    Retrieve full ready-to-render DAG and exact configuration for a given blueprint ID.
    """
    normalized_id = template_id.lower().replace("-", "_")
    # Alias mappings for frontend convenience
    if normalized_id in ["ml", "ml_template", "supervised", "classification", "regression"]:
        normalized_id = "ml_supervised"
    elif normalized_id in ["forecast", "forecasting", "prophet", "ts", "timeseries"]:
        normalized_id = "time_series_forecasting"
    elif normalized_id in ["anomaly", "outlier", "isolation"]:
        normalized_id = "anomaly_detection"
    elif normalized_id in ["governance", "audit", "compliance"]:
        normalized_id = "enterprise_governance"

    template = TEMPLATES_CATALOG.get(normalized_id)
    if not template:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Template blueprint '{template_id}' not found. Available: {list(TEMPLATES_CATALOG.keys())}"
        )
    return template
