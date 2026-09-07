import os
import tempfile
import joblib
from typing import Dict, Any, Optional
from backend.app.recipes.base.recipe import BaseRecipe

try:
    import mlflow
    from mlflow.exceptions import MlflowException
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def normalize_tracking_uri(uri: str) -> str:
    """Normalizes tracking URI to a valid MLflow scheme (especially on Windows)."""
    if not uri or uri == "./mlruns":
        return "sqlite:///mlflow.db"
    if uri.startswith(("http://", "https://", "sqlite://", "file://", "postgresql://", "mysql://")):
        return uri
    # If a Windows file path like C:\..., convert to sqlite file or file URI
    clean_path = uri.replace("\\", "/")
    if not clean_path.startswith("/"):
        clean_path = "/" + clean_path
    return f"sqlite:///{uri.replace('\\', '/')}/mlflow.db" if os.path.isdir(uri) else f"file://{clean_path}"


class MLflowTrackerRecipe(BaseRecipe):
    recipe_id = "mlflow_tracker"
    name = "MLflow Governance & Model Registry"
    version = "1.0.0"
    category = "governance"
    description = "Logs experiment parameters, evaluation metrics, and registers versioned models to the MLflow Model Registry (Tier-1 Governance Standard)."
    input_types = ["model", "metrics"]
    output_types = ["governance_record"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "experiment_name": {
                    "type": "string",
                    "title": "MLflow Experiment Name",
                    "default": "Enterprise_ML_Pipelines"
                },
                "registered_model_name": {
                    "type": "string",
                    "title": "Registered Model Name",
                    "default": "Production_Champion_Model",
                    "description": "The model entity name inside MLflow Model Registry."
                },
                "stage": {
                    "type": "string",
                    "title": "Promotion Stage",
                    "enum": ["Staging", "Production", "Archived", "None"],
                    "default": "Staging"
                },
                "tracking_uri": {
                    "type": "string",
                    "title": "MLflow Tracking URI",
                    "default": "sqlite:///mlflow.db",
                    "description": "SQLite database URI or remote MLflow server endpoint."
                }
            }
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        if not MLFLOW_AVAILABLE:
            raise ValueError("mlflow is not installed. Please run 'pip install mlflow'.")

        model = inputs.get("model")
        metrics = inputs.get("metrics") or {}

        exp_name = config.get("experiment_name", "Enterprise_ML_Pipelines")
        model_name = config.get("registered_model_name", "Production_Champion_Model")
        stage = config.get("stage", "Staging")
        raw_uri = config.get("tracking_uri", "sqlite:///mlflow.db")
        tracking_uri = normalize_tracking_uri(raw_uri)

        # Set tracking URI and experiment
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(exp_name)

        run_id = "unknown"
        with mlflow.start_run(run_name=f"run_{model_name}") as run:
            run_id = run.info.run_id

            # 1. Log Metrics
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    mlflow.log_metric(k, float(v))
                elif isinstance(v, str):
                    mlflow.log_param(k, v)

            # 2. Log Model Artifact
            if model is not None:
                try:
                    mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=model_name if stage != "None" else None)
                except Exception:
                    try:
                        mlflow.sklearn.log_model(model, artifact_path="model")
                    except Exception:
                        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
                            joblib.dump(model, tmp.name)
                            mlflow.log_artifact(tmp.name, artifact_path="model_artifact")

            mlflow.set_tag("governance_stage", stage)

        governance_record = {
            "mlflow_run_id": run_id,
            "experiment_name": exp_name,
            "registered_model_name": model_name,
            "stage": stage,
            "tracking_uri": tracking_uri,
            "metrics_logged": len(metrics)
        }

        return {
            "governance_record": governance_record,
            "metrics": metrics,
            "model": model
        }
