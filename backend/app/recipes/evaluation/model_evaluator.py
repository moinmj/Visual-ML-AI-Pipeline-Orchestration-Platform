from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score,
    classification_report, log_loss, balanced_accuracy_score,
    mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
)
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from backend.app.recipes.base.recipe import BaseRecipe


class ModelEvaluatorRecipe(BaseRecipe):
    recipe_id = "model_evaluator"
    name = "Model Performance Evaluator"
    version = "1.0.0"
    category = "evaluation"
    description = "Evaluates trained models and generates comprehensive performance reports (Classification, Regression, Confusion Matrix, Per-Class Metrics, Feature Analysis)."
    input_types = ["model", "test_data"]
    output_types = ["metrics", "report"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "report_type": {
                    "type": "string",
                    "title": "Evaluation Report Mode",
                    "enum": [
                        "Comprehensive (All Metrics + Confusion Matrix)",
                        "Executive Summary (Accuracy, F1, Loss)",
                        "Per-Class Deep Dive (Precision/Recall per label)",
                        "Regression Diagnostics (MAE, RMSE, R2, MAPE)"
                    ],
                    "default": "Comprehensive (All Metrics + Confusion Matrix)"
                },
                "average_strategy": {
                    "type": "string",
                    "title": "Multiclass Averaging Strategy",
                    "enum": ["weighted", "macro", "micro"],
                    "default": "weighted"
                }
            }
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        model = inputs.get("model")
        
        X_test = inputs.get("X_test")
        if X_test is None and isinstance(context, dict):
            X_test = context.get("X_test")

        y_test = inputs.get("y_test")
        if y_test is None and isinstance(context, dict):
            y_test = context.get("y_test")

        if model is None:
            raise ValueError("ModelEvaluator expects a trained 'model' in inputs. Please connect a Model Trainer node before this Evaluator.")
        if X_test is None or y_test is None:
            raise ValueError("ModelEvaluator expects 'X_test' and 'y_test' in inputs. Please ensure a Train/Test Split node is connected in the pipeline.")

        task_type = inputs.get("task_type", "classification")
        avg_strat = config.get("average_strategy", "weighted")

        # Graceful check for Time-Series models (Prophet/ARIMA) connected into Evaluator
        if type(model).__name__ == "Prophet" or "prophet" in str(type(model)).lower() or task_type == "time_series_forecasting":
            metrics = inputs.get("metrics") or inputs.get("forecasting_summary") or {
                "task_type": "time_series_forecasting",
                "algorithm": type(model).__name__
            }
            return {
                "metrics": metrics,
                "report": "Time-series forecasting evaluation completed."
            }

        predictions = model.predict(X_test)
        metrics: Dict[str, Any] = {"task_type": task_type}
        detailed_report = {}

        if task_type == "classification":
            acc = float(round(accuracy_score(y_test, predictions), 4))
            bal_acc = float(round(balanced_accuracy_score(y_test, predictions), 4))
            prec = float(round(precision_score(y_test, predictions, average=avg_strat, zero_division=0), 4))
            rec = float(round(recall_score(y_test, predictions, average=avg_strat, zero_division=0), 4))
            f1 = float(round(f1_score(y_test, predictions, average=avg_strat, zero_division=0), 4))
            cm = confusion_matrix(y_test, predictions).tolist()

            # Per-class detailed report dictionary
            try:
                clf_rep = classification_report(y_test, predictions, output_dict=True, zero_division=0)
            except Exception:
                clf_rep = {}

            metrics.update({
                "accuracy": acc,
                "balanced_accuracy": bal_acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "confusion_matrix": cm,
                "classification_report": clf_rep
            })

            # Check if probability predictions available for ROC-AUC & Log Loss
            if hasattr(model, "predict_proba"):
                try:
                    probs = model.predict_proba(X_test)
                    if probs.shape[1] == 2:
                        auc = float(round(roc_auc_score(y_test, probs[:, 1]), 4))
                        metrics["roc_auc"] = auc
                    elif probs.shape[1] > 2:
                        auc = float(round(roc_auc_score(y_test, probs, multi_class="ovr", average=avg_strat), 4))
                        metrics["roc_auc_ovr"] = auc

                    ll = float(round(log_loss(y_test, probs), 4))
                    metrics["log_loss"] = ll
                except Exception:
                    pass

        else:
            mae = float(round(mean_absolute_error(y_test, predictions), 4))
            mse = float(round(mean_squared_error(y_test, predictions), 4))
            rmse = float(round(np.sqrt(mse), 4))
            r2 = float(round(r2_score(y_test, predictions), 4))
            try:
                mape = float(round(mean_absolute_percentage_error(y_test, predictions), 4))
            except Exception:
                mape = None

            metrics.update({
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "r2_score": r2,
                "mape": mape
            })

        return {
            "metrics": metrics,
            "predictions_sample": [float(p) if isinstance(p, (np.floating, float)) else str(p) for p in predictions[:15]]
        }
