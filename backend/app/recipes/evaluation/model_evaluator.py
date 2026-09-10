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
    version = "1.1.0"
    category = "evaluation"
    description = "Evaluates models with comprehensive performance reports, custom decision thresholds (sensitivity tuning), and business cost/loss impact metrics."
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
                },
                "decision_threshold": {
                    "type": "number",
                    "title": "Decision Probability Threshold",
                    "default": 0.5,
                    "minimum": 0.05,
                    "maximum": 0.95,
                    "description": "Business threshold for positive class (e.g. lower to 0.2 to catch more fraud/churn; raise to 0.8 to minimize false alarms)."
                },
                "cost_false_positive": {
                    "type": "number",
                    "title": "Business Cost per False Positive ($)",
                    "default": 0.0,
                    "minimum": 0.0,
                    "description": "Cost of a false alarm (e.g. customer support call or investigation cost)."
                },
                "cost_false_negative": {
                    "type": "number",
                    "title": "Business Cost per False Negative ($)",
                    "default": 0.0,
                    "minimum": 0.0,
                    "description": "Financial cost of a missed positive (e.g. unprevented fraud loss or customer churn value)."
                }
            }
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        model = inputs.get("model") or inputs.get("trained_model")
        if model is None and isinstance(context, dict):
            model = context.get("model") or context.get("trained_model")
        
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

        task_type = inputs.get("task_type") or (context.get("task_type") if isinstance(context, dict) else "classification") or "classification"
        avg_strat = config.get("average_strategy", "weighted")
        decision_threshold = float(config.get("decision_threshold", 0.5))
        cost_fp = float(config.get("cost_false_positive", 0.0))
        cost_fn = float(config.get("cost_false_negative", 0.0))

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

        # Ensure X_test non-numeric columns are safely encoded and aligned with model features
        if isinstance(X_test, pd.DataFrame):
            expected_features = None
            if hasattr(model, "feature_names_in_"):
                expected_features = list(model.feature_names_in_)
            elif isinstance(context, dict) and context.get("feature_names"):
                expected_features = context.get("feature_names")

            non_numeric = [c for c in X_test.columns if not pd.api.types.is_numeric_dtype(X_test[c])]
            if non_numeric:
                from backend.app.recipes.training.encoder_utils import safe_prepare_training_data
                X_train_ctx = context.get("X_train") if isinstance(context, dict) else None
                if X_train_ctx is not None and isinstance(X_train_ctx, pd.DataFrame):
                    _, X_test_prep = safe_prepare_training_data(X_train_ctx, X_test)
                    if X_test_prep is not None:
                        X_test = X_test_prep
                else:
                    X_test_prep, _ = safe_prepare_training_data(X_test, None)
                    if X_test_prep is not None:
                        X_test = X_test_prep

            if expected_features:
                X_test = X_test.reindex(columns=expected_features, fill_value=0)

        predictions = model.predict(X_test)
        metrics: Dict[str, Any] = {"task_type": task_type}

        if task_type == "classification":
            # Check if probability predictions available and apply custom decision threshold
            if hasattr(model, "predict_proba"):
                try:
                    probs = model.predict_proba(X_test)
                    if probs.shape[1] == 2 and decision_threshold != 0.5:
                        predictions = (probs[:, 1] >= decision_threshold).astype(int)

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

            # Align y_test types with predictions if needed
            try:
                if len(predictions) > 0 and len(y_test) > 0:
                    first_pred = predictions[0]
                    first_y = y_test.iloc[0] if hasattr(y_test, "iloc") else y_test[0]
                    if isinstance(first_pred, (int, np.integer)) and isinstance(first_y, str):
                        from sklearn.preprocessing import LabelEncoder
                        y_test = LabelEncoder().fit_transform(y_test)
            except Exception:
                pass

            acc = float(round(accuracy_score(y_test, predictions), 4))
            bal_acc = float(round(balanced_accuracy_score(y_test, predictions), 4))
            prec = float(round(precision_score(y_test, predictions, average=avg_strat, zero_division=0), 4))
            rec = float(round(recall_score(y_test, predictions, average=avg_strat, zero_division=0), 4))
            f1 = float(round(f1_score(y_test, predictions, average=avg_strat, zero_division=0), 4))
            cm = confusion_matrix(y_test, predictions).tolist()

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
                "decision_threshold_used": decision_threshold,
                "confusion_matrix": cm,
                "classification_report": clf_rep
            })

            # Calculate Business Dollar Impact if cost values provided
            if (cost_fp > 0 or cost_fn > 0) and len(cm) == 2 and len(cm[0]) == 2:
                tn, fp = cm[0][0], cm[0][1]
                fn, tp = cm[1][0], cm[1][1]
                total_business_cost = round((fp * cost_fp) + (fn * cost_fn), 2)
                metrics["business_loss_impact"] = {
                    "cost_per_false_positive": cost_fp,
                    "cost_per_false_negative": cost_fn,
                    "false_positive_count": fp,
                    "false_negative_count": fn,
                    "total_estimated_business_loss": total_business_cost
                }

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

            # Check for temporal / date / year column to generate chronological Actual vs Predicted trajectory
            trajectory = []
            time_col = None
            time_sort_key = None
            formatted_dates = None

            if isinstance(X_test, pd.DataFrame):
                year_cols = [c for c in X_test.columns if "year" in c.lower()]
                month_cols = [c for c in X_test.columns if "month" in c.lower()]
                day_cols = [c for c in X_test.columns if "day" in c.lower()]
                candidates = [c for c in X_test.columns if any(k in c.lower() for k in ["date", "time", "year", "timestamp", "period", "ds", "month"])]

                if year_cols and month_cols:
                    # Construct clean compound date representation and sort keys
                    time_col = year_cols[0]
                    y_s = pd.to_numeric(X_test[year_cols[0]], errors="coerce").fillna(2000).astype(int)
                    m_s = pd.to_numeric(X_test[month_cols[0]], errors="coerce").fillna(1).astype(int)
                    if day_cols:
                        d_s = pd.to_numeric(X_test[day_cols[0]], errors="coerce").fillna(1).astype(int)
                        formatted_dates = [f"{y}-{m:02d}-{d:02d}" for y, m, d in zip(y_s, m_s, d_s)]
                        time_sort_key = [y * 10000 + m * 100 + d for y, m, d in zip(y_s, m_s, d_s)]
                    else:
                        formatted_dates = [f"{y}-{m:02d}" for y, m in zip(y_s, m_s)]
                        time_sort_key = [y * 100 + m for y, m in zip(y_s, m_s)]
                elif candidates:
                    time_col = candidates[0]
                    t_vals = X_test[time_col]
                    formatted_dates = [str(v) for v in t_vals.values]
                    # Attempt robust sorting key
                    try:
                        parsed_dt = pd.to_datetime(t_vals, errors="coerce")
                        if parsed_dt.notna().sum() > len(parsed_dt) * 0.5:
                            time_sort_key = parsed_dt.values
                        else:
                            time_sort_key = pd.to_numeric(t_vals, errors="coerce").fillna(0).values
                    except Exception:
                        time_sort_key = list(range(len(t_vals)))
                elif isinstance(X_test.index, pd.DatetimeIndex):
                    time_col = "__index__"
                    formatted_dates = [str(v) for v in X_test.index.values]
                    time_sort_key = X_test.index.values

            y_actuals = y_test.values if hasattr(y_test, "values") else list(y_test)
            preds_list = predictions.values if hasattr(predictions, "values") else list(predictions)

            if len(y_actuals) > 0:
                if not formatted_dates:
                    formatted_dates = [f"Step {i+1}" for i in range(len(y_actuals))]
                    time_sort_key = list(range(len(y_actuals)))

                # Sort chronologically if sort keys exist
                if time_sort_key is not None and len(time_sort_key) == len(y_actuals):
                    order = np.argsort(time_sort_key)
                    sorted_dates = [formatted_dates[i] for i in order]
                    sorted_actuals = [y_actuals[i] for i in order]
                    sorted_preds = [preds_list[i] for i in order]
                    combined = list(zip(sorted_dates, sorted_actuals, sorted_preds))
                else:
                    combined = list(zip(formatted_dates, y_actuals, preds_list))

                step = max(1, len(combined) // 200)
                sampled = combined[::step]

                for ds_val, act_val, pred_val in sampled:
                    act_f = float(act_val) if pd.notna(act_val) else 0.0
                    pred_f = float(round(float(pred_val), 4)) if pd.notna(pred_val) else 0.0
                    trajectory.append({
                        "ds": str(ds_val),
                        "actual": act_f,
                        "yhat": pred_f,
                        "residual": float(round(act_f - pred_f, 4))
                    })

                metrics["trajectory"] = trajectory
                metrics["actual_vs_predicted_time_series"] = trajectory
                if time_col:
                    metrics["temporal_column"] = time_col
                    metrics["is_temporal"] = True

        ret = {
            "metrics": metrics,
            "predictions_sample": [float(p) if isinstance(p, (np.floating, float)) else str(p) for p in predictions[:15]]
        }
        if model is not None:
            ret["model"] = model
        return ret
