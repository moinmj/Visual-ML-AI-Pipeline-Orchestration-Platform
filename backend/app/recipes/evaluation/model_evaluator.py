from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score,
    classification_report, log_loss, balanced_accuracy_score,
    mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
)
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from backend.app.recipes.base.recipe import BaseRecipe


def _is_valid_calendar_datetime_series(series: Optional[pd.Series]) -> Tuple[bool, Optional[pd.Series]]:
    """
    Production-grade validation for temporal calendar series.
    Detects native datetime dtypes, ISO string dates, and valid epoch timestamps.
    Rejects degenerate numeric nanosecond Epoch artifacts (e.g. small integers mapped to 1970-01-01).
    """
    if series is None or len(series) == 0:
        return False, None

    # Case 1: Native datetime64 dtype
    if pd.api.types.is_datetime64_any_dtype(series):
        return True, series

    # Case 2: Numeric dtype (integers or floats)
    if pd.api.types.is_numeric_dtype(series):
        s_clean = series.dropna()
        if s_clean.empty:
            return False, None
        v_min = float(s_clean.min())
        # Only values >= 1e8 can represent valid Unix timestamps in seconds (e.g. 1.6e9 -> year 2020+)
        if v_min < 1e8:
            return False, None
        try:
            parsed = pd.to_datetime(s_clean, unit="s", errors="coerce")
            if parsed.notna().sum() > 0.5 * len(s_clean):
                return True, parsed
        except Exception:
            return False, None

    # Case 3: Object / String / Categorical series
    try:
        s_str = series.astype(str).str.strip()
        parsed = pd.to_datetime(s_str, dayfirst=True, errors="coerce")
        if parsed.notna().sum() <= 0.5 * len(s_str):
            parsed = pd.to_datetime(s_str, errors="coerce")

        valid_count = parsed.notna().sum()
        if valid_count > 0.5 * len(s_str):
            # Production Guard: Detect degenerate conversions where distinct input features collapse to 1 constant date
            formatted_dates = parsed.dt.strftime("%Y-%m-%d").dropna()
            if len(s_str) > 1 and s_str.nunique() > 1 and formatted_dates.nunique() == 1:
                return False, None
            return True, parsed
    except Exception:
        pass

    return False, None


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

        if (X_test is None or y_test is None) and "test_data" in inputs and isinstance(inputs["test_data"], dict):
            X_test = inputs["test_data"].get("X_test", X_test)
            y_test = inputs["test_data"].get("y_test", y_test)

        if model is None:
            raise ValueError("ModelEvaluator expects a trained 'model' in inputs. Please connect a Model Trainer node before this Evaluator.")
        if X_test is None or y_test is None:
            raise ValueError("ModelEvaluator expects 'X_test' and 'y_test' in inputs. Please ensure a Train/Test Split node is connected in the pipeline.")

        task_type = inputs.get("task_type") or (context.get("task_type") if isinstance(context, dict) else "classification") or "classification"
        avg_strat = config.get("average_strategy", "weighted")
        decision_threshold = float(config.get("decision_threshold", 0.5))
        cost_fp = float(config.get("cost_false_positive", 0.0))
        cost_fn = float(config.get("cost_false_negative", 0.0))

        # Graceful handler for Time-Series models (Prophet/ARIMA) connected into Evaluator.
        # The forecasting recipes themselves perform chronological out-of-sample backtesting
        # and embed the resulting accuracy metrics directly in their output.
        # We pass those through enriched with an evaluation summary.
        if type(model).__name__ == "Prophet" or "prophet" in str(type(model)).lower() or task_type == "time_series_forecasting":
            metrics = inputs.get("metrics") or inputs.get("forecasting_summary") or {
                "task_type": "time_series_forecasting",
                "algorithm": type(model).__name__
            }
            # Surface evaluation metadata so UI can distinguish OOS vs in-sample
            eval_type = metrics.get("eval_type", "unknown")
            eval_note = metrics.get("evaluation_note", "")
            if not eval_note:
                if eval_type == "out_of_sample_holdout":
                    train_sz = metrics.get("train_size", "?")
                    test_sz  = metrics.get("test_size", "?")
                    eval_note = (
                        f"Out-of-sample holdout evaluation: trained on {train_sz} points, "
                        f"evaluated on chronologically latest {test_sz} unseen observations."
                    )
                elif eval_type == "in_sample_fallback":
                    eval_note = "In-sample fallback: dataset was too small for a holdout split."
                else:
                    eval_note = "Evaluation metrics sourced from forecasting recipe output."
            metrics["evaluation_note"] = eval_note
            report_lines = [
                f"Algorithm : {metrics.get('algorithm', 'N/A')}",
                f"Eval Type : {eval_type}",
                f"MAE       : {metrics.get('mae', 'N/A')}",
                f"RMSE      : {metrics.get('rmse', 'N/A')}",
                f"MAPE      : {metrics.get('mape', 'N/A')}%",
            ]
            if "coverage_95pct" in metrics:
                report_lines.append(f"95% CI Cov: {metrics['coverage_95pct']}%")
            if "aic" in metrics:
                report_lines.append(f"AIC       : {metrics['aic']}")
            if "bic" in metrics:
                report_lines.append(f"BIC       : {metrics['bic']}")
            report_lines.append(f"Note      : {eval_note}")
            return {
                "metrics": metrics,
                "report":  "\n".join(report_lines)
            }

        # Capture split_mode from upstream TrainTestSplit recipe for reporting
        split_mode = inputs.get("split_mode") or (context.get("split_mode") if isinstance(context, dict) else None)

        # ── Capture the date x-axis for the trajectory chart ───────────────
        # Priority 0 (best): explicit test_dates list threaded from the splitter.
        #   The splitter has access to the full df_test *before* the Date column
        #   is dropped from the feature set, so it can reliably extract real dates.
        #   Using this avoids any heuristic column-name scanning in the evaluator.
        #
        # Priority 1 (fallback): scan X_test columns with STRICT matching only
        #   (exact match, prefix, or suffix on known date keywords — NOT substring).
        #   This prevents "Weekly_Sales_lag_1" from matching the keyword "week",
        #   which was the root cause of the "1970-01-01" epoch chart bug.
        #
        # Priority 2 (last resort): step indices ("Step 1", "Step 2", …).
        _temporal_snapshot: Optional[pd.Series] = None
        _temporal_snap_col: Optional[str] = None

        # Priority 0: explicit sidecar from splitter ────────────────────────
        _explicit_test_dates = inputs.get("test_dates") or (
            context.get("test_dates") if isinstance(context, dict) else None
        )
        _explicit_date_col = inputs.get("date_column_name") or (
            context.get("date_column_name") if isinstance(context, dict) else None
        )
        if _explicit_test_dates and len(_explicit_test_dates) > 0:
            _temporal_snapshot = pd.Series(_explicit_test_dates)
            _temporal_snap_col = _explicit_date_col or "Date"

        # Priority 1: strict column-name scan in X_test ─────────────────────
        elif isinstance(X_test, pd.DataFrame):
            # Strict keywords: a column qualifies only if the cleaned name IS one
            # of these words, or starts/ends with one separated by an underscore.
            # This deliberately excludes "Weekly_Sales_lag_1" (contains "week"
            # as an interior substring of "Weekly").
            _strict_kws = ["date", "timestamp", "period", "ds", "time", "datetime"]
            for _c in X_test.columns:
                _cl = _c.lower().strip()
                _matches = any(
                    _cl == kw
                    or _cl.startswith(kw + "_")
                    or _cl.endswith("_" + kw)
                    for kw in _strict_kws
                )
                if not _matches:
                    continue
                _is_valid, _parsed = _is_valid_calendar_datetime_series(X_test[_c])
                if _is_valid and _parsed is not None:
                    _temporal_snapshot = _parsed
                    _temporal_snap_col = _c
                    break
            # Also check dtype-detected datetime columns as a final fallback
            if _temporal_snapshot is None:
                for _c in X_test.columns:
                    if pd.api.types.is_datetime64_any_dtype(X_test[_c]):
                        _temporal_snapshot = X_test[_c]
                        _temporal_snap_col = _c
                        break

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

            # ── Build Actual vs Predicted trajectory.
            # Priority 1: use the pre-encoding date snapshot captured before safe_prepare_training_data
            # ran (it would have replaced e.g. "Date" with Date_year/month/day integers).
            # Priority 2: reconstruct from year/month/day integer columns in the post-encoded X_test.
            # Priority 3: fall back to step indices.
            trajectory = []
            time_col = None
            time_sort_key = None
            formatted_dates = None

            def _fmt_date_val(v) -> str:
                """Normalise any date-like value to a clean ISO-8601 string."""
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return "N/A"
                try:
                    ts = pd.Timestamp(v)
                    if pd.isna(ts):
                        return str(v)
                    return ts.strftime("%Y-%m-%d")
                except Exception:
                    return str(v)

            # ── Priority 1: pre-encoding snapshot ────────────────────────────────
            if _temporal_snapshot is not None and _temporal_snap_col is not None:
                time_col = _temporal_snap_col
                _is_valid, _parsed = _is_valid_calendar_datetime_series(_temporal_snapshot)
                if _is_valid and _parsed is not None:
                    formatted_dates = [_fmt_date_val(v) for v in _parsed]
                    time_sort_key = _parsed.values
                else:
                    formatted_dates = list(_temporal_snapshot.astype(str))
                    time_sort_key = list(range(len(formatted_dates)))

            # ── Priority 2: post-encoded year/month/day columns ──────────────────
            elif isinstance(X_test, pd.DataFrame):
                year_cols = [c for c in X_test.columns if "year" in c.lower()]
                month_cols = [c for c in X_test.columns if "month" in c.lower()]
                day_cols = [c for c in X_test.columns if "day" in c.lower()]
                candidates = [c for c in X_test.columns if any(k in c.lower() for k in ["date", "time", "timestamp", "period", "ds"])]

                if year_cols and month_cols:
                    time_col = year_cols[0]
                    y_s = pd.to_numeric(X_test[year_cols[0]], errors="coerce").fillna(2000).astype(int)
                    m_s = pd.to_numeric(X_test[month_cols[0]], errors="coerce").fillna(1).astype(int)
                    # Validate year values are real years (1990-2100), not ordinal row-numbers
                    if y_s.between(1990, 2100).mean() > 0.5:
                        if day_cols:
                            d_s = pd.to_numeric(X_test[day_cols[0]], errors="coerce").fillna(1).astype(int)
                            formatted_dates = [f"{y}-{m:02d}-{d:02d}" for y, m, d in zip(y_s, m_s, d_s)]
                            time_sort_key = [y * 10000 + m * 100 + d for y, m, d in zip(y_s, m_s, d_s)]
                        else:
                            formatted_dates = [f"{y}-{m:02d}" for y, m in zip(y_s, m_s)]
                            time_sort_key = [y * 100 + m for y, m in zip(y_s, m_s)]
                elif candidates:
                    for cand in candidates:
                        t_vals = X_test[cand]
                        _is_v, _parsed_dt = _is_valid_calendar_datetime_series(t_vals)
                        if _is_v and _parsed_dt is not None:
                            time_col = cand
                            formatted_dates = [_fmt_date_val(v) for v in _parsed_dt]
                            time_sort_key = _parsed_dt.values
                            break
                if formatted_dates is None and isinstance(X_test.index, pd.DatetimeIndex):
                    time_col = "__index__"
                    formatted_dates = [_fmt_date_val(v) for v in X_test.index]
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
                    # Strip internal __index__ sentinel before surfacing to UI
                    metrics["temporal_column"] = time_col if time_col != "__index__" else "index"
                    metrics["is_temporal"] = True

        if split_mode:
            metrics["split_mode"] = split_mode

        ret = {
            "metrics": metrics,
            "predictions_sample": [float(p) if isinstance(p, (np.floating, float)) else str(p) for p in predictions[:15]]
        }
        if model is not None:
            ret["model"] = model
        return ret
