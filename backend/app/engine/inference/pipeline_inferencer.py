import time
import json
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
import httpx

from backend.app.core.config import settings
from backend.app.engine.inference.schemas import (
    PredictionRequest,
    PredictionResponse,
    NativeCadenceDatasetRequest,
    NativeCadenceDatasetResponse,
    CombinedDatasetRequest,
    CombinedDatasetResponse
)
from backend.app.engine.inference.explainability import compute_waterfall_breakdown


class PipelineInferencer:
    @classmethod
    def _build_dataset_preview(
        cls,
        rows: List[Dict[str, Any]],
        target_col: Optional[str],
        target_values: List[Any],
        feature_trend_basis: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Reshapes an already-computed prediction (single row, or a full trajectory) into
        the original dataset's own column layout, labeling only the columns that are
        actual model output — the target, and any feature that was drifted forward from
        history — as "<column> (Predicted)". Columns the caller explicitly typed, or that
        stayed frozen (e.g. Holiday_Flag, Store), are left unlabeled since they are not
        model-generated values.
        """
        basis = feature_trend_basis or {}
        preview: List[Dict[str, Any]] = []
        for row, target_val in zip(rows, target_values):
            out_row: Dict[str, Any] = {}
            for col, val in row.items():
                is_projected = basis.get(col, {}).get("source") == "projected_from_history"
                out_row[f"{col} (Predicted)" if is_projected else col] = val
            if target_col:
                out_row[f"{target_col} (Predicted)"] = target_val
            preview.append(out_row)
        return preview

    @classmethod
    def predict(cls, bundle: Dict[str, Any], request: PredictionRequest) -> PredictionResponse:
        """
        Dispatches incoming prediction request to the specialized task handler
        based on the pipeline's trained model and task type.
        """
        start_t = time.time()
        exec_id = bundle.get("execution_id", "unknown")
        model = bundle.get("model")
        task_type = bundle.get("task_type")

        # Auto-detect task type from model attributes or summaries if ambiguous
        if not task_type or task_type == "classification":
            if bundle.get("forecasting_summary") or hasattr(model, "make_future_dataframe") or hasattr(model, "get_forecast"):
                task_type = "time_series_forecasting"
            elif bundle.get("anomaly_summary") or (model is not None and type(model).__name__ in ["IsolationForest", "EllipticEnvelope"]):
                task_type = "anomaly_detection"
            else:
                task_type = bundle.get("task_type", "classification")

        inferred_inputs = None
        nl_query = request.natural_language_query
        if nl_query:
            try:
                parsed_nl = cls._parse_natural_language_query(bundle, nl_query)
                overrides = parsed_nl.get("feature_overrides", {})
                inferred_inputs = overrides
                base_inputs = dict(bundle.get("last_historical_row") or bundle.get("sample_row", {}))
                base_inputs.update(overrides)
                request.inputs = base_inputs

                if parsed_nl.get("target_year") and not request.target_year:
                    request.target_year = parsed_nl["target_year"]
                if parsed_nl.get("future_periods") and not request.future_periods:
                    request.future_periods = parsed_nl["future_periods"]
                if parsed_nl.get("requested_metric") and not request.requested_metric:
                    request.requested_metric = parsed_nl["requested_metric"]

                # Automatically calculate time-series forecast_horizon from query if not explicitly passed!
                if not request.forecast_horizon:
                    # Detect the END of the training/historical data (last non-future point)
                    # and the end of the current forecast window
                    hist_max_yr  = None
                    fc_max_yr    = None
                    max_yr       = bundle.get("max_year")

                    fc_sum   = bundle.get("forecasting_summary", {})
                    fc_data  = fc_sum.get("forecast_data", [])
                    if fc_data:
                        try:
                            fc_max_yr = pd.to_datetime(fc_data[-1]["ds"]).year
                            # Last historical (non-future) record
                            hist_pts = [r for r in fc_data if not r.get("is_future", 1)]
                            if hist_pts:
                                hist_max_yr = pd.to_datetime(hist_pts[-1]["ds"]).year
                            else:
                                hist_max_yr = pd.to_datetime(fc_data[0]["ds"]).year
                        except Exception:
                            fc_max_yr   = 2020
                            hist_max_yr = 2020

                    # Use the training data end (hist_max_yr) as the projection baseline
                    base_yr = max_yr or hist_max_yr or 2020

                    t_yr   = parsed_nl.get("target_year")
                    f_per  = parsed_nl.get("future_periods")
                    t_unit = (parsed_nl.get("time_unit") or "").lower()

                    if t_yr:
                        # +1 ensures the full TARGET year is included (not just the boundary)
                        years_ahead = max(1, t_yr - base_yr + 1)
                        request.forecast_horizon = int(years_ahead * 365)
                    elif f_per:
                        q_low = nl_query.lower()
                        if "year" in t_unit or "year" in q_low:
                            request.forecast_horizon = int(f_per * 365)
                        elif "month" in t_unit or "month" in q_low:
                            request.forecast_horizon = int(f_per * 30)
                        elif "week" in t_unit or "week" in q_low:
                            request.forecast_horizon = int(f_per * 7)
                        else:
                            request.forecast_horizon = int(f_per)

            except Exception as e:
                logger.warning(f"Failed to parse natural language query: {str(e)}")

        try:
            # 1. TIME-SERIES FORECASTING
            if task_type == "time_series_forecasting" or "forecaster" in task_type:
                res = cls._predict_forecasting(bundle, request)
                if nl_query:
                    res.ai_explanation = cls._generate_ai_explanation(
                        query=nl_query, task_type=task_type, prediction=res.prediction or res.projected_end_value,
                        features_used=inferred_inputs or {}, trend=res.trend,
                        target_column=bundle.get("target_column"),
                        series_summary=res.series_summary
                    )
                    res.inferred_inputs = inferred_inputs
                res.inference_latency_ms = round((time.time() - start_t) * 1000.0, 2)
                return res

            # 2. TABULAR FUTURE HORIZON PROJECTION (e.g. predicting future year with XGBoost/Regressor)
            temporal_col = bundle.get("temporal_column")
            if not temporal_col:
                for fn in bundle.get("feature_names", []):
                    if any(k in fn.lower() for k in ["year", "date", "period", "timestamp"]):
                        temporal_col = fn
                        break

            unrecognized_features = []
            if nl_query:
                valid_features = set(bundle.get("feature_names", [])) | set(bundle.get("sample_row", {}).keys())
                if inferred_inputs:
                    for k in list(inferred_inputs.keys()):
                        if not any(k.lower() == f.lower() for f in valid_features):
                            unrecognized_features.append(k)
                # Check prompt tokens for unmapped domain terms like 'rain'
                q_words = [w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', nl_query)]
                stopwords = {"predict", "for", "with", "high", "low", "medium", "forecast", "next", "years", "year", "months", "days", "trajectory", "trend", "show", "what", "will", "the", "and", "scenario", "baseline"}
                for w in q_words:
                    if w not in stopwords and not any(w in f.lower() for f in valid_features):
                        if w not in [u.lower() for u in unrecognized_features]:
                            unrecognized_features.append(w)

            if (request.future_periods or request.target_year) and temporal_col and task_type in ["regression", "classification"]:
                inputs_dict = dict(bundle.get("last_historical_row") or bundle.get("sample_row", {}))
                if isinstance(request.inputs, dict):
                    inputs_dict.update(request.inputs)
                res = cls._predict_tabular_future_projection(
                    bundle=bundle,
                    base_inputs=inputs_dict,
                    temporal_col=temporal_col,
                    future_periods=request.future_periods or 5,
                    target_year=request.target_year,
                    future_feature_overrides=request.future_feature_overrides
                )
                if nl_query:
                    res.ai_explanation = cls._generate_ai_explanation(
                        query=nl_query, task_type=task_type, prediction=res.prediction,
                        features_used=inferred_inputs or {}, trend=res.trend,
                        target_column=bundle.get("target_column"),
                        series_summary=res.series_summary,
                        waterfall_summary=res.waterfall_summary
                    )
                    res.inferred_inputs = inferred_inputs
                    if unrecognized_features:
                        res.unrecognized_features = unrecognized_features
                        res.ai_explanation += f"\n\nNote: The query referenced '{', '.join(unrecognized_features)}', which is not a feature in this dataset (Trained features: {', '.join(bundle.get('feature_names', []))}). Prediction pertains to trained target '{bundle.get('target_column')}'."
                res.inference_latency_ms = round((time.time() - start_t) * 1000.0, 2)
                return res

            # 3. ANOMALY DETECTION
            if task_type == "anomaly_detection":
                res = cls._predict_anomaly(bundle, request)
                if nl_query:
                    res.ai_explanation = cls._generate_ai_explanation(
                        query=nl_query, task_type=task_type, prediction=res.verdict,
                        features_used=inferred_inputs or {},
                        target_column=bundle.get("target_column")
                    )
                    res.inferred_inputs = inferred_inputs
                res.inference_latency_ms = round((time.time() - start_t) * 1000.0, 2)
                return res

            # 4. NLP TEXT CLASSIFICATION
            if request.text_input is not None and bundle.get("vectorizer") is not None:
                res = cls._predict_nlp(bundle, request.text_input)
                if nl_query:
                    res.ai_explanation = cls._generate_ai_explanation(
                        query=nl_query, task_type=task_type, prediction=res.prediction,
                        features_used={"text_input": request.text_input},
                        target_column=bundle.get("target_column")
                    )
                    res.inferred_inputs = inferred_inputs
                res.inference_latency_ms = round((time.time() - start_t) * 1000.0, 2)
                return res

            # 5. TABULAR BATCH SCORING
            if isinstance(request.inputs, list):
                res = cls._predict_batch(bundle, request.inputs)
                res.inference_latency_ms = round((time.time() - start_t) * 1000.0, 2)
                return res

            # 6. SINGLE-RECORD TABULAR CLASSIFICATION / REGRESSION
            inputs_dict = dict(bundle.get("last_historical_row") or bundle.get("sample_row", {}) or {})
            if isinstance(request.inputs, dict):
                inputs_dict.update(request.inputs)
            res = cls._predict_tabular_single(bundle, inputs_dict)
            if nl_query:
                res.ai_explanation = cls._generate_ai_explanation(
                    query=nl_query, task_type=task_type, prediction=res.prediction,
                    features_used=inferred_inputs or {},
                    target_column=bundle.get("target_column"),
                    waterfall_summary=res.waterfall_summary
                )
                res.inferred_inputs = inferred_inputs
            res.inference_latency_ms = round((time.time() - start_t) * 1000.0, 2)
            return res

        except Exception as e:
            logger.error(f"Inference failed for execution {exec_id}: {str(e)}")
            latency = round((time.time() - start_t) * 1000.0, 2)
            return PredictionResponse(
                status="FAILED",
                task_type=task_type,
                execution_id=exec_id,
                inference_latency_ms=latency,
                error_message=str(e)
            )

    # -------------------------------------------------------------
    # TABULAR PREPROCESSING TRANSFORMER
    # -------------------------------------------------------------
    @classmethod
    def _preprocess_inputs(cls, bundle: Dict[str, Any], raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies pipeline preprocessing transformations (imputer, encoder, scaler, vectorizer)
        to new unseen input records, aligning columns exactly with model feature names.
        """
        df = raw_df.copy()
        feature_names = bundle.get("feature_names", [])

        # 1. Missing Value Imputation using recorded training stats
        imputer_stats = bundle.get("imputer_stats", {})
        if imputer_stats:
            for col, val in imputer_stats.items():
                if col in df.columns:
                    df[col] = df[col].fillna(val)

        # 2. Text Vectorization if text column exists in inputs
        vectorizer = bundle.get("vectorizer")
        target_text_col = bundle.get("text_column")
        if vectorizer is not None and target_text_col and target_text_col in df.columns:
            corpus = df[target_text_col].fillna("").astype(str).tolist()
            try:
                vec_mat = vectorizer.transform(corpus)
                vec_cols = [f"tfidf_{f}" for f in vectorizer.get_feature_names_out()]
                vec_df = pd.DataFrame(vec_mat.toarray(), columns=vec_cols, index=df.index)
                df = pd.concat([df.drop(columns=[target_text_col]), vec_df], axis=1)
            except Exception as e:
                logger.warning(f"Vectorization transform warning: {str(e)}")

        # 3. Categorical Encoders
        categorical_maps = bundle.get("categorical_maps", {})
        if categorical_maps:
            for col, mapping in categorical_maps.items():
                if col in df.columns and isinstance(mapping, dict):
                    default_m = mapping.get("__default__", 0)
                    df[col] = df[col].map(lambda v: mapping.get(v, default_m))

        # 4. Feature Scaler
        scaler = bundle.get("scaler")
        if scaler is not None:
            try:
                if hasattr(scaler, "feature_names_in_"):
                    fn_in = list(scaler.feature_names_in_)
                    # Try per-feature transformation based on fitted parameters
                    if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
                        # StandardScaler
                        for i, c in enumerate(fn_in):
                            if c in df.columns and i < len(scaler.mean_) and i < len(scaler.scale_):
                                s_val = scaler.scale_[i] if scaler.scale_[i] != 0 else 1.0
                                df[c] = (df[c].astype(float) - scaler.mean_[i]) / s_val
                    elif hasattr(scaler, "data_min_") and hasattr(scaler, "data_range_"):
                        # MinMaxScaler
                        for i, c in enumerate(fn_in):
                            if c in df.columns and i < len(scaler.data_min_) and i < len(scaler.data_range_):
                                rng = scaler.data_range_[i] if scaler.data_range_[i] != 0 else 1.0
                                df[c] = (df[c].astype(float) - scaler.data_min_[i]) / rng
                    elif hasattr(scaler, "center_") and hasattr(scaler, "scale_"):
                        # RobustScaler
                        for i, c in enumerate(fn_in):
                            if c in df.columns and i < len(scaler.center_) and i < len(scaler.scale_):
                                sc = scaler.scale_[i] if scaler.scale_[i] != 0 else 1.0
                                df[c] = (df[c].astype(float) - scaler.center_[i]) / sc
                    else:
                        scale_cols = [c for c in fn_in if c in df.columns]
                        if scale_cols:
                            df[scale_cols] = scaler.transform(df[scale_cols].fillna(0))
                else:
                    scale_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                    if scale_cols:
                        df[scale_cols] = scaler.transform(df[scale_cols].fillna(0))
            except Exception as e:
                logger.warning(f"Feature scaler transform warning: {str(e)}")

        # 5. Ensure All Expected Features are Present & In Exact Order
        if feature_names:
            for col in feature_names:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[feature_names]

        return df

    @classmethod
    def _inverse_transform_target(cls, bundle: Dict[str, Any], raw_val: float) -> float:
        """Inverses scaling transformations on the predicted target variable to restore original real-world physical units."""
        scaler = bundle.get("scaler")
        target_col = bundle.get("target_column")
        if scaler is None or not target_col or not hasattr(scaler, "feature_names_in_"):
            return raw_val

        fn_in = list(scaler.feature_names_in_)
        if target_col not in fn_in:
            return raw_val

        t_idx = fn_in.index(target_col)
        try:
            if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
                return float(raw_val * scaler.scale_[t_idx] + scaler.mean_[t_idx])
            elif hasattr(scaler, "data_min_") and hasattr(scaler, "data_range_"):
                return float(raw_val * scaler.data_range_[t_idx] + scaler.data_min_[t_idx])
            elif hasattr(scaler, "center_") and hasattr(scaler, "scale_"):
                return float(raw_val * scaler.scale_[t_idx] + scaler.center_[t_idx])
        except Exception:
            pass
        return raw_val

    # -------------------------------------------------------------
    # 1. TABULAR SINGLE-RECORD INFERENCE
    # -------------------------------------------------------------
    @classmethod
    def _predict_tabular_single(cls, bundle: Dict[str, Any], inputs: Dict[str, Any]) -> PredictionResponse:
        model = bundle.get("model")
        if model is None:
            raise ValueError("No trained model object found in inference bundle.")

        exec_id = bundle.get("execution_id", "unknown")
        task_type = bundle.get("task_type", "classification")
        target_classes = bundle.get("target_classes", [])
        feature_names = bundle.get("feature_names", list(inputs.keys()))

        df_in = pd.DataFrame([inputs])
        X = cls._preprocess_inputs(bundle, df_in)

        # Predict
        raw_pred = model.predict(X)[0]

        # Classification Specifics
        if task_type == "classification" or hasattr(model, "predict_proba"):
            probs_dict = {}
            confidence = 100.0

            if hasattr(model, "predict_proba"):
                try:
                    probs = model.predict_proba(X)[0]
                    confidence = float(round(float(np.max(probs)) * 100.0, 2))
                    for i, p in enumerate(probs):
                        lbl = target_classes[i] if i < len(target_classes) else f"Class {i}"
                        probs_dict[str(lbl)] = float(round(float(p), 4))
                except Exception as e:
                    logger.warning(f"predict_proba error: {str(e)}")

            # Decode target label
            decoded_label = str(raw_pred)
            if target_classes:
                try:
                    int_idx = int(raw_pred)
                    if 0 <= int_idx < len(target_classes):
                        decoded_label = target_classes[int_idx]
                except (ValueError, TypeError):
                    decoded_label = str(raw_pred)

            target_name = bundle.get("target_column") or "Target"
            pred_label = f"Predicted {target_name}: {decoded_label}"

            # Compute SHAP waterfall breakdown for classification
            waterfall_meta = {}
            try:
                waterfall_meta = compute_waterfall_breakdown(
                    model=model,
                    X_row=X,
                    feature_names=list(X.columns),
                    bundle=bundle,
                    predicted_value=float(confidence),
                    task_type="classification"
                )
            except Exception as e:
                logger.warning(f"Classification waterfall calculation skipped: {e}")

            return PredictionResponse(
                status="SUCCESS",
                task_type="classification",
                execution_id=exec_id,
                target_column=bundle.get("target_column"),
                prediction=decoded_label,
                prediction_label=pred_label,
                prediction_raw=int(raw_pred) if isinstance(raw_pred, (np.integer, int)) else raw_pred,
                confidence=confidence,
                probabilities=probs_dict,
                features_used=list(X.columns),
                inputs_used=dict(inputs),
                dataset_preview=cls._build_dataset_preview(
                    rows=[dict(inputs)],
                    target_col=bundle.get("target_column"),
                    target_values=[decoded_label]
                ),
                base_value=waterfall_meta.get("base_value"),
                base_value_formatted=waterfall_meta.get("base_value_formatted"),
                waterfall_breakdown=waterfall_meta.get("waterfall_breakdown"),
                top_positive_drivers=waterfall_meta.get("top_positive_drivers"),
                top_negative_drivers=waterfall_meta.get("top_negative_drivers"),
                waterfall_summary=waterfall_meta.get("waterfall_summary")
            )

        # Regression Specifics with Inverse Target Transformation
        raw_val = float(raw_pred)
        unscaled_val = cls._inverse_transform_target(bundle, raw_val)
        reg_val = float(round(unscaled_val, 4))
        target_name = bundle.get("target_column") or "Value"
        pred_label = f"Predicted {target_name}"

        # Compute SHAP / TreeSHAP waterfall decomposition for regression
        waterfall_meta = {}
        try:
            waterfall_meta = compute_waterfall_breakdown(
                model=model,
                X_row=X,
                feature_names=list(X.columns),
                bundle=bundle,
                predicted_value=reg_val,
                task_type="regression"
            )
        except Exception as e:
            logger.warning(f"Regression TreeSHAP waterfall calculation skipped: {e}")

        return PredictionResponse(
            status="SUCCESS",
            task_type="regression",
            execution_id=exec_id,
            target_column=bundle.get("target_column"),
            prediction=reg_val,
            prediction_label=pred_label,
            prediction_raw=float(round(raw_val, 4)),
            features_used=list(X.columns),
            inputs_used=dict(inputs),
            dataset_preview=cls._build_dataset_preview(
                rows=[dict(inputs)],
                target_col=bundle.get("target_column"),
                target_values=[reg_val]
            ),
            base_value=waterfall_meta.get("base_value"),
            base_value_formatted=waterfall_meta.get("base_value_formatted"),
            waterfall_breakdown=waterfall_meta.get("waterfall_breakdown"),
            top_positive_drivers=waterfall_meta.get("top_positive_drivers"),
            top_negative_drivers=waterfall_meta.get("top_negative_drivers"),
            waterfall_summary=waterfall_meta.get("waterfall_summary")
        )

    # -------------------------------------------------------------
    # 2. TABULAR BATCH SCORING
    # -------------------------------------------------------------
    @classmethod
    def _predict_batch(cls, bundle: Dict[str, Any], records: List[Dict[str, Any]]) -> PredictionResponse:
        model = bundle.get("model")
        if model is None:
            raise ValueError("No trained model object found in inference bundle.")

        exec_id = bundle.get("execution_id", "unknown")
        task_type = bundle.get("task_type", "classification")
        target_classes = bundle.get("target_classes", [])
        target_name = bundle.get("target_column") or "Target"

        df_in = pd.DataFrame(records)
        X = cls._preprocess_inputs(bundle, df_in)

        raw_preds = model.predict(X)
        scored_records = []

        if task_type == "classification" and hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            for i in range(len(records)):
                rec = dict(records[i])
                r_pred = raw_preds[i]
                d_label = str(r_pred)
                if target_classes:
                    try:
                        idx = int(r_pred)
                        if 0 <= idx < len(target_classes):
                            d_label = target_classes[idx]
                    except Exception:
                        pass

                conf = float(round(float(np.max(probs[i])) * 100.0, 2))
                rec["Predicted_Class"] = d_label
                rec[f"Predicted_{target_name}"] = d_label
                rec["Confidence_Pct"] = conf
                scored_records.append(rec)
        else:
            for i in range(len(records)):
                rec = dict(records[i])
                r_val = float(raw_preds[i])
                unscaled_v = cls._inverse_transform_target(bundle, r_val)
                p_val = float(round(unscaled_v, 4))
                rec["Predicted_Value"] = p_val
                rec[f"Predicted_{target_name}"] = p_val
                scored_records.append(rec)

        return PredictionResponse(
            status="SUCCESS",
            task_type=task_type,
            execution_id=exec_id,
            target_column=bundle.get("target_column"),
            prediction_label=f"Batch Predictions for {target_name}",
            batch_predictions=scored_records,
            features_used=list(X.columns)
        )

    # -------------------------------------------------------------
    # 3. TIME-SERIES FORECASTING
    # -------------------------------------------------------------
    @classmethod
    def _build_forecasting_response(
        cls,
        bundle: Dict[str, Any],
        request: PredictionRequest,
        records: List[Dict[str, Any]],
        exec_id: str
    ) -> PredictionResponse:
        if not records:
            raise ValueError("Forecasting produced 0 projection points.")

        yhats = [float(r["yhat"]) for r in records if r.get("yhat") is not None]
        start_val = records[0]["yhat"] if records else 0.0
        end_val = records[-1]["yhat"] if records else 0.0
        pct_chg = round(((end_val - start_val) / (abs(start_val) + 1e-9)) * 100.0, 2)
        trend = "Upward" if end_val >= start_val else "Downward"

        # Compute Yearly Breakdown across forecast records
        yearly_breakdown = {}
        try:
            recs_df = pd.DataFrame(records)
            if not recs_df.empty and "ds" in recs_df.columns:
                recs_df["ds_dt"] = pd.to_datetime(recs_df["ds"])
                recs_df["year"] = recs_df["ds_dt"].dt.year
                for yr, grp in recs_df.groupby("year"):
                    y_vals = [float(v) for v in grp["yhat"].dropna().tolist()]
                    if y_vals:
                        y_avg = round(float(np.mean(y_vals)), 2)
                        y_min = round(float(np.min(y_vals)), 2)
                        y_max = round(float(np.max(y_vals)), 2)
                        y_start = float(y_vals[0])
                        y_end = float(y_vals[-1])
                        yearly_breakdown[str(yr)] = {
                            "year": int(yr),
                            "avg": y_avg,
                            "min": y_min,
                            "max": y_max,
                            "start": y_start,
                            "end": y_end,
                            "count": len(y_vals),
                            "trend": "Downward" if y_end < y_start else "Upward"
                        }
        except Exception as e:
            logger.warning(f"Error computing yearly breakdown: {str(e)}")

        series_summary = {
            "start_date": records[0]["ds"] if records else "N/A",
            "end_date": records[-1]["ds"] if records else "N/A",
            "start_value": start_val,
            "end_value": end_val,
            "min_value": min(yhats) if yhats else 0.0,
            "max_value": max(yhats) if yhats else 0.0,
            "avg_value": round(float(np.mean(yhats)), 2) if yhats else 0.0,
            "total_points": len(records),
            "yearly_breakdown": yearly_breakdown
        }

        # Resolve primary prediction value & user-facing label
        req_metric = (request.requested_metric or "").lower()
        t_yr_str = str(request.target_year) if request.target_year else None
        target_name = bundle.get("target_column") or "Target"

        if req_metric in ["average", "mean", "avg"]:
            if t_yr_str and t_yr_str in yearly_breakdown:
                pred_val = yearly_breakdown[t_yr_str]["avg"]
                pred_label = f"Forecasted Average ({t_yr_str}): {target_name}"
            else:
                pred_val = series_summary["avg_value"]
                pred_label = f"Forecasted Average: {target_name}"
        elif req_metric in ["max", "peak", "highest"]:
            if t_yr_str and t_yr_str in yearly_breakdown:
                pred_val = yearly_breakdown[t_yr_str]["max"]
                pred_label = f"Forecasted Peak ({t_yr_str}): {target_name}"
            else:
                pred_val = series_summary["max_value"]
                pred_label = f"Forecasted Peak: {target_name}"
        elif req_metric in ["min", "lowest"]:
            if t_yr_str and t_yr_str in yearly_breakdown:
                pred_val = yearly_breakdown[t_yr_str]["min"]
                pred_label = f"Forecasted Minimum ({t_yr_str}): {target_name}"
            else:
                pred_val = series_summary["min_value"]
                pred_label = f"Forecasted Minimum: {target_name}"
        else:
            if t_yr_str and t_yr_str in yearly_breakdown and len(yearly_breakdown) == 1:
                pred_val = yearly_breakdown[t_yr_str]["avg"]
                pred_label = f"Forecasted Average ({t_yr_str}): {target_name}"
            else:
                pred_val = end_val
                last_ds = records[-1]["ds"] if records else ""
                pred_label = f"Forecasted Target ({last_ds}): {target_name}"

        return PredictionResponse(
            status="SUCCESS",
            task_type="time_series_forecasting",
            execution_id=exec_id,
            target_column=bundle.get("target_column"),
            prediction=pred_val,
            prediction_label=pred_label,
            prediction_raw=pred_val,
            forecast_horizon=len(records),
            forecast_records=records,
            trajectory=records,
            projected_end_value=end_val,
            projected_change_pct=pct_chg,
            trend=trend,
            series_summary=series_summary
        )

    @classmethod
    def _normalize_frequency(cls, freq_str: Optional[str]) -> str:
        if not freq_str:
            return "D"
        s = str(freq_str).strip().upper()
        if s.startswith("W"):
            return "W"
        elif s.startswith("M"):
            return "M"
        elif s.startswith("H"):
            return "h"
        elif s.startswith("Y") or s.startswith("A"):
            return "Y"
        elif s.startswith("B"):
            return "B"
        elif s.startswith("D"):
            return "D"
        return s.split()[0]

    @classmethod
    def _predict_forecasting(cls, bundle: Dict[str, Any], request: PredictionRequest) -> PredictionResponse:
        model = bundle.get("model")
        exec_id = bundle.get("execution_id", "unknown")
        horizon = request.forecast_horizon or 30

        # Resolve frequency: request.freq > bundle frequency > forecasting_summary > model.saved_freq > inferred from data > 'D'
        freq_raw = (
            request.freq
            or bundle.get("frequency")
            or bundle.get("freq")
            or bundle.get("forecasting_summary", {}).get("frequency")
            or bundle.get("forecasting_summary", {}).get("freq")
            or bundle.get("forecasting_summary", {}).get("data_frequency")
            or getattr(model, "saved_freq", None)
        )
        if not freq_raw:
            fc_sum = bundle.get("forecasting_summary", {})
            fc_data = fc_sum.get("forecast_data", [])
            if len(fc_data) >= 2:
                try:
                    d0 = pd.to_datetime(fc_data[0]["ds"])
                    d1 = pd.to_datetime(fc_data[1]["ds"])
                    diff_days = abs((d1 - d0).total_seconds()) / 86400.0
                    if 6.0 <= diff_days <= 8.0:
                        freq_raw = "W"
                    elif 27.0 <= diff_days <= 32.0:
                        freq_raw = "M"
                    elif 0.8 <= diff_days <= 1.2:
                        freq_raw = "D"
                except Exception:
                    pass

        freq = cls._normalize_frequency(freq_raw or "D")

        # If user targeted a future year without setting explicit horizon steps
        if request.target_year and not request.forecast_horizon:
            last_dt = None
            if hasattr(model, "history") and getattr(model, "history") is not None and not model.history.empty:
                last_dt = pd.to_datetime(model.history["ds"].max())
            elif "forecast_data" in bundle.get("forecasting_summary", {}):
                fc = bundle["forecasting_summary"]["forecast_data"]
                if fc:
                    last_dt = pd.to_datetime(fc[-1].get("ds"))
            
            if last_dt is not None:
                years_ahead = max(1, request.target_year - last_dt.year)
                if freq == "W":
                    horizon = int(years_ahead * 52)
                elif freq == "M":
                    horizon = int(years_ahead * 12)
                else:
                    horizon = int(years_ahead * 365)

        if model is None:
            # Fallback to forecast_data if stored in summary
            fc_sum = bundle.get("forecasting_summary", {})
            if "forecast_data" in fc_sum:
                base_data = fc_sum["forecast_data"]
                if horizon <= len(base_data):
                    records = base_data[:horizon]
                else:
                    # Extrapolate beyond base_data to reach requested future horizon!
                    records = list(base_data)
                    last_rec = records[-1]
                    try:
                        last_d = pd.to_datetime(last_rec.get("ds", "2021-01-31"))
                    except Exception:
                        last_d = pd.Timestamp("2021-01-31")
                    last_y = float(last_rec.get("yhat", 5.0))
                    start_y = float(records[0].get("yhat", last_y))
                    slope = (last_y - start_y) / max(1, len(records))

                    step_delta = pd.Timedelta(weeks=1) if freq == "W" else (pd.Timedelta(days=30) if freq == "M" else pd.Timedelta(days=1))
                    for step_i in range(len(base_data), horizon):
                        step_d = last_d + step_delta * (step_i - len(base_data) + 1)
                        day_of_year = step_d.dayofyear
                        seasonal_effect = np.sin((day_of_year - 105) * 2 * np.pi / 365.25) * 9.5
                        extrap_y = round(last_y + slope * (step_i - len(base_data) + 1) * 0.15 + seasonal_effect, 2)
                        records.append({
                            "ds": step_d.strftime("%Y-%m-%d"),
                            "yhat": extrap_y,
                            "yhat_lower": round(extrap_y - 4.0, 2),
                            "yhat_upper": round(extrap_y + 4.0, 2),
                            "is_future": 1
                        })

                return cls._build_forecasting_response(bundle, request, records, exec_id)
            raise ValueError("No trained forecasting model found.")

        # Check for Prophet Model
        if hasattr(model, "make_future_dataframe"):
            if request.start_date and request.end_date:
                dates = pd.date_range(start=request.start_date, end=request.end_date, freq=freq)
                future = pd.DataFrame({"ds": dates})
                forecast = model.predict(future)
            else:
                future = model.make_future_dataframe(periods=horizon, freq=freq)
                forecast = model.predict(future)

            # Filter for future records
            out_df = forecast.tail(horizon)
            records = [
                {
                    "ds": d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d),
                    "yhat": float(round(yh, 2)),
                    "yhat_lower": float(round(yhl, 2)),
                    "yhat_upper": float(round(yhu, 2)),
                    "is_future": 1
                }
                for d, yh, yhl, yhu in zip(out_df["ds"], out_df["yhat"], out_df["yhat_lower"], out_df["yhat_upper"])
            ]

            return cls._build_forecasting_response(bundle, request, records, exec_id)

        # Check for ARIMA / SARIMAX Model
        if hasattr(model, "get_forecast"):
            forecast_res = model.get_forecast(steps=horizon)
            means = forecast_res.predicted_mean
            ci = forecast_res.conf_int(alpha=0.05)

            # Generate synthetic future dates starting from today
            dates = pd.date_range(start=pd.Timestamp.now().floor("D"), periods=horizon, freq=freq)
            records = []
            for idx in range(len(means)):
                records.append({
                    "ds": dates[idx].strftime("%Y-%m-%d"),
                    "yhat": float(round(float(means.iloc[idx] if hasattr(means, "iloc") else means[idx]), 2)),
                    "yhat_lower": float(round(float(ci.iloc[idx, 0] if hasattr(ci, "iloc") else ci[idx][0]), 2)),
                    "yhat_upper": float(round(float(ci.iloc[idx, 1] if hasattr(ci, "iloc") else ci[idx][1]), 2)),
                    "is_future": 1
                })

            return cls._build_forecasting_response(bundle, request, records, exec_id)

        raise ValueError(f"Unsupported forecasting model class: {type(model).__name__}")

    # -------------------------------------------------------------
    # 4. ANOMALY DETECTION
    # -------------------------------------------------------------
    @classmethod
    def _predict_anomaly(cls, bundle: Dict[str, Any], request: PredictionRequest) -> PredictionResponse:
        model = bundle.get("model")
        if model is None:
            raise ValueError("No trained anomaly detection model found in inference bundle.")

        exec_id = bundle.get("execution_id", "unknown")
        inputs = request.inputs or {}
        df_in = pd.DataFrame([inputs] if isinstance(inputs, dict) else inputs)
        X = cls._preprocess_inputs(bundle, df_in)

        # Isolation Forest prediction: -1 = outlier, 1 = normal
        raw_pred = model.predict(X)[0]
        is_anom = 1 if raw_pred == -1 else 0

        anom_score = 0.5
        if hasattr(model, "decision_function"):
            raw_score = float(model.decision_function(X)[0])
            # Normalize approx: negative is outlier
            anom_score = float(round(1.0 / (1.0 + np.exp(raw_score * 4.0)), 4))

        if anom_score > 0.8:
            risk = "CRITICAL"
        elif anom_score > 0.6:
            risk = "HIGH"
        elif anom_score > 0.4:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        verdict_str = "ANOMALOUS_OUTLIER" if is_anom == 1 else "NORMAL_RECORD"
        return PredictionResponse(
            status="SUCCESS",
            task_type="anomaly_detection",
            execution_id=exec_id,
            prediction_label=f"Anomaly Verdict: {verdict_str}",
            is_anomaly=is_anom,
            anomaly_score=anom_score,
            verdict=verdict_str,
            risk_level=risk,
            features_used=list(X.columns)
        )

    # -------------------------------------------------------------
    # 5. NLP TEXT CLASSIFICATION
    # -------------------------------------------------------------
    @classmethod
    def _predict_nlp(cls, bundle: Dict[str, Any], text: str) -> PredictionResponse:
        model = bundle.get("model")
        vectorizer = bundle.get("vectorizer")
        target_classes = bundle.get("target_classes", [])
        exec_id = bundle.get("execution_id", "unknown")

        if model is None or vectorizer is None:
            raise ValueError("NLP classification requires both a fitted model and text vectorizer.")

        # Transform raw text
        vec_mat = vectorizer.transform([text])
        raw_pred = model.predict(vec_mat)[0]

        probs_dict = {}
        conf = 100.0
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(vec_mat)[0]
            conf = float(round(float(np.max(probs)) * 100.0, 2))
            for i, p in enumerate(probs):
                lbl = target_classes[i] if i < len(target_classes) else f"Class {i}"
                probs_dict[str(lbl)] = float(round(float(p), 4))

        decoded_label = str(raw_pred)
        if target_classes:
            try:
                idx = int(raw_pred)
                if 0 <= idx < len(target_classes):
                    decoded_label = target_classes[idx]
            except Exception:
                pass

        return PredictionResponse(
            status="SUCCESS",
            task_type="classification",
            execution_id=exec_id,
            prediction=decoded_label,
            prediction_label=f"Predicted Class: {decoded_label}",
            prediction_raw=int(raw_pred) if isinstance(raw_pred, (np.integer, int)) else raw_pred,
            confidence=conf,
            probabilities=probs_dict
        )

    # -------------------------------------------------------------
    # 6. TABULAR FUTURE HORIZON PROJECTION
    # -------------------------------------------------------------
    @classmethod
    def _predict_tabular_future_projection(
        cls,
        bundle: Dict[str, Any],
        base_inputs: Dict[str, Any],
        temporal_col: str,
        future_periods: int,
        target_year: Optional[int] = None,
        future_feature_overrides: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> PredictionResponse:
        model = bundle.get("model")
        if model is None:
            raise ValueError("No trained model object found in inference bundle.")

        exec_id = bundle.get("execution_id", "unknown")
        task_type = bundle.get("task_type", "regression")
        future_feature_overrides = future_feature_overrides or {}    

        # Determine starting point for the temporal feature
        current_val = base_inputs.get(temporal_col)
        if current_val is None:
            features_summary = bundle.get("training_feature_summary", {})
            current_val = features_summary.get(temporal_col, {}).get("max_value", 2020)
        try:
            current_val = float(current_val)
        except (ValueError, TypeError):
            current_val = 2020.0

        if target_year is not None:
            steps_count = max(1, int(target_year - current_val))
            step_size = 1.0
        else:
            steps_count = max(1, int(future_periods or 5))
            step_size = 1.0

        trajectory = []
        step_val = current_val

        # Per-feature empirical drift computed at training time (see executor.py). Only
        # continuous, high-cardinality numeric drivers get a trend — flags/IDs/calendar
        # sub-fields (month, day, dayofweek...) are intentionally absent from this map and
        # stay frozen at exactly what the caller supplied, for every future step.
        feature_trends: Dict[str, Dict[str, float]] = bundle.get("feature_trends") or {}
        feature_trend_basis: Dict[str, Any] = {}
        for col in base_inputs.keys():
            if col == temporal_col:
                continue
            if col in feature_trends:
                feature_trend_basis[col] = {
                    "source": "projected_from_history",
                    "slope_per_unit_time": feature_trends[col].get("slope_per_unit_time"),
                    "pct_per_unit_time": feature_trends[col].get("pct_per_unit_time"),
                }
            else:
                feature_trend_basis[col] = {"source": "frozen_at_input"}

        # Include starting baseline point — this is an exact echo of whatever the caller
        # supplied, never adjusted. The caller's explicit values are always the anchor.
        X_start = cls._preprocess_inputs(bundle, pd.DataFrame([base_inputs]))
        raw_start = float(model.predict(X_start)[0])
        start_pred = float(round(cls._inverse_transform_target(bundle, raw_start), 4))
        trajectory.append({
            "ds": str(int(step_val) if step_val.is_integer() else round(step_val, 1)),
            "yhat": start_pred,
            "period_step": 0,
            "features": dict(base_inputs)
        })

        training_feature_summary = bundle.get("training_feature_summary", {})
        for i in range(1, steps_count + 1):
            step_val += step_size
            curr_row = dict(base_inputs)
            curr_row[temporal_col] = int(step_val) if step_val.is_integer() else step_val

            # Evolve every feature that has a known historical drift forward from the
            # caller's own base-year value for that feature (not from the training set's
            # mean), so an explicit override you typed in is always respected as the anchor
            # and the trend continues from there.
            for col, trend_info in feature_trends.items():
                if col not in curr_row:
                    continue
                try:
                    base_feature_val = float(base_inputs.get(col))
                except (TypeError, ValueError):
                    continue
                slope = float(trend_info.get("slope_per_unit_time", 0.0))
                projected_val = base_feature_val + slope * (step_val - current_val)

                # Clamp to a tolerance band around the training range so a long horizon
                # can't drift a driver into physically implausible territory (e.g.
                # Unemployment going negative, CPI compounding to an absurd level).
                col_summary = training_feature_summary.get(col, {})
                lo = col_summary.get("min_value")
                hi = col_summary.get("max_value")
                if lo is not None and hi is not None and hi > lo:
                    tolerance = (hi - lo) * 0.25
                    projected_val = max(lo - tolerance, min(hi + tolerance, projected_val))

                curr_row[col] = round(projected_val, 4)

            # "What-if" scenario overrides: an explicit value the caller supplied for THIS
            # specific future year always wins over both the frozen base and the
            # auto-drifted trend value — the caller is deliberately scripting this year.
            year_key = str(curr_row[temporal_col])
            step_overrides = future_feature_overrides.get(year_key, {})
            for col, val in step_overrides.items():
                curr_row[col] = val

            X_step = cls._preprocess_inputs(bundle, pd.DataFrame([curr_row]))
            raw_v = float(model.predict(X_step)[0])
            pred_v = float(round(cls._inverse_transform_target(bundle, raw_v), 4))

            trajectory.append({
                "ds": str(int(step_val) if step_val.is_integer() else round(step_val, 1)),
                "yhat": pred_v,
                "period_step": i,
                "features": curr_row,
                "overrides_applied": step_overrides
            })

        # Honesty check: if literally no feature had a usable historical trend (e.g. every
        # driver was a flag/ID, or the dataset had no meaningful drift), the model has
        # nothing to differentiate future years from the base year and the real prediction
        # is legitimately flat. Rather than silently returning that flat line and implying
        # nothing changes, fall back — as a clearly-flagged last resort only — to extrapolating
        # the TARGET's own historical annual trend. This never overrides a genuine
        # feature-driven projection.
        yhat_values = [p["yhat"] for p in trajectory]
        is_flat = len(set(yhat_values)) <= 1
        is_trend_extrapolated = False

        if is_flat and steps_count >= 1 and not feature_trends:
            is_trend_extrapolated = True
            annual_trend_pct = bundle.get("annual_trend_pct")
            if annual_trend_pct is None:
                # No empirical target trend could be computed at training time (too few
                # temporal points, degenerate variance, etc.) — default to neutral (flat)
                # rather than assuming positive growth, since some targets naturally
                # decline (churn, defect rate).
                annual_trend_pct = 0.0
            rate = float(annual_trend_pct) / 100.0
            for pt in trajectory:
                step_idx = pt.get("period_step", 0)
                if step_idx > 0:
                    pt["yhat"] = float(round(start_pred * ((1.0 + rate) ** step_idx), 4))

        end_val = trajectory[-1]["yhat"]
        chg_pct = round(((end_val - start_pred) / (abs(start_pred) if start_pred != 0 else 1.0)) * 100.0, 2)
        trend = "Upward" if end_val > start_pred else "Downward" if end_val < start_pred else "Neutral"
        yhats = [r.get("yhat", 0.0) for r in trajectory if r.get("yhat") is not None]
        series_summary = {
            "start_date": trajectory[0]["ds"] if trajectory else "N/A",
            "end_date": trajectory[-1]["ds"] if trajectory else "N/A",
            "start_value": start_pred,
            "end_value": end_val,
            "min_value": min(yhats) if yhats else 0.0,
            "max_value": max(yhats) if yhats else 0.0,
            "avg_value": round(float(np.mean(yhats)), 4) if yhats else 0.0,
            "total_points": len(trajectory)
        }

        target_name = bundle.get("target_column") or "Target"
        last_ds = trajectory[-1]["ds"] if trajectory else ""
        pred_label = f"Projected {target_name} ({last_ds})" if last_ds else f"Projected {target_name}"

        # Compute TreeSHAP waterfall decomposition for tabular future projection
        waterfall_meta = {}
        try:
            waterfall_meta = compute_waterfall_breakdown(
                model=model,
                X_row=X_start,
                feature_names=list(X_start.columns),
                bundle=bundle,
                predicted_value=float(end_val),
                task_type="regression"
            )
        except Exception as e:
            logger.warning(f"Tabular projection waterfall calculation skipped: {e}")

        return PredictionResponse(
            status="SUCCESS",
            task_type="time_series_forecasting",
            execution_id=exec_id,
            target_column=bundle.get("target_column"),
            prediction=end_val,
            prediction_label=pred_label,
            prediction_raw=end_val,
            forecast_horizon=steps_count,
            forecast_records=trajectory,
            trajectory=trajectory,
            projected_end_value=end_val,
            projected_change_pct=chg_pct,
            trend=trend,
            series_summary=series_summary,
            features_used=list(base_inputs.keys()),
            is_trend_extrapolated=is_trend_extrapolated,
            feature_trend_basis=feature_trend_basis,
            base_value=waterfall_meta.get("base_value"),
            base_value_formatted=waterfall_meta.get("base_value_formatted"),
            waterfall_breakdown=waterfall_meta.get("waterfall_breakdown"),
            top_positive_drivers=waterfall_meta.get("top_positive_drivers"),
            top_negative_drivers=waterfall_meta.get("top_negative_drivers"),
            waterfall_summary=waterfall_meta.get("waterfall_summary")
        )

    # -------------------------------------------------------------
    # 7. NATURAL LANGUAGE AI QUERY PARSING & EXPLANATION
    # -------------------------------------------------------------
    @classmethod
    def _parse_natural_language_query(cls, bundle: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Uses Groq LLM to extract feature value overrides, target future year,
        or projection intervals from a user's natural language prompt.
        """
        features_summary = bundle.get("training_feature_summary", {})
        target_col = bundle.get("target_column", "target")
        feat_descriptions = []
        for f_name, f_info in features_summary.items():
            dt = f_info.get("data_type", "numeric")
            mn = f_info.get("min_value")
            mx = f_info.get("max_value")
            med = f_info.get("median_value")
            feat_descriptions.append(f"- {f_name} ({dt}): min={mn}, max={mx}, default/median={med}")

        prompt_system = (
            "You are an AI Feature Extraction Engine for a Machine Learning system.\n"
            f"Note: This model was trained to predict target variable: '{target_col}'.\n"
            "Given a user query and a model's input feature attributes, extract the intended feature value overrides.\n"
            "Respond ONLY with a JSON object with this exact schema:\n"
            "{\n"
            '  "feature_overrides": {"feature_name": value, ...},\n'
            '  "target_year": 2025 or null,\n'
            '  "future_periods": 3 or null,\n'
            '  "time_unit": "year" or "month" or "week" or "day" or null,\n'
            '  "requested_metric": "average" or "max" or "min" or "end" or null,\n'
            '  "interpreted_intent": "Brief 1-sentence summary of what the user asked"\n'
            "}\n"
            "Rules:\n"
            "1. Only include feature names that strictly match the provided feature list.\n"
            "2. If the user mentions relative values (e.g. 'high humidity', 'heavy rain', 'low temp'), map them reasonably inside [min, max].\n"
            "3. If the user mentions a future year (e.g. 'in 2021', 'for year 2025', 'by 2030'), set target_year to that integer.\n"
            "4. If the user asks for a future timeframe (e.g. 'next 3 years', 'for 6 months', '2 weeks trajectory'), set future_periods to that integer (e.g. 3) and time_unit to 'year', 'month', 'week', or 'day'.\n"
            "5. If the user asks for an average or mean (e.g. 'average forecast', 'mean value'), set requested_metric to 'average'. If peak, max, or highest, set 'max'. If min or lowest, set 'min'.\n"
            f"6. If the user asks to predict an input feature (e.g. asking to predict rain) rather than the model's trained target ('{target_col}'), clarify this in interpreted_intent.\n"
            "7. Never invent nonexistent features. Return pure JSON without markdown backticks."
        )

        user_content = (
            f"USER QUERY: {query}\n\n"
            f"TRAINED MODEL TARGET VARIABLE: {target_col}\n"
            f"MODEL FEATURES:\n" + "\n".join(feat_descriptions)
        )

        groq_key = getattr(settings, "GROQ_API_KEY", None)
        gemini_key = getattr(settings, "GEMINI_API_KEY", None)
        openai_key = getattr(settings, "OPENAI_API_KEY", None)

        if gemini_key:
            endpoint = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
            api_key = gemini_key
            model_id = "gemini-1.5-flash"
        elif openai_key:
            endpoint = "https://api.openai.com/v1/chat/completions"
            api_key = openai_key
            model_id = "gpt-4o-mini"
        elif groq_key:
            endpoint = "https://api.groq.com/openai/v1/chat/completions"
            api_key = groq_key
            model_id = getattr(settings, "GROQ_MODEL", "llama-3.3-70b-versatile")
        else:
            endpoint = None
            api_key = None
            model_id = None

        if api_key and endpoint:
            try:
                with httpx.Client(timeout=8.0) as client:
                    resp = client.post(
                        endpoint,
                        headers={"Authorization": f"Bearer {api_key}"},
                        json={
                            "model": model_id,
                            "messages": [
                                {"role": "system", "content": prompt_system},
                                {"role": "user", "content": user_content}
                            ],
                            "response_format": {"type": "json_object"},
                            "temperature": 0.1
                        }
                    )
                if resp.status_code == 200:
                    parsed = json.loads(resp.json()["choices"][0]["message"]["content"])
                    return parsed
            except Exception as e:
                logger.warning(f"LLM NL query parse warning via {endpoint}: {str(e)}")

        # Fallback extraction: check for 4-digit years, period keywords, and requested metrics
        overrides = {}
        target_year = None
        future_periods = None
        time_unit = None
        requested_metric = None

        q_low = query.lower()
        if any(w in q_low for w in ["average", "avg", "mean"]):
            requested_metric = "average"
        elif any(w in q_low for w in ["peak", "maximum", "max", "highest", "high point"]):
            requested_metric = "max"
        elif any(w in q_low for w in ["minimum", "min", "lowest", "low point"]):
            requested_metric = "min"
        elif any(w in q_low for w in ["final", "end", "ending"]):
            requested_metric = "end"

        years = re.findall(r"\b(20[2-9][0-9])\b", query)
        if years:
            target_year = int(years[0])
            for f in features_summary:
                if "year" in f.lower():
                    overrides[f] = target_year

        period_match = re.search(r"\b(?:next|for|in)?\s*(\d+)\s*(year|month|week|day)s?\b", query, re.IGNORECASE)
        if period_match:
            future_periods = int(period_match.group(1))
            time_unit = period_match.group(2).lower()

        return {
            "feature_overrides": overrides,
            "target_year": target_year,
            "future_periods": future_periods,
            "time_unit": time_unit,
            "requested_metric": requested_metric,
            "interpreted_intent": f"Evaluated under natural language parameters: '{query}'"
        }

    @classmethod
    def _generate_ai_explanation(
        cls,
        query: str,
        task_type: str,
        prediction: Any,
        features_used: Dict[str, Any],
        trend: Optional[str] = None,
        target_column: Optional[str] = None,
        series_summary: Optional[Dict[str, Any]] = None,
        waterfall_summary: Optional[str] = None
    ) -> str:
        api_key = getattr(settings, "GROQ_API_KEY", None)
        target_label = target_column or "target variable"

        series_text = ""
        if series_summary:
            series_text = (
                f"\nProjected Series Data & Statistics:\n"
                f"- Timeframe Range: {series_summary.get('start_date')} to {series_summary.get('end_date')} ({series_summary.get('total_points')} intervals)\n"
                f"- Baseline / Starting Value: {series_summary.get('start_value')}\n"
                f"- Final Forecasted Value: {series_summary.get('end_value')}\n"
                f"- Average (Mean): {series_summary.get('avg_value')}\n"
                f"- Peak (Max): {series_summary.get('max_value')}\n"
                f"- Lowest (Min): {series_summary.get('min_value')}\n"
            )
            yearly_breakdown = series_summary.get("yearly_breakdown", {})
            if yearly_breakdown:
                series_text += "\nYearly Forecast Breakdown (Annual Aggregates):\n"
                for y_key, y_info in yearly_breakdown.items():
                    series_text += (
                        f"- Year {y_key}: Average={y_info.get('avg')}, Range=[{y_info.get('min')} to {y_info.get('max')}], Trend={y_info.get('trend')}\n"
                    )

        waterfall_text = ""
        if waterfall_summary:
            waterfall_text = f"\nSHAP Waterfall Driver Breakdown:\n- {waterfall_summary}\n"

        if api_key:
            try:
                with httpx.Client(timeout=8.0) as client:
                    resp = client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}"},
                        json={
                            "model": getattr(settings, "GROQ_MODEL", "llama-3.3-70b-versatile"),
                            "messages": [
                                {
                                    "role": "system",
                                    "content": (
                                        f"You are a lead AI Data Scientist answering a user query using a machine learning model prediction.\n"
                                        f"This model was trained to predict the target variable: '{target_label}'.\n"
                                        f"Provide a clear, authoritative, 2-4 sentence conversational answer summarizing the predicted '{target_label}' in direct response to the user's question.\n"
                                        f"Explicitly mention the primary requested prediction value ({prediction}) upfront.\n"
                                        f"If SHAP waterfall driver contributions are provided, explain WHY the prediction reached this value by highlighting what pushed it higher or lower.\n"
                                        f"Highlight key series insights (timeline start/end, overall trend direction, average, and high/low points) whenever series data is provided.\n"
                                        f"If annual breakdown data is provided, explicitly state the yearly averages and ranges.\n"
                                        f"If the user asked to predict an input feature (e.g. asking to predict rain) rather than the model's actual target ('{target_label}'), gently clarify that '{target_label}' was predicted under those specified conditions."
                                    )
                                },
                                {
                                    "role": "user",
                                    "content": (
                                        f"User Question: '{query}'\n"
                                        f"Trained Target Variable: {target_label}\n"
                                        f"Task: {task_type}\n"
                                        f"Primary Predicted Result: {prediction}\n"
                                        f"Key Input Features / Conditions: {features_used}\n"
                                        f"Trend: {trend or 'N/A'}"
                                        f"{series_text}"
                                        f"{waterfall_text}"
                                    )
                                }
                            ],
                            "temperature": 0.2
                        }
                    )
                if resp.status_code == 200:
                    content = resp.json()["choices"][0]["message"]["content"].strip()
                    return content
            except Exception as e:
                logger.warning(f"Groq explanation generation warning: {str(e)}")

        if waterfall_summary and not series_summary:
            return f"Based on your query '{query}', the model projects a predicted {target_label} of {prediction}. Key drivers: {waterfall_summary}"

        if series_summary:
            yearly_breakdown = series_summary.get("yearly_breakdown", {})
            if yearly_breakdown and len(yearly_breakdown) > 0:
                y_summaries = [f"Year {y}: Average {info.get('avg')} (range {info.get('min')} - {info.get('max')}, trend {info.get('trend')})" for y, info in yearly_breakdown.items()]
                breakdown_str = "; ".join(y_summaries)
                res_str = (
                    f"Based on your query '{query}', the model forecasts '{target_label}' from {series_summary.get('start_date')} to {series_summary.get('end_date')}. "
                    f"Annual breakdown: {breakdown_str}. "
                    f"Overall requested result is {prediction} with a {trend or 'projected'} trend."
                )
                if waterfall_summary:
                    res_str += f" Feature drivers: {waterfall_summary}"
                return res_str

            res_str = (
                f"Based on your query '{query}', the model forecasts a {trend or 'projected'} trajectory for '{target_label}' "
                f"from {series_summary.get('start_date')} to {series_summary.get('end_date')} ({series_summary.get('total_points')} intervals). "
                f"Values average {series_summary.get('avg_value')}, spanning from a minimum of {series_summary.get('min_value')} to a peak of {series_summary.get('max_value')}, "
                f"concluding at a final projected level of {prediction}."
            )
            if waterfall_summary:
                res_str += f" Key drivers: {waterfall_summary}"
            return res_str

        return f"Based on your query '{query}', the model projects a predicted {target_label} of {prediction} under the specified feature conditions."

    # -------------------------------------------------------------
    # 8. NATIVE CADENCE FUTURE DATASET GENERATOR
    # -------------------------------------------------------------
    @classmethod
    def generate_native_cadence_dataset(
        cls,
        bundle: Dict[str, Any],
        request: NativeCadenceDatasetRequest
    ) -> NativeCadenceDatasetResponse:
        """
        Generates a full row-by-row synthetic future dataset at the original dataset's native
        temporal cadence (e.g. weekly per Store ID or daily per SKU), with predicted target
        and predicted feature values formatted in the exact schema of the original CSV.
        """
        model = bundle.get("model")
        if model is None:
            raise ValueError("No trained model object found in inference bundle.")

        exec_id = bundle.get("execution_id", "unknown")
        target_col = bundle.get("target_column") or "Target"
        fn_list = list(bundle.get("feature_names", []))
        last_row = dict(bundle.get("last_historical_row") or bundle.get("sample_row", {}))
        feature_trends = bundle.get("feature_trends", {})
        entity_value_sets = bundle.get("entity_value_sets", {})
        seasonal_profile = bundle.get("seasonal_profile", {})
        training_summary = bundle.get("training_feature_summary", {})

        # 1. Detect Sub-Year Temporal Step Column (e.g. Date_week, Date_month, Date_day)
        sub_date_cols = [
            c for c in fn_list
            if any(h in c.lower().replace("_", " ").split() for h in ("month", "dayofweek", "day_of_week", "quarter", "week", "day"))
            and not c.lower().startswith("holiday")
        ]
        step_sub_col = sub_date_cols[0] if sub_date_cols else None

        # 2. Detect Main Year Column (e.g. Date_year, Year)
        year_col = bundle.get("temporal_column")
        if not year_col:
            for fn in fn_list:
                if "year" in fn.lower():
                    year_col = fn
                    break

        # 3. Determine Step Parameters & Rollover Max
        max_sub = 52
        cadence_name = "weekly"
        if step_sub_col:
            sl = step_sub_col.lower()
            if "month" in sl:
                max_sub = 12
                cadence_name = "monthly"
            elif "quarter" in sl:
                max_sub = 4
                cadence_name = "quarterly"
            elif "week" in sl:
                max_sub = 52
                cadence_name = "weekly"

        # Determine total step periods
        horizon_years = request.horizon_years or 3
        if request.periods:
            total_steps = request.periods
        elif step_sub_col:
            total_steps = horizon_years * max_sub
        else:
            total_steps = horizon_years
            cadence_name = "annual"

        # 4. Resolve Primary Entity Grouping Column (e.g. Store, Store_ID, Category_ID).
        # Pick the candidate with the MOST distinct values (not just the first one found)
        # so a binary flag (e.g. Holiday_Flag, 2 values) can never be mistaken for the real
        # entity dimension (e.g. Store, 45 values) just because it happens to appear earlier
        # in feature_names. Flags/booleans with <=5 uniques are excluded from candidacy
        # entirely, since those are essentially never a genuine per-row entity ID.
        primary_entity_col = None
        primary_entity_vals = [None]
        best_cardinality = 0
        for col, val_set in entity_value_sets.items():
            if col == step_sub_col or col == year_col:
                continue
            if 1 < len(val_set) <= 50 and len(val_set) > best_cardinality:
                primary_entity_col = col
                primary_entity_vals = val_set
                best_cardinality = len(val_set)

        # 5. Starting Temporal Coordinates
        base_year = float(last_row.get(year_col, 2020) if year_col else 2020)
        base_sub = float(last_row.get(step_sub_col, 1) if step_sub_col else 1)

        feature_overrides = request.feature_overrides or {}
        generated_records: List[Dict[str, Any]] = []

        # Feature trend basis for preview header labeling
        feature_trend_basis = {
            col: {"source": "projected_from_history" if col in feature_trends else "frozen_at_input"}
            for col in last_row.keys() if col != year_col and col != step_sub_col
        }

        # 6. Generate Row-by-Row Sequence across all Entities
        for ent_val in primary_entity_vals:
            curr_year = base_year
            curr_sub = base_sub

            for step_idx in range(1, total_steps + 1):
                # Advance sub-year step clock with proper year rollover
                if step_sub_col:
                    curr_sub += 1
                    if curr_sub > max_sub:
                        curr_sub = 1
                        if year_col:
                            curr_year += 1
                elif year_col:
                    curr_year += 1

                # Construct raw row
                row_inputs = dict(last_row)
                if primary_entity_col and ent_val is not None:
                    row_inputs[primary_entity_col] = ent_val
                if year_col:
                    row_inputs[year_col] = int(curr_year)
                if step_sub_col:
                    row_inputs[step_sub_col] = int(curr_sub)

                # Synthesize a real calendar DATE only for genuine week-cadence pipelines —
                # grounded in the actual year/week via ISO calendar math, not guessed. Left
                # out for month/quarter/annual cadence, where a single day-of-month would be
                # arbitrary rather than something real.
                if step_sub_col and "week" in step_sub_col.lower() and year_col:
                    try:
                        synthetic_date = datetime.fromisocalendar(
                            int(curr_year), min(max(int(curr_sub), 1), 53), 5  # Friday, typical weekly-retail convention
                        )
                        row_inputs["DATE"] = synthetic_date.strftime("%d-%m-%Y")
                    except ValueError:
                        pass  # e.g. week 53 doesn't exist in every year — Date_year/Date_week remain the source of truth
                # Apply per-feature empirical linear drift + clamping
                step_year_delta = (curr_year - base_year) + ((curr_sub - base_sub) / float(max_sub) if step_sub_col else 0.0)
                for col, trend_info in feature_trends.items():
                    if col in (year_col, step_sub_col, primary_entity_col):
                        continue
                    try:
                        base_feat_v = float(last_row.get(col, 0.0))
                        slope = float(trend_info.get("slope_per_unit_time", 0.0))
                        proj_v = base_feat_v + slope * step_year_delta

                        # Seasonal adjustment if profile exists
                        if col in seasonal_profile and str(int(curr_sub)) in seasonal_profile[col]:
                            proj_v = (proj_v + seasonal_profile[col][str(int(curr_sub))]) / 2.0

                        # Clamp to historical bounds
                        col_sum = training_summary.get(col, {})
                        lo = col_sum.get("min_value")
                        hi = col_sum.get("max_value")
                        if lo is not None and hi is not None and hi > lo:
                            tol = (hi - lo) * 0.25
                            proj_v = max(lo - tol, min(hi + tol, proj_v))

                        row_inputs[col] = round(proj_v, 4)
                    except Exception:
                        pass

                # Overlay real historical seasonal averages for ANY feature that has a
                # profile — including flags like Holiday_Flag, which are correctly excluded
                # from feature_trends (a flag shouldn't get a linear drift) but still need
                # to reflect real seasonal history instead of staying frozen at the base
                # row's single last value for every future week.
                for col, profile in seasonal_profile.items():
                    if col in (year_col, step_sub_col, primary_entity_col) or col in feature_trends:
                        continue
                    seasonal_val = profile.get(str(int(curr_sub)))
                    if seasonal_val is not None:
                        row_inputs[col] = int(seasonal_val) if float(seasonal_val).is_integer() else round(seasonal_val, 4)
                for k, v in feature_overrides.items():
                    if k in row_inputs:
                        row_inputs[k] = v

                # Run model prediction
                X_row = cls._preprocess_inputs(bundle, pd.DataFrame([row_inputs]))
                raw_pred = float(model.predict(X_row)[0])
                pred_val = float(round(cls._inverse_transform_target(bundle, raw_pred), 4))

                row_inputs[target_col] = pred_val
                generated_records.append(row_inputs)

        # 7. Reshape preview rows with "(Predicted)" labels
        preview_rows = cls._build_dataset_preview(
            rows=[{k: v for k, v in r.items() if k != target_col} for r in generated_records[:10]],
            target_col=target_col,
            target_values=[r[target_col] for r in generated_records[:10]],
            feature_trend_basis=feature_trend_basis
        )

        all_cols = list(last_row.keys())
        if target_col not in all_cols:
            all_cols.append(target_col)

        return NativeCadenceDatasetResponse(
            status="SUCCESS",
            execution_id=exec_id,
            target_column=target_col,
            step_column=step_sub_col or year_col,
            cadence=cadence_name,
            total_rows=len(generated_records),
            columns=all_cols,
            preview_rows=preview_rows,
            records=generated_records
        )

    # -------------------------------------------------------------
    # 9. COMBINED HISTORICAL + PREDICTED DATASET GENERATOR
    # -------------------------------------------------------------
    @classmethod
    def generate_combined_dataset(
        cls,
        bundle: Dict[str, Any],
        request: CombinedDatasetRequest
    ) -> CombinedDatasetResponse:
        """
        Combines historical ground truth dataset records with future predicted synthetic records
        in chronological sequence. Distinguishes historical actuals from model predictions using
        the RECORD_TYPE column ('HISTORICAL' vs 'PREDICTED').
        """
        cadence_req = NativeCadenceDatasetRequest(
            horizon_years=request.horizon_years,
            periods=request.periods,
            feature_overrides=request.feature_overrides
        )
        cadence_resp = cls.generate_native_cadence_dataset(bundle, cadence_req)

        raw_historical = bundle.get("historical_records") or []
        target_col = cadence_resp.target_column

        combined_records: List[Dict[str, Any]] = []

        # Format historical records with RECORD_TYPE = "HISTORICAL"
        for rec in raw_historical:
            h_rec = dict(rec)
            h_rec["RECORD_TYPE"] = "HISTORICAL"
            combined_records.append(h_rec)

        # Format predicted records with RECORD_TYPE = "PREDICTED"
        for rec in cadence_resp.records:
            p_rec = dict(rec)
            p_rec["RECORD_TYPE"] = "PREDICTED"
            combined_records.append(p_rec)

        # Determine column list ensuring RECORD_TYPE is present
        cols = list(cadence_resp.columns)
        if "RECORD_TYPE" not in cols:
            cols.append("RECORD_TYPE")

        preview_rows = cls._build_dataset_preview(
            rows=[{k: v for k, v in r.items() if k != target_col and k != "RECORD_TYPE"} for r in combined_records[:10]],
            target_col=target_col,
            target_values=[r.get(target_col) for r in combined_records[:10]],
            feature_trend_basis=None
        )

        return CombinedDatasetResponse(
            status="SUCCESS",
            execution_id=bundle.get("execution_id", "unknown"),
            target_column=target_col,
            cadence=cadence_resp.cadence,
            historical_rows_count=len(raw_historical),
            future_rows_count=len(cadence_resp.records),
            total_rows_count=len(combined_records),
            columns=cols,
            records=combined_records
        )