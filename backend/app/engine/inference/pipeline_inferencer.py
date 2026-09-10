import time
import json
import re
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
import httpx

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.engine.inference.schemas import PredictionRequest, PredictionResponse


class PipelineInferencer:
    """
    Universal Inference Engine for Visual ML/AI Pipelines.
    Executes live real-time predictions, batch scoring, and time-series projections
    using artifacts preserved from completed pipeline executions.
    """

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
                base_inputs = dict(bundle.get("sample_row", {}))
                base_inputs.update(overrides)
                request.inputs = base_inputs

                if parsed_nl.get("target_year") and not request.target_year:
                    request.target_year = parsed_nl["target_year"]
                if parsed_nl.get("future_periods") and not request.future_periods:
                    request.future_periods = parsed_nl["future_periods"]
            except Exception as e:
                logger.warning(f"Failed to parse natural language query: {str(e)}")

        try:
            # 1. TIME-SERIES FORECASTING
            if task_type == "time_series_forecasting" or "forecaster" in task_type:
                res = cls._predict_forecasting(bundle, request)
                if nl_query:
                    res.ai_explanation = cls._generate_ai_explanation(
                        query=nl_query, task_type=task_type, prediction=res.prediction or res.projected_end_value,
                        features_used=inferred_inputs or {}, trend=res.trend
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

            if (request.future_periods or request.target_year) and temporal_col and task_type in ["regression", "classification"]:
                inputs_dict = request.inputs if isinstance(request.inputs, dict) else dict(bundle.get("sample_row", {}))
                res = cls._predict_tabular_future_projection(
                    bundle=bundle,
                    base_inputs=inputs_dict,
                    temporal_col=temporal_col,
                    future_periods=request.future_periods or 5,
                    target_year=request.target_year
                )
                if nl_query:
                    res.ai_explanation = cls._generate_ai_explanation(
                        query=nl_query, task_type=task_type, prediction=res.prediction,
                        features_used=inferred_inputs or {}, trend=res.trend
                    )
                    res.inferred_inputs = inferred_inputs
                res.inference_latency_ms = round((time.time() - start_t) * 1000.0, 2)
                return res

            # 3. ANOMALY DETECTION
            if task_type == "anomaly_detection":
                res = cls._predict_anomaly(bundle, request)
                if nl_query:
                    res.ai_explanation = cls._generate_ai_explanation(
                        query=nl_query, task_type=task_type, prediction=res.verdict,
                        features_used=inferred_inputs or {}
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
                        features_used={"text_input": request.text_input}
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
            inputs_dict = request.inputs if isinstance(request.inputs, dict) else (bundle.get("sample_row", {}) or {})
            res = cls._predict_tabular_single(bundle, inputs_dict)
            if nl_query:
                res.ai_explanation = cls._generate_ai_explanation(
                    query=nl_query, task_type=task_type, prediction=res.prediction,
                    features_used=inferred_inputs or {}
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
                # Find common numeric features to scale
                scale_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                if hasattr(scaler, "feature_names_in_"):
                    scale_cols = [c for c in scaler.feature_names_in_ if c in df.columns]
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

            return PredictionResponse(
                status="SUCCESS",
                task_type="classification",
                execution_id=exec_id,
                prediction=decoded_label,
                prediction_raw=int(raw_pred) if isinstance(raw_pred, (np.integer, int)) else raw_pred,
                confidence=confidence,
                probabilities=probs_dict,
                features_used=list(X.columns)
            )

        # Regression Specifics
        reg_val = float(round(float(raw_pred), 4))
        return PredictionResponse(
            status="SUCCESS",
            task_type="regression",
            execution_id=exec_id,
            prediction=reg_val,
            prediction_raw=reg_val,
            features_used=list(X.columns)
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
                rec["Confidence_Pct"] = conf
                scored_records.append(rec)
        else:
            for i in range(len(records)):
                rec = dict(records[i])
                rec["Predicted_Value"] = float(round(float(raw_preds[i]), 4))
                scored_records.append(rec)

        return PredictionResponse(
            status="SUCCESS",
            task_type=task_type,
            execution_id=exec_id,
            batch_predictions=scored_records,
            features_used=list(X.columns)
        )

    # -------------------------------------------------------------
    # 3. TIME-SERIES FORECASTING
    # -------------------------------------------------------------
    @classmethod
    def _predict_forecasting(cls, bundle: Dict[str, Any], request: PredictionRequest) -> PredictionResponse:
        model = bundle.get("model")
        exec_id = bundle.get("execution_id", "unknown")
        horizon = request.forecast_horizon or 30
        freq = request.freq or "D"

        if model is None:
            # Fallback to forecast_data if stored in summary
            fc_sum = bundle.get("forecasting_summary", {})
            if "forecast_data" in fc_sum:
                records = fc_sum["forecast_data"][:horizon]
                return PredictionResponse(
                    status="SUCCESS",
                    task_type="time_series_forecasting",
                    execution_id=exec_id,
                    forecast_horizon=len(records),
                    forecast_records=records,
                    projected_end_value=records[-1].get("yhat"),
                    trend=fc_sum.get("trend_direction", "Neutral")
                )
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

            end_val = records[-1]["yhat"] if records else 0.0
            start_val = records[0]["yhat"] if records else 0.0
            pct_chg = round(((end_val - start_val) / (start_val + 1e-9)) * 100.0, 2)

            return PredictionResponse(
                status="SUCCESS",
                task_type="time_series_forecasting",
                execution_id=exec_id,
                forecast_horizon=len(records),
                forecast_records=records,
                projected_end_value=end_val,
                projected_change_pct=pct_chg,
                trend="Upward" if end_val >= start_val else "Downward"
            )

        # Check for ARIMA / SARIMAX Model
        if hasattr(model, "get_forecast"):
            forecast_res = model.get_forecast(steps=horizon)
            means = forecast_res.predicted_mean
            ci = forecast_res.conf_int(alpha=0.05)
            ci_cols = ci.columns if hasattr(ci, "columns") else [0, 1]

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

            end_val = records[-1]["yhat"] if records else 0.0
            start_val = records[0]["yhat"] if records else 0.0
            pct_chg = round(((end_val - start_val) / (start_val + 1e-9)) * 100.0, 2)

            return PredictionResponse(
                status="SUCCESS",
                task_type="time_series_forecasting",
                execution_id=exec_id,
                forecast_horizon=len(records),
                forecast_records=records,
                projected_end_value=end_val,
                projected_change_pct=pct_chg,
                trend="Upward" if end_val >= start_val else "Downward"
            )

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

        return PredictionResponse(
            status="SUCCESS",
            task_type="anomaly_detection",
            execution_id=exec_id,
            is_anomaly=is_anom,
            anomaly_score=anom_score,
            verdict="ANOMALOUS_OUTLIER" if is_anom == 1 else "NORMAL_RECORD",
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
        target_year: Optional[int] = None
    ) -> PredictionResponse:
        model = bundle.get("model")
        if model is None:
            raise ValueError("No trained model object found in inference bundle.")

        exec_id = bundle.get("execution_id", "unknown")
        task_type = bundle.get("task_type", "regression")

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

        # Include starting baseline point
        X_start = cls._preprocess_inputs(bundle, pd.DataFrame([base_inputs]))
        start_pred = float(round(float(model.predict(X_start)[0]), 4))
        trajectory.append({
            "ds": str(int(step_val) if step_val.is_integer() else round(step_val, 1)),
            "yhat": start_pred,
            "period_step": 0
        })

        for i in range(1, steps_count + 1):
            step_val += step_size
            curr_row = dict(base_inputs)
            curr_row[temporal_col] = int(step_val) if step_val.is_integer() else step_val

            X_step = cls._preprocess_inputs(bundle, pd.DataFrame([curr_row]))
            pred_v = float(round(float(model.predict(X_step)[0]), 4))

            trajectory.append({
                "ds": str(int(step_val) if step_val.is_integer() else round(step_val, 1)),
                "yhat": pred_v,
                "period_step": i
            })

        end_val = trajectory[-1]["yhat"]
        chg_pct = round(((end_val - start_pred) / (abs(start_pred) if start_pred != 0 else 1.0)) * 100.0, 2)
        trend = "Upward" if end_val > start_pred else "Downward" if end_val < start_pred else "Neutral"

        return PredictionResponse(
            status="SUCCESS",
            task_type="time_series_forecasting",
            execution_id=exec_id,
            prediction=end_val,
            prediction_raw=end_val,
            forecast_horizon=steps_count,
            forecast_records=trajectory,
            trajectory=trajectory,
            projected_end_value=end_val,
            projected_change_pct=chg_pct,
            trend=trend,
            features_used=list(base_inputs.keys())
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
        feat_descriptions = []
        for f_name, f_info in features_summary.items():
            dt = f_info.get("data_type", "numeric")
            mn = f_info.get("min_value")
            mx = f_info.get("max_value")
            med = f_info.get("median_value")
            feat_descriptions.append(f"- {f_name} ({dt}): min={mn}, max={mx}, default/median={med}")

        prompt_system = (
            "You are an AI Feature Extraction Engine for a Machine Learning system.\n"
            "Given a user query and a model's input feature attributes, extract the intended feature value overrides.\n"
            "Respond ONLY with a JSON object with this exact schema:\n"
            "{\n"
            '  "feature_overrides": {"feature_name": value, ...},\n'
            '  "target_year": 2025 or null,\n'
            '  "future_periods": 5 or null,\n'
            '  "interpreted_intent": "Brief 1-sentence summary of what the user asked"\n'
            "}\n"
            "Rules:\n"
            "1. Only include feature names that strictly match the provided feature list.\n"
            "2. If the user mentions relative values (e.g. 'high humidity', 'heavy rain', 'low temp'), map them reasonably inside [min, max].\n"
            "3. If the user mentions a future year (e.g. 'in 2025', 'by 2030'), set target_year to that integer.\n"
            "4. Never invent nonexistent features. Return pure JSON without markdown backticks."
        )

        user_content = (
            f"USER QUERY: {query}\n\n"
            f"MODEL FEATURES:\n" + "\n".join(feat_descriptions)
        )

        api_key = getattr(settings, "GROQ_API_KEY", None)
        if api_key:
            try:
                with httpx.Client(timeout=8.0) as client:
                    resp = client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}"},
                        json={
                            "model": getattr(settings, "GROQ_MODEL", "openai/gpt-oss-120b"),
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
                logger.warning(f"Groq NL query parse warning: {str(e)}")

        # Fallback extraction: check for 4-digit years (e.g. 2024, 2025) and basic keywords
        overrides = {}
        target_year = None
        years = re.findall(r"\b(20[2-9][0-9])\b", query)
        if years:
            target_year = int(years[0])
            for f in features_summary:
                if "year" in f.lower():
                    overrides[f] = target_year

        return {
            "feature_overrides": overrides,
            "target_year": target_year,
            "future_periods": None,
            "interpreted_intent": f"Evaluated under natural language parameters: '{query}'"
        }

    @classmethod
    def _generate_ai_explanation(
        cls,
        query: str,
        task_type: str,
        prediction: Any,
        features_used: Dict[str, Any],
        trend: Optional[str] = None
    ) -> str:
        api_key = getattr(settings, "GROQ_API_KEY", None)
        if api_key:
            try:
                with httpx.Client(timeout=8.0) as client:
                    resp = client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}"},
                        json={
                            "model": getattr(settings, "GROQ_MODEL", "openai/gpt-oss-120b"),
                            "messages": [
                                {
                                    "role": "system",
                                    "content": (
                                        "You are an executive AI Data Scientist answering a user query using a machine learning model prediction.\n"
                                        "Provide a clear, authoritative, 2-sentence conversational answer summarizing the prediction in direct response to the user's question."
                                    )
                                },
                                {
                                    "role": "user",
                                    "content": (
                                        f"User Question: '{query}'\n"
                                        f"Task: {task_type}\n"
                                        f"Predicted Result: {prediction}\n"
                                        f"Key Input Features: {features_used}\n"
                                        f"Trend: {trend or 'N/A'}"
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

        return f"Based on your query '{query}', the model projects an outcome of {prediction} using the specified feature attributes."
