import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from backend.app.profiling.profiler import DataProfiler


from backend.app.core.config import settings

class AIRecommender:
    """
    Analyzes dataset profile characteristics and user intent to detect problem types
    and rank optimal preprocessing recipes and ML models (Section 8 of spec).
    Delegates to LLMRecommender when Groq API Key is available.
    """

    @classmethod
    def recommend_pipeline(
        cls,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        task_type: Optional[str] = None,
        query: Optional[str] = None
    ) -> Dict[str, Any]:
        has_llm_key = any([
            getattr(settings, "GROQ_API_KEY", None),
            getattr(settings, "GEMINI_API_KEY", None),
            getattr(settings, "OPENAI_API_KEY", None),
        ])
        _fallback_reason = None
        if has_llm_key:
            try:
                from backend.app.recommendation.llm_recommender import LLMRecommender
                return LLMRecommender.recommend_pipeline(
                    df=df, query=query, target_column=target_column, task_type=task_type
                )
            except Exception as _llm_err:
                import logging
                logging.getLogger(__name__).warning(
                    f"LLMRecommender.recommend_pipeline raised, falling back to heuristic: {_llm_err}"
                )
                _fallback_reason = f"LLMRecommender.recommend_pipeline raised: {_llm_err}"
        else:
            _fallback_reason = "No LLM API key configured (GROQ_API_KEY / GEMINI_API_KEY / OPENAI_API_KEY all unset)."
        return cls._heuristic_recommend_pipeline(df, target_column=target_column, task_type=task_type, fallback_reason=_fallback_reason)

    @classmethod
    def _heuristic_recommend_pipeline(
        cls,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        task_type: Optional[str] = None,
        fallback_reason: Optional[str] = None
    ) -> Dict[str, Any]:
        profile = DataProfiler.profile_dataframe(df)
        columns = profile.get("columns", {})
        row_count = profile.get("row_count", 0)
        missing_cells = profile.get("total_missing_cells", 0)

        date_cols = [c for c, m in columns.items() if m.get("inferred_type") == "datetime"]
        cat_cols  = [c for c, m in columns.items() if m.get("inferred_type") == "categorical"]
        num_cols  = [c for c, m in columns.items() if m.get("inferred_type") == "numeric"]

        # Also detect integer year-range columns (e.g. a 'Year' column containing 2000-2050)
        # as temporal, since the data profiler treats them as numeric, not datetime.
        year_cols = [
            c for c in df.columns
            if c not in date_cols
            and any(kw in c.lower() for kw in ["year", "date", "time", "period", "timestamp", "month", "ds", "week"])
            and pd.api.types.is_numeric_dtype(df[c])
            and df[c].dropna().between(1800, 2200).all()
        ] if len(df) > 0 else []
        # Treat these as usable temporal columns
        all_temporal_cols = date_cols + [c for c in year_cols if c not in date_cols]

        # 1. Determine Target Column
        primary_metrics = [c for c in df.columns if any(kw in c.lower() for kw in ["weekly_sales", "sales", "revenue", "demand", "price", "amount"])]
        if target_column and target_column in df.columns:
            if target_column.lower() in ["unemployment", "cpi", "fuel_price", "temperature", "store"] and primary_metrics:
                selected_target = primary_metrics[0]
            else:
                selected_target = target_column
        else:
            # Pick logical target using domain-agnostic semantic keywords or statistical variance
            candidates = [c for c in df.columns if c not in date_cols]
            primary_keywords = ["weekly_sales", "sales", "revenue", "demand", "target", "label", "y", "class", "churn", "survived", "price", "amount", "score", "value"]
            found_target = None
            for kw in primary_keywords:
                matched = [c for c in candidates if kw == c.lower().strip() or kw in c.lower()]
                if matched:
                    found_target = matched[0]
                    break

            if not found_target and candidates:
                # Statistical dynamic fallback: pick candidate numeric feature with highest variance (excluding static ID columns and trailing metadata)
                num_candidates = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 1]
                if num_candidates:
                    # Filter out likely ID columns and static flags
                    non_id = [c for c in num_candidates if not (df[c].nunique() == len(df) and "id" in c.lower()) and df[c].nunique() > 2]
                    if non_id:
                        variances = {c: float(df[c].var()) for c in non_id if not pd.isna(df[c].var())}
                        if variances:
                            found_target = max(variances, key=variances.get)

            selected_target = found_target or (candidates[-1] if candidates else list(df.columns)[-1])

        # 2. Determine / Infer Task Type
        if task_type in ["classification", "regression", "time_series_forecasting", "anomaly_detection"]:
            detected_task = task_type
            explanation = f"User specified problem task as **{task_type.replace('_', ' ').title()}** on target `{selected_target}`."
        else:
            if date_cols and len(num_cols) >= 1 and (selected_target in num_cols):
                detected_task = "time_series_forecasting"
                explanation = f"Detected chronological time-series with timestamp `{date_cols[0]}` and numeric metric `{selected_target}`."
            else:
                target_meta = columns.get(selected_target, {})
                inferred = target_meta.get("inferred_type")
                nunique = target_meta.get("unique_count", df[selected_target].nunique() if selected_target in df.columns else 0)
                series = df[selected_target].dropna() if selected_target in df.columns else pd.Series()

                if inferred in ["categorical", "text", "boolean"] or nunique == 2:
                    detected_task = "classification"
                    explanation = f"Detected discrete classification on `{selected_target}` ({nunique} unique classes)."
                elif inferred == "numeric":
                    # Check if integer classification vs continuous regression
                    is_float_continuous = any(series % 1 != 0) if not series.empty and pd.api.types.is_numeric_dtype(series) else False
                    if is_float_continuous or nunique > 20:
                        detected_task = "regression"
                        explanation = f"Detected continuous numerical regression on target `{selected_target}`."
                    elif nunique <= 10 and nunique < (row_count * 0.1):
                        detected_task = "classification"
                        explanation = f"Detected multi-class classification on discrete target `{selected_target}` ({nunique} distinct categories)."
                    else:
                        detected_task = "regression"
                        explanation = f"Detected numeric regression on target `{selected_target}`."
                else:
                    detected_task = "anomaly_detection"
                    explanation = "Unlabeled or high-dimensional continuous feature space suitable for anomaly detection."

        # 3. Recommended Preprocessing Chain
        cleaning_steps = []
        if missing_cells > 0:
            cleaning_steps.append({
                "recipe_id": "missing_value_imputer",
                "name": "Missing Value Imputer",
                "recipe_name": "Missing Value Imputer",
                "config": {"strategy": "median"},
                "reason": f"Dataset contains {missing_cells} missing cells requiring imputation."
            })

        # Exclude target and temporal date/time columns from encoding/scaling lists
        date_kws = ["date", "time", "timestamp", "period", "ds", "datetime"]
        feature_cats = [
            c for c in cat_cols 
            if c != selected_target 
            and not any(kw == c.lower().strip() or c.lower().strip().startswith(kw + "_") or c.lower().strip().endswith("_" + kw) for kw in date_kws)
        ]
        feature_nums = [
            c for c in num_cols 
            if c != selected_target 
            and not any(kw == c.lower().strip() or c.lower().strip().startswith(kw + "_") or c.lower().strip().endswith("_" + kw) for kw in date_kws)
        ]

        if feature_cats:
            cleaning_steps.append({
                "recipe_id": "categorical_encoder",
                "name": "Categorical One-Hot Encoder",
                "recipe_name": "Categorical One-Hot Encoder",
                "config": {"method": "one_hot", "columns": feature_cats},
                "reason": f"Found {len(feature_cats)} categorical features ({', '.join(feature_cats[:3])}) requiring numerical encoding."
            })

        if feature_nums:
            cleaning_steps.append({
                "recipe_id": "feature_scaler",
                "name": "Feature Scaler",
                "recipe_name": "Feature Scaler",
                "config": {"method": "standard"},
                "reason": "Standardizing variance across numerical features for model stability."
            })

        # 4. Recommended Algorithm Rankings (Dynamic Tiering based on Dataset Profile)
        recommended_models = []
        if detected_task in ["classification", "regression"]:
            # A regression task with a temporal driver (Year/Date alongside real business
            # features like CPI/Fuel_Price/Unemployment) will later need future-year
            # predictions that extrapolate PAST the training range. Standard tree splits
            # plateau once a projected feature value exceeds the highest split ever learned,
            # so LightGBM's linear_tree mode (a linear fit per leaf) is put first for this
            # case, since it keeps extrapolating sensibly instead of flattening.
            needs_extrapolation = bool(all_temporal_cols) and detected_task == "regression"

            xgb_entry = {
                "recipe_id": "xgboost_trainer",
                "name": f"XGBoost {detected_task.title()}",
                "tier": "Gold Standard",
                "reason": "Highest accuracy regularized gradient boosting for tabular datasets."
            }
            lgb_entry = {
                "recipe_id": "lightgbm_trainer",
                "name": f"LightGBM {detected_task.title()}",
                "tier": "High Speed",
                "config": {"linear_tree": True} if needs_extrapolation else {},
                "reason": (
                    "Trains with linear_tree mode enabled: this dataset has a temporal "
                    "column, so future-year predictions need to extrapolate past the "
                    "training range rather than plateau like a standard tree split would."
                ) if needs_extrapolation else "Optimal for ultra-fast training with histogram-based leaf growth."
            }
            if needs_extrapolation:
                recommended_models.append(lgb_entry)
                recommended_models.append(xgb_entry)
            else:
                recommended_models.append(xgb_entry)
                recommended_models.append(lgb_entry)

            if feature_cats:
                recommended_models.append({
                    "recipe_id": "catboost_trainer",
                    "name": f"CatBoost {detected_task.title()}",
                    "tier": "Categorical",
                    "reason": "Native handling of high-cardinality categorical features without one-hot expansion."
                })
            if date_cols:
                recommended_models.append({
                    "recipe_id": "prophet_forecaster" if row_count >= 60 else "arima_forecaster",
                    "name": "Meta Prophet Forecaster" if row_count >= 60 else "ARIMA / SARIMAX",
                    "tier": "Time Series",
                    "reason": "Captures trend and seasonality for long time series." if row_count >= 60 else "Optimal classical statistical forecaster for short time series."
                })
            else:
                recommended_models.append({
                    "recipe_id": "random_forest_trainer",
                    "name": "Random Forest",
                    "tier": "Ensemble Baseline",
                    "reason": "Robust non-linear bagging baseline."
                })

        elif detected_task == "time_series_forecasting":
            if row_count < 60:
                recommended_models.append({
                    "recipe_id": "arima_forecaster",
                    "name": "ARIMA / SARIMAX",
                    "tier": "Statistical Baseline",
                    "reason": "Optimal classical statistical baseline for small time series datasets (< 60 points)."
                })
                recommended_models.append({
                    "recipe_id": "prophet_forecaster",
                    "name": "Meta Prophet",
                    "tier": "Trend & Seasonality",
                    "reason": "Decomposes trend and yearly seasonality."
                })
            else:
                recommended_models.append({
                    "recipe_id": "prophet_forecaster",
                    "name": "Meta Prophet",
                    "tier": "Business Standard",
                    "reason": "Decomposes trend, weekly/yearly seasonality, and handles irregular intervals with prediction bands."
                })
                recommended_models.append({
                    "recipe_id": "arima_forecaster",
                    "name": "ARIMA / SARIMAX",
                    "tier": "Statistical Baseline",
                    "reason": "Rigorous classical statistical baseline with lag and error differencing."
                })
            recommended_models.append({
                "recipe_id": "xgboost_trainer",
                "name": "XGBoost Regressor (Temporal)",
                "tier": "Gradient Boosting",
                "reason": "Tree-based gradient boosting on temporal feature lags."
            })
            recommended_models.append({
                "recipe_id": "lightgbm_trainer",
                "name": "LightGBM Regressor (Temporal)",
                "tier": "High Speed",
                "reason": "Ultra-fast histogram gradient boosting with native categorical and temporal handling."
            })

        else: # Anomaly Detection
            recommended_models.append({
                "recipe_id": "isolation_forest",
                "name": "Isolation Forest",
                "tier": "Outlier Standard",
                "reason": "Linear-time unsupervised isolation partitioning that scales to high dimensions."
            })
            recommended_models.append({
                "recipe_id": "statistical_guardrail",
                "name": "Statistical Guardrail",
                "tier": "ELT Filter",
                "reason": "Z-Score / IQR standard-deviation thresholding for data quality filtering."
            })

        rec_result = {
            "task_type": detected_task,
            "explanation": explanation,
            "target_column": selected_target,
            "profile_summary": {
                "rows": row_count,
                "columns": len(columns),
                "missing_cells": missing_cells,
                "categorical_columns": len(cat_cols),
                "numeric_columns": len(num_cols),
                "date_columns": len(date_cols)
            },
            "preprocessing_recommendations": cleaning_steps,
            "model_rankings": recommended_models
        }

        rec_result["recommended_dag"] = cls.build_recommended_dag(
            rec_result,
            df=df,
            target_column=selected_target,
            date_column=all_temporal_cols[0] if all_temporal_cols else None
        )
        # Honest provenance tagging: this pipeline came from the deterministic rule-based
        # recommender, not the LLM. Callers should never have to guess which path produced
        # a recommendation. See LLMRecommender for where "llm" / a fallback_reason gets set.
        rec_result["llm_generated"] = False
        rec_result["recommendation_source"] = "heuristic_fallback"
        rec_result["fallback_reason"] = fallback_reason
        return rec_result

    @classmethod
    def build_recommended_dag(
        cls,
        recommendation: Dict[str, Any],
        df: Optional[pd.DataFrame] = None,
        target_column: Optional[str] = None,
        date_column: Optional[str] = None,
        dataset_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Synthesizes a visual, ready-to-render DAG (nodes, edges, layout coordinates,
        and node configs) from an AI recommendation analysis.
        """
        task = recommendation.get("task_type", "classification")
        target_col = target_column or recommendation.get("target_column")
        nodes = []
        edges = []
        node_configs = {}

        # 1. Ingestion Node
        csv_config = {}
        if dataset_id:
            csv_config["dataset_id"] = dataset_id

        nodes.append({
            "id": "node_csv",
            "recipe_id": "csv_loader",
            "label": "Data Ingestion",
            "position": {"x": 40, "y": 100},
            "config": csv_config
        })
        node_configs["node_csv"] = {
            "recipe_id": "csv_loader",
            "label": "Dataset Ingestion",
            "config": csv_config
        }

        prev_node_id = "node_csv"
        cur_x = 280

        if task == "time_series_forecasting":
            # Time-Series Pipeline
            # 2. Imputer
            nodes.append({
                "id": "node_impute",
                "recipe_id": "missing_value_imputer",
                "label": "Time Imputer (ffill)",
                "position": {"x": cur_x, "y": 100},
                "config": {"strategy": "ffill"}
            })
            node_configs["node_impute"] = {
                "recipe_id": "missing_value_imputer",
                "label": "Imputer",
                "config": {"strategy": "ffill"}
            }
            edges.append({"id": "e_csv_impute", "source": "node_csv", "target": "node_impute", "animated": True})
            prev_node_id = "node_impute"
            cur_x += 280

            # 3. Dynamic Forecaster Node (ARIMA for small series, Prophet for long series)
            rankings = recommendation.get("model_rankings", [])
            chosen_recipe = rankings[0].get("recipe_id") if rankings else "prophet_forecaster"
            if chosen_recipe not in ["prophet_forecaster", "arima_forecaster"]:
                chosen_recipe = "prophet_forecaster"

            fc_label = "ARIMA / SARIMAX" if chosen_recipe == "arima_forecaster" else "Prophet Forecaster"
            p_config = {
                "date_column": date_column or "Date",
                "target_column": target_col or "Value",
                "horizon_periods": 30
            }
            nodes.append({
                "id": "node_forecaster",
                "recipe_id": chosen_recipe,
                "label": fc_label,
                "position": {"x": cur_x, "y": 100},
                "config": p_config
            })
            node_configs["node_forecaster"] = {
                "recipe_id": chosen_recipe,
                "label": fc_label,
                "config": p_config
            }
            edges.append({"id": "e_impute_fc", "source": prev_node_id, "target": "node_forecaster", "animated": True})

        elif task == "anomaly_detection":
            # Anomaly Detection Pipeline
            # 2. Imputer
            nodes.append({
                "id": "node_impute",
                "recipe_id": "missing_value_imputer",
                "label": "Imputer (Median)",
                "position": {"x": cur_x, "y": 100},
                "config": {"strategy": "median"}
            })
            node_configs["node_impute"] = {
                "recipe_id": "missing_value_imputer",
                "label": "Imputer",
                "config": {"strategy": "median"}
            }
            edges.append({"id": "e_csv_impute", "source": "node_csv", "target": "node_impute", "animated": True})
            prev_node_id = "node_impute"
            cur_x += 280

            # 3. Isolation Forest
            nodes.append({
                "id": "node_iso",
                "recipe_id": "isolation_forest",
                "label": "Isolation Forest",
                "position": {"x": cur_x, "y": 100},
                "config": {"contamination": 0.05, "n_estimators": 100}
            })
            node_configs["node_iso"] = {
                "recipe_id": "isolation_forest",
                "label": "Isolation Forest",
                "config": {"contamination": 0.05, "n_estimators": 100}
            }
            edges.append({"id": "e_impute_iso", "source": prev_node_id, "target": "node_iso", "animated": True})

        else:
            # Classification / Regression Pipeline
            # 2. Add recommended preprocessing steps
            pre_steps = recommendation.get("preprocessing_recommendations", [])
            for idx, step in enumerate(pre_steps):
                step_id = f"node_prep_{idx+1}"
                r_id = step["recipe_id"]
                step_label = step.get("name", r_id.replace("_", " ").title())
                step_cfg = dict(step.get("config", {}))

                if r_id == "feature_scaler":
                    step_cfg["target_column"] = target_col
                    step_cfg["exclude_target"] = True
                elif r_id == "data_type_converter":
                    if date_column and "conversions" not in step_cfg:
                        step_cfg["conversions"] = {date_column: "datetime"}

                nodes.append({
                    "id": step_id,
                    "recipe_id": r_id,
                    "label": step_label,
                    "position": {"x": cur_x, "y": 100},
                    "config": step_cfg
                })
                node_configs[step_id] = {
                    "recipe_id": r_id,
                    "label": step_label,
                    "config": step_cfg
                }
                edges.append({
                    "id": f"e_{prev_node_id}_{step_id}",
                    "source": prev_node_id,
                    "target": step_id,
                    "animated": True
                })
                prev_node_id = step_id
                cur_x += 240

            # 3. Intelligent Train/Test Splitter Selection
            split_id = "node_split"
            is_discrete_cls = False
            if task == "classification" and df is not None and target_col and target_col in df.columns:
                s = df[target_col].dropna()
                nunique = s.nunique()
                is_float_cont = any(s % 1 != 0) if not s.empty and pd.api.types.is_numeric_dtype(s) else False
                if not is_float_cont and nunique <= 20:
                    is_discrete_cls = True

            if date_column:
                split_recipe_id = "time_series_split"
                split_label = "Time-Series Split"
                split_cfg = {
                    "target_column": target_col or "target",
                    "date_column": date_column,
                    "test_size": 0.2
                }
            elif is_discrete_cls:
                split_recipe_id = "stratified_split"
                split_label = "Stratified Split"
                split_cfg = {
                    "target_column": target_col or "target",
                    "test_size": 0.2,
                    "random_state": 42
                }
            else:
                split_recipe_id = "train_test_split"
                split_label = "Train/Test Split"
                split_cfg = {
                    "target_column": target_col or "target",
                    "test_size": 0.2,
                    "random_state": 42
                }

            nodes.append({
                "id": split_id,
                "recipe_id": split_recipe_id,
                "label": split_label,
                "position": {"x": cur_x, "y": 100},
                "config": split_cfg
            })
            node_configs[split_id] = {
                "recipe_id": split_recipe_id,
                "label": split_label,
                "config": split_cfg
            }
            edges.append({
                "id": f"e_{prev_node_id}_{split_id}",
                "source": prev_node_id,
                "target": split_id,
                "animated": True
            })
            cur_x += 240

            # 4. Top Ranked Model Trainer
            top_model = recommendation.get("model_rankings", [{}])[0]
            model_recipe = top_model.get("recipe_id", "xgboost_trainer")
            model_name = top_model.get("name", "XGBoost Trainer")
            model_id = "node_model"
            model_cfg = {
                "task_type": "regression" if task == "regression" else "classification",
                "n_estimators": 100,
                "max_depth": 6
            }
            model_cfg.update(top_model.get("config", {}))
            nodes.append({
                "id": model_id,
                "recipe_id": model_recipe,
                "label": model_name,
                "position": {"x": cur_x, "y": 50},
                "config": model_cfg
            })
            node_configs[model_id] = {
                "recipe_id": model_recipe,
                "label": model_name,
                "config": model_cfg
            }
            edges.append({
                "id": f"e_{split_id}_{model_id}",
                "source": split_id,
                "target": model_id,
                "animated": True
            })
            cur_x += 240

            # 5. Model Evaluator
            eval_id = "node_eval"
            eval_recipe = "model_evaluator"
            nodes.append({
                "id": eval_id,
                "recipe_id": eval_recipe,
                "label": "Model Evaluator",
                "position": {"x": cur_x, "y": 100},
                "config": {"report_type": "Comprehensive"}
            })
            node_configs[eval_id] = {
                "recipe_id": eval_recipe,
                "label": "Evaluator",
                "config": {"report_type": "Comprehensive"}
            }
            edges.append({
                "id": f"e_{split_id}_{eval_id}",
                "source": split_id,
                "target": eval_id,
                "animated": True
            })
            edges.append({
                "id": f"e_{model_id}_{eval_id}",
                "source": model_id,
                "target": eval_id,
                "animated": True
            })

        return {
            "nodes": nodes,
            "edges": edges,
            "node_configs": node_configs
        }