import time
import re
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field

from backend.app.engine.dag.graph import WorkflowGraph, WorkflowNode
from backend.app.recipes.base.registry import recipe_registry
from backend.app.core.exceptions import ExecutionException
from backend.app.core.logging import logger


class NodeExecutionResult(BaseModel):
    node_id: str
    recipe_id: str
    status: str  # "SUCCESS", "FAILED", "SKIPPED"
    duration_ms: float
    error_message: Optional[str] = None
    error_title: Optional[str] = None
    error_suggestion: Optional[str] = None
    output_summary: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}


def diagnose_execution_error(
    recipe: Any,
    error: Exception,
    inputs: Dict[str, Any],
    config: Dict[str, Any],
    context: Dict[str, Any]
) -> Dict[str, str]:
    """
    Translates cryptic low-level Python/C++/ML library errors into plain-English root causes
    and concrete, actionable step-by-step UI suggestions for the visual pipeline canvas.
    """
    err_str = str(error)
    rec_name = getattr(recipe, "name", "Processor")

    # 1. Unencoded Categorical/String Columns in ML Trainers
    if any(k in err_str for k in ["enable_categorical", "DataFrame.dtypes for data must be int, float", "could not convert string to float", "cannot convert string"]):
        return {
            "title": "Unencoded Categorical Features",
            "message": f"Trainer '{rec_name}' received non-numeric text columns. Machine learning algorithms require categorical features to be encoded into numbers.",
            "suggestion": "Insert a 'Categorical Feature Encoder' processor before 'Train / Test Splitter' to choose an encoding strategy (One-Hot, Target, Label, or Binary)."
        }

    # 2. Missing Train/Test Split or Missing Partition Keys
    if any(k in err_str for k in ["expects 'X_train' and 'y_train'", "expects 'X_test' and 'y_test'"]):
        return {
            "title": "Missing Dataset Split",
            "message": f"'{rec_name}' requires train/test dataset partitions but did not find them in upstream inputs.",
            "suggestion": "Connect a 'Train / Test Splitter' processor before this component and ensure a target column is selected."
        }

    # 3. Missing Model in Model Evaluator
    if "expects a trained 'model'" in err_str:
        return {
            "title": "Missing Trained Model",
            "message": f"'{rec_name}' requires a trained machine learning model to evaluate.",
            "suggestion": "Connect a model trainer (e.g. XGBoost, Random Forest, LightGBM, CatBoost) to this Evaluator."
        }

    # 4. Time Series Forecaster Missing Date Column
    if any(k in err_str for k in ["'ds'", "time-series observations", "date_column"]):
        return {
            "title": "Time-Series Date Column Issue",
            "message": f"Forecaster '{rec_name}' could not identify a valid sequential date column.",
            "suggestion": "Open processor settings and configure 'date_column' to your date/timestamp column (e.g., 'OrderDate', 'Date')."
        }

    # 5. Target Column Missing or Not Found
    if any(k in err_str for k in ["Target column", "target_column", "not found in dataframe"]):
        return {
            "title": "Target Column Not Found",
            "message": f"The target variable specified does not exist in the incoming dataset.",
            "suggestion": "Open processor configuration and select a valid target column from your dataset."
        }

    # 6. Feature Dimension Mismatch
    if "not aligned" in err_str or "feature_names" in err_str:
        return {
            "title": "Feature Dimension Mismatch",
            "message": f"The feature columns in test data do not match what the model was trained on.",
            "suggestion": "Ensure the exact same preprocessing steps (Imputer, Scaler, Encoder) are applied before both training and testing."
        }

    # 7. Dataset Join / Merge Column Missing
    if any(k in err_str for k in ["Join key", "join_type", "KeyError: 'left_on'", "MergeError", "requires two incoming datasets"]):
        return {
            "title": "Dataset Join Error",
            "message": f"Dataset Join processor '{rec_name}' failed: {err_str}",
            "suggestion": "Verify that two datasets are connected and that the join keys (left_on / right_on) exist in both datasets."
        }

    # Default fallback
    return {
        "title": f"Execution Error in {rec_name}",
        "message": err_str,
        "suggestion": "Check incoming connection handles and verify the processor configuration parameters."
    }


class WorkflowExecutionResult(BaseModel):
    execution_id: str
    workflow_id: Optional[str] = None
    status: str  # "SUCCESS", "FAILED"
    total_duration_ms: float
    node_results: List[Any] = Field(default_factory=list)
    final_metrics: Optional[Dict[str, Any]] = None
    anomaly_summary: Optional[Dict[str, Any]] = None
    forecasting_summary: Optional[Dict[str, Any]] = None
    governance_summary: Optional[Dict[str, Any]] = None
    logs: List[str] = Field(default_factory=list)
    node_outputs: Dict[str, Any] = Field(default_factory=dict)
    step_snapshots: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    inference_schema: Optional[Dict[str, Any]] = None

    model_config = {"arbitrary_types_allowed": True}


def make_json_safe(obj: Any) -> Any:
    """
    Recursively converts arbitrary Python/ML objects (DataFrames, ndarrays, numpy scalars,
    models) into standard, JSON-serializable primitives for FastAPI Pydantic responses.
    """
    if obj is None:
        return None
    if isinstance(obj, pd.DataFrame):
        clean_df = obj.replace({float("nan"): None, float("inf"): None, float("-inf"): None})
        return {
            "type": "DataFrame",
            "shape": list(obj.shape),
            "columns": list(obj.columns),
            "records": clean_df.to_dict(orient="records")
        }
    elif isinstance(obj, pd.Series):
        clean_s = obj.replace({float("nan"): None, float("inf"): None, float("-inf"): None})
        return clean_s.to_dict()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    elif isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [make_json_safe(v) for v in obj]
    elif hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        try:
            return make_json_safe(obj.to_dict())
        except Exception:
            return str(obj)
    elif type(obj).__module__ != "builtins":
        return f"<{type(obj).__name__} Object>"
    else:
        return obj


class DAGExecutor:
    """
    Unified In-Memory DAG Execution Engine.
    Executes nodes in topological order, manages context passing,
    and captures metrics, summaries, diagnostic artifacts, and step snapshots.
    """

    @classmethod
    def execute_workflow(
        cls,
        execution_id: str,
        workflow: WorkflowGraph,
        initial_df: Optional[pd.DataFrame] = None,
        context: Optional[Dict[str, Any]] = None,
        include_node_outputs: bool = False
    ) -> WorkflowExecutionResult:
        start_time = time.time()
        logs: List[str] = []
        node_results: List[NodeExecutionResult] = []
        node_outputs: Dict[str, Dict[str, Any]] = {}
        step_snapshots: Dict[str, Dict[str, Any]] = {}
        
        final_metrics = None
        anomaly_summary = None
        forecasting_summary = None
        governance_summary = None

        pipeline_context = dict(context or {})
        if initial_df is not None:
            pipeline_context["dataframe"] = initial_df.copy()

        logs.append(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] Starting execution '{execution_id}'")

        # 1. Topological Sorting
        try:
            ordered_nodes = workflow.get_topological_order()
        except Exception as e:
            if hasattr(e, "errors") and getattr(e, "errors"):
                for err in getattr(e, "errors"):
                    logs.append(f"❌ DAG Resolution Error: {err}")
            else:
                logs.append(f"❌ DAG Resolution Error: {str(e)}")
            return WorkflowExecutionResult(
                execution_id=execution_id,
                status="FAILED",
                total_duration_ms=0.0,
                node_results=[],
                logs=logs
            )

        # 2. Build In-Edge map to find parent nodes and incoming edge details
        parent_map: Dict[str, List[str]] = {n.id: [] for n in workflow.nodes}
        in_edges_map: Dict[str, List[Any]] = {n.id: [] for n in workflow.nodes}
        for edge in workflow.edges:
            parent_map[edge.target].append(edge.source)
            in_edges_map[edge.target].append(edge)

        overall_status = "SUCCESS"

        # 3. Step-by-Step Node Execution
        for node in ordered_nodes:
            node_start = time.time()
            parents = parent_map[node.id]

            try:
                recipe = recipe_registry.get(node.recipe_id)
            except Exception as e:
                err_msg = f"Recipe '{node.recipe_id}' not found in registry: {str(e)}"
                logs.append(f"❌ Node '{node.id}' failed: {err_msg}")
                overall_status = "FAILED"
                node_results.append(NodeExecutionResult(
                    node_id=node.id,
                    recipe_id=node.recipe_id,
                    status="FAILED",
                    duration_ms=0.0,
                    error_message=err_msg
                ))
                break

            # Collect inputs from parents
            node_inputs: Dict[str, Any] = {}
            if not parents:
                # If root node, supply initial dataframe only if recipe accepts dataframe
                if "dataframe" in recipe.input_types or "any" in recipe.input_types or not recipe.input_types:
                    if initial_df is not None:
                        node_inputs = {"dataframe": initial_df.copy()}
                else:
                    # Model Trainer or Evaluator dropped without parents!
                    err_msg = (
                        f"Node '{node.id}' [{recipe.name}] requires inputs {recipe.input_types}, "
                        f"but has 0 incoming connections. It cannot run as an unparented root node."
                    )
                    duration_ms = round((time.time() - node_start) * 1000.0, 2)
                    logs.append(f"❌ Node '{node.id}' failed: {err_msg}")
                    node_results.append(NodeExecutionResult(
                        node_id=node.id,
                        recipe_id=node.recipe_id,
                        status="FAILED",
                        duration_ms=duration_ms,
                        error_message=err_msg
                    ))
                    overall_status = "FAILED"
                    break
            else:
                incoming_edges = in_edges_map.get(node.id, [])
                parent_outputs_map: Dict[str, Dict[str, Any]] = {}
                parent_dfs: List[pd.DataFrame] = []

                for parent_id in parents:
                    parent_out = node_outputs.get(parent_id, {})
                    parent_outputs_map[parent_id] = parent_out
                    if "dataframe" in parent_out and isinstance(parent_out["dataframe"], pd.DataFrame):
                        parent_dfs.append(parent_out["dataframe"])
                    node_inputs.update(parent_out)

                # Store multi-parent collections for multi-input nodes (e.g. joins, unions, ensembles)
                node_inputs["parent_outputs"] = parent_outputs_map
                node_inputs["parent_dataframes"] = parent_dfs

                # Handle handle-aware assignment (e.g. left and right handles)
                for edge in incoming_edges:
                    p_out = node_outputs.get(edge.source, {})
                    p_df = p_out.get("dataframe")
                    if isinstance(p_df, pd.DataFrame):
                        th = (getattr(edge, "target_handle", None) or "").lower()
                        if th in ["left", "left_dataset", "df_left", "upstream_left"]:
                            node_inputs["left_dataframe"] = p_df
                            node_inputs["left_parent_id"] = edge.source
                        elif th in ["right", "right_dataset", "df_right", "upstream_right"]:
                            node_inputs["right_dataframe"] = p_df
                            node_inputs["right_parent_id"] = edge.source

                # If left/right dataframes are not explicitly bound by handles, default from parent_dfs
                if "left_dataframe" not in node_inputs and len(parent_dfs) >= 1:
                    node_inputs["left_dataframe"] = parent_dfs[0]
                    if len(parents) >= 1:
                        node_inputs["left_parent_id"] = parents[0]
                if "right_dataframe" not in node_inputs and len(parent_dfs) >= 2:
                    node_inputs["right_dataframe"] = parent_dfs[1]
                    if len(parents) >= 2:
                        node_inputs["right_parent_id"] = parents[1]

                # Fallback to pipeline_context if node requires model/scaler/encoder but immediate parent didn't pass it
                for ctx_key in ["model", "scaler", "encoder", "target_classes", "target_encoder", "feature_names", "imputer_stats", "vectorizer"]:
                    if ctx_key not in node_inputs or node_inputs[ctx_key] is None:
                        if pipeline_context.get(ctx_key) is not None:
                            node_inputs[ctx_key] = pipeline_context[ctx_key]

            # Execution
            try:
                outputs = recipe.execute(inputs=node_inputs, config=node.config, context=pipeline_context)
                node_outputs[node.id] = outputs

                # Propagate standard artifacts to shared context (never overwrite existing valid object with None)
                for key in [
                    "X_test", "y_test", "X_train", "y_train", "dataframe", "forecast_df",
                    "model", "scaler", "encoder", "task_type", "feature_names", "feature_importances",
                    "target_classes", "target_encoder", "target_column", "imputer_stats",
                    "vectorizer", "text_column", "split_mode", "categorical_maps"
                ]:
                    if key in outputs and outputs[key] is not None:
                        pipeline_context[key] = outputs[key]

                if "target_column" in node.config and node.config["target_column"]:
                    pipeline_context["target_column"] = node.config["target_column"]

                # Capture summaries & KPIs
                if "metrics" in outputs:
                    final_metrics = outputs["metrics"]
                if "anomaly_summary" in outputs:
                    anomaly_summary = outputs["anomaly_summary"]
                if "forecasting_summary" in outputs:
                    forecasting_summary = outputs["forecasting_summary"]
                if "governance_record" in outputs:
                    governance_summary = outputs["governance_record"]

                duration_ms = round((time.time() - node_start) * 1000.0, 2)
                logs.append(f"✅ Node `{node.id}` [{recipe.name}] ➔ Finished in {duration_ms}ms (SUCCESS)")

                # Create serializable summary & Step Snapshot (n8n/Boomi step inspection)
                summary: Dict[str, Any] = {}
                snapshot_info: Dict[str, Any] = {
                    "node_id": node.id,
                    "recipe_id": node.recipe_id,
                    "recipe_name": recipe.name,
                    "duration_ms": duration_ms,
                    "input_keys": list(node_inputs.keys()),
                    "output_keys": list(outputs.keys()),
                    "row_count": None,
                    "columns": [],
                    "preview_rows": []
                }

                # Prioritize primary "dataframe" for node snapshot inspection, or first DataFrame found
                primary_df = outputs.get("dataframe")
                if primary_df is None or not isinstance(primary_df, pd.DataFrame):
                    for v in outputs.values():
                        if isinstance(v, pd.DataFrame):
                            primary_df = v
                            break

                for k, v in outputs.items():
                    if isinstance(v, pd.DataFrame):
                        summary[k] = {"shape": list(v.shape), "type": "DataFrame"}
                    elif hasattr(v, "shape"):
                        summary[k] = {"shape": list(v.shape), "type": "Array"}
                    elif k in ["metrics", "anomaly_summary", "forecasting_summary", "feature_importances", "output_summary"]:
                        summary[k] = make_json_safe(v)
                    else:
                        summary[k] = {"type": type(v).__name__}

                if primary_df is not None and isinstance(primary_df, pd.DataFrame):
                    snapshot_info["row_count"] = int(primary_df.shape[0])
                    snapshot_info["columns"] = list(primary_df.columns)
                    try:
                        clean_v = primary_df.head(5).replace({float("nan"): None, float("inf"): None, float("-inf"): None})
                        snapshot_info["preview_rows"] = clean_v.to_dict(orient="records")
                    except Exception:
                        pass

                if "output_summary" in outputs:
                    snapshot_info["output_summary"] = make_json_safe(outputs["output_summary"])

                step_snapshots[node.id] = snapshot_info

                node_results.append(NodeExecutionResult(
                    node_id=node.id,
                    recipe_id=node.recipe_id,
                    status="SUCCESS",
                    duration_ms=duration_ms,
                    output_summary=summary
                ))

            except Exception as e:
                duration_ms = round((time.time() - node_start) * 1000.0, 2)
                diag = diagnose_execution_error(recipe, e, node_inputs, node.config, pipeline_context)
                err_title = diag["title"]
                err_msg = diag["message"]
                err_sugg = diag["suggestion"]

                logs.append(f"❌ Node `{node.id}` [{recipe.name}] failed in {duration_ms}ms: {err_title} ➔ {err_msg}")
                logs.append(f"💡 Suggestion: {err_sugg}")
                logger.error(f"Execution failed on node {node.id} ({err_title}): {traceback.format_exc()}")

                node_results.append(NodeExecutionResult(
                    node_id=node.id,
                    recipe_id=node.recipe_id,
                    status="FAILED",
                    duration_ms=duration_ms,
                    error_message=err_msg,
                    error_title=err_title,
                    error_suggestion=err_sugg
                ))
                overall_status = "FAILED"
                break

        total_duration = round((time.time() - start_time) * 1000.0, 2)
        logs.append(f"🏁 Execution finished with status '{overall_status}' in {total_duration}ms")

        # 4. Construct Inference Bundle & Dynamic Schema
        features_schema = []
        sample_payload = {}
        fn_list = list(pipeline_context.get("feature_names", []))
        
        # Combine train and test splits to compute feature bounds over the full dataset
        X_tr = pipeline_context.get("X_train")
        X_te = pipeline_context.get("X_test")
        if X_tr is not None and X_te is not None and isinstance(X_tr, pd.DataFrame) and isinstance(X_te, pd.DataFrame):
            X_eval = pd.concat([X_tr, X_te], ignore_index=True)
        elif X_te is not None and isinstance(X_te, pd.DataFrame):
            X_eval = X_te
        else:
            X_eval = X_tr

        y_tr = pipeline_context.get("y_train")
        y_te = pipeline_context.get("y_test")
        if y_tr is not None and y_te is not None:
            try:
                y_eval = pd.concat([pd.Series(y_tr), pd.Series(y_te)], ignore_index=True)
            except Exception:
                y_eval = pd.Series(y_tr)
        elif y_tr is not None:
            y_eval = pd.Series(y_tr)
        elif y_te is not None:
            y_eval = pd.Series(y_te)
        else:
            y_eval = None

        # General fallback so feature_trends / seasonal_profile / entity_value_sets /
        # last_historical_row work for EVERY model type, not only tabular
        # classifiers/regressors trained via a train_test_split node. Forecasting recipes
        # (Prophet, ARIMA) train directly on (ds, y) and never produce X_train/X_test, so
        # without this fallback that entire section of the bundle silently stays empty for
        # every forecasting pipeline. When no tabular split exists, fall back to computing
        # everything from the raw input dataset instead.
        if X_eval is None:
            # NOTE: pipeline_context["dataframe"] is NOT safe to use here — it gets
            # overwritten by whatever the last executed node returns in its own
            # "dataframe" output key (e.g. Prophet's own forecast table), so by this point
            # it may no longer be the original uploaded data at all. The initial_df
            # function parameter is never reassigned during execution, so it's the only
            # reliable reference to the true raw input dataset.
            raw_df = initial_df
            if isinstance(raw_df, pd.DataFrame) and not raw_df.empty:
                target_col_name = pipeline_context.get("target_column")
                exclude_cols = {c for c in (target_col_name, "y") if c}
                X_eval = raw_df.drop(columns=[c for c in exclude_cols if c in raw_df.columns], errors="ignore")
                if y_eval is None and target_col_name and target_col_name in raw_df.columns:
                    y_eval = raw_df[target_col_name]

                # Forecasting recipes train on a date column (commonly 'ds') that is a
                # string, not a numeric feature — has_temporal detection and the
                # slope-based feature_trends computation below both need it as a number.
                # Convert it to a fractional-year axis, the same role a numeric Date_year
                # column already plays for tabular pipelines.
                date_col_candidate = "ds" if "ds" in X_eval.columns else next(
                    (c for c in X_eval.columns if "date" in c.lower()), None
                )
                if date_col_candidate:
                    try:
                        parsed = pd.to_datetime(X_eval[date_col_candidate], errors="coerce")
                        if parsed.notna().sum() >= 5:
                            X_eval[date_col_candidate] = parsed.dt.year + (parsed.dt.dayofyear - 1) / 365.25
                    except Exception:
                        pass

        if X_eval is not None and isinstance(X_eval, pd.DataFrame):
            if not fn_list:
                fn_list = list(X_eval.columns)
            for col in fn_list:
                if col in X_eval.columns:
                    s = X_eval[col]
                    if pd.api.types.is_numeric_dtype(s):
                        mn = float(s.min()) if not s.empty and pd.notna(s.min()) else 0.0
                        mx = float(s.max()) if not s.empty and pd.notna(s.max()) else 100.0
                        med = float(s.median()) if not s.empty and pd.notna(s.median()) else 0.0
                        mean_v = float(s.mean()) if not s.empty and pd.notna(s.mean()) else 0.0
                        features_schema.append({
                            "name": col,
                            "data_type": "numeric",
                            "min_value": round(mn, 2),
                            "max_value": round(mx, 2),
                            "median_value": round(med, 2),
                            "mean_value": round(mean_v, 2),
                            "default_value": round(med, 2)
                        })
                        sample_payload[col] = round(med, 2)
                    else:
                        cats = [str(v) for v in s.dropna().unique()[:30]]
                        def_v = cats[0] if cats else "Unknown"
                        features_schema.append({
                            "name": col,
                            "data_type": "categorical",
                            "allowed_categories": cats,
                            "default_value": def_v
                        })
                        sample_payload[col] = def_v
        elif fn_list:
            for col in fn_list:
                features_schema.append({
                    "name": col,
                    "data_type": "numeric",
                    "default_value": 0.0
                })
                sample_payload[col] = 0.0

        # Detect temporal / year feature for future projection support
        has_temporal = False
        temporal_col = None
        min_year = None
        max_year = None
        annual_trend_pct = None

        for f in features_schema:
            fn_low = f["name"].lower()
            if any(k in fn_low for k in ["year", "date", "time", "timestamp", "period", "ds"]):
                has_temporal = True
                if not temporal_col:
                    temporal_col = f["name"]
                if "year" in fn_low and f.get("min_value") is not None and f.get("max_value") is not None:
                    if 1900 <= f["min_value"] <= 2100:
                        min_year = int(f["min_value"])
                        max_year = int(f["max_value"])

        # Compute empirical annual trend % if temporal regression target exists
        if has_temporal and temporal_col and y_eval is not None and X_eval is not None and temporal_col in X_eval.columns:
            try:
                t_series = pd.to_numeric(X_eval[temporal_col], errors="coerce")
                y_series = pd.to_numeric(y_eval, errors="coerce")
                valid_mask = t_series.notna() & y_series.notna()
                if valid_mask.sum() >= 5:
                    t_clean = t_series[valid_mask].values
                    y_clean = y_series[valid_mask].values
                    if np.std(t_clean) > 0 and abs(np.mean(y_clean)) > 1e-6:
                        slope, _ = np.polyfit(t_clean, y_clean, 1)
                        calc_trend = (slope / abs(np.mean(y_clean))) * 100.0
                        annual_trend_pct = round(float(calc_trend), 2)
            except Exception as e:
                logger.warning(f"Could not compute annual_trend_pct: {str(e)}")

        # Compute empirical per-feature drift (slope vs. the temporal column) for every
        # continuous numeric driver feature. This lets future-year projections evolve
        # each feature forward using its own historical trend (e.g. CPI, Unemployment,
        # Fuel_Price naturally drifting) instead of freezing every input at the value the
        # caller supplied for the base year. Calendar-position sub-features (month, day,
        # day-of-week, quarter...) and low-cardinality numeric columns (flags, IDs, store
        # numbers) are intentionally excluded and left frozen at the caller-supplied value,
        # since "trending" those would be meaningless.
        feature_trends: Dict[str, Dict[str, float]] = {}
        _DATE_SUBCOMPONENT_HINTS = ("month", "day", "dayofweek", "day_of_week", "quarter", "week", "hour", "minute", "second")
        if has_temporal and temporal_col and X_eval is not None and temporal_col in X_eval.columns:
            try:
                t_series_full = pd.to_numeric(X_eval[temporal_col], errors="coerce")
                for col in fn_list:
                    if col == temporal_col or col not in X_eval.columns:
                        continue
                    col_low = col.lower()
                    if any(h in col_low for h in _DATE_SUBCOMPONENT_HINTS):
                        continue
                    s = X_eval[col]
                    if not pd.api.types.is_numeric_dtype(s):
                        continue
                    unique_vals = s.dropna().unique()
                    if len(unique_vals) <= 10:
                        # Looks like a flag/ID/coded-category rather than a continuous driver.
                        continue
                    v_series = pd.to_numeric(s, errors="coerce")
                    valid_mask = t_series_full.notna() & v_series.notna()
                    if valid_mask.sum() < 5:
                        continue
                    t_clean = t_series_full[valid_mask].values
                    v_clean = v_series[valid_mask].values
                    if np.std(t_clean) > 0:
                        slope, _intercept = np.polyfit(t_clean, v_clean, 1)
                        mean_v = float(np.mean(v_clean))
                        feature_trends[col] = {
                            "slope_per_unit_time": round(float(slope), 6),
                            "pct_per_unit_time": round((float(slope) / abs(mean_v)) * 100.0, 4) if abs(mean_v) > 1e-9 else 0.0
                        }
            except Exception as e:
                logger.warning(f"Could not compute feature_trends: {str(e)}")
        # Anchor point for a "pure" future forecast (caller supplied no explicit inputs):
        # the actual last real historical record, not a synthetic dataset-wide median.
        # sample_payload above is intentionally a median/default row for cold-start single
        # predictions; last_historical_row reflects "where the real data actually left off"
        # so an unprompted forecast genuinely continues from history.
        last_historical_row: Dict[str, Any] = dict(sample_payload)
        if has_temporal and temporal_col and X_eval is not None and temporal_col in X_eval.columns:
            try:
                t_series_last = pd.to_numeric(X_eval[temporal_col], errors="coerce")
                if t_series_last.notna().any():
                    last_idx = t_series_last.idxmax()
                    last_row = X_eval.loc[last_idx]
                    for col in fn_list:
                        if col not in last_row.index:
                            continue
                        val = last_row[col]
                        if pd.isna(val):
                            continue
                        try:
                            last_historical_row[col] = round(float(val), 2)
                        except (TypeError, ValueError):
                            last_historical_row[col] = val
            except Exception as e:
                logger.warning(f"Could not compute last_historical_row: {str(e)}")

        # Capture distinct entity value sets (e.g. Stores 1..45) and seasonal profile (historical feature averages per week/month)
        entity_value_sets: Dict[str, List[Any]] = {}
        seasonal_profile: Dict[str, Dict[str, float]] = {}
        if X_eval is not None and isinstance(X_eval, pd.DataFrame):
            try:
                for col in fn_list:
                    if col not in X_eval.columns:
                        continue
                    s = X_eval[col].dropna()
                    uniques = s.unique()
                    # Low-cardinality entity grouping column (e.g., Store IDs, Category IDs, Flag columns)
                    if 1 < len(uniques) <= 50:
                        entity_value_sets[col] = [
                            int(v) if isinstance(v, (np.integer, int)) else (float(v) if isinstance(v, (np.floating, float)) else str(v))
                            for v in uniques
                        ]

                # Seasonal profiling: group continuous numeric features by sub-year calendar components (e.g. Date_week, Date_month)
                sub_date_cols = [
                    c for c in fn_list
                    if c in X_eval.columns and any(
                        h in c.lower().replace("_", " ").split() for h in ("month", "dayofweek", "day_of_week", "quarter", "week", "day")
                    ) and not c.lower().startswith("holiday")
                ]
                if sub_date_cols:
                    step_sub_col = sub_date_cols[0]
                    for col in fn_list:
                        if col == step_sub_col or col not in X_eval.columns:
                            continue
                        if pd.api.types.is_numeric_dtype(X_eval[col]):
                            grp_means = X_eval.groupby(step_sub_col)[col].mean().to_dict()
                            seasonal_profile[col] = {
                                str(k): round(float(v), 4)
                                for k, v in grp_means.items() if pd.notna(v)
                            }
            except Exception as e:
                logger.warning(f"Could not compute entity_value_sets or seasonal_profile: {str(e)}")

        # Resolve time-series frequency if available
        resolved_freq = (
            pipeline_context.get("frequency")
            or pipeline_context.get("freq")
            or pipeline_context.get("forecasting_summary", {}).get("frequency")
            or pipeline_context.get("forecasting_summary", {}).get("freq")
            or pipeline_context.get("forecasting_summary", {}).get("data_frequency")
        )

        inference_schema = {
            "execution_id": execution_id,
            "task_type": pipeline_context.get("task_type", "classification"),
            "target_column": pipeline_context.get("target_column"),
            "target_classes": pipeline_context.get("target_classes", []),
            "features": features_schema,
            "sample_payload": sample_payload,
            "last_historical_row": last_historical_row,
            "time_series_meta": pipeline_context.get("forecasting_summary"),
            "has_temporal_feature": has_temporal,
            "temporal_column": temporal_col,
            "min_year": min_year,
            "max_year": max_year,
            "annual_trend_pct": annual_trend_pct,
            "feature_trends": feature_trends,
            "entity_value_sets": entity_value_sets,
            "seasonal_profile": seasonal_profile,
            "frequency": resolved_freq,
            "freq": resolved_freq,
        }

        # Ensure model is preserved even if a downstream node produced outputs or was ordered differently
        resolved_model = pipeline_context.get("model")
        if resolved_model is None:
            for n_out in node_outputs.values():
                if isinstance(n_out, dict) and n_out.get("model") is not None:
                    resolved_model = n_out["model"]
                    break

        if resolved_model is not None and not resolved_freq:
            resolved_freq = getattr(resolved_model, "saved_freq", None)

        inference_bundle = {
            "execution_id": execution_id,
            "task_type": pipeline_context.get("task_type", "classification"),
            "model": resolved_model,
            "feature_names": fn_list,
            "target_column": pipeline_context.get("target_column"),
            "target_classes": pipeline_context.get("target_classes", []),
            "target_encoder": pipeline_context.get("target_encoder"),
            "scaler": pipeline_context.get("scaler"),
            "encoder": pipeline_context.get("encoder"),
            "vectorizer": pipeline_context.get("vectorizer"),
            "text_column": pipeline_context.get("text_column"),
            "imputer_stats": pipeline_context.get("imputer_stats", {}),
            "forecasting_summary": pipeline_context.get("forecasting_summary", {}),
            "sample_row": sample_payload,
            "last_historical_row": last_historical_row,
            "training_feature_summary": {f["name"]: f for f in features_schema},
            "has_temporal_feature": has_temporal,
            "temporal_column": temporal_col,
            "min_year": min_year,
            "max_year": max_year,
            "annual_trend_pct": annual_trend_pct,
            "feature_trends": feature_trends,
            "entity_value_sets": entity_value_sets,
            "seasonal_profile": seasonal_profile,
            "frequency": resolved_freq,
            "freq": resolved_freq,
            "split_mode": pipeline_context.get("split_mode"),
            "categorical_maps": pipeline_context.get("categorical_maps", {}),
            "historical_records": (
                pd.concat([X_eval, pd.Series(y_eval, name=pipeline_context.get("target_column") or "target")], axis=1)
                .tail(1000)
                .replace({float("nan"): None, float("inf"): None, float("-inf"): None})
                .to_dict(orient="records")
            ) if (X_eval is not None and isinstance(X_eval, pd.DataFrame) and y_eval is not None and len(y_eval) == len(X_eval)) else (
                X_eval.tail(1000).replace({float("nan"): None, float("inf"): None, float("-inf"): None}).to_dict(orient="records")
                if (X_eval is not None and isinstance(X_eval, pd.DataFrame)) else []
            ),
        }

        try:
            from backend.app.engine.execution.job_manager import job_manager
            job_manager.register_inference_bundle(execution_id, inference_bundle)
        except Exception as e:
            logger.warning(f"Could not register inference bundle with job_manager: {str(e)}")

        # Sanitize all outputs to be 100% JSON serializable for FastAPI responses
        safe_node_outputs = make_json_safe(node_outputs) if include_node_outputs else {}
        safe_final_metrics = make_json_safe(final_metrics)
        safe_anomaly_summary = make_json_safe(anomaly_summary)
        safe_forecasting_summary = make_json_safe(forecasting_summary)
        safe_governance_summary = make_json_safe(governance_summary)
        safe_step_snapshots = make_json_safe(step_snapshots)
        safe_inference_schema = make_json_safe(inference_schema)

        return WorkflowExecutionResult(
            execution_id=execution_id,
            status=overall_status,
            total_duration_ms=total_duration,
            node_results=node_results,
            final_metrics=safe_final_metrics,
            anomaly_summary=safe_anomaly_summary,
            forecasting_summary=safe_forecasting_summary,
            governance_summary=safe_governance_summary,
            logs=logs,
            node_outputs=safe_node_outputs,
            step_snapshots=safe_step_snapshots,
            inference_schema=safe_inference_schema
        )