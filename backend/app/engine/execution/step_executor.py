import os
import json
import time
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import defaultdict, deque

import joblib
import pandas as pd
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from backend.app.infrastructure.storage.storage_manager import storage_manager
from backend.app.recipes.base.registry import recipe_registry
from backend.app.datasets.models import Dataset
from backend.app.engine.execution.executor import diagnose_execution_error, make_json_safe
from backend.app.workflows.schemas import (
    NodeColumnInfo,
    NodePortSchema,
    NodeTestStepResponse,
    SessionClearResponse
)

logger = logging.getLogger(__name__)


class NodeStepExecutor:
    """
    Interactive Single-Node / Step-by-Step Execution Engine.
    Powers n8n/Boomi-style "Test Step" functionality in the canvas.
    Caches intermediate DataFrames and ML artifacts per test session,
    auto-resolves upstream dependencies, and provides instant table/schema previews.
    """

    @classmethod
    def get_session_dir(cls, session_id: str) -> Path:
        """Returns the absolute filesystem directory for this test session's cache."""
        clean_sid = "".join(c for c in session_id if c.isalnum() or c in ("-", "_")).strip()
        if not clean_sid:
            clean_sid = "default_session"
        # Per-tenant folder: the same session_id in another tenant is a different session.
        from backend.app.core.tenant_context import tenant_cache_dir
        p = storage_manager.get_absolute_path(f"{tenant_cache_dir()}/test_sessions/{clean_sid}")
        p.mkdir(parents=True, exist_ok=True)
        return p

    @classmethod
    def is_node_cached(cls, session_id: str, node_id: str) -> bool:
        """Checks if a node has successfully executed and cached its outputs in this session."""
        sdir = cls.get_session_dir(session_id)
        meta_file = sdir / f"node_{node_id}_meta.json"
        if not meta_file.exists():
            return False
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            return meta.get("status") == "SUCCESS"
        except Exception:
            return False

    @classmethod
    def clear_session(cls, session_id: str) -> SessionClearResponse:
        """Cleans up all cached parquet files and artifacts for a session."""
        sdir = cls.get_session_dir(session_id)
        try:
            if sdir.exists():
                shutil.rmtree(sdir)
            return SessionClearResponse(
                success=True,
                session_id=session_id,
                message=f"Session cache '{session_id}' cleared successfully."
            )
        except Exception as e:
            logger.warning(f"Error clearing session '{session_id}': {e}")
            return SessionClearResponse(
                success=False,
                session_id=session_id,
                message=f"Failed to clear session: {str(e)}"
            )

    @classmethod
    def save_node_output(
        cls,
        session_id: str,
        node_id: str,
        recipe_id: str,
        outputs: Dict[str, Any],
        duration_ms: float = 0.0,
        available_left_columns: Optional[List[str]] = None,
        available_right_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Saves DataFrames to Parquet, ML artifacts to joblib, and summary metadata to JSON.
        """
        sdir = cls.get_session_dir(session_id)
        
        # 1. Save Primary DataFrame
        primary_df = outputs.get("dataframe")
        if primary_df is None or not isinstance(primary_df, pd.DataFrame):
            for v in outputs.values():
                if isinstance(v, pd.DataFrame):
                    primary_df = v
                    break

        if isinstance(primary_df, pd.DataFrame):
            df_file = sdir / f"node_{node_id}_df.parquet"
            primary_df.to_parquet(df_file, index=False)

        # 2. Save Port-Specific DataFrames
        port_mappings = {
            "dataframe_train": "train",
            "dataframe_test": "test",
            "matched_dataframe": "joined",
            "left_unmatched_dataframe": "unmatched_left",
            "right_unmatched_dataframe": "unmatched_right"
        }
        for out_key, port_name in port_mappings.items():
            if out_key in outputs and isinstance(outputs[out_key], pd.DataFrame):
                p_file = sdir / f"node_{node_id}_port_{port_name}.parquet"
                outputs[out_key].to_parquet(p_file, index=False)

        # Also save any other named DataFrames
        for k, v in outputs.items():
            if isinstance(v, pd.DataFrame) and k not in port_mappings and k != "dataframe":
                clean_k = "".join(c for c in k if c.isalnum() or c == "_")
                p_file = sdir / f"node_{node_id}_port_{clean_k}.parquet"
                v.to_parquet(p_file, index=False)

        # 3. Save Non-DataFrame ML Artifacts via Joblib
        artifacts = {}
        for key in [
            "model", "scaler", "encoder", "task_type", "feature_names", "feature_importances",
            "target_classes", "target_encoder", "target_column", "imputer_stats",
            "vectorizer", "text_column", "split_mode", "categorical_maps",
            "X_train", "y_train", "X_test", "y_test", "metrics", "output_summary"
        ]:
            if key in outputs and outputs[key] is not None:
                val = outputs[key]
                if not isinstance(val, pd.DataFrame):
                    artifacts[key] = val

        if artifacts:
            art_file = sdir / f"node_{node_id}_artifacts.joblib"
            try:
                joblib.dump(artifacts, art_file)
            except Exception as e:
                logger.warning(f"Failed to dump artifacts for node '{node_id}': {e}")

        # 4. Construct Metadata & Preview
        columns = list(primary_df.columns) if isinstance(primary_df, pd.DataFrame) else []
        col_types = {str(col): str(dtype) for col, dtype in primary_df.dtypes.items()} if isinstance(primary_df, pd.DataFrame) else {}
        row_count = int(primary_df.shape[0]) if isinstance(primary_df, pd.DataFrame) else None
        
        preview_rows = []
        if isinstance(primary_df, pd.DataFrame):
            try:
                clean_head = primary_df.head(10).replace({float("nan"): None, float("inf"): None, float("-inf"): None})
                preview_rows = clean_head.to_dict(orient="records")
            except Exception:
                pass

        output_summary = make_json_safe(outputs.get("output_summary") or {})
        metrics = make_json_safe(outputs.get("metrics") or {})

        meta = {
            "node_id": node_id,
            "recipe_id": recipe_id,
            "status": "SUCCESS",
            "duration_ms": duration_ms,
            "row_count": row_count,
            "columns": columns,
            "column_types": col_types,
            "preview_rows": preview_rows,
            "output_summary": output_summary,
            "metrics": metrics,
            "available_left_columns": available_left_columns or [],
            "available_right_columns": available_right_columns or [],
            "output_keys": list(outputs.keys()),
            "timestamp": time.time()
        }

        meta_file = sdir / f"node_{node_id}_meta.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return meta

    @classmethod
    def load_node_output(cls, session_id: str, node_id: str) -> Dict[str, Any]:
        """
        Reconstructs the outputs dictionary for a cached node from disk.
        """
        sdir = cls.get_session_dir(session_id)
        outputs: Dict[str, Any] = {}

        # 1. Primary DataFrame
        df_file = sdir / f"node_{node_id}_df.parquet"
        if df_file.exists():
            outputs["dataframe"] = pd.read_parquet(df_file)

        # 2. Port DataFrames
        for p_file in sdir.glob(f"node_{node_id}_port_*.parquet"):
            port_name = p_file.stem.replace(f"node_{node_id}_port_", "")
            try:
                p_df = pd.read_parquet(p_file)
                if port_name == "train":
                    outputs["dataframe_train"] = p_df
                elif port_name == "test":
                    outputs["dataframe_test"] = p_df
                elif port_name == "joined":
                    outputs["matched_dataframe"] = p_df
                elif port_name == "unmatched_left":
                    outputs["left_unmatched_dataframe"] = p_df
                elif port_name == "unmatched_right":
                    outputs["right_unmatched_dataframe"] = p_df
                else:
                    outputs[port_name] = p_df
            except Exception as e:
                logger.warning(f"Error loading port parquet {p_file}: {e}")

        # 3. Artifacts (joblib)
        art_file = sdir / f"node_{node_id}_artifacts.joblib"
        if art_file.exists():
            try:
                artifacts = joblib.load(art_file)
                outputs.update(artifacts)
            except Exception as e:
                logger.warning(f"Error loading artifacts for node {node_id}: {e}")

        return outputs

    @classmethod
    def load_node_meta(cls, session_id: str, node_id: str) -> Optional[Dict[str, Any]]:
        """Reads node metadata JSON if cached."""
        sdir = cls.get_session_dir(session_id)
        meta_file = sdir / f"node_{node_id}_meta.json"
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    @classmethod
    async def execute_node_step(
        cls,
        target_node_id: str,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        node_configs: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        sample_size: Optional[int] = None,
        auto_run_ancestors: bool = True,
        dry_run: bool = False,
        db: Optional[AsyncSession] = None,
        tenant_id: Optional[int] = None
    ) -> NodeTestStepResponse:
        """
        Executes a single visual canvas node in isolation or auto-resolves unexecuted ancestors.
        Returns immediate schema, preview rows, metrics, and execution diagnostics.
        """
        from backend.app.core.tenant_context import get_current_tenant_id, with_tenant_config
        if tenant_id is None:
            tenant_id = get_current_tenant_id()
        session_id = session_id or "default_session"
        node_configs = node_configs or {}
        node_map: Dict[str, Dict[str, Any]] = {str(n.get("id")): n for n in nodes if n.get("id")}

        if target_node_id not in node_map:
            return NodeTestStepResponse(
                success=False,
                node_id=target_node_id,
                recipe_id="unknown",
                status="FAILED",
                error_message=f"Target node '{target_node_id}' does not exist in workflow."
            )

        # 1. Build Graph Dependency Maps
        in_degree: Dict[str, int] = {nid: 0 for nid in node_map}
        adj_list: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        in_edges_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        parent_map: Dict[str, List[str]] = {nid: [] for nid in node_map}

        for edge in edges:
            src = str(edge.get("source", ""))
            tgt = str(edge.get("target", ""))
            if src in node_map and tgt in node_map:
                adj_list[src].append(edge)
                in_edges_map[tgt].append(edge)
                parent_map[tgt].append(src)
                in_degree[tgt] += 1

        # 2. Find All Ancestor Nodes for Target Node (Reverse BFS/DFS)
        ancestors: Set[str] = set()
        queue = deque(parent_map[target_node_id])
        while queue:
            curr = queue.popleft()
            if curr not in ancestors:
                ancestors.add(curr)
                for p in parent_map[curr]:
                    if p not in ancestors:
                        queue.append(p)

        # 3. Topological Sort of Ancestors
        # Kahn's algorithm restricted to the ancestors subgraph
        subgraph_nodes = ancestors | {target_node_id}
        sub_in_degree = {nid: 0 for nid in subgraph_nodes}
        for src in subgraph_nodes:
            for edge in adj_list[src]:
                tgt = str(edge.get("target", ""))
                if tgt in subgraph_nodes:
                    sub_in_degree[tgt] += 1

        sub_queue = deque([nid for nid in subgraph_nodes if sub_in_degree[nid] == 0])
        topo_order: List[str] = []
        while sub_queue:
            curr = sub_queue.popleft()
            topo_order.append(curr)
            for edge in adj_list[curr]:
                tgt = str(edge.get("target", ""))
                if tgt in subgraph_nodes:
                    sub_in_degree[tgt] -= 1
                    if sub_in_degree[tgt] == 0:
                        sub_queue.append(tgt)

        # Add any disconnected ancestors
        for nid in subgraph_nodes:
            if nid not in topo_order:
                topo_order.append(nid)

        # 4. Check Ancestors Execution & Auto-Run Unexecuted Ancestors
        ancestors_to_run: List[str] = []
        for nid in topo_order:
            if nid != target_node_id:
                if not cls.is_node_cached(session_id, nid):
                    ancestors_to_run.append(nid)

        if ancestors_to_run and not auto_run_ancestors:
            missing_names = [f"'{nid}' ({node_map[nid].get('recipe_id')})" for nid in ancestors_to_run]
            return NodeTestStepResponse(
                success=False,
                node_id=target_node_id,
                recipe_id=str(node_map[target_node_id].get("recipe_id", "")),
                status="FAILED",
                error_message=(
                    f"Upstream ancestor node(s) {', '.join(missing_names)} have not been executed yet in this session. "
                    f"Enable 'auto_run_ancestors=True' to automatically run them, or test upstream nodes first."
                )
            )

        ancestors_executed: List[str] = []

        # Helper to execute a single node (ancestor or target)
        async def _execute_single(nid: str, is_target: bool = False) -> Tuple[Optional[Dict[str, Any]], Optional[NodeTestStepResponse]]:
            n_start = time.time()
            n_obj = node_map[nid]
            r_id = str(n_obj.get("recipe_id", "")).lower().strip()

            try:
                recipe = recipe_registry.get(r_id)
            except Exception as e:
                from backend.app.core.exceptions import ValidationException
                err_msg = e.message if isinstance(e, ValidationException) else f"Recipe '{r_id}' not found in registry: {str(e)}"
                return None, NodeTestStepResponse(
                    success=False,
                    node_id=nid,
                    recipe_id=r_id,
                    status="FAILED",
                    error_message=err_msg
                )

            # Build Node Config
            merged_config = {**n_obj.get("config", {}), **node_configs.get(nid, {})}
            if dry_run and is_target:
                merged_config["dry_run"] = True

            # Resolve Inputs
            node_inputs: Dict[str, Any] = {}
            incoming_edges = in_edges_map.get(nid, [])
            parents = parent_map.get(nid, [])

            available_left_cols: List[str] = []
            available_right_cols: List[str] = []

            if not parents:
                # Root Loader Node
                if r_id in ["csv_loader", "dataset_loader", "file_ingestion"]:
                    # If dataset_id is present, check DB if not in inputs
                    dataset_id = merged_config.get("dataset_id") or n_obj.get("dataset_id")
                    if not dataset_id and "dataset" in merged_config:
                        dataset_id = merged_config["dataset"]
                    if dataset_id and db and "dataframe" not in node_inputs:
                        try:
                            ds_query = select(Dataset).where(Dataset.id == str(dataset_id))
                            if tenant_id is not None:
                                # Only the caller's own tenant's datasets can be loaded.
                                ds_query = ds_query.where(Dataset.tenant_id == tenant_id)
                            res = await db.execute(ds_query)
                            ds_obj = res.scalar_one_or_none()
                            if ds_obj and ds_obj.storage_path:
                                # Datasets live in Ozone (s3://bucket/ml-workflow/t<id>/...);
                                # read_dataframe also enforces the tenant-folder check.
                                load_df = storage_manager.read_dataframe(
                                    ds_obj.storage_path,
                                    tenant_id if tenant_id is not None else ds_obj.tenant_id
                                )
                                if sample_size:
                                    load_df = load_df.head(sample_size)
                                node_inputs["dataframe"] = load_df
                        except Exception as ex:
                            logger.warning(f"Could not preload dataset '{dataset_id}': {ex}")
            else:
                # Load cached outputs from parents
                parent_dfs: List[pd.DataFrame] = []
                parent_outputs_map: Dict[str, Dict[str, Any]] = {}

                for pid in parents:
                    p_out = cls.load_node_output(session_id, pid)
                    parent_outputs_map[pid] = p_out
                    if "dataframe" in p_out and isinstance(p_out["dataframe"], pd.DataFrame):
                        parent_dfs.append(p_out["dataframe"])
                    node_inputs.update(p_out)

                node_inputs["parent_outputs"] = parent_outputs_map
                node_inputs["parent_dataframes"] = parent_dfs

                # Handle handle-aware assignment
                for edge in incoming_edges:
                    src_id = edge.get("source")
                    p_out = parent_outputs_map.get(src_id, {})
                    sh = (edge.get("source_handle") or "").lower().strip()

                    # Train/test partition routing
                    if sh in ["output_2", "test_data", "test", "te"]:
                        if "X_test" in p_out:
                            node_inputs["X_test"] = p_out["X_test"]
                        if "y_test" in p_out:
                            node_inputs["y_test"] = p_out["y_test"]
                        if "test_data" in p_out:
                            node_inputs["test_data"] = p_out["test_data"]
                    elif sh in ["output_1", "train_data", "train", "tr"]:
                        if "X_train" in p_out:
                            node_inputs["X_train"] = p_out["X_train"]
                        if "y_train" in p_out:
                            node_inputs["y_train"] = p_out["y_train"]
                        if "train_data" in p_out:
                            node_inputs["train_data"] = p_out["train_data"]

                    # Resolve output dataframe partition
                    p_df = None
                    if sh in ["unmatched_left", "left_unmatched"]:
                        p_df = p_out.get("left_unmatched_dataframe")
                        if p_df is None:
                            p_df = p_out.get("dataframe")
                    elif sh in ["unmatched_right", "right_unmatched"]:
                        p_df = p_out.get("right_unmatched_dataframe")
                        if p_df is None:
                            p_df = p_out.get("dataframe")
                    elif sh in ["joined", "matched", "matched_dataframe"]:
                        p_df = p_out.get("matched_dataframe")
                        if p_df is None:
                            p_df = p_out.get("dataframe")
                    elif sh in ["output_2", "test_data", "test", "te"] and "dataframe_test" in p_out:
                        p_df = p_out["dataframe_test"]
                    elif sh in ["output_1", "train_data", "train", "tr"] and "dataframe_train" in p_out:
                        p_df = p_out["dataframe_train"]
                    elif sh and sh in p_out and isinstance(p_out[sh], pd.DataFrame):
                        p_df = p_out[sh]
                    else:
                        p_df = p_out.get("dataframe")

                    if isinstance(p_df, pd.DataFrame):
                        th = (edge.get("target_handle") or "").lower().strip()
                        if th in ["left", "left_dataset", "df_left", "upstream_left"]:
                            node_inputs["left_dataframe"] = p_df
                            node_inputs["left_parent_id"] = src_id
                            available_left_cols = list(p_df.columns)
                        elif th in ["right", "right_dataset", "df_right", "upstream_right"]:
                            node_inputs["right_dataframe"] = p_df
                            node_inputs["right_parent_id"] = src_id
                            available_right_cols = list(p_df.columns)
                        else:
                            if len(incoming_edges) == 1 or not th or th in ["input", "in", "dataframe"]:
                                node_inputs["dataframe"] = p_df

                # Fallback to positional parent dataframes
                if "left_dataframe" not in node_inputs and len(parent_dfs) >= 1:
                    node_inputs["left_dataframe"] = parent_dfs[0]
                    available_left_cols = list(parent_dfs[0].columns)
                    if len(parents) >= 1:
                        node_inputs["left_parent_id"] = parents[0]
                if "right_dataframe" not in node_inputs and len(parent_dfs) >= 2:
                    node_inputs["right_dataframe"] = parent_dfs[1]
                    available_right_cols = list(parent_dfs[1].columns)
                    if len(parents) >= 2:
                        node_inputs["right_parent_id"] = parents[1]

            # Execute Recipe
            try:
                # Session Context for model/scaler propagation
                session_context = dict(node_inputs)
                outputs = recipe.execute(inputs=node_inputs, config=with_tenant_config(merged_config), context=session_context)
                duration_ms = round((time.time() - n_start) * 1000.0, 2)

                # Save output to session cache
                meta = cls.save_node_output(
                    session_id=session_id,
                    node_id=nid,
                    recipe_id=r_id,
                    outputs=outputs,
                    duration_ms=duration_ms,
                    available_left_columns=available_left_cols,
                    available_right_columns=available_right_cols
                )

                response = NodeTestStepResponse(
                    success=True,
                    session_id=session_id,
                    node_id=nid,
                    recipe_id=r_id,
                    recipe_name=getattr(recipe, "name", r_id),
                    status="SUCCESS",
                    duration_ms=duration_ms,
                    row_count=meta.get("row_count"),
                    columns=meta.get("columns", []),
                    column_types=meta.get("column_types", {}),
                    preview_rows=meta.get("preview_rows", []),
                    output_summary=meta.get("output_summary", {}),
                    metrics=meta.get("metrics"),
                    available_left_columns=available_left_cols,
                    available_right_columns=available_right_cols
                )
                return outputs, response

            except Exception as ex:
                duration_ms = round((time.time() - n_start) * 1000.0, 2)
                diag = diagnose_execution_error(
                    recipe=recipe,
                    error=ex,
                    inputs=node_inputs,
                    config=merged_config,
                    context=node_inputs
                )
                return None, NodeTestStepResponse(
                    success=False,
                    node_id=nid,
                    recipe_id=r_id,
                    recipe_name=getattr(recipe, "name", r_id),
                    status="FAILED",
                    duration_ms=duration_ms,
                    error_message=str(ex),
                    error_title=diag.get("title"),
                    error_suggestion=diag.get("suggestion")
                )

        # 5. Run Ancestors that Need Running
        for anc_id in ancestors_to_run:
            logger.info(f"Auto-running ancestor node '{anc_id}' for target '{target_node_id}' in session '{session_id}'")
            anc_out, anc_resp = await _execute_single(anc_id, is_target=False)
            if anc_resp and anc_resp.status == "FAILED":
                anc_resp.error_message = f"Ancestor node '{anc_id}' failed: {anc_resp.error_message}"
                return anc_resp
            ancestors_executed.append(anc_id)

        # 6. Run the Target Node
        logger.info(f"Executing target test step '{target_node_id}' in session '{session_id}'")
        target_out, target_resp = await _execute_single(target_node_id, is_target=True)
        if target_resp:
            target_resp.ancestors_executed = ancestors_executed
            return target_resp

        return NodeTestStepResponse(
            success=False,
            node_id=target_node_id,
            recipe_id=str(node_map[target_node_id].get("recipe_id", "")),
            status="FAILED",
            error_message="Unexpected error during node step execution."
        )