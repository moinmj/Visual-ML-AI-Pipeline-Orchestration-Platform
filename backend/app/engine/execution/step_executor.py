import os
import asyncio
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
        if not p.exists():
            for fb_prefix in ["t_none/test_sessions", "test_sessions"]:
                fb = storage_manager.get_absolute_path(f"{fb_prefix}/{clean_sid}")
                if fb.exists() and (fb / f"node_loader_api_df.parquet").exists() or (fb.exists() and any(fb.glob("node_*.*"))):
                    return fb
        p.mkdir(parents=True, exist_ok=True)
        return p

    @classmethod
    def is_node_cached(cls, session_id: str, node_id: str) -> bool:
        """Checks if a node has successfully executed and cached its outputs in this session."""
        sdir = cls.get_session_dir(session_id)
        clean_nid = "".join(c for c in str(node_id) if c.isalnum() or c in ("-", "_")).strip() or "node"
        meta_file = sdir / f"node_{clean_nid}_meta.json"
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
        clean_nid = "".join(c for c in str(node_id) if c.isalnum() or c in ("-", "_")).strip() or "node"
        
        # 1. Save Primary DataFrame
        primary_df = outputs.get("dataframe")
        if primary_df is None or not isinstance(primary_df, pd.DataFrame):
            for v in outputs.values():
                if isinstance(v, pd.DataFrame):
                    primary_df = v
                    break

        # Fallback: synthesize tabular view for governance/tracking/evaluator nodes
        if primary_df is None or not isinstance(primary_df, pd.DataFrame):
            if outputs.get("governance_record") and isinstance(outputs["governance_record"], dict):
                primary_df = pd.DataFrame([outputs["governance_record"]])
            elif outputs.get("metrics") and isinstance(outputs["metrics"], dict) and len(outputs["metrics"]) > 0:
                scalar_m = {k: v for k, v in outputs["metrics"].items() if isinstance(v, (int, float, str, bool))}
                if scalar_m:
                    primary_df = pd.DataFrame([scalar_m])

        if isinstance(primary_df, pd.DataFrame):
            df_file = sdir / f"node_{clean_nid}_df.parquet"
            storage_manager.safe_to_parquet(primary_df, df_file)

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
                p_file = sdir / f"node_{clean_nid}_port_{port_name}.parquet"
                storage_manager.safe_to_parquet(outputs[out_key], p_file)

        # Also save any other named DataFrames
        for k, v in outputs.items():
            if isinstance(v, pd.DataFrame) and k not in port_mappings and k != "dataframe":
                clean_k = "".join(c for c in k if c.isalnum() or c == "_")
                p_file = sdir / f"node_{clean_nid}_port_{clean_k}.parquet"
                storage_manager.safe_to_parquet(v, p_file)

        # 3. Save Non-DataFrame ML Artifacts via Joblib
        artifacts = {}
        for key in [
            "model", "scaler", "encoder", "task_type", "feature_names", "feature_importances",
            "target_classes", "target_encoder", "target_column", "imputer_stats",
            "vectorizer", "text_column", "split_mode", "is_temporal", "temporal_column", "categorical_maps",
            "X_train", "y_train", "X_test", "y_test", "metrics", "output_summary"
        ]:
            if key in outputs and outputs[key] is not None:
                val = outputs[key]
                if not isinstance(val, pd.DataFrame):
                    artifacts[key] = val

        if artifacts:
            art_file = sdir / f"node_{clean_nid}_artifacts.joblib"
            try:
                joblib.dump(artifacts, art_file)
            except Exception as e:
                logger.warning(f"Failed to dump artifacts for node '{node_id}': {e}")

        # 4. Construct Metadata & Preview
        columns = [str(col) for col in primary_df.columns] if isinstance(primary_df, pd.DataFrame) else []
        col_types = {str(col): str(dtype) for col, dtype in primary_df.dtypes.items()} if isinstance(primary_df, pd.DataFrame) else {}
        row_count = int(primary_df.shape[0]) if isinstance(primary_df, pd.DataFrame) else None
        
        preview_rows = []
        if isinstance(primary_df, pd.DataFrame):
            try:
                clean_head = primary_df.head(10).copy()
                for c in clean_head.select_dtypes(include=["datetime", "datetimetz"]).columns:
                    clean_head[c] = clean_head[c].dt.strftime("%Y-%m-%d %H:%M:%S")
                try:
                    clean_head = clean_head.replace({float("nan"): None, float("inf"): None, float("-inf"): None})
                    clean_head = clean_head.where(pd.notnull(clean_head), None)
                    raw_head = clean_head.to_dict(orient="records")
                except Exception:
                    raw_head = clean_head.to_dict(orient="records")
                preview_rows = [make_json_safe(r) for r in raw_head]
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

        meta_file = sdir / f"node_{clean_nid}_meta.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(make_json_safe(meta), f, indent=2, default=str)

        return meta

    @classmethod
    def load_node_output(cls, session_id: str, node_id: str) -> Dict[str, Any]:
        """
        Reconstructs the outputs dictionary for a cached node from disk.
        """
        sdir = cls.get_session_dir(session_id)
        clean_nid = "".join(c for c in str(node_id) if c.isalnum() or c in ("-", "_")).strip() or "node"
        outputs: Dict[str, Any] = {}

        # 1. Primary DataFrame
        df_file = sdir / f"node_{clean_nid}_df.parquet"
        if df_file.exists():
            try:
                outputs["dataframe"] = pd.read_parquet(df_file)
            except Exception as e:
                logger.warning(f"Error loading primary parquet {df_file}: {e}")

        # 2. Port DataFrames
        for p_file in sdir.glob(f"node_{clean_nid}_port_*.parquet"):
            port_name = p_file.stem.replace(f"node_{clean_nid}_port_", "")
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
        art_file = sdir / f"node_{clean_nid}_artifacts.joblib"
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
        clean_nid = "".join(c for c in str(node_id) if c.isalnum() or c in ("-", "_")).strip() or "node"
        meta_file = sdir / f"node_{clean_nid}_meta.json"
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
        from backend.app.core.tenant_context import (
            get_current_tenant_id,
            with_tenant_config,
            get_current_workflow_id,
            get_current_execution_id,
            get_current_session_id,
        )
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

        target_recipe_id = str(node_map[target_node_id].get("recipe_id", "")).lower().strip()
        try:
            target_recipe = recipe_registry.get(target_recipe_id)
            target_recipe_name = target_recipe.name
        except Exception:
            target_recipe_name = target_recipe_id

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

        # Check for cycles in ancestor graph
        if len(topo_order) < len(subgraph_nodes):
            cyclic_nodes = [nid for nid in subgraph_nodes if nid not in topo_order]
            return NodeTestStepResponse(
                success=False,
                session_id=session_id,
                node_id=target_node_id,
                recipe_id=target_recipe_id,
                recipe_name=target_recipe_name,
                status="FAILED",
                error_message=f"Cycle detected in workflow graph ancestors: {cyclic_nodes}. Workflows must be Directed Acyclic Graphs (DAGs).",
                error_title="Circular Dependency Error",
                error_suggestion="Remove or reverse the cyclic connection among the upstream nodes leading to this step."
            )

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
                    session_id=session_id,
                    node_id=nid,
                    recipe_id=r_id,
                    status="FAILED",
                    error_message=err_msg
                )

            # Build Node Config
            override_cfg = node_configs.get(nid, {})
            if isinstance(override_cfg, dict) and "config" in override_cfg and isinstance(override_cfg["config"], dict):
                inner_cfg = override_cfg["config"]
                override_cfg = {**override_cfg, **inner_cfg}
            node_base_cfg = (
                n_obj.get("config")
                or (n_obj.get("data", {}).get("config") if isinstance(n_obj.get("data"), dict) else None)
                or (n_obj.get("data", {}).get("parameters") if isinstance(n_obj.get("data"), dict) else None)
                or n_obj.get("parameters")
                or {}
            )
            merged_config = {**node_base_cfg, **override_cfg}
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
                                effective_cap = sample_size or 5000
                                if effective_cap and len(load_df) > effective_cap:
                                    load_df = load_df.head(effective_cap)
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
                    sh = (
                        edge.get("sourceHandle")
                        or edge.get("source_handle")
                        or edge.get("source_port")
                        or ""
                    ).lower().strip()

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
                        th = (
                            edge.get("targetHandle")
                            or edge.get("target_handle")
                            or edge.get("target_port")
                            or ""
                        ).lower().strip()
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
                # Session Context for model/scaler propagation & endpoint URLs
                session_context = dict(node_inputs)
                session_context["session_id"] = session_id
                session_context["workflow_id"] = get_current_workflow_id() or session_id
                session_context["execution_id"] = get_current_execution_id() or session_id
                session_context["auth_session_id"] = get_current_session_id()

                # Resolve credential from vault if credential_id is provided
                cred_id = str(merged_config.get("credential_id", "")).strip()
                if cred_id:
                    try:
                        from backend.app.credentials.service import CredentialService
                        if db is not None:
                            stored_cred = await CredentialService.get_credential_data(db=db, credential_id=cred_id, tenant_id=tenant_id)
                        else:
                            from backend.app.infrastructure.database.session import AsyncSessionLocal
                            async with AsyncSessionLocal() as session:
                                stored_cred = await CredentialService.get_credential_data(db=session, credential_id=cred_id, tenant_id=tenant_id)
                        if stored_cred and isinstance(stored_cred, dict):
                            if "credentials" not in session_context:
                                session_context["credentials"] = {}
                            session_context["credentials"][cred_id] = stored_cred
                            for ck, cv in stored_cred.items():
                                if ck not in merged_config or not merged_config[ck]:
                                    merged_config[ck] = cv
                    except Exception as cred_res_err:
                        logger.warning(f"Failed to resolve credential '{cred_id}' for node '{nid}': {cred_res_err}")

                # Add Workflow: load the child through the tenant-guarded ORM session, with its
                # datasets and saved credentials, instead of letting the recipe query the DB itself
                from backend.app.workflows.sub_workflows import (
                    ADD_WORKFLOW_RECIPE_IDS, SUB_WORKFLOW_DEFINITION_KEY, SUB_WORKFLOW_ERROR_KEY,
                    SUB_WORKFLOW_VERSION_INFO_KEY, load_sub_workflow_graph,
                )
                child_wf_id = str(merged_config.get("workflow_id") or "").strip()
                if r_id in ADD_WORKFLOW_RECIPE_IDS and child_wf_id:
                    try:
                        child_version = merged_config.get("workflow_version")
                        if db is not None:
                            child_graph, child_info = await load_sub_workflow_graph(
                                child_wf_id, db, tenant_id, version=child_version)
                        else:
                            from backend.app.infrastructure.database.session import AsyncSessionLocal
                            async with AsyncSessionLocal() as session:
                                child_graph, child_info = await load_sub_workflow_graph(
                                    child_wf_id, session, tenant_id, version=child_version)
                        merged_config[SUB_WORKFLOW_DEFINITION_KEY] = child_graph
                        merged_config[SUB_WORKFLOW_VERSION_INFO_KEY] = child_info
                    except ValueError as sub_err:
                        merged_config[SUB_WORKFLOW_ERROR_KEY] = str(sub_err)

                from backend.app.core.crypto import decrypt_node_config_secrets
                node_exec_cfg = decrypt_node_config_secrets(with_tenant_config(merged_config))
                outputs = await asyncio.to_thread(
                    recipe.execute,
                    inputs=node_inputs,
                    config=node_exec_cfg,
                    context=session_context
                )
                duration_ms = round((time.time() - n_start) * 1000.0, 2)

                # Auto-save credential if save_credential toggle was enabled and execution succeeded
                saved_cred_id = None
                saved_cred_name = None
                if bool(merged_config.get("save_credential", False)):
                    try:
                        from backend.app.credentials.service import CredentialService
                        provider = r_id.replace("_connector", "").lower()
                        saved_cred_name = (
                            str(merged_config.get("credential_name", "")).strip()
                            or f"{getattr(recipe, 'name', r_id)} ({merged_config.get('host') or merged_config.get('database') or merged_config.get('username') or 'Connection'})"
                        )
                        cred_payload = {
                            k: v for k, v in merged_config.items()
                            if k not in (
                                "save_credential", "credential_name", "tenant_id", "dry_run",
                                "sample_size", "auto_run_ancestors", "simulation_mode"
                            ) and v is not None and v != ""
                        }
                        safe_meta = {
                            k: merged_config[k]
                            for k in ("host", "port", "database", "username", "account", "warehouse", "ssl_mode", "bucket", "project_id", "email")
                            if k in merged_config and merged_config[k]
                        }
                        auth_type = "database" if "database" in getattr(recipe, "subgroup", "").lower() else "basic"

                        if db is not None:
                            cred_obj = await CredentialService.create_credential(
                                db=db,
                                provider=provider,
                                name=saved_cred_name,
                                data=cred_payload,
                                tenant_id=tenant_id,
                                metadata_info=safe_meta,
                                auth_type=auth_type,
                            )
                        else:
                            from backend.app.infrastructure.database.session import AsyncSessionLocal
                            async with AsyncSessionLocal() as session:
                                cred_obj = await CredentialService.create_credential(
                                    db=session,
                                    provider=provider,
                                    name=saved_cred_name,
                                    data=cred_payload,
                                    tenant_id=tenant_id,
                                    metadata_info=safe_meta,
                                    auth_type=auth_type,
                                )
                        if cred_obj:
                            saved_cred_id = cred_obj.id
                            saved_cred_name = cred_obj.name
                    except Exception as save_err:
                        logger.warning(f"Failed to auto-save credential for node '{nid}': {save_err}")

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

                if saved_cred_id and isinstance(meta.get("output_summary"), dict):
                    meta["output_summary"]["saved_credential_id"] = saved_cred_id
                    meta["output_summary"]["saved_credential_name"] = saved_cred_name

                # A step that ended with a warning (e.g. a failed sub-process with 'Halt Pipeline on
                # Sub-Workflow Error' off) is reported as WARNING, not SUCCESS
                is_warning = isinstance(outputs, dict) and outputs.get("_node_status") == "WARNING"
                response = NodeTestStepResponse(
                    success=True,
                    session_id=session_id,
                    node_id=nid,
                    recipe_id=r_id,
                    recipe_name=getattr(recipe, "name", r_id),
                    status="WARNING" if is_warning else "SUCCESS",
                    error_message=str(outputs.get("_warning")) if is_warning else None,
                    duration_ms=duration_ms,
                    row_count=meta.get("row_count"),
                    columns=[str(c) for c in meta.get("columns", [])],
                    column_types=meta.get("column_types", {}),
                    preview_rows=meta.get("preview_rows", []),
                    output_summary=meta.get("output_summary", {}),
                    metrics=meta.get("metrics"),
                    available_left_columns=available_left_cols,
                    available_right_columns=available_right_cols,
                    saved_credential_id=saved_cred_id,
                    saved_credential_name=saved_cred_name
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
                err_title = diag.get("title") or f"Execution Failed: {getattr(recipe, 'name', r_id)}"
                err_suggestion = diag.get("suggestion") or "Please verify database connection parameters, credentials, and network accessibility."
                err_summary = {
                    "title": err_title,
                    "status": "FAILED",
                    "message": str(ex),
                    "error": str(ex),
                    "suggestion": err_suggestion,
                    "duration_ms": duration_ms
                }
                # Record failure metadata and clear any stale dataframe cache from previous runs
                try:
                    sdir = cls.get_session_dir(session_id)
                    stale_df = sdir / f"node_{nid}_df.parquet"
                    if stale_df.exists():
                        stale_df.unlink()
                    meta_fail = {
                        "node_id": nid,
                        "recipe_id": r_id,
                        "status": "FAILED",
                        "duration_ms": duration_ms,
                        "row_count": 0,
                        "columns": [],
                        "column_types": {},
                        "preview_rows": [],
                        "error_message": str(ex),
                        "error_title": err_title,
                        "error_suggestion": err_suggestion,
                        "output_summary": err_summary,
                        "timestamp": time.time()
                    }
                    meta_file = sdir / f"node_{nid}_meta.json"
                    with open(meta_file, "w", encoding="utf-8") as f:
                        json.dump(make_json_safe(meta_fail), f, indent=2, default=str)
                except Exception as clean_err:
                    logger.debug(f"Could not record failure metadata for node {nid}: {clean_err}")

                return None, NodeTestStepResponse(
                    success=False,
                    session_id=session_id,
                    node_id=nid,
                    recipe_id=r_id,
                    recipe_name=getattr(recipe, "name", r_id),
                    status="FAILED",
                    duration_ms=duration_ms,
                    error_message=str(ex),
                    error_title=err_title,
                    error_suggestion=err_suggestion,
                    output_summary=err_summary
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
            session_id=session_id,
            node_id=target_node_id,
            recipe_id=str(node_map[target_node_id].get("recipe_id", "")),
            status="FAILED",
            error_message="Unexpected error during node step execution.",
            output_summary={
                "title": "Execution Error",
                "status": "FAILED",
                "message": "Unexpected error during node step execution.",
                "error": "Unexpected error during node step execution.",
                "suggestion": "Check node configuration and recipe parameters."
            }
        )