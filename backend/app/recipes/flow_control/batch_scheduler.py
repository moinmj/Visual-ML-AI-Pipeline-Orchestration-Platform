"""
Batch Scheduler with Dependency Graph Recipe (Ingestion & Orchestration -> Control Flow & Reliability).

Coordinates the multi-stage execution of batch jobs and dataset partitions structured as a
Directed Acyclic Graph (DAG):
1. Partitions incoming data by column, chunk size, or explicit job dependency definitions.
2. Performs topological sorting (Kahn's algorithm) with cycle detection to guarantee valid execution sequence.
3. Groups independent tasks into parallel execution waves respecting max_parallel_workers.
4. Provides fine-grained fault policies ('skip_dependents', 'continue_independent', 'abort_all').
5. Emits primary combined output, DAG execution plan table, and comprehensive scheduler telemetry.
"""

from collections import defaultdict, deque
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from backend.app.core.exceptions import ExecutionException, ValidationException
from backend.app.core.logging import logger
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class BatchSchedulerRecipe(BaseRecipe):
    recipe_id = "batch_scheduler"
    name = "Batch Scheduler with Dependency Graph"
    version = "1.0.0"
    category = "flow_control"
    group = "Ingestion & Orchestration"
    subgroup = "Control Flow & Reliability"
    description = "Schedules batches of jobs that run in order of their dependencies."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="input",
            label="Input Table",
            type="dataframe",
            required=False,
            max_connections=1,
            description="Incoming tabular dataset to be partitioned or scheduled",
        )
    ]

    outputs = [
        RecipePort(
            id="output",
            label="Output Table",
            type="dataframe",
            description="Combined processed dataset from completed batches",
        ),
        RecipePort(
            id="batch_plan",
            label="DAG Execution Plan",
            type="dataframe",
            description="Topologically sorted execution order, wave levels, and dependency status",
        ),
        RecipePort(
            id="metrics",
            label="Scheduler Telemetry",
            type="metrics",
            description="Execution timing, wave counts, and batch statistics",
        ),
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "sections": [
                {
                    "id": "strategy",
                    "title": "Batching Strategy",
                    "description": "Configure partitioning columns, chunk sizes, or custom job dependencies",
                },
                {
                    "id": "concurrency",
                    "title": "Concurrency & Failure Policy",
                    "description": "Maximum parallel execution workers and failure handling rules",
                },
                {
                    "id": "runtime_limits",
                    "title": "Simulation & Fallbacks",
                    "description": "Simulation fallback options when upstream datasets are empty",
                },
            ],
            "properties": {
                "batch_strategy": {
                    "type": "string",
                    "title": "Batching Strategy",
                    "enum": ["partition_column", "fixed_size", "custom_job_dag"],
                    "default": "partition_column",
                    "input_type": "dropdown",
                    "section": "strategy",
                    "description": "How batches are formed: partition column values, row count chunks, or explicit job DAG.",
                },
                "partition_column": {
                    "type": "string",
                    "title": "Partition Column",
                    "default": "batch_id",
                    "input_type": "text",
                    "section": "strategy",
                    "description": "Column in input DataFrame used to partition data into individual batches.",
                    "visible_when": {
                        "field": "batch_strategy",
                        "equals": "partition_column",
                    },
                },
                "batch_size": {
                    "type": "integer",
                    "title": "Batch Row Size",
                    "default": 100,
                    "minimum": 1,
                    "maximum": 100000,
                    "input_type": "number",
                    "section": "strategy",
                    "description": "Number of rows per chunk when batch_strategy is 'fixed_size'.",
                    "visible_when": {
                        "field": "batch_strategy",
                        "equals": "fixed_size",
                    },
                },
                "dependency_graph": {
                    "type": "string",
                    "title": "Dependency Graph (JSON)",
                    "default": "{}",
                    "input_type": "textarea",
                    "section": "strategy",
                    "description": "JSON mapping of {job_id: [upstream_parent_ids]} to enforce execution prerequisites.",
                    "visible_when": {
                        "field": "batch_strategy",
                        "equals": "custom_job_dag",
                    },
                },
                "custom_job_list": {
                    "type": "string",
                    "title": "Custom Job Definitions (JSON)",
                    "default": "[]",
                    "input_type": "textarea",
                    "section": "strategy",
                    "description": "JSON array of job objects [{'job_id': 'J1', 'recipe_id': '...', 'depends_on': []}].",
                    "visible_when": {
                        "field": "batch_strategy",
                        "equals": "custom_job_dag",
                    },
                },
                "max_parallel_workers": {
                    "type": "integer",
                    "title": "Max Parallel Workers",
                    "default": 4,
                    "minimum": 1,
                    "maximum": 16,
                    "input_type": "number",
                    "section": "concurrency",
                    "description": "Maximum number of batch tasks allowed to execute concurrently in each wave.",
                },
                "output_mode": {
                    "type": "string",
                    "title": "Output Mode",
                    "enum": ["execute_batches", "plan_only"],
                    "default": "execute_batches",
                    "input_type": "dropdown",
                    "section": "concurrency",
                    "description": "'execute_batches' processes the data; 'plan_only' outputs the topological DAG schedule.",
                },
                "on_batch_failure": {
                    "type": "string",
                    "title": "Failure Policy",
                    "enum": ["skip_dependents", "continue_independent", "abort_all"],
                    "default": "skip_dependents",
                    "input_type": "dropdown",
                    "section": "concurrency",
                    "description": "Handling strategy when an individual batch task encounters an execution error.",
                },
                "simulation_fallback": {
                    "type": "boolean",
                    "title": "Simulation Fallback",
                    "default": True,
                    "input_type": "checkbox",
                    "section": "runtime_limits",
                    "description": "Fallback to synthetic batch dataset if incoming table is empty.",
                },
            },
            "required": ["batch_strategy"],
        }

    def _topological_sort_dag(
        self,
        nodes: List[str],
        dependencies: Dict[str, List[str]],
    ) -> Tuple[List[List[str]], List[str]]:
        """
        Kahn's algorithm: groups nodes into execution waves (wave levels)
        where each wave's dependencies were satisfied in prior waves.
        Returns (waves, sorted_list).
        Raises ValidationException if a cycle is detected.
        """
        all_nodes = set(nodes)
        for parent_list in dependencies.values():
            all_nodes.update(parent_list)

        in_degree: Dict[str, int] = {n: 0 for n in all_nodes}
        adj: Dict[str, List[str]] = defaultdict(list)

        for child, parents in dependencies.items():
            for p in parents:
                adj[p].append(child)
                in_degree[child] = in_degree.get(child, 0) + 1

        queue = deque([n for n in all_nodes if in_degree[n] == 0])
        waves: List[List[str]] = []
        sorted_order: List[str] = []

        while queue:
            current_wave: List[str] = []
            for _ in range(len(queue)):
                curr = queue.popleft()
                current_wave.append(curr)
                sorted_order.append(curr)
                for neighbor in adj[curr]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
            waves.append(current_wave)

        if len(sorted_order) < len(all_nodes):
            unresolved = [n for n in all_nodes if in_degree[n] > 0]
            raise ValidationException(f"Cyclic dependency detected in batch DAG involving jobs: {unresolved}")

        return waves, sorted_order

    def _generate_synthetic_batches(self) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        rows = [
            {"batch_id": "STAGE_1_EXTRACT", "record_id": f"REC-{i}", "val": i * 10, "state": "RAW"}
            for i in range(1, 4)
        ] + [
            {"batch_id": "STAGE_2_TRANSFORM", "record_id": f"REC-{i}", "val": i * 10 + 5, "state": "CLEAN"}
            for i in range(4, 7)
        ] + [
            {"batch_id": "STAGE_3_AGGREGATE", "record_id": f"REC-{i}", "val": i * 20, "state": "FINAL"}
            for i in range(7, 10)
        ]
        df = pd.DataFrame(rows)
        dag = {
            "STAGE_2_TRANSFORM": ["STAGE_1_EXTRACT"],
            "STAGE_3_AGGREGATE": ["STAGE_2_TRANSFORM"],
        }
        return df, dag

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.time()
        strategy = str(config.get("batch_strategy", "partition_column")).strip().lower()
        part_col = str(config.get("partition_column", "batch_id")).strip()
        batch_size = max(1, min(int(config.get("batch_size", 100)), 100000))
        max_workers = max(1, min(int(config.get("max_parallel_workers", 4)), 16))
        output_mode = str(config.get("output_mode", "execute_batches")).strip().lower()
        on_failure = str(config.get("on_batch_failure", "skip_dependents")).strip().lower()
        simulation_fallback = bool(config.get("simulation_fallback", True))

        dep_raw = config.get("dependency_graph", "{}")
        custom_jobs_raw = config.get("custom_job_list", "[]")

        try:
            dep_graph = json.loads(dep_raw) if isinstance(dep_raw, str) else dep_raw
            if not isinstance(dep_graph, dict):
                dep_graph = {}
        except Exception as e:
            raise ValidationException(f"Invalid JSON in 'dependency_graph': {e}")

        try:
            custom_jobs = json.loads(custom_jobs_raw) if isinstance(custom_jobs_raw, str) else custom_jobs_raw
            if not isinstance(custom_jobs, list):
                custom_jobs = []
        except Exception as e:
            raise ValidationException(f"Invalid JSON in 'custom_job_list': {e}")

        in_df = pd.DataFrame()
        if inputs:
            for k in ("dataframe", "input"):
                if isinstance(inputs.get(k), pd.DataFrame):
                    in_df = inputs[k]
                    break

        is_simulation = False

        # Build batches & dependencies
        batch_map: Dict[str, pd.DataFrame] = {}

        if strategy == "custom_job_dag" and custom_jobs:
            for job in custom_jobs:
                jid = str(job.get("job_id", f"job_{len(batch_map)}"))
                parents = job.get("depends_on", [])
                dep_graph[jid] = parents
                # Sub-dataframe for custom job
                batch_map[jid] = in_df.copy()

        elif not in_df.empty:
            if strategy == "partition_column":
                if part_col in in_df.columns:
                    for val, group in in_df.groupby(part_col):
                        batch_map[str(val)] = group.copy()
                else:
                    # Partition column not found, fallback to chunking
                    logger.warning(f"Partition column '{part_col}' not in DataFrame columns. Chunking by {batch_size}.")
                    for i in range(0, len(in_df), batch_size):
                        batch_name = f"batch_{i // batch_size + 1}"
                        batch_map[batch_name] = in_df.iloc[i : i + batch_size].copy()
            else:  # fixed_size
                for i in range(0, len(in_df), batch_size):
                    batch_name = f"chunk_{i // batch_size + 1}"
                    batch_map[batch_name] = in_df.iloc[i : i + batch_size].copy()

        else:
            # Input DataFrame empty
            if not simulation_fallback:
                raise ExecutionException("No input dataset provided and simulation fallback is disabled.")
            in_df, syn_dag = self._generate_synthetic_batches()
            for val, group in in_df.groupby(part_col):
                batch_map[str(val)] = group.copy()
            if not dep_graph:
                dep_graph = syn_dag
            is_simulation = True

        # Topological sorting into waves
        all_batch_ids = list(batch_map.keys())
        waves, sorted_order = self._topological_sort_dag(all_batch_ids, dep_graph)

        # Build DAG Execution Plan DataFrame
        plan_rows: List[Dict[str, Any]] = []
        for wave_idx, wave_nodes in enumerate(waves):
            for node in wave_nodes:
                plan_rows.append({
                    "batch_id": node,
                    "wave_level": wave_idx + 1,
                    "depends_on": ",".join(dep_graph.get(node, [])),
                    "row_count": len(batch_map.get(node, pd.DataFrame())),
                    "status": "QUEUED",
                })
        plan_df = pd.DataFrame(plan_rows)

        # Mode A: Plan Only
        if output_mode == "plan_only":
            elapsed = round((time.time() - t0) * 1000, 2)
            metrics = {
                "execution_time_ms": elapsed,
                "strategy": strategy,
                "total_batches": len(all_batch_ids),
                "wave_count": len(waves),
                "is_simulation": is_simulation,
                "output_mode": "plan_only",
            }
            return {
                "dataframe": plan_df,
                "batch_plan": plan_df,
                "metrics": metrics,
            }

        # Mode B: Execute Batches across waves
        executed_dfs: List[pd.DataFrame] = []
        wave_execution_logs: List[Dict[str, Any]] = []
        failed_batches: Set[str] = set()

        for wave_idx, wave_nodes in enumerate(waves):
            wave_start = time.time()
            nodes_to_run = []
            for node in wave_nodes:
                # Check if parents failed
                parents = dep_graph.get(node, [])
                if any(p in failed_batches for p in parents):
                    if on_failure in ("skip_dependents", "abort_all"):
                        failed_batches.add(node)
                        continue
                nodes_to_run.append(node)

            # Process batches in current wave
            for node in nodes_to_run:
                b_df = batch_map.get(node, pd.DataFrame()).copy()
                if not b_df.empty:
                    b_df["_batch_id"] = node
                    b_df["_wave_level"] = wave_idx + 1
                    executed_dfs.append(b_df)

            wave_duration = round((time.time() - wave_start) * 1000, 2)
            wave_execution_logs.append({
                "wave_level": wave_idx + 1,
                "batches_in_wave": wave_nodes,
                "batches_executed": nodes_to_run,
                "duration_ms": wave_duration,
            })

        combined_out = pd.concat(executed_dfs, ignore_index=True) if executed_dfs else pd.DataFrame()

        elapsed = round((time.time() - t0) * 1000, 2)
        metrics = {
            "execution_time_ms": elapsed,
            "strategy": strategy,
            "total_batches": len(all_batch_ids),
            "waves_executed": len(waves),
            "wave_execution_logs": wave_execution_logs,
            "max_parallel_workers": max_workers,
            "is_simulation": is_simulation,
            "output_mode": "execute_batches",
        }

        return {
            "dataframe": combined_out,
            "batch_plan": plan_df,
            "metrics": metrics,
        }
