"""
Dead-Letter Queue (DLQ) Handler Recipe (Ingestion & Orchestration -> Control Flow & Reliability).

Collects, isolates, and diagnoses records or events that failed pipeline processing, schema validation,
or downstream API mutations.
Key capabilities:
1. Segregates clean records from corrupt/failed records using dual-routing input/output ports.
2. Enriches quarantined rows with forensic diagnostics (_dlq_id, _dlq_timestamp, _error_category, _error_detail, _replay_status).
3. Optional file-backed DLQ persistence (JSONL/CSV) for audit trails and offline triage.
4. Replay mechanism allowing previously quarantined and remediated records to re-enter pipeline execution.
5. Emits detailed DLQ telemetry (quarantine rate, category breakdown, error histograms).
"""

import io
import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from backend.app.core.exceptions import ExecutionException, ValidationException
from backend.app.core.logging import logger
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class DeadLetterQueueRecipe(BaseRecipe):
    recipe_id = "dead_letter_queue"
    name = "Dead-Letter Queue Handler"
    version = "1.0.0"
    category = "flow_control"
    group = "Ingestion & Orchestration"
    subgroup = "Control Flow & Reliability"
    description = "Collects rows or events that failed processing for review and replay."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="input",
            label="Clean / Primary Data",
            type="dataframe",
            required=False,
            max_connections=1,
            description="Upstream primary dataset to be scanned or passed through",
        ),
        RecipePort(
            id="failed_input",
            label="Failed / Error Records",
            type="dataframe",
            required=False,
            max_connections=1,
            description="Stream of failed records from upstream error handlers or validators",
        ),
    ]

    outputs = [
        RecipePort(
            id="output",
            label="Clean Output",
            type="dataframe",
            description="Sanitized dataset continuing through the normal pipeline flow",
        ),
        RecipePort(
            id="dlq_output",
            label="Quarantined DLQ Records",
            type="dataframe",
            description="Quarantined poison pills and failed records with diagnostic metadata",
        ),
        RecipePort(
            id="metrics",
            label="DLQ Telemetry",
            type="metrics",
            description="Quarantine metrics, error taxonomy, and persistence stats",
        ),
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "sections": [
                {
                    "id": "mode",
                    "title": "DLQ Routing & Categorization",
                    "description": "Configure operation mode, required clean check columns, and error tags",
                },
                {
                    "id": "storage",
                    "title": "Quarantine Persistence",
                    "description": "Configure persistent file storage for audit logs and forensic replay",
                },
                {
                    "id": "runtime_limits",
                    "title": "Capacity & Simulation",
                    "description": "Buffer limits and simulation fallback options",
                },
            ],
            "properties": {
                "dlq_mode": {
                    "type": "string",
                    "title": "DLQ Operation Mode",
                    "enum": ["trap_and_isolate", "pass_clean_only", "replay_from_storage"],
                    "default": "trap_and_isolate",
                    "input_type": "dropdown",
                    "section": "mode",
                    "description": "'trap_and_isolate' routes failed input to DLQ; 'pass_clean_only' filters corrupt rows from primary; 'replay_from_storage' re-ingests saved DLQ records.",
                },
                "validation_column": {
                    "type": "string",
                    "title": "Required Column (Clean Check)",
                    "default": "",
                    "input_type": "text",
                    "section": "mode",
                    "description": "Optional column required to be non-null in primary data; null rows are routed to DLQ.",
                    "visible_when": {
                        "field": "dlq_mode",
                        "equals": "pass_clean_only",
                    },
                },
                "error_category_default": {
                    "type": "string",
                    "title": "Default Error Category",
                    "default": "PROCESSING_ERROR",
                    "input_type": "text",
                    "section": "mode",
                    "description": "Category tag assigned to quarantined records if not already present.",
                },
                "persist_to_disk": {
                    "type": "boolean",
                    "title": "Persist DLQ to Disk",
                    "default": False,
                    "input_type": "checkbox",
                    "section": "storage",
                    "description": "Append quarantined records to a persistent JSONL or CSV file on the server.",
                },
                "storage_file_path": {
                    "type": "string",
                    "title": "Storage File Path",
                    "default": "./data/dlq/quarantine_log.jsonl",
                    "input_type": "text",
                    "section": "storage",
                    "description": "Disk path where quarantined records are stored when persistence is enabled.",
                    "visible_when": {
                        "field": "persist_to_disk",
                        "equals": True,
                    },
                },
                "max_dlq_capacity": {
                    "type": "integer",
                    "title": "Max DLQ Buffer Capacity",
                    "default": 10000,
                    "minimum": 100,
                    "maximum": 1000000,
                    "input_type": "number",
                    "section": "runtime_limits",
                    "description": "Maximum number of quarantined records retained in a single run.",
                },
                "simulation_fallback": {
                    "type": "boolean",
                    "title": "Simulation Fallback",
                    "default": True,
                    "input_type": "checkbox",
                    "section": "runtime_limits",
                    "description": "Generate synthetic dirty stream with simulated DLQ traps if inputs are empty.",
                },
            },
            "required": ["dlq_mode"],
        }

    def _generate_synthetic_stream(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Clean primary events
        clean_rows = [
            {"event_id": f"EVT-OK-{i}", "user_id": f"USR-{100 + i}", "payload": f"action_{i}", "status": "CONFIRMED"}
            for i in range(1, 9)
        ]
        # Corrupt / poison pill events
        failed_rows = [
            {"event_id": "EVT-BAD-901", "user_id": None, "payload": "MALFORMED_BINARY_PAYLOAD", "error_code": "NULL_USER_ID"},
            {"event_id": "EVT-BAD-902", "user_id": "USR-999", "payload": "{broken_json}", "error_code": "JSON_SYNTAX_ERROR"},
        ]
        return pd.DataFrame(clean_rows), pd.DataFrame(failed_rows)

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.time()
        dlq_mode = str(config.get("dlq_mode", "trap_and_isolate")).strip().lower()
        valid_col = str(config.get("validation_column", "")).strip()
        default_cat = str(config.get("error_category_default", "PROCESSING_ERROR")).strip()
        persist = bool(config.get("persist_to_disk", False))
        storage_path = str(config.get("storage_file_path", "./data/dlq/quarantine_log.jsonl")).strip()
        max_capacity = max(100, min(int(config.get("max_dlq_capacity", 10000)), 1000000))
        simulation_fallback = bool(config.get("simulation_fallback", True))

        primary_df = pd.DataFrame()
        failed_df = pd.DataFrame()

        if inputs:
            for k in ("dataframe", "input"):
                if isinstance(inputs.get(k), pd.DataFrame):
                    primary_df = inputs[k]
                    break
            if isinstance(inputs.get("failed_input"), pd.DataFrame):
                failed_df = inputs["failed_input"]

        is_simulation = False

        # Fallback to simulation if both inputs are empty
        if primary_df.empty and failed_df.empty:
            if not simulation_fallback and dlq_mode != "replay_from_storage":
                raise ExecutionException("No input data provided to Dead-Letter Queue handler.")
            if dlq_mode != "replay_from_storage":
                primary_df, failed_df = self._generate_synthetic_stream()
                is_simulation = True

        now_iso = datetime.now(timezone.utc).isoformat()

        # Branch A: Replay from Storage Mode
        if dlq_mode == "replay_from_storage":
            if os.path.exists(storage_path) and os.path.isfile(storage_path):
                try:
                    if storage_path.endswith(".csv"):
                        replayed_df = pd.read_csv(storage_path)
                    else:
                        replayed_df = pd.read_json(storage_path, lines=True)
                    replayed_df["_replay_status"] = "REPLAYED"
                    replayed_df["_replayed_at"] = now_iso
                except Exception as e:
                    raise ExecutionException(f"Failed reading DLQ storage file '{storage_path}': {e}")
            else:
                # Synthetic replayed items
                replayed_df = pd.DataFrame([
                    {"_dlq_id": "REPLAY-001", "original_event": "EVT-REPLAYED-1", "_replay_status": "REPLAYED", "_replayed_at": now_iso},
                    {"_dlq_id": "REPLAY-002", "original_event": "EVT-REPLAYED-2", "_replay_status": "REPLAYED", "_replayed_at": now_iso},
                ])
                is_simulation = True

            elapsed = round((time.time() - t0) * 1000, 2)
            metrics = {
                "execution_time_ms": elapsed,
                "dlq_mode": dlq_mode,
                "replayed_records_count": len(replayed_df),
                "is_simulation": is_simulation,
            }
            return {
                "dataframe": replayed_df,
                "dlq_output": pd.DataFrame(),
                "metrics": metrics,
            }

        # Branch B: Standard Trap / Pass Clean Flow
        clean_records: List[Dict[str, Any]] = []
        quarantine_records: List[Dict[str, Any]] = []

        # 1. Process explicit failed_input
        if not failed_df.empty:
            for _, row in failed_df.iterrows():
                row_dict = row.to_dict()
                row_dict["_dlq_id"] = str(uuid.uuid4())[:8]
                row_dict["_dlq_timestamp"] = now_iso
                row_dict["_error_category"] = row_dict.get("_error_category", default_cat)
                row_dict["_error_detail"] = row_dict.get("_error_detail", row_dict.get("error_message", "Explicit failure routed from upstream"))
                row_dict["_replay_status"] = "PENDING"
                quarantine_records.append(row_dict)

        # 2. Check primary_df for missing validation_column or corrupt entries
        if not primary_df.empty:
            if valid_col and valid_col in primary_df.columns:
                for _, row in primary_df.iterrows():
                    val = row.get(valid_col)
                    if pd.isna(val) or val is None or str(val).strip() == "":
                        row_dict = row.to_dict()
                        row_dict["_dlq_id"] = str(uuid.uuid4())[:8]
                        row_dict["_dlq_timestamp"] = now_iso
                        row_dict["_error_category"] = "VALIDATION_FAILED"
                        row_dict["_error_detail"] = f"Required column '{valid_col}' was null or empty."
                        row_dict["_replay_status"] = "PENDING"
                        quarantine_records.append(row_dict)
                    else:
                        clean_records.append(row.to_dict())
            else:
                clean_records = primary_df.to_dict(orient="records")

        # Cap quarantine records to buffer capacity
        quarantine_records = quarantine_records[:max_capacity]

        out_clean_df = pd.DataFrame(clean_records) if clean_records else pd.DataFrame()
        out_dlq_df = pd.DataFrame(quarantine_records) if quarantine_records else pd.DataFrame()

        # 3. Optional Persistence to Disk
        if persist and not out_dlq_df.empty:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(storage_path)), exist_ok=True)
                if storage_path.endswith(".csv"):
                    out_dlq_df.to_csv(storage_path, mode="a", index=False, header=not os.path.exists(storage_path))
                else:
                    out_dlq_df.to_json(storage_path, mode="a", orient="records", lines=True)
                logger.info(f"Persisted {len(out_dlq_df)} DLQ records to {storage_path}")
            except Exception as e:
                logger.warning(f"Failed to persist DLQ records to {storage_path}: {e}")

        elapsed = round((time.time() - t0) * 1000, 2)
        total_seen = len(out_clean_df) + len(out_dlq_df)
        quarantine_rate = round((len(out_dlq_df) / total_seen) * 100, 2) if total_seen > 0 else 0.0

        # Category taxonomy breakdown
        cat_counts: Dict[str, int] = {}
        if not out_dlq_df.empty and "_error_category" in out_dlq_df.columns:
            cat_counts = out_dlq_df["_error_category"].value_counts().to_dict()

        metrics = {
            "execution_time_ms": elapsed,
            "dlq_mode": dlq_mode,
            "clean_records_forwarded": len(out_clean_df),
            "quarantined_records_count": len(out_dlq_df),
            "quarantine_rate_pct": quarantine_rate,
            "error_categories": cat_counts,
            "is_simulation": is_simulation,
            "persisted_to_disk": persist,
            "storage_file_path": storage_path if persist else None,
        }

        return {
            "dataframe": out_clean_df,
            "dlq_output": out_dlq_df,
            "metrics": metrics,
        }
