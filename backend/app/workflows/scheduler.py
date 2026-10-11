import asyncio
import datetime
import logging
import uuid
from typing import Dict, Any, List, Optional
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore

from sqlalchemy.future import select
from fastapi.encoders import jsonable_encoder

from backend.app.infrastructure.database.session import AsyncSessionLocal
from backend.app.workflows.models import Workflow
from backend.app.core.logging import logger
from backend.app.core.tenant_context import set_current_tenant_id, set_current_user_id

CRON_PRESETS = {
    "Every 15 Minutes": "*/15 * * * *",
    "Hourly": "0 * * * *",
    "Daily at Midnight": "0 0 * * *",
    "Weekly on Sunday": "0 0 * * 0",
}


def resolve_timezone(tz_name: Optional[str]) -> ZoneInfo:
    """Robust, case-insensitive IANA timezone resolver with regional aliases."""
    if not tz_name or not str(tz_name).strip():
        try:
            local_tz = datetime.datetime.now().astimezone().tzinfo
            if isinstance(local_tz, ZoneInfo):
                return local_tz
        except Exception:
            pass
        return ZoneInfo("UTC")

    clean = str(tz_name).strip()
    try:
        return ZoneInfo(clean)
    except Exception:
        pass

    clean_lower = clean.lower()
    aliases = {
        "asia": "Asia/Kolkata",
        "india": "Asia/Kolkata",
        "ist": "Asia/Kolkata",
        "calcutta": "Asia/Kolkata",
        "kolkata": "Asia/Kolkata",
        "pakistan": "Asia/Karachi",
        "karachi": "Asia/Karachi",
        "pkt": "Asia/Karachi",
        "dubai": "Asia/Dubai",
        "uae": "Asia/Dubai",
        "singapore": "Asia/Singapore",
        "tokyo": "Asia/Tokyo",
        "japan": "Asia/Tokyo",
        "london": "Europe/London",
        "gmt": "Etc/GMT",
        "utc": "UTC",
        "est": "America/New_York",
        "pst": "America/Los_Angeles",
        "cst": "America/Chicago",
    }
    if clean_lower in aliases:
        try:
            return ZoneInfo(aliases[clean_lower])
        except Exception:
            pass

    try:
        import zoneinfo as zi
        tz_map = {z.lower(): z for z in zi.available_timezones()}
        if clean_lower in tz_map:
            return ZoneInfo(tz_map[clean_lower])
        for k, v in tz_map.items():
            if k.startswith(clean_lower):
                return ZoneInfo(v)
    except Exception:
        pass

    try:
        return ZoneInfo("Asia/Kolkata") if "asia" in clean_lower else ZoneInfo("UTC")
    except Exception:
        return ZoneInfo("UTC")



def match_cron_field(field_spec: str, val: int, min_val: int = 0, max_val: int = 59) -> bool:
    """Matches a single cron expression field against an integer value."""
    field_spec = field_spec.strip()
    if field_spec == "*":
        return True

    # Comma-separated list: e.g. "1,5,15"
    if "," in field_spec:
        return any(match_cron_field(sub.strip(), val, min_val, max_val) for sub in field_spec.split(","))

    # Step values: e.g. "*/15" or "10-30/5"
    if "/" in field_spec:
        parts = field_spec.split("/", 1)
        base_range = parts[0]
        try:
            step = int(parts[1])
        except ValueError:
            return False
        if step <= 0:
            return False
        if base_range == "*":
            return (val - min_val) % step == 0
        if "-" in base_range:
            try:
                start, end = map(int, base_range.split("-"))
                return start <= val <= end and (val - start) % step == 0
            except ValueError:
                return False
        return False

    # Range values: e.g. "1-5"
    if "-" in field_spec:
        try:
            start, end = map(int, field_spec.split("-"))
            return start <= val <= end
        except ValueError:
            return False

    # Exact integer value
    try:
        return int(field_spec) == val
    except ValueError:
        return False


def is_cron_due(cron_expr: str, dt: datetime.datetime) -> bool:
    """Evaluates whether a standard 5-part cron expression is due at the specified datetime."""
    parts = cron_expr.strip().split()
    if len(parts) != 5:
        return False
    minute_expr, hour_expr, dom_expr, month_expr, dow_expr = parts

    # Standard cron day of week: 0=Sunday..6=Saturday (7=Sunday also allowed)
    cron_dow = (dt.weekday() + 1) % 7

    return (
        match_cron_field(minute_expr, dt.minute, 0, 59) and
        match_cron_field(hour_expr, dt.hour, 0, 23) and
        match_cron_field(dom_expr, dt.day, 1, 31) and
        match_cron_field(month_expr, dt.month, 1, 12) and
        (match_cron_field(dow_expr, cron_dow, 0, 6) or (dow_expr == "7" and cron_dow == 0))
    )


class WorkflowCronScheduler:
    """
    Automated Background Cron Scheduler Daemon.
    Monitors saved active workflows containing 'cron_trigger' nodes,
    evaluates recurring schedule expressions (e.g. '*/15 * * * *'),
    and executes them automatically in the background.
    """

    def __init__(self, check_interval_seconds: int = 30):
        self.check_interval_seconds = check_interval_seconds
        self._running = False
        self._last_triggered: Dict[str, str] = {}  # workflow_id -> minute_key

    def stop(self):
        self._running = False

    async def start(self):
        self._running = True
        logger.info(f"WorkflowCronScheduler: Started background daemon (poll interval: {self.check_interval_seconds}s).")
        while self._running:
            try:
                await self.tick()
            except Exception as e:
                logger.error(f"WorkflowCronScheduler: Error during schedule tick: {e}", exc_info=True)
            await asyncio.sleep(self.check_interval_seconds)

    async def tick(self, force: bool = False, target_workflow_id: Optional[str] = None) -> List[str]:
        """Runs a single schedule check cycle. Returns list of triggered workflow IDs."""
        triggered_ids: List[str] = []
        async with AsyncSessionLocal() as db:
            if target_workflow_id:
                stmt = (
                    select(Workflow)
                    .where(Workflow.id == target_workflow_id, Workflow.deleted_at.is_(None))
                    .execution_options(skip_tenant_filter=True)
                )
            else:
                stmt = (
                    select(Workflow)
                    .where(Workflow.is_active == True, Workflow.deleted_at.is_(None))
                    .execution_options(skip_tenant_filter=True)
                )
            result = await db.execute(stmt)
            workflows = result.scalars().all()

            for wf in workflows:
                cron_cfg = self._extract_cron_config(wf)
                if not cron_cfg:
                    continue

                cron_expr = cron_cfg["cron_expression"]
                tz_name = cron_cfg.get("timezone", "UTC")
                tz = resolve_timezone(tz_name)

                now_tz = datetime.datetime.now(tz)
                minute_key = now_tz.strftime("%Y-%m-%d %H:%M")

                is_due = force or is_cron_due(cron_expr, now_tz)
                if is_due:
                    if force or self._last_triggered.get(wf.id) != minute_key:
                        self._last_triggered[wf.id] = minute_key
                        logger.info(
                            f"WorkflowCronScheduler: Firing scheduled pipeline '{wf.name}' ({wf.id}) "
                            f"on schedule '{cron_expr}' ({tz_name}) [force={force}]."
                        )
                        asyncio.create_task(self._execute_scheduled_workflow(wf.id, wf.tenant_id, cron_expr))
                        triggered_ids.append(wf.id)

        # Cleanup old minute keys
        if len(self._last_triggered) > 500:
            current_prefix = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
            self._last_triggered = {
                k: v for k, v in self._last_triggered.items() if v.startswith(current_prefix)
            }

        return triggered_ids

    def _extract_cron_config(self, wf: Workflow) -> Optional[Dict[str, Any]]:
        """Finds any active cron_trigger node in the workflow and returns its resolved schedule."""
        import json
        saved_configs = wf.node_configs or {}
        if isinstance(saved_configs, str):
            try:
                saved_configs = json.loads(saved_configs)
            except Exception:
                saved_configs = {}

        raw_nodes = wf.nodes or []
        if isinstance(raw_nodes, str):
            try:
                raw_nodes = json.loads(raw_nodes)
            except Exception:
                raw_nodes = []

        for node in raw_nodes:
            if not isinstance(node, dict):
                continue
            recipe_id = node.get("recipe_id") or (saved_configs.get(node.get("id"), {}) if isinstance(saved_configs, dict) else {}).get("recipe_id")
            if recipe_id == "cron_trigger":
                node_id = node.get("id")
                node_saved = saved_configs.get(node_id, {}) if isinstance(saved_configs, dict) else {}
                cfg = dict(node.get("config") or node_saved.get("config") or {})

                preset = cfg.get("interval_preset", "")
                expr = cfg.get("cron_expression", "").strip()

                # Automatically resolve preset if specified
                if preset in CRON_PRESETS and (not expr or expr == "0 0 * * *"):
                    expr = CRON_PRESETS[preset]
                elif not expr:
                    expr = "0 0 * * *"

                return {
                    "node_id": node_id,
                    "cron_expression": expr,
                    "interval_preset": preset,
                    "timezone": cfg.get("timezone", "UTC")
                }
        return None

    async def _execute_scheduled_workflow(self, workflow_id: str, tenant_id: Optional[int], cron_expr: str):
        """Asynchronously executes a triggered workflow and updates its execution history."""
        from backend.app.engine.execution.executor import DAGExecutor
        from backend.app.workflows.router import (
            db_workflow_to_graph,
            preload_workflow_datasets,
            workflow_graph_to_db_payload,
            record_workflow_execution_history
        )

        try:
            async with AsyncSessionLocal() as db:
                set_current_tenant_id(tenant_id)
                set_current_user_id("cron_scheduler")

                res = await db.execute(
                    select(Workflow)
                    .where(Workflow.id == workflow_id)
                    .execution_options(skip_tenant_filter=True)
                )
                wf = res.scalar_one_or_none()
                if not wf:
                    return

                graph = db_workflow_to_graph(wf)
                await preload_workflow_datasets(workflow=graph, db=db, tenant_id=tenant_id)

                exec_id = f"cron_{uuid.uuid4().hex[:12]}"
                result = DAGExecutor.execute_workflow(
                    execution_id=exec_id,
                    workflow=graph,
                    include_node_outputs=False
                )
                result.workflow_id = workflow_id

                # Save execution result
                exec_nodes, exec_edges, exec_configs = workflow_graph_to_db_payload(graph)
                safe_node_results = [
                    nr.model_dump() if hasattr(nr, "model_dump") else (nr.dict() if hasattr(nr, "dict") else nr)
                    for nr in (result.node_results or [])
                ]

                wf.last_execution = jsonable_encoder({
                    "execution_id": result.execution_id,
                    "status": result.status,
                    "total_duration_ms": result.total_duration_ms,
                    "final_metrics": result.final_metrics,
                    "anomaly_summary": result.anomaly_summary,
                    "forecasting_summary": result.forecasting_summary,
                    "governance_summary": result.governance_summary,
                    "node_results": safe_node_results,
                    "execution_logs": result.logs,
                    "step_snapshots": result.step_snapshots,
                    "inference_schema": getattr(result, "inference_schema", None),
                    "trigger_type": "cron_schedule",
                    "cron_expression": cron_expr
                })
                await db.commit()

                # Record immutable history entry
                await record_workflow_execution_history(
                    db=db,
                    workflow_id=workflow_id,
                    execution_id=result.execution_id,
                    status_str=result.status,
                    total_duration_ms=result.total_duration_ms or 0.0,
                    nodes=exec_nodes,
                    edges=exec_edges,
                    node_configs=exec_configs,
                    metrics=result.final_metrics,
                    reports={
                        "anomaly_summary": result.anomaly_summary,
                        "forecasting_summary": result.forecasting_summary,
                        "governance_summary": result.governance_summary,
                        "node_results": safe_node_results,
                        "inference_schema": getattr(result, "inference_schema", None),
                    },
                    step_snapshots=result.step_snapshots,
                    logs=result.logs,
                    run_label=f"Scheduled Cron Run ({cron_expr})"
                )
                logger.info(f"WorkflowCronScheduler: Completed scheduled execution for '{wf.name}' ({result.status}).")

        except Exception as ex:
            logger.error(f"WorkflowCronScheduler: Failed scheduled execution for workflow '{workflow_id}': {ex}", exc_info=True)


workflow_scheduler = WorkflowCronScheduler(check_interval_seconds=30)
