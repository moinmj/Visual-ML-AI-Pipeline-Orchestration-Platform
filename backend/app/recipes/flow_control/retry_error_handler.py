"""
Retry / Error-Handling Wrapper Recipe (Ingestion & Orchestration -> Control Flow & Reliability).

Wraps any pipeline recipe step with enterprise fault-tolerance:
1. Configurable retry attempts (1 to 10) with exponential backoff and randomized jitter.
2. Filterable error matching (only retry transient timeouts, connection resets, 429s/503s).
3. Resilient fallback paths when retries are exhausted (pass-through original data, fallback recipe, error table).
4. Dual-routed outputs: Primary Output for successful or recovered datasets, Error Route for quarantine/alerts.
5. Detailed retry telemetry metrics (attempt count, delays, final status).
"""

import fnmatch
import json
import random
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from backend.app.core.exceptions import ExecutionException, ValidationException
from backend.app.core.logging import logger
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class RetryErrorHandlerRecipe(BaseRecipe):
    recipe_id = "retry_error_handler"
    name = "Retry / Error-Handling Wrapper"
    version = "1.0.0"
    category = "flow_control"
    group = "Ingestion & Orchestration"
    subgroup = "Control Flow & Reliability"
    description = "Retries a failing step with backoff and routes errors to a fallback path."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="input",
            label="Input Table",
            type="dataframe",
            required=False,
            max_connections=1,
            description="Dataset passed to the protected target recipe",
        )
    ]

    outputs = [
        RecipePort(
            id="output",
            label="Output Table",
            type="dataframe",
            description="Dataset from successful execution or recovered fallback",
        ),
        RecipePort(
            id="error_output",
            label="Error Route",
            type="dataframe",
            description="Quarantined records or error diagnostics when retries are exhausted",
        ),
        RecipePort(
            id="metrics",
            label="Retry Telemetry",
            type="metrics",
            description="Attempt history, delays, and fault-tolerance statistics",
        ),
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "target_recipe_id": {
                    "type": "string",
                    "title": "Target Recipe to Protect",
                    "default": "custom_python_step",
                    "input_type": "text",
                    "description": "Recipe ID of the node to wrap with retry and fault-tolerance protection.",
                },
                "target_recipe_config": {
                    "type": "string",
                    "title": "Target Recipe Config (JSON)",
                    "default": "{}",
                    "input_type": "textarea",
                    "description": "JSON dictionary of configuration parameters for the target recipe.",
                },
                "max_retries": {
                    "type": "integer",
                    "title": "Max Retries",
                    "default": 3,
                    "minimum": 1,
                    "maximum": 10,
                    "input_type": "number",
                    "description": "Maximum number of retry attempts before routing to the fallback strategy.",
                },
                "initial_delay_seconds": {
                    "type": "number",
                    "title": "Initial Backoff Delay (Seconds)",
                    "default": 0.1,
                    "minimum": 0.0,
                    "maximum": 60.0,
                    "input_type": "number",
                    "description": "Initial wait delay before the first retry attempt.",
                },
                "backoff_factor": {
                    "type": "number",
                    "title": "Exponential Backoff Multiplier",
                    "default": 2.0,
                    "minimum": 1.0,
                    "maximum": 5.0,
                    "input_type": "number",
                    "description": "Multiplier applied to delay on each subsequent retry (e.g. 2.0 doubles delay each attempt).",
                },
                "jitter": {
                    "type": "boolean",
                    "title": "Add Random Jitter",
                    "default": True,
                    "input_type": "checkbox",
                    "description": "Add randomized jitter to avoid synchronized retry spikes across workers.",
                },
                "fallback_strategy": {
                    "type": "string",
                    "title": "Fallback Strategy",
                    "enum": ["return_input", "empty_dataframe", "error_table", "fallback_recipe"],
                    "default": "return_input",
                    "input_type": "select",
                    "description": "Recovery action when all retries are exhausted without success.",
                },
                "fallback_recipe_id": {
                    "type": "string",
                    "title": "Fallback Recipe ID",
                    "default": "",
                    "input_type": "text",
                    "description": "Optional recipe ID to execute if fallback_strategy is 'fallback_recipe'.",
                },
                "fallback_recipe_config": {
                    "type": "string",
                    "title": "Fallback Recipe Config (JSON)",
                    "default": "{}",
                    "input_type": "textarea",
                    "description": "Optional configuration for the fallback recipe.",
                },
                "retryable_error_patterns": {
                    "type": "string",
                    "title": "Retryable Error Patterns",
                    "default": "*Timeout*;*Connection*;*503*;*RateLimit*;*Temporary*",
                    "input_type": "text",
                    "description": "Semicolon-separated glob patterns of exception messages that trigger a retry.",
                },
                "simulate_failure_attempts": {
                    "type": "integer",
                    "title": "Simulate Transient Failure Attempts",
                    "default": 0,
                    "minimum": 0,
                    "maximum": 10,
                    "input_type": "number",
                    "description": "For testing: forces the first N attempts to fail to verify retry backoff behavior.",
                },
            },
            "required": ["target_recipe_id"],
        }

    def _is_retryable_error(self, err_msg: str, patterns_str: str) -> bool:
        if not patterns_str or patterns_str.strip() == "*":
            return True
        patterns = [p.strip() for p in patterns_str.split(";") if p.strip()]
        for pat in patterns:
            if fnmatch.fnmatch(err_msg, pat) or fnmatch.fnmatch(err_msg.lower(), pat.lower()):
                return True
            try:
                if re.search(pat, err_msg, re.IGNORECASE):
                    return True
            except Exception:
                pass
        return False

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.time()
        target_rid = str(config.get("target_recipe_id", "custom_python_step")).strip()
        if not target_rid:
            raise ValidationException("Parameter 'target_recipe_id' is required.")

        target_cfg_raw = config.get("target_recipe_config", "{}")
        fallback_cfg_raw = config.get("fallback_recipe_config", "{}")

        try:
            target_cfg = json.loads(target_cfg_raw) if isinstance(target_cfg_raw, str) else target_cfg_raw
            if not isinstance(target_cfg, dict):
                target_cfg = {}
        except Exception as e:
            raise ValidationException(f"Invalid JSON in 'target_recipe_config': {e}")

        try:
            fallback_cfg = json.loads(fallback_cfg_raw) if isinstance(fallback_cfg_raw, str) else fallback_cfg_raw
            if not isinstance(fallback_cfg, dict):
                fallback_cfg = {}
        except Exception as e:
            raise ValidationException(f"Invalid JSON in 'fallback_recipe_config': {e}")

        max_retries = max(1, min(int(config.get("max_retries", 3)), 10))
        initial_delay = max(0.0, min(float(config.get("initial_delay_seconds", 0.1)), 60.0))
        backoff_factor = max(1.0, min(float(config.get("backoff_factor", 2.0)), 5.0))
        use_jitter = bool(config.get("jitter", True))
        fallback_strategy = str(config.get("fallback_strategy", "return_input")).strip().lower()
        fallback_rid = str(config.get("fallback_recipe_id", "")).strip()
        error_patterns = str(config.get("retryable_error_patterns", "*"))
        sim_failures = max(0, int(config.get("simulate_failure_attempts", 0)))

        # Resolve upstream input dataframe
        in_df = pd.DataFrame()
        if inputs:
            for k in ("dataframe", "input"):
                if isinstance(inputs.get(k), pd.DataFrame):
                    in_df = inputs[k]
                    break

        from backend.app.recipes.base.registry import recipe_registry

        attempt_history: List[Dict[str, Any]] = []
        success = False
        last_error: Optional[Exception] = None
        result_df = pd.DataFrame()
        total_delay_slept = 0.0

        for attempt in range(1, max_retries + 1):
            att_t0 = time.time()

            # Simulated transient failure for verification testing
            if attempt <= sim_failures:
                err_text = f"Simulated transient ConnectionTimeout error on attempt {attempt}"
                attempt_history.append({
                    "attempt": attempt,
                    "status": "FAILED",
                    "error": err_text,
                    "duration_ms": round((time.time() - att_t0) * 1000, 2),
                })
                last_error = ConnectionError(err_text)
            else:
                try:
                    target_recipe = recipe_registry.get(target_rid)
                    step_inputs = {"dataframe": in_df, "input": in_df}
                    res = target_recipe.execute(inputs=step_inputs, config=target_cfg)

                    if isinstance(res, dict) and "dataframe" in res:
                        result_df = res["dataframe"]
                    elif isinstance(res, pd.DataFrame):
                        result_df = res
                    else:
                        result_df = in_df.copy()

                    attempt_history.append({
                        "attempt": attempt,
                        "status": "SUCCESS",
                        "duration_ms": round((time.time() - att_t0) * 1000, 2),
                    })
                    success = True
                    break

                except Exception as e:
                    last_error = e
                    err_text = str(e)
                    attempt_history.append({
                        "attempt": attempt,
                        "status": "FAILED",
                        "error": err_text,
                        "duration_ms": round((time.time() - att_t0) * 1000, 2),
                    })

            # Check if we should retry
            if attempt < max_retries:
                if not self._is_retryable_error(str(last_error), error_patterns):
                    logger.info(f"Non-retryable error on attempt {attempt}: {last_error}")
                    break

                # Calculate exponential backoff delay
                delay = initial_delay * (backoff_factor ** (attempt - 1))
                if use_jitter:
                    delay += random.uniform(0.0, 0.05)
                # Sleep briefly (cap delay to 0.5s max in execution loop to avoid slow pipelines)
                actual_sleep = min(delay, 0.5)
                time.sleep(actual_sleep)
                total_delay_slept += actual_sleep

        error_df = pd.DataFrame()

        # If retries exhausted or non-retryable error occurred
        if not success:
            logger.warning(f"Retries exhausted for '{target_rid}' after {len(attempt_history)} attempts: {last_error}")
            error_row = {
                "target_recipe_id": target_rid,
                "error_type": type(last_error).__name__ if last_error else "UnknownError",
                "error_message": str(last_error),
                "attempts_made": len(attempt_history),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            error_df = pd.DataFrame([error_row])

            # Apply recovery strategy
            if fallback_strategy == "return_input":
                result_df = in_df.copy()
            elif fallback_strategy == "empty_dataframe":
                result_df = pd.DataFrame()
            elif fallback_strategy == "error_table":
                result_df = error_df
            elif fallback_strategy == "fallback_recipe" and fallback_rid:
                try:
                    fb_recipe = recipe_registry.get(fallback_rid)
                    fb_res = fb_recipe.execute(inputs={"dataframe": in_df, "input": in_df}, config=fallback_cfg)
                    result_df = fb_res["dataframe"] if isinstance(fb_res, dict) and "dataframe" in fb_res else fb_res
                except Exception as fb_err:
                    logger.error(f"Fallback recipe '{fallback_rid}' also failed: {fb_err}")
                    result_df = in_df.copy()
            else:
                result_df = in_df.copy()

        elapsed = round((time.time() - t0) * 1000, 2)
        metrics = {
            "execution_time_ms": elapsed,
            "target_recipe_id": target_rid,
            "success": success,
            "attempts_made": len(attempt_history),
            "max_retries": max_retries,
            "attempt_history": attempt_history,
            "total_backoff_delay_ms": round(total_delay_slept * 1000, 2),
            "fallback_strategy_applied": fallback_strategy if not success else None,
            "exhausted_retries": not success,
        }

        return {
            "dataframe": result_df,
            "error_output": error_df,
            "metrics": metrics,
        }
