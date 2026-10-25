"""
Code Execution Sandbox Recipe.
(Agents & LLM Orchestration -> Core Agents)

Executes Python code snippets in a controlled, isolated execution sandbox.
Captures stdout, return value, error tracebacks, and execution metrics.
Supports data injection (DataFrames, dictionaries) into the sandbox namespace.
"""

import io
import sys
import time
import math
import json
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np

from backend.app.recipes.base.recipe import BaseRecipe

_DEFAULT_ALLOWED_MODULES = [
    "math", "json", "re", "pandas", "pd", "numpy", "np",
    "datetime", "collections", "itertools", "statistics", "random"
]

_RESTRICTED_BUILTINS = [
    "eval", "exec", "compile", "__import__", "open", "input",
    "breakpoint", "memoryview"
]


def _safe_import_wrapper(allowed_modules: List[str]):
    """Returns a restricted __import__ function that only allows specified modules."""
    orig_import = __import__

    def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        base_name = name.split(".")[0]
        if base_name in allowed_modules:
            return orig_import(name, globals, locals, fromlist, level)
        raise ImportError(f"Importing '{name}' is disallowed in safe execution sandbox.")

    return restricted_import


def _execute_snippet(
    code: str,
    context_vars: Optional[Dict[str, Any]] = None,
    allowed_modules: Optional[List[str]] = None,
    safe_mode: bool = True
) -> Dict[str, Any]:
    """
    Executes a single Python code snippet within a restricted global scope.
    Captures stdout and the 'result' variable or the last statement value.
    """
    clean_code = str(code).strip() if code is not None else ""
    if not clean_code:
        return {
            "status": "success",
            "stdout": "",
            "result": None,
            "error": None,
            "execution_time_ms": 0.0
        }

    modules = allowed_modules if allowed_modules is not None else _DEFAULT_ALLOWED_MODULES

    # Prepare execution environment
    stdout_buffer = io.StringIO()
    sandbox_globals: Dict[str, Any] = {
        "pd": pd,
        "pandas": pd,
        "np": np,
        "numpy": np,
        "math": math,
        "json": json,
    }

    if safe_mode:
        safe_builtins = {k: v for k, v in __builtins__.items() if k not in _RESTRICTED_BUILTINS} if isinstance(__builtins__, dict) else {k: getattr(__builtins__, k) for k in dir(__builtins__) if k not in _RESTRICTED_BUILTINS}
        safe_builtins["__import__"] = _safe_import_wrapper(modules)
        sandbox_globals["__builtins__"] = safe_builtins
    else:
        sandbox_globals["__builtins__"] = __builtins__

    # Inject context variables
    if context_vars and isinstance(context_vars, dict):
        for k, v in context_vars.items():
            if str(k).isidentifier():
                sandbox_globals[str(k)] = v

    sandbox_locals: Dict[str, Any] = {}

    start_time = time.perf_counter()
    saved_stdout = sys.stdout

    try:
        sys.stdout = stdout_buffer

        # Compile snippet to check syntax first
        compiled = compile(clean_code, "<sandbox>", "exec")
        exec(compiled, sandbox_globals, sandbox_locals)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        captured_stdout = stdout_buffer.getvalue()

        # Extract returned result: check local 'result' first, or last defined variable
        result_val = sandbox_locals.get("result")
        if result_val is None:
            # Pick the last created variable that doesn't start with '_'
            user_vars = [k for k in sandbox_locals.keys() if not k.startswith("_")]
            if user_vars:
                result_val = sandbox_locals[user_vars[-1]]

        # Serialize if not primitive
        if isinstance(result_val, (pd.DataFrame, pd.Series)):
            result_val = result_val.to_dict()
        elif isinstance(result_val, np.ndarray):
            result_val = result_val.tolist()

        return {
            "status": "success",
            "stdout": captured_stdout,
            "result": result_val,
            "error": None,
            "execution_time_ms": elapsed_ms
        }

    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        captured_stdout = stdout_buffer.getvalue()
        err_msg = f"{type(exc).__name__}: {str(exc)}"
        return {
            "status": "error",
            "stdout": captured_stdout,
            "result": None,
            "error": err_msg,
            "execution_time_ms": elapsed_ms
        }
    finally:
        sys.stdout = saved_stdout


class CodeExecutionSandboxRecipe(BaseRecipe):
    recipe_id = "code_execution_sandbox"
    name = "Code Execution Sandbox"
    version = "1.0.0"
    category = "Agents & LLM Orchestration"
    group = "Agents & LLM Orchestration"
    subgroup = "Core Agents"
    media_type = "agents"
    description = "Safely compiles and runs code blocks generated by LLM agents or user scripts in an isolated sandbox."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code_column": {
                    "type": "string",
                    "title": "Code Snippet Column",
                    "default": "code",
                    "description": "Column containing code to execute."
                },
                "output_column": {
                    "type": "string",
                    "title": "Execution Output Column",
                    "default": "execution_result",
                    "description": "Base column name for stdout and results."
                },
                "safe_mode": {
                    "type": "boolean",
                    "title": "Enable Safe Sandbox Mode",
                    "default": True,
                    "description": "Restricts dangerous builtins and unauthorized modules."
                },
                "allowed_modules": {
                    "type": "array",
                    "items": {"type": "string"},
                    "title": "Allowed Import Modules",
                    "default": _DEFAULT_ALLOWED_MODULES,
                    "description": "Modules permitted for import."
                }
            }
        }

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        return []

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        """
        Executes code sandbox logic.

        Inputs:
            - code / script / python_code: str snippet, or
            - dataframe / input / df: DataFrame containing code column, or
            - context / variables / data: optional injected variables

        Config:
            - allowed_modules: List[str] of allowed modules
            - safe_mode: bool (default True)
            - code_column: str (default 'code')
            - output_column: str (default 'execution_result')
        """
        # Safe extraction without truth-value ambiguity
        raw_input = None
        for k in ["dataframe", "input", "df", "data", "corpus"]:
            if k in inputs and inputs[k] is not None:
                raw_input = inputs[k]
                break

        code_direct = None
        for k in ["code", "script", "python_code"]:
            if k in inputs and inputs[k] is not None:
                code_direct = inputs[k]
                break

        context_vars = inputs.get("context") or inputs.get("variables") or {}
        if not isinstance(context_vars, dict):
            context_vars = {"input_data": context_vars}

        allowed_modules = config.get("allowed_modules") or _DEFAULT_ALLOWED_MODULES
        safe_mode = bool(config.get("safe_mode", True))
        code_col = config.get("code_column", "code")
        output_col = config.get("output_column", "execution_result")

        # Case A: Direct single code snippet execution
        if code_direct is not None and not isinstance(code_direct, pd.DataFrame):
            res = _execute_snippet(
                code=str(code_direct),
                context_vars=context_vars,
                allowed_modules=allowed_modules,
                safe_mode=safe_mode
            )
            df_out = pd.DataFrame([{
                "code": str(code_direct),
                "status": res["status"],
                "stdout": res["stdout"],
                "result": json.dumps(res["result"]) if isinstance(res["result"], (dict, list)) else str(res["result"]),
                "error": res["error"],
                "execution_time_ms": res["execution_time_ms"]
            }])
            return {
                "dataframe": df_out,
                "output": res,
                "status": res["status"],
                "result": res["result"],
                "stdout": res["stdout"],
                "error": res["error"],
                "metrics": {
                    "total_scripts": 1,
                    "successful_runs": 1 if res["status"] == "success" else 0,
                    "failed_runs": 0 if res["status"] == "success" else 1,
                    "execution_time_ms": res["execution_time_ms"]
                }
            }

        # Case B: DataFrame input containing multiple snippets
        if isinstance(raw_input, pd.DataFrame):
            df = raw_input.copy()
            if df.empty:
                return {
                    "dataframe": df,
                    "output": [],
                    "metrics": {"total_scripts": 0, "successful_runs": 0, "failed_runs": 0}
                }

            # If target column missing, pick first text column or fallback
            if code_col not in df.columns:
                text_cols = [c for c in df.columns if df[c].dtype == object]
                code_col = text_cols[0] if text_cols else df.columns[0]

            statuses = []
            stdouts = []
            results = []
            errors = []
            times = []

            for _, row in df.iterrows():
                row_code = row.get(code_col, "")
                # Inject row fields as context
                row_context = {**context_vars, **{k: v for k, v in row.items() if str(k).isidentifier()}}
                res = _execute_snippet(
                    code=str(row_code),
                    context_vars=row_context,
                    allowed_modules=allowed_modules,
                    safe_mode=safe_mode
                )
                statuses.append(res["status"])
                stdouts.append(res["stdout"])
                results.append(json.dumps(res["result"]) if isinstance(res["result"], (dict, list)) else str(res["result"]))
                errors.append(res["error"])
                times.append(res["execution_time_ms"])

            df[f"{output_col}_status"] = statuses
            df[f"{output_col}_stdout"] = stdouts
            df[f"{output_col}_result"] = results
            df[f"{output_col}_error"] = errors
            df[f"{output_col}_time_ms"] = times

            success_cnt = sum(1 for s in statuses if s == "success")
            fail_cnt = len(statuses) - success_cnt

            return {
                "dataframe": df,
                "metrics": {
                    "total_scripts": len(df),
                    "successful_runs": success_cnt,
                    "failed_runs": fail_cnt,
                    "avg_execution_time_ms": round(sum(times) / len(times), 2) if times else 0.0
                }
            }

        # Case C: Fallback string/empty input
        code_str = str(raw_input or "")
        res = _execute_snippet(
            code=code_str,
            context_vars=context_vars,
            allowed_modules=allowed_modules,
            safe_mode=safe_mode
        )
        df_out = pd.DataFrame([{
            "code": code_str,
            "status": res["status"],
            "stdout": res["stdout"],
            "result": str(res["result"]),
            "error": res["error"],
            "execution_time_ms": res["execution_time_ms"]
        }])

        return {
            "dataframe": df_out,
            "output": res,
            "status": res["status"],
            "result": res["result"],
            "stdout": res["stdout"],
            "error": res["error"],
            "metrics": {
                "total_scripts": 1,
                "successful_runs": 1 if res["status"] == "success" else 0,
                "failed_runs": 0 if res["status"] == "success" else 1,
                "execution_time_ms": res["execution_time_ms"]
            }
        }
