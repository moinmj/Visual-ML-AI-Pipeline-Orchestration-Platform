import ast
import io
import sys
import math
import json
import re
import random
import datetime
import subprocess
import importlib
import importlib.util
from pathlib import Path
from contextlib import redirect_stdout
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from backend.app.recipes.base.recipe import BaseRecipe
from backend.app.core.logging import logger


FORBIDDEN_MODULES = {"subprocess", "shutil", "socket", "http", "urllib", "requests", "ctypes", "multiprocessing", "threading"}
FORBIDDEN_CALLS = {"exit", "quit", "breakpoint"}
PACKAGE_NAME_REGEX = re.compile(r"^[a-zA-Z0-9_\-\.<=>~!+]+$")


def _create_safe_import():
    import builtins

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        root_pkg = name.split(".")[0]
        if root_pkg in FORBIDDEN_MODULES:
            raise ImportError(f"Importing module '{root_pkg}' is restricted for security.")
        return builtins.__import__(name, globals, locals, fromlist, level)

    return safe_import


class CustomPythonStepRecipe(BaseRecipe):
    recipe_id = "custom_python_step"
    name = "Custom Python Step"
    version = "1.1.0"
    category = "preprocessing"
    description = (
        "Runs user-written Python code to transform or generate dataframes, "
        "with n8n-style items/records and pandas DataFrame support, built-in standard libraries, "
        "package management (internet PyPI or local wheels), and live console log capture."
    )
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "packages": {
                    "type": "string",
                    "title": "Additional Python Packages",
                    "input_type": "text",
                    "default": "",
                    "description": (
                        "Comma-separated list of extra Python packages to install/import (e.g. 'scipy, faker, polars'). "
                        "Pre-bundled libraries (pandas, numpy, math, json, re, random, datetime) do not need to be listed."
                    )
                },
                "install_source": {
                    "type": "string",
                    "title": "Package Installation Source",
                    "input_type": "dropdown",
                    "enum": ["internet", "local"],
                    "default": "internet",
                    "description": "Choose 'internet' to install packages from PyPI or 'local' to install from offline wheel files."
                },
                "local_wheels_dir": {
                    "type": "string",
                    "title": "Local Wheels Directory (for offline install)",
                    "input_type": "text",
                    "default": "./data/packages",
                    "description": "Path to local directory containing .whl or source archive files when install_source is 'local'."
                },
                "code": {
                    "type": "textarea",
                    "title": "Python Code",
                    "input_type": "textarea",
                    "default": (
                        "# Access data as 'df' (pandas DataFrame) or 'items' (list of dicts, n8n style)\n"
                        "# Example:\n"
                        "# df['family_size'] = df['SibSp'] + df['Parch'] + 1\n"
                        "df['processed'] = True\n"
                    ),
                    "description": (
                        "Write Python code to transform your data. "
                        "Incoming data is available as 'df' (pandas DataFrame) or 'items' / 'records' (list of dicts). "
                        "You can modify 'df' in-place, return a DataFrame, or assign to 'output_df' / 'result'."
                    )
                }
            },
            "required": ["code"],
            "templates": [
                {
                    "label": "Add Calculated Column",
                    "code": "df['calculated_metric'] = df.iloc[:, 0] * 1.1\n"
                },
                {
                    "label": "Filter Rows",
                    "code": "df = df.dropna().head(100)\n"
                },
                {
                    "label": "Group By & Aggregate",
                    "code": "df = df.describe().reset_index()\n"
                },
                {
                    "label": "String Clean & Standardize",
                    "code": "for col in df.select_dtypes(include='object').columns:\n    df[col] = df[col].astype(str).str.strip().str.upper()\n"
                },
                {
                    "label": "n8n Items / Record Loop",
                    "code": "for item in items:\n    item['is_valid'] = True\nreturn items\n"
                },
                {
                    "label": "Use Scipy / External Library",
                    "code": "import scipy.stats as stats\ndf['z_score'] = stats.zscore(df.select_dtypes(include='number').iloc[:, 0])\n"
                },
                {
                    "label": "Create New Output DataFrame",
                    "code": "output_df = pd.DataFrame({'metric': ['A', 'B', 'C'], 'value': [100, 250, 400]})\n"
                }
            ]
        }

    def _parse_packages(self, val: Any) -> List[str]:
        if not val:
            return []
        if isinstance(val, str):
            return [p.strip() for p in val.split(",") if p.strip()]
        if isinstance(val, (list, tuple)):
            res = []
            for item in val:
                if item and str(item).strip():
                    parts = str(item).split(",")
                    res.extend([p.strip() for p in parts if p.strip()])
            return res
        return []

    def _ensure_packages_installed(
        self,
        packages: List[str],
        install_source: str = "internet",
        local_wheels_dir: Optional[str] = None
    ) -> List[str]:
        """Installs requested packages via pip from PyPI or local wheels directory if not already available."""
        installed_now: List[str] = []
        install_source = (install_source or "internet").lower().strip()

        for pkg in packages:
            clean_name = re.split(r"[><=~!+]", pkg.strip())[0].strip().replace("-", "_")
            if not clean_name:
                continue

            # Check if module is already present in python environment
            if importlib.util.find_spec(clean_name) is not None:
                continue

            logger.info(f"CustomPythonStep: Installing package '{pkg}' via '{install_source}'...")
            cmd = [sys.executable, "-m", "pip", "install", "--no-warn-script-location"]

            if install_source == "local":
                wheels_path = Path(local_wheels_dir or "./data/packages").resolve()
                if not wheels_path.exists():
                    raise FileNotFoundError(
                        f"Local wheels directory '{wheels_path}' does not exist. "
                        "Please place your offline .whl files in that directory."
                    )
                cmd.extend(["--no-index", f"--find-links={wheels_path}"])

            cmd.append(pkg.strip())

            try:
                subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=180,
                    check=True
                )
                importlib.invalidate_caches()
                installed_now.append(pkg.strip())
            except subprocess.CalledProcessError as e:
                err_msg = (e.stderr or e.stdout or str(e)).strip()
                raise RuntimeError(f"Failed to install package '{pkg}' via {install_source}: {err_msg}")
            except subprocess.TimeoutExpired:
                raise TimeoutError(f"Timed out installing package '{pkg}' via {install_source} after 180s.")

        return installed_now

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors = []
        code = (config.get("code") or "").strip()
        if not code:
            errors.append("Python code cannot be empty.")
            return errors

        # Validate package names
        pkgs = self._parse_packages(config.get("packages"))
        for p in pkgs:
            if not PACKAGE_NAME_REGEX.match(p):
                errors.append(f"Invalid package specification '{p}'. Must contain only valid package name characters.")

        # Syntax check via AST
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
            return errors

        # AST-level safety scanning
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_pkg = alias.name.split(".")[0]
                    if root_pkg in FORBIDDEN_MODULES:
                        errors.append(f"Importing module '{root_pkg}' is restricted for security.")
            elif isinstance(node, ast.ImportFrom):
                root_pkg = (node.module or "").split(".")[0]
                if root_pkg in FORBIDDEN_MODULES:
                    errors.append(f"Importing from module '{root_pkg}' is restricted for security.")
            elif isinstance(node, ast.Name):
                if node.id in FORBIDDEN_CALLS:
                    errors.append(f"Function call '{node.id}()' is not permitted in pipeline steps.")

        return errors

    def _convert_to_dataframe(self, val: Any) -> Optional[pd.DataFrame]:
        """Converts diverse user return types into a valid pandas DataFrame."""
        if val is None:
            return None
        if isinstance(val, pd.DataFrame):
            return val
        if isinstance(val, pd.Series):
            return val.to_frame()
        if isinstance(val, list):
            if len(val) == 0:
                return pd.DataFrame()
            if isinstance(val[0], dict):
                return pd.DataFrame(val)
            return pd.DataFrame({"value": val})
        if isinstance(val, dict):
            try:
                return pd.DataFrame(val)
            except Exception:
                return pd.DataFrame([val])
        return None

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        df: pd.DataFrame = inputs.get("dataframe")
        if df is None:
            raise ValueError("CustomPythonStepRecipe expects 'dataframe' in inputs.")

        code = (config.get("code") or "").strip()
        if not code:
            raise ValueError("Python code cannot be empty.")

        # Ensure AST is valid
        try:
            ast.parse(code)
        except SyntaxError as e:
            raise ValueError(f"Python code syntax error at line {e.lineno}: {e.msg}")

        # Install requested packages if needed
        pkgs = self._parse_packages(config.get("packages"))
        install_src = config.get("install_source", "internet")
        wheels_dir = config.get("local_wheels_dir", "./data/packages")
        installed_packages = []
        if pkgs:
            installed_packages = self._ensure_packages_installed(pkgs, install_src, wheels_dir)

        # Setup runtime execution scope with Safe Import Hook
        df_input_copy = df.copy()
        items_list = df.to_dict(orient="records")

        stdout_buf = io.StringIO()
        result_df: Optional[pd.DataFrame] = None
        safe_import = _create_safe_import()

        safe_globals = {
            "__builtins__": {
                "abs": abs, "all": all, "any": any, "bin": bin, "bool": bool, "bytes": bytes,
                "callable": callable, "chr": chr, "complex": complex, "dict": dict, "dir": dir,
                "divmod": divmod, "enumerate": enumerate, "filter": filter, "float": float,
                "format": format, "frozenset": frozenset, "getattr": getattr, "hasattr": hasattr,
                "setattr": setattr, "delattr": delattr, "hash": hash, "hex": hex, "id": id,
                "int": int, "isinstance": isinstance, "issubclass": issubclass, "iter": iter,
                "len": len, "list": list, "map": map, "max": max, "min": min, "next": next,
                "oct": oct, "ord": ord, "pow": pow, "print": print, "range": range, "repr": repr,
                "reversed": reversed, "round": round, "set": set, "slice": slice, "sorted": sorted,
                "str": str, "sum": sum, "tuple": tuple, "type": type, "zip": zip,
                "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
                "KeyError": KeyError, "IndexError": IndexError, "AttributeError": AttributeError,
                "ImportError": ImportError, "RuntimeError": RuntimeError, "ZeroDivisionError": ZeroDivisionError,
                "StopIteration": StopIteration, "__import__": safe_import
            },
            "pd": pd,
            "pandas": pd,
            "np": np,
            "numpy": np,
            "math": math,
            "re": re,
            "json": json,
            "random": random,
            "datetime": datetime,
            "date": datetime.date,
            "time": datetime.time,
            "timedelta": datetime.timedelta,
        }

        local_scope = {
            "df": df_input_copy,
            "input_df": df_input_copy,
            "items": items_list,
            "records": items_list,
            "result": None,
            "output_df": None,
            "out": None,
        }

        has_top_level_return = bool(re.search(r"^\s*return\b", code, flags=re.MULTILINE))

        with redirect_stdout(stdout_buf):
            try:
                if has_top_level_return:
                    # Wrap in callable function to support top-level 'return ...'
                    indented_code = "\n".join("    " + line for line in code.splitlines())
                    wrapped_code = (
                        "def __user_script__(df, input_df, items, records):\n"
                        f"{indented_code}\n"
                    )
                    exec(wrapped_code, safe_globals, local_scope)
                    user_fn = local_scope["__user_script__"]
                    return_val = user_fn(df_input_copy, df_input_copy, items_list, items_list)
                    result_df = self._convert_to_dataframe(return_val)
                else:
                    exec(code, safe_globals, local_scope)

                # If no direct return value, resolve from variables
                if result_df is None:
                    for var_name in ["output_df", "result", "out"]:
                        candidate = local_scope.get(var_name)
                        converted = self._convert_to_dataframe(candidate)
                        if converted is not None:
                            result_df = converted
                            break

                # If still None, check if items was updated or if df was mutated
                if result_df is None:
                    current_items = local_scope.get("items") or local_scope.get("records")
                    current_df = local_scope.get("df")
                    if isinstance(current_df, pd.DataFrame):
                        result_df = current_df
                    elif current_items is not None:
                        result_df = pd.DataFrame(current_items)
                    else:
                        result_df = df_input_copy

            except Exception as e:
                exc_type, exc_value, exc_tb = sys.exc_info()
                tb_lines = []
                while exc_tb:
                    if exc_tb.tb_frame.f_code.co_name in ("<module>", "__user_script__"):
                        tb_lines.append(f"line {exc_tb.tb_lineno}")
                    exc_tb = exc_tb.tb_next
                loc = f" at {', '.join(tb_lines)}" if tb_lines else ""
                raise ValueError(f"Custom Python step error{loc}: {str(e)}") from e

        console_output = stdout_buf.getvalue()

        # Sanitize output DataFrame
        result_df = result_df.reset_index(drop=True)
        result_df.columns = [str(c) for c in result_df.columns]

        return {
            "dataframe": result_df,
            "feature_names": list(result_df.columns),
            "output_summary": {
                "row_count": len(result_df),
                "columns": list(result_df.columns),
                "installed_packages": installed_packages,
                "console_output": console_output.strip() if console_output else None
            },
            "metrics": {
                "row_count": len(result_df),
                "column_count": len(result_df.columns),
                "installed_packages_count": len(installed_packages),
                "console_output": console_output.strip() if console_output else None
            }
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        code = config.get("code", "df['processed'] = True").strip()
        pkgs = self._parse_packages(config.get("packages"))
        pkg_comment = f"# Required packages: {', '.join(pkgs)}\n" if pkgs else ""
        return (
            "# Custom Python Step\n"
            f"{pkg_comment}"
            f"{code}\n"
        )
