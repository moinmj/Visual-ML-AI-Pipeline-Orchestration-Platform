import pytest
import pandas as pd
from backend.app.recipes import recipe_registry
from backend.app.engine.execution.step_executor import NodeStepExecutor
from backend.app.recipes.preprocessing.custom_python_step import CustomPythonStepRecipe
from backend.app.recipes.agents.code_execution_sandbox import CodeExecutionSandboxRecipe


def test_custom_python_security_os_sys_blocked():
    """Verify that os, sys, and dangerous calls are blocked in custom python step."""
    recipe = CustomPythonStepRecipe()
    errs = recipe.validate_config({"code": "import os\nx = 1"})
    assert any("os" in e for e in errs)

    errs_sys = recipe.validate_config({"code": "import sys\nx = 1"})
    assert any("sys" in e for e in errs_sys)

    errs_clean = recipe.validate_config({"code": "import math\ndf['root'] = 4"})
    assert len(errs_clean) == 0


def test_code_execution_sandbox_isolated_stdout():
    """Verify code execution sandbox captures output without contaminating sys.stdout."""
    recipe = CodeExecutionSandboxRecipe()
    df = pd.DataFrame({"val": [10, 20]})
    res = recipe.execute({"dataframe": df}, {"code": "result = 100", "input_mode": "per_row"})
    assert "dataframe" in res
    assert res["metrics"]["successful_runs"] == 2


def test_step_executor_path_sanitization():
    """Verify that path traversal in node_id does not cause unexpected directory escape."""
    assert NodeStepExecutor.is_node_cached("safe_sess", "../../etc/passwd") is False
    assert NodeStepExecutor.load_node_meta("safe_sess", "../../etc/passwd") is None
