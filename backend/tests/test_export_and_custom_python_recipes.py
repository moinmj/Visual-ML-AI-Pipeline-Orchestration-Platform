import os
import json
import pytest
import pandas as pd
from pathlib import Path

from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.ingestion.export_writer import ExportWriterRecipe
from backend.app.recipes.preprocessing.custom_python_step import CustomPythonStepRecipe


def test_registry_registration():
    """Verify both export_writer and custom_python_step are registered and enabled."""
    r_export = recipe_registry.get("export_writer")
    assert r_export is not None
    meta_export = recipe_registry.get_metadata("export_writer")
    assert meta_export.enabled is True
    assert meta_export.group == "Ingestion & Orchestration"
    assert meta_export.subgroup == "Output"

    r_py = recipe_registry.get("custom_python_step")
    assert r_py is not None
    meta_py = recipe_registry.get_metadata("custom_python_step")
    assert meta_py.enabled is True
    assert meta_py.group == "Data Preparation & Transformation"
    assert meta_py.subgroup == "Custom Logic"


def test_custom_python_step_df_mutation():
    """Test standard pandas in-place mutation."""
    recipe = CustomPythonStepRecipe()
    df = pd.DataFrame({"name": ["alice", "bob"], "score": [80, 95]})

    res = recipe.execute(
        {"dataframe": df},
        {"code": "df['score_pct'] = df['score'] / 100.0\nprint('Processed', len(df), 'rows')"}
    )
    out_df = res["dataframe"]
    assert "score_pct" in out_df.columns
    assert list(out_df["score_pct"]) == [0.8, 0.95]
    # Check stdout captured
    assert "Processed 2 rows" in res["output_summary"]["console_output"]


def test_custom_python_step_return_statement():
    """Test top-level return statement."""
    recipe = CustomPythonStepRecipe()
    df = pd.DataFrame({"category": ["A", "B", "A", "C"], "val": [10, 20, 30, 40]})

    res = recipe.execute(
        {"dataframe": df},
        {"code": "return df[df['category'] == 'A']"}
    )
    out_df = res["dataframe"]
    assert len(out_df) == 2
    assert set(out_df["category"]) == {"A"}


def test_custom_python_step_n8n_items_style():
    """Test n8n style items / records dictionary iteration."""
    recipe = CustomPythonStepRecipe()
    df = pd.DataFrame({"id": [1, 2], "val": [100, 200]})

    res = recipe.execute(
        {"dataframe": df},
        {"code": (
            "for item in items:\n"
            "    item['doubled'] = item['val'] * 2\n"
            "return items\n"
        )}
    )
    out_df = res["dataframe"]
    assert "doubled" in out_df.columns
    assert list(out_df["doubled"]) == [200, 400]


def test_custom_python_step_output_df_assignment():
    """Test assigning output_df explicitly."""
    recipe = CustomPythonStepRecipe()
    df = pd.DataFrame({"x": [1, 2, 3]})

    res = recipe.execute(
        {"dataframe": df},
        {"code": "output_df = pd.DataFrame({'a': [10, 20], 'b': [30, 40]})"}
    )
    out_df = res["dataframe"]
    assert list(out_df.columns) == ["a", "b"]
    assert len(out_df) == 2


def test_custom_python_step_safety_and_errors():
    """Test AST safety checks and line-number error reporting."""
    recipe = CustomPythonStepRecipe()

    # 1. Validation catches forbidden modules
    errs = recipe.validate_config({"code": "import subprocess\nsubprocess.run('ls')"})
    assert any("subprocess" in e for e in errs)

    # 2. Syntax errors detected
    errs_syntax = recipe.validate_config({"code": "def bad_syntax(:"})
    assert any("Syntax error" in e for e in errs_syntax)

    # 3. Runtime error includes line info
    df = pd.DataFrame({"x": [1, 2]})
    with pytest.raises(ValueError) as exc:
        recipe.execute({"dataframe": df}, {"code": "x = 10\ny = x / 0"})
    assert "division by zero" in str(exc.value)


def test_export_writer_csv(tmp_path):
    """Test CSV file export and passthrough output."""
    recipe = ExportWriterRecipe()
    df = pd.DataFrame({"city": ["Paris", "Tokyo"], "pop": [2.1, 14.0]})

    res = recipe.execute(
        {"dataframe": df},
        {
            "file_format": "csv",
            "file_name": "test_cities",
            "delimiter": ","
        }
    )
    # 1. Passthrough output
    assert res["dataframe"].equals(df)

    # 2. File written
    summary = res["output_summary"]
    abs_path = Path(summary["absolute_path"])
    assert abs_path.exists()
    assert abs_path.name == "test_cities.csv"
    assert summary["row_count"] == 2
    assert summary["file_size_bytes"] > 0

    # 3. Verify content
    read_back = pd.read_csv(abs_path)
    assert list(read_back["city"]) == ["Paris", "Tokyo"]


def test_export_writer_parquet(tmp_path):
    """Test Parquet export."""
    recipe = ExportWriterRecipe()
    df = pd.DataFrame({"num": [1, 2, 3], "str": ["a", "b", "c"]})

    res = recipe.execute(
        {"dataframe": df},
        {
            "file_format": "parquet",
            "file_name": "test_pq"
        }
    )
    abs_path = Path(res["output_summary"]["absolute_path"])
    assert abs_path.exists()
    assert abs_path.suffix == ".parquet"

    read_back = pd.read_parquet(abs_path)
    assert len(read_back) == 3


def test_export_writer_excel(tmp_path):
    """Test Excel export."""
    recipe = ExportWriterRecipe()
    df = pd.DataFrame({"item": ["Apple", "Banana"], "price": [1.5, 0.8]})

    res = recipe.execute(
        {"dataframe": df},
        {
            "file_format": "excel",
            "file_name": "test_sheet",
            "excel_sheet_name": "Groceries"
        }
    )
    abs_path = Path(res["output_summary"]["absolute_path"])
    assert abs_path.exists()
    assert abs_path.suffix == ".xlsx"

    read_back = pd.read_excel(abs_path, sheet_name="Groceries")
    assert list(read_back["item"]) == ["Apple", "Banana"]


def test_export_writer_json(tmp_path):
    """Test JSON export."""
    recipe = ExportWriterRecipe()
    df = pd.DataFrame({"user": ["U1", "U2"], "active": [True, False]})

    res = recipe.execute(
        {"dataframe": df},
        {
            "file_format": "json",
            "file_name": "test_json",
            "json_orient": "records"
        }
    )
    abs_path = Path(res["output_summary"]["absolute_path"])
    assert abs_path.exists()
    assert abs_path.suffix == ".json"

    with open(abs_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 2
    assert data[0]["user"] == "U1"
