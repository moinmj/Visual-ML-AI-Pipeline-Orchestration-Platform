import pytest
import pandas as pd
import numpy as np

from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.ingestion.manual_data_entry import ManualDataEntryRecipe
from backend.app.core.exceptions import ValidationException


def test_manual_data_entry_registration():
    """Verify recipe is registered, enabled, and assigned to Ingestion & Orchestration > Load."""
    recipe = recipe_registry.get("manual_data_entry")
    assert isinstance(recipe, ManualDataEntryRecipe)
    assert recipe.recipe_id == "manual_data_entry"
    assert recipe_registry.has("manual_data_entry")

    meta = recipe_registry._to_metadata_with_group(recipe)
    assert meta.enabled is True
    assert meta.group == "Ingestion & Orchestration"
    assert meta.subgroup == "Load"


def test_manual_data_entry_aliases():
    """Verify registered aliases resolve to manual_data_entry."""
    assert recipe_registry.get("manual_entry").recipe_id == "manual_data_entry"
    assert recipe_registry.get("manual_table").recipe_id == "manual_data_entry"
    assert recipe_registry.get("inline_data").recipe_id == "manual_data_entry"
    assert recipe_registry.get("editable_table").recipe_id == "manual_data_entry"


def test_manual_data_entry_schema_contract():
    """Verify all schema properties have input_type and description."""
    recipe = ManualDataEntryRecipe()
    schema = recipe.get_schema()
    props = schema.get("properties", {})
    assert len(props) > 0
    for name, p_data in props.items():
        assert "input_type" in p_data, f"Property '{name}' is missing 'input_type'"
        assert "description" in p_data, f"Property '{name}' is missing 'description'"


def test_manual_data_entry_json_records_string():
    """Test parsing a JSON array of record objects passed as string."""
    recipe = ManualDataEntryRecipe()
    config = {
        "data_format": "json_records",
        "data": '[{"id": 1, "name": "Alice", "score": 92.5}, {"id": 2, "name": "Bob", "score": 85.0}]',
    }
    result = recipe.execute(inputs={}, config=config)
    df = result["dataframe"]
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == ["id", "name", "score"]
    assert df.iloc[0]["name"] == "Alice"
    assert df.iloc[1]["score"] == 85.0


def test_manual_data_entry_json_records_list():
    """Test passing direct list of dictionaries in data or dataframe_records."""
    recipe = ManualDataEntryRecipe()
    config = {
        "data_format": "json_records",
        "data": [
            {"dept": "Engineering", "headcount": 45},
            {"dept": "Product", "headcount": 12},
        ],
    }
    result = recipe.execute(inputs={}, config=config)
    df = result["dataframe"]
    assert len(df) == 2
    assert "dept" in df.columns
    assert df.iloc[0]["headcount"] == 45


def test_manual_data_entry_csv_text_default():
    """Test standard comma-separated text with headers."""
    recipe = ManualDataEntryRecipe()
    csv_payload = "city,temp,rainfall\nBerlin,18.5,45.2\nTokyo,26.0,80.1\nNew York,22.1,30.0"
    config = {
        "data_format": "csv_text",
        "data": csv_payload,
        "delimiter": ",",
        "has_header": True,
    }
    result = recipe.execute(inputs={}, config=config)
    df = result["dataframe"]
    assert len(df) == 3
    assert list(df.columns) == ["city", "temp", "rainfall"]
    assert df.iloc[0]["city"] == "Berlin"


def test_manual_data_entry_csv_custom_delimiter_and_no_header():
    """Test pipe-delimited CSV text without header row."""
    recipe = ManualDataEntryRecipe()
    pipe_payload = "alpha|100|true\nbeta|200|false\ngamma|300|true"
    config = {
        "data_format": "csv_text",
        "data": pipe_payload,
        "delimiter": "|",
        "has_header": False,
    }
    result = recipe.execute(inputs={}, config=config)
    df = result["dataframe"]
    assert len(df) == 3
    assert list(df.columns) == ["column_1", "column_2", "column_3"]
    assert df.iloc[1]["column_1"] == "beta"


def test_manual_data_entry_grid_rows():
    """Test 2D grid array with header row."""
    recipe = ManualDataEntryRecipe()
    grid = [
        ["sku", "qty", "price"],
        ["ITEM-A", 10, 5.99],
        ["ITEM-B", 25, 12.50],
    ]
    config = {
        "data_format": "grid_rows",
        "data": grid,
        "has_header": True,
    }
    result = recipe.execute(inputs={}, config=config)
    df = result["dataframe"]
    assert len(df) == 2
    assert list(df.columns) == ["sku", "qty", "price"]
    assert df.iloc[0]["sku"] == "ITEM-A"


def test_manual_data_entry_jagged_grid_rows():
    """Test 2D grid with ragged / irregular row lengths being padded."""
    recipe = ManualDataEntryRecipe()
    grid = [
        ["col1", "col2", "col3"],
        ["val1"],
        ["val2", "val3", "val4", "extra_ignored"],
    ]
    config = {
        "data_format": "grid_rows",
        "data": grid,
        "has_header": True,
    }
    result = recipe.execute(inputs={}, config=config)
    df = result["dataframe"]
    assert len(df) == 2
    assert len(df.columns) == 4  # max columns across all rows
    assert df.iloc[0]["col1"] == "val1"


def test_manual_data_entry_empty_inputs():
    """Test various empty inputs return empty DataFrames safely."""
    recipe = ManualDataEntryRecipe()
    for empty_val in ["", "   \n\t  ", None, [], {}]:
        result = recipe.execute(inputs={}, config={"data": empty_val})
        assert isinstance(result["dataframe"], pd.DataFrame)
        assert result["row_count"] == 0

    # With column_types requested, empty result preserves column names
    result_with_cols = recipe.execute(
        inputs={},
        config={"data": "", "column_types": {"id": "int", "name": "str"}},
    )
    assert list(result_with_cols["dataframe"].columns) == ["id", "name"]
    assert len(result_with_cols["dataframe"]) == 0


def test_manual_data_entry_null_token_handling():
    """Test configured null tokens are replaced with NaN."""
    recipe = ManualDataEntryRecipe()
    csv_text = "id,status,note\n1,NA,active\n2,null,N/A\n3,completed,None"
    config = {
        "data_format": "csv_text",
        "data": csv_text,
        "null_values": ["NA", "null", "N/A", "None"],
    }
    result = recipe.execute(inputs={}, config=config)
    df = result["dataframe"]
    assert pd.isna(df.iloc[0]["status"])
    assert pd.isna(df.iloc[1]["status"])
    assert pd.isna(df.iloc[1]["note"])
    assert pd.isna(df.iloc[2]["note"])
    assert df.iloc[2]["status"] == "completed"


def test_manual_data_entry_explicit_column_types():
    """Test explicit column type casting."""
    recipe = ManualDataEntryRecipe()
    csv_text = "id,price,is_active,created_at\n1,99.95,true,2026-01-01\n2,149.00,0,2026-02-15"
    config = {
        "data_format": "csv_text",
        "data": csv_text,
        "column_types": {
            "id": "int",
            "price": "float",
            "is_active": "bool",
            "created_at": "datetime",
        },
    }
    result = recipe.execute(inputs={}, config=config)
    df = result["dataframe"]
    assert pd.api.types.is_integer_dtype(df["id"])
    assert pd.api.types.is_float_dtype(df["price"])
    assert pd.api.types.is_bool_dtype(df["is_active"])
    assert pd.api.types.is_datetime64_any_dtype(df["created_at"])
    assert df.iloc[0]["is_active"] is True or df.iloc[0]["is_active"] == 1
    assert df.iloc[1]["is_active"] is False or df.iloc[1]["is_active"] == 0


def test_manual_data_entry_whitespace_stripping():
    """Test trimming whitespace from column headers and string values."""
    recipe = ManualDataEntryRecipe()
    csv_text = "  name  ,   city   \n  Alice  ,   Paris   \n  Bob  ,   London  "
    config = {
        "data_format": "csv_text",
        "data": csv_text,
        "strip_whitespace": True,
    }
    result = recipe.execute(inputs={}, config=config)
    df = result["dataframe"]
    assert list(df.columns) == ["name", "city"]
    assert df.iloc[0]["name"] == "Alice"
    assert df.iloc[0]["city"] == "Paris"


def test_manual_data_entry_invalid_json_raises():
    """Test malformed JSON text raises ValidationException."""
    recipe = ManualDataEntryRecipe()
    config = {
        "data_format": "json_records",
        "data": '{"unclosed": [1, 2, 3',
    }
    with pytest.raises(ValidationException) as exc_info:
        recipe.execute(inputs={}, config=config)
    assert "Invalid JSON format" in str(exc_info.value)


def test_manual_data_entry_csv_fallback_when_in_json_mode():
    """Test graceful fallback to CSV when user pastes CSV text into json_records mode."""
    recipe = ManualDataEntryRecipe()
    csv_text = "item,count\nWidget,15\nGadget,30"
    config = {
        "data_format": "json_records",
        "data": csv_text,
    }
    result = recipe.execute(inputs={}, config=config)
    df = result["dataframe"]
    assert len(df) == 2
    assert list(df.columns) == ["item", "count"]
    assert df.iloc[0]["item"] == "Widget"


def test_manual_data_entry_passthrough():
    """Test DataFrame passed via inputs or config is returned cleanly."""
    recipe = ManualDataEntryRecipe()
    existing_df = pd.DataFrame({"x": [10, 20], "y": ["a", "b"]})
    res_inputs = recipe.execute(inputs={"dataframe": existing_df}, config={})
    assert res_inputs["dataframe"].equals(existing_df)

    res_config = recipe.execute(inputs={}, config={"dataframe": existing_df})
    assert res_config["dataframe"].equals(existing_df)

