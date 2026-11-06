import pandas as pd
import pytest

from backend.app.recipes.reporting.bi_tool_export import BiToolExportRecipe


def test_bi_tool_export_metadata_and_schema():
    recipe = BiToolExportRecipe()
    assert recipe.recipe_id == "bi_tool_export"
    assert recipe.category == "reporting"
    assert "dataframe" in recipe.output_types

    schema = recipe.get_schema()
    assert schema["type"] == "object"
    props = schema["properties"]
    assert "target_tool" in props
    assert "dataset_name" in props
    assert "tableau_format" in props
    assert "powerbi_mode" in props
    assert "generate_lookml_model" in props

    for prop_name, prop_data in props.items():
        assert "description" in prop_data, f"Property {prop_name} missing description"
        assert "input_type" in prop_data, f"Property {prop_name} missing input_type"


def test_bi_tool_export_fallback_simulation():
    recipe = BiToolExportRecipe()
    result = recipe.execute({}, {})

    assert "output" in result
    assert "dataframe" in result
    assert "export_manifest" in result
    assert "lookml_view" in result
    assert "powerbi_schema" in result
    assert "tableau_tds" in result

    manifest = result["export_manifest"]
    assert manifest["status"] == "COMPLETED"
    assert manifest["rows_exported"] > 0
    assert len(manifest["measure_columns"]) > 0
    assert len(manifest["dimension_columns"]) > 0

    # Verify TDS XML
    tds = result["tableau_tds"]
    assert "<datasource" in tds
    assert "role=\"measure\"" in tds

    # Verify LookML
    lookml = result["lookml_view"]
    assert "view: analytics_view {" in lookml
    assert "measure: count {" in lookml

    # Verify PowerBI
    pbi = result["powerbi_schema"]
    assert pbi["defaultMode"] == "Push"
    assert len(pbi["tables"][0]["columns"]) > 0


def test_bi_tool_export_custom_dataframe():
    recipe = BiToolExportRecipe()
    df = pd.DataFrame({
        "order_id": ["ORD-1", "ORD-2", "ORD-3"],
        "revenue": [250.0, 480.5, 99.0],
        "is_refunded": [False, False, True],
    })

    result = recipe.execute(
        {"input": df},
        {
            "target_tool": "looker",
            "dataset_name": "Orders_Analytics",
            "looker_view_name": "orders_view",
            "cloud_storage_bucket": "s3://company-bi-lake/orders/",
        },
    )

    manifest = result["export_manifest"]
    assert manifest["dataset_name"] == "Orders_Analytics"
    assert manifest["rows_exported"] == 3
    assert "revenue" in manifest["measure_columns"]
    assert "order_id" in manifest["dimension_columns"]
    assert "view: orders_view {" in result["lookml_view"]
    assert result["dataframe"].equals(df)
