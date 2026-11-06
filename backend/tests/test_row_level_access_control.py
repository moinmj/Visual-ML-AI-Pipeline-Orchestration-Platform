import pandas as pd
import pytest

from backend.app.recipes.security.row_level_access_control import RowLevelAccessControlRecipe


def test_row_level_access_control_metadata_and_schema():
    recipe = RowLevelAccessControlRecipe()
    assert recipe.recipe_id == "row_level_access_control"
    assert recipe.category == "security"
    assert "dataframe" in recipe.output_types

    schema = recipe.get_schema()
    assert schema["type"] == "object"
    props = schema["properties"]
    assert "user_role" in props
    assert "policy_enforcement_mode" in props
    assert "masking_placeholder" in props

    for prop_name, prop_data in props.items():
        assert "description" in prop_data, f"Property {prop_name} missing description"
        assert "input_type" in prop_data, f"Property {prop_name} missing input_type"


def test_row_level_access_control_strict_filter():
    recipe = RowLevelAccessControlRecipe()
    # Default synthetic dataset has 6 rows: 2 viewer rows for tenant-default
    result = recipe.execute(
        {},
        {
            "user_role": "viewer",
            "user_department": "All",
            "user_tenant_id": "tenant-default",
            "policy_enforcement_mode": "strict_filter",
        },
    )

    df_out = result["dataframe"]
    audit = result["security_audit"]
    assert len(df_out) == 2
    assert audit["authorized_rows_count"] == 2
    assert audit["unauthorized_rows_count"] == 4


def test_row_level_access_control_admin_bypass():
    recipe = RowLevelAccessControlRecipe()
    result = recipe.execute({}, {"user_role": "admin", "bypass_for_admin": True})
    df_out = result["dataframe"]
    audit = result["security_audit"]
    assert len(df_out) == 6
    assert audit["admin_bypassed"] is True
    assert audit["unauthorized_rows_count"] == 0


def test_row_level_access_control_masking():
    recipe = RowLevelAccessControlRecipe()
    result = recipe.execute(
        {},
        {
            "user_role": "viewer",
            "user_tenant_id": "tenant-default",
            "policy_enforcement_mode": "mask_disallowed_rows",
            "masking_placeholder": "[LOCKED]",
        },
    )

    df_out = result["dataframe"]
    # Keeps all rows but masks sensitive columns on unauthorized rows
    assert len(df_out) == 6
    assert "[LOCKED]" in df_out["client_name"].values


def test_row_level_access_control_raise_error():
    recipe = RowLevelAccessControlRecipe()
    with pytest.raises(PermissionError) as exc_info:
        recipe.execute(
            {},
            {
                "user_role": "guest",
                "policy_enforcement_mode": "raise_error_if_unauthorized",
            },
        )
    assert "RowLevelAccessControl violation" in str(exc_info.value)
