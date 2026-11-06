import pandas as pd
import pytest

from backend.app.recipes.security.credential_rotation_manager import CredentialRotationManagerRecipe


def test_credential_rotation_manager_metadata_and_schema():
    recipe = CredentialRotationManagerRecipe()
    assert recipe.recipe_id == "credential_rotation_manager"
    assert recipe.category == "security"
    assert "report" in recipe.output_types

    schema = recipe.get_schema()
    assert schema["type"] == "object"
    props = schema["properties"]
    assert "credential_name" in props
    assert "vault_provider" in props
    assert "rotation_interval_days" in props
    assert "grace_period_hours" in props

    for prop_name, prop_data in props.items():
        assert "description" in prop_data, f"Property {prop_name} missing description"
        assert "input_type" in prop_data, f"Property {prop_name} missing input_type"


def test_credential_rotation_manager_execution():
    recipe = CredentialRotationManagerRecipe()
    result = recipe.execute(
        {},
        {
            "credential_name": "SNOWFLAKE_SVC_PWD",
            "vault_provider": "aws_secrets_manager",
            "rotation_interval_days": 60,
            "grace_period_hours": 48,
            "force_rotation_now": True,
        },
    )

    assert "output" in result
    assert "report" in result
    report = result["report"]
    assert report["credential_name"] == "SNOWFLAKE_SVC_PWD"
    assert report["vault_provider"] == "aws_secrets_manager"
    assert report["status"] == "ROTATED"
    assert report["active_version"] == "v2"
    assert report["masked_key"].startswith("sec_****")
    assert len(report["key_fingerprint"]) == 64
