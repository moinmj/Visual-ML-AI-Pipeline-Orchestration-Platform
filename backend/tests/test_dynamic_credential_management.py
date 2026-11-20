"""
Unit and Integration Tests for Dynamic Credential Management via Recipe Schema & Step Executor.
Verifies:
1. Automatic schema enrichment with 'save_credential' toggle and 'credential_name' input on recipes that require credentials.
2. Non-credential recipes (e.g. CSV loader, row filter) remain unaffected.
3. Auto-saving credentials during node test-step execution when 'save_credential' is True.
4. Automatic resolution of stored credentials when 'credential_id' is supplied in node config.
5. Auto-saving in inspect_recipe when 'save_credential' is included in node config.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.ingestion.postgresql_connector import PostgreSQLConnectorRecipe
from backend.app.recipes.ingestion.email_attachment_extractor import EmailAttachmentExtractorRecipe
from backend.app.recipes.ingestion.kafka_consumer import KafkaConsumerRecipe
from backend.app.recipes.ingestion.mqtt_iot_stream import MQTTStreamRecipe
from backend.app.recipes.ingestion.csv_loader import CSVLoaderRecipe
from backend.app.engine.execution.step_executor import NodeStepExecutor
from backend.app.workflows.schemas import NodeTestStepResponse


def test_schema_auto_enrichment_for_credential_recipes():
    """Verify that recipes accepting credentials automatically have 'save_credential' and 'credential_name' properties."""
    recipes = [
        PostgreSQLConnectorRecipe(),
        EmailAttachmentExtractorRecipe(),
        KafkaConsumerRecipe(),
        MQTTStreamRecipe(),
    ]

    for recipe in recipes:
        schema = recipe.get_schema()
        assert "properties" in schema, f"Recipe {recipe.recipe_id} must have schema properties"
        props = schema["properties"]

        # 1. save_credential toggle exists
        assert "save_credential" in props, f"Recipe {recipe.recipe_id} should have 'save_credential'"
        save_prop = props["save_credential"]
        assert save_prop["type"] == "boolean"
        assert save_prop["input_type"] == "toggle"
        assert save_prop["default"] is False

        # If connection_mode is present, save_credential must be conditionally visible
        if "connection_mode" in props:
            assert "visible_when" in save_prop
            assert save_prop["visible_when"] == {
                "field": "connection_mode",
                "not_equals": "credential"
            }

        # 2. credential_name exists and is conditionally visible
        assert "credential_name" in props, f"Recipe {recipe.recipe_id} should have 'credential_name'"
        name_prop = props["credential_name"]
        assert name_prop["type"] == "string"
        assert name_prop["input_type"] == "text"
        assert "visible_when" in name_prop
        assert name_prop["visible_when"] == {
            "field": "save_credential",
            "equals": True
        }

        # 3. Position: Ensure credential_id and save_credential appear ABOVE (not at the bottom)
        assert "credential_id" in props, f"Recipe {recipe.recipe_id} should have 'credential_id' for GET credentials integration"
        prop_keys = list(props.keys())
        cred_id_idx = prop_keys.index("credential_id")
        save_cred_idx = prop_keys.index("save_credential")
        cred_name_idx = prop_keys.index("credential_name")

        assert save_cred_idx == cred_id_idx + 1, "save_credential must immediately follow credential_id"
        assert cred_name_idx == save_cred_idx + 1, "credential_name must immediately follow save_credential"
        assert save_cred_idx < 5, f"save_credential must be near the top (index < 5), got {save_cred_idx}"


def test_schema_non_credential_recipes_not_enriched():
    """Verify recipes without credentials (e.g. CSV Loader) do NOT get 'save_credential'."""
    csv_recipe = CSVLoaderRecipe()
    schema = csv_recipe.get_schema()
    props = schema["properties"]
    assert "save_credential" not in props, "CSV loader should not have save_credential"
    assert "credential_name" not in props, "CSV loader should not have credential_name"


@pytest.mark.asyncio
async def test_execute_node_step_auto_saves_credential():
    """Verify that execute_node_step automatically encrypts and saves credentials when save_credential is True."""
    mock_cred = MagicMock()
    mock_cred.id = "cred-saved-uuid-1234"
    mock_cred.name = "My Analytics PostgreSQL"

    nodes = [
        {
            "id": "node_pg_1",
            "recipe_id": "postgresql_connector",
            "data": {
                "config": {
                    "connection_mode": "host_port",
                    "host": "postgres.corp.internal",
                    "port": 5432,
                    "database": "analytics",
                    "username": "data_engineer",
                    "password": "SecurePassword123!",
                    "save_credential": True,
                    "credential_name": "My Analytics PostgreSQL"
                }
            }
        }
    ]

    with patch("backend.app.recipes.ingestion.postgresql_connector.PostgreSQLConnectorRecipe.execute") as mock_exec, \
         patch("backend.app.credentials.service.CredentialService.create_credential", new_callable=AsyncMock) as mock_create_cred:

        mock_exec.return_value = {
            "dataframe": MagicMock(columns=["id", "val"]),
            "output_summary": {"status": "SUCCESS"}
        }
        mock_create_cred.return_value = mock_cred

        resp = await NodeStepExecutor.execute_node_step(
            target_node_id="node_pg_1",
            nodes=nodes,
            edges=[],
            tenant_id=1,
            session_id="test_sess_cred_save"
        )

        assert resp.success is True
        assert resp.saved_credential_id == "cred-saved-uuid-1234"
        assert resp.saved_credential_name == "My Analytics PostgreSQL"
        assert resp.output_summary.get("saved_credential_id") == "cred-saved-uuid-1234"
        assert resp.output_summary.get("saved_credential_name") == "My Analytics PostgreSQL"

        # Verify create_credential was called with correct parameters
        mock_create_cred.assert_called_once()
        call_kwargs = mock_create_cred.call_args.kwargs
        assert call_kwargs["provider"] == "postgresql"
        assert call_kwargs["name"] == "My Analytics PostgreSQL"
        assert call_kwargs["tenant_id"] == 1
        assert call_kwargs["data"]["host"] == "postgres.corp.internal"
        assert call_kwargs["data"]["username"] == "data_engineer"
        assert call_kwargs["data"]["password"] == "SecurePassword123!"
        assert "save_credential" not in call_kwargs["data"]


@pytest.mark.asyncio
async def test_execute_node_step_resolves_existing_credential():
    """Verify that execute_node_step resolves credentials from vault when credential_id is provided."""
    stored_cred_data = {
        "host": "resolved-db.internal",
        "port": 5432,
        "database": "sales_db",
        "username": "sales_reader",
        "password": "DecryptedPasswordXYZ"
    }

    nodes = [
        {
            "id": "node_pg_stored",
            "recipe_id": "postgresql_connector",
            "data": {
                "config": {
                    "connection_mode": "credential",
                    "credential_id": "vault-cred-999"
                }
            }
        }
    ]

    captured_config = {}

    def side_effect_exec(inputs, config, context=None):
        nonlocal captured_config
        captured_config.update(config)
        return {
            "dataframe": MagicMock(columns=["order_id"]),
            "output_summary": {"status": "SUCCESS"}
        }

    with patch("backend.app.recipes.ingestion.postgresql_connector.PostgreSQLConnectorRecipe.execute", side_effect=side_effect_exec), \
         patch("backend.app.credentials.service.CredentialService.get_credential_data", new_callable=AsyncMock) as mock_get_cred:

        mock_get_cred.return_value = stored_cred_data

        resp = await NodeStepExecutor.execute_node_step(
            target_node_id="node_pg_stored",
            nodes=nodes,
            edges=[],
            tenant_id=1,
            session_id="test_sess_cred_resolve"
        )

        assert resp.success is True
        mock_get_cred.assert_called_once()
        # Verify the resolved credential fields were merged into the recipe's config
        assert captured_config.get("host") == "resolved-db.internal"
        assert captured_config.get("username") == "sales_reader"
        assert captured_config.get("password") == "DecryptedPasswordXYZ"


@pytest.mark.asyncio
async def test_inspect_recipe_auto_saves_from_node_config():
    """Verify that inspect_recipe saves credentials when save_credential is true in config."""
    from backend.app.recipes.router import inspect_recipe, RecipeInspectRequest
    from backend.app.core.security import TokenData

    mock_cred = MagicMock()
    mock_cred.id = "cred-inspect-uuid-5678"
    mock_cred.name = "Inspect Saved Postgres"

    user = TokenData(sub="user-1", tenant_id=10, roles=["admin"], permissions=["*"], email="user@aiiotgeeks.com")
    db_mock = AsyncMock()

    body = RecipeInspectRequest(
        action="test_connection",
        config={
            "host": "db.prod.internal",
            "port": 5432,
            "database": "prod_db",
            "username": "prod_user",
            "password": "SecretPassword!",
            "save_credential": True,
            "credential_name": "Inspect Saved Postgres"
        }
    )

    with patch("backend.app.recipes.ingestion.postgresql_connector.PostgreSQLConnectorRecipe.inspect") as mock_inspect, \
         patch("backend.app.credentials.service.CredentialService.create_credential", new_callable=AsyncMock) as mock_create_cred:

        mock_inspect.return_value = {"success": True, "message": "Connection verified"}
        mock_create_cred.return_value = mock_cred

        res = await inspect_recipe(recipe_id="postgresql_connector", body=body, user=user, db=db_mock)

        assert res.success is True
        assert res.saved_credential_id == "cred-inspect-uuid-5678"
        assert res.saved_credential_name == "Inspect Saved Postgres"
        mock_create_cred.assert_called_once()
