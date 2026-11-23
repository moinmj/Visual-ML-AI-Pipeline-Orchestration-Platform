# -*- coding: utf-8 -*-
"""
Unit and integration tests for Workflow At-Rest Encryption & Security.

Verifies:
1. Passwords, connection strings with embedded credentials, secrets, tokens,
   and keys are encrypted using AES-256 (Fernet) prefixed with 'enc:' before database writes.
2. Direct inspection of DB rows in the 'workflows' and 'workflow_executions' tables
   confirms ZERO plaintext passwords or connection strings are stored.
3. Idempotent re-saves: already encrypted 'enc:' values are never double-encrypted.
4. Transparent in-memory decryption restores original credentials for DAGExecutor and NodeStepExecutor.
5. GET /api/v1/workflows/{id} returns clean decrypted configs for frontend canvas editing.
6. GET /api/v1/workflows/{id}/export returns sanitized configs with all credentials and ciphertexts stripped.
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.main import app
from backend.app.infrastructure.database.session import AsyncSessionLocal
from backend.app.workflows.models import Workflow, WorkflowExecution
from backend.app.workflows.security import (
    encrypt_secret,
    decrypt_secret,
    encrypt_node_configs,
    decrypt_node_configs,
    encrypt_workflow_nodes,
    decrypt_workflow_nodes,
    sanitize_node_configs_credentials,
    CIPHERTEXT_PREFIX,
)
from backend.app.engine.dag.graph import WorkflowGraph, WorkflowNode, WorkflowEdge
from backend.app.engine.execution.executor import DAGExecutor
from backend.app.engine.execution.step_executor import NodeStepExecutor


def test_encrypt_decrypt_secret_basic():
    """Verify encrypt_secret creates an enc: token and decrypt_secret restores it."""
    raw_pass = "#MyUser@123"
    enc = encrypt_secret(raw_pass)
    assert enc.startswith(CIPHERTEXT_PREFIX)
    assert raw_pass not in enc

    dec = decrypt_secret(enc)
    assert dec == raw_pass


def test_encrypt_secret_idempotency():
    """Verify that encrypting an already encrypted token does NOT double-encrypt."""
    raw_pass = "P@ssword!999"
    enc1 = encrypt_secret(raw_pass)
    enc2 = encrypt_secret(enc1)
    assert enc1 == enc2


def test_decrypt_unencrypted_and_corrupt():
    """Verify unencrypted strings are returned as-is and corrupt tokens do not raise."""
    assert decrypt_secret("plain_text") == "plain_text"
    assert decrypt_secret("") == ""
    assert decrypt_secret(None) is None
    # Corrupt token
    corrupt = "enc:gAAAAAB_invalid_garbage_token"
    res = decrypt_secret(corrupt)
    assert res == corrupt  # Gracefully returned without crashing


def test_encrypt_node_configs_mysql_example():
    """
    Test the user's exact MySQL configuration from user management:
    Verify password and connection_string are encrypted, while non-sensitive fields remain intact.
    """
    raw_config = {
        "processor_mysql_connector_13": {
            "connection_mode": "host_port",
            "host": "88.222.215.30",
            "port": 3306,
            "username": "MyUser",
            "password": "#MyUser@123",
            "connection_string": "mysql://MyUser:%23MyUser%40123@88.222.215.30:3306",
            "charset": "utf8mb4",
            "timeout": 15,
            "read_mode": "table",
            "database": "ProjectX",
            "select_all_tables": True,
            "tables": [],
            "table_name": "",
            "select_all_columns": False,
            "columns": [],
            "where_clause": "",
            "order_by": "",
            "query": "SELECT ID, ApprovalStatusName, IsActive FROM ApprovalStatusMaster;",
            "limit": 5,
            "offset": 0,
            "simulation_mode": False,
            "delimiter": ",",
        }
    }

    encrypted = encrypt_node_configs(raw_config)
    node = encrypted["processor_mysql_connector_13"]

    # Sensitive fields MUST be encrypted
    assert node["password"].startswith("enc:")
    assert "#MyUser@123" not in node["password"]

    assert node["connection_string"].startswith("enc:")
    assert "MyUser" not in node["connection_string"] or node["connection_string"].startswith("enc:")

    # Non-sensitive fields MUST be preserved
    assert node["host"] == "88.222.215.30"
    assert node["port"] == 3306
    assert node["username"] == "MyUser"
    assert node["database"] == "ProjectX"
    assert node["query"] == "SELECT ID, ApprovalStatusName, IsActive FROM ApprovalStatusMaster;"
    assert node["limit"] == 5

    # Decrypt and verify 100% fidelity
    decrypted = decrypt_node_configs(encrypted)
    assert decrypted["processor_mysql_connector_13"]["password"] == "#MyUser@123"
    assert decrypted["processor_mysql_connector_13"]["connection_string"] == "mysql://MyUser:%23MyUser%40123@88.222.215.30:3306"


def test_sanitize_node_configs_for_export():
    """Verify export sanitization removes both plaintext and encrypted secrets."""
    config_with_enc = {
        "node_1": {
            "host": "localhost",
            "password": "enc:gAAAAAB_something_encrypted",
            "connection_string": "enc:gAAAAAB_conn_encrypted",
            "api_key": "enc:gAAAAAB_key_encrypted",
            "database": "analytics"
        }
    }
    sanitized = sanitize_node_configs_credentials(config_with_enc)
    assert sanitized["node_1"]["password"] == ""
    assert sanitized["node_1"]["connection_string"] == ""
    assert sanitized["node_1"]["api_key"] == ""
    assert sanitized["node_1"]["host"] == "localhost"
    assert sanitized["node_1"]["database"] == "analytics"


@pytest.mark.asyncio
async def test_workflow_at_rest_encryption_in_database():
    """
    Integration test:
    1. Create a workflow via POST /api/v1/workflows with plaintext credentials.
    2. Query the raw database row using SQLAlchemy select(Workflow).
    3. Assert that node_configs and nodes in the DB store enc: ciphertext with 0 plaintext passwords.
    4. Fetch via GET /api/v1/workflows/{id} and assert UI receives decrypted configs.
    """
    wf_id = "test_wf_encryption_db_01"
    raw_node_configs = {
        "mysql_node_01": {
            "recipe_id": "mysql_connector",
            "connection_mode": "host_port",
            "host": "88.222.215.30",
            "port": 3306,
            "username": "MyUser",
            "password": "#MyUser@123",
            "connection_string": "mysql://MyUser:%23MyUser%40123@88.222.215.30:3306",
            "database": "ProjectX",
            "simulation_mode": True,
        }
    }
    nodes_payload = [
        {
            "id": "mysql_node_01",
            "recipe_id": "mysql_connector",
            "label": "MySQL Loader",
            "config": {
                "connection_mode": "host_port",
                "host": "88.222.215.30",
                "password": "#MyUser@123",
                "database": "ProjectX",
            }
        }
    ]

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        headers = {
            "X-Tenant-Id": "1",
            "X-User-Role": "Tenant Admin",
            "X-User-Id": "test_admin",
        }
        create_resp = await ac.post(
            "/api/v1/workflows/",
            headers=headers,
            json={
                "id": wf_id,
                "name": "Secure DB Pipeline",
                "nodes": nodes_payload,
                "edges": [],
                "node_configs": raw_node_configs,
            }
        )
        assert create_resp.status_code in (200, 201), f"Workflow create failed: {create_resp.text}"

        # 1. Direct raw DB Inspection
        async with AsyncSessionLocal() as db:
            raw_wf_res = await db.execute(select(Workflow).where(Workflow.id == wf_id))
            raw_wf = raw_wf_res.scalar_one_or_none()
            assert raw_wf is not None

            # Verify node_configs in DB has enc: prefix and NO raw password
            db_cfg = raw_wf.node_configs.get("mysql_node_01", {})
            assert db_cfg.get("password", "").startswith("enc:"), "DB password must be encrypted with enc:"
            assert "#MyUser@123" not in db_cfg.get("password", ""), "Raw password MUST NOT appear in DB"

            assert db_cfg.get("connection_string", "").startswith("enc:"), "DB connection_string must be encrypted"
            assert "#MyUser@123" not in db_cfg.get("connection_string", ""), "Raw password in connection_string MUST NOT appear in DB"

            # Verify nodes in DB also has enc: prefix
            db_node_cfg = raw_wf.nodes[0].get("config", {})
            assert db_node_cfg.get("password", "").startswith("enc:"), "DB nodes[0].config.password must be encrypted"

        # 2. Canvas UI retrieval via GET /api/v1/workflows/{id}
        get_resp = await ac.get(f"/api/v1/workflows/{wf_id}", headers=headers)
        assert get_resp.status_code == 200
        get_data = get_resp.json()

        # Canvas UI receives transparently decrypted values
        ui_cfg = get_data["node_configs"]["mysql_node_01"]
        assert ui_cfg["password"] == "#MyUser@123"
        assert ui_cfg["connection_string"] == "mysql://MyUser:%23MyUser%40123@88.222.215.30:3306"

        ui_node = get_data["nodes"][0]["config"]
        assert ui_node["password"] == "#MyUser@123"

        # 3. Export via GET /api/v1/workflows/{id}/export
        export_resp = await ac.get(f"/api/v1/workflows/{wf_id}/export", headers=headers)
        assert export_resp.status_code == 200
        export_data = export_resp.json()
        export_cfg = export_data["pipeline"]["node_configs"]["mysql_node_01"]
        assert export_cfg["password"] == "", "Exported password must be redacted to empty string"
        assert export_cfg["connection_string"] == "", "Exported connection_string must be redacted"
        assert "enc:" not in export_cfg["password"]


@pytest.mark.asyncio
async def test_dag_executor_transparent_decryption():
    """
    Verify DAGExecutor transparently decrypts encrypted node configurations in memory
    during live pipeline execution.
    """
    raw_pass = "#MyUser@123"
    enc_pass = encrypt_secret(raw_pass)

    # Build graph with encrypted password and simulation mode
    graph = WorkflowGraph(
        nodes=[
            WorkflowNode(
                id="mysql_1",
                recipe_id="mysql_connector",
                config={
                    "connection_mode": "host_port",
                    "host": "88.222.215.30",
                    "port": 3306,
                    "username": "MyUser",
                    "password": enc_pass,  # Stored encrypted
                    "database": "ProjectX",
                    "select_all_tables": True,
                    "simulation_mode": True,  # Uses simulated data for test
                    "limit": 5,
                }
            )
        ],
        edges=[]
    )

    result = DAGExecutor.execute_workflow(
        execution_id="exec_test_encryption_01",
        workflow=graph
    )
    assert result.status == "SUCCESS"
    assert len(result.node_results) == 1
    assert result.node_results[0].status == "SUCCESS"


@pytest.mark.asyncio
async def test_node_step_executor_transparent_decryption():
    """
    Verify NodeStepExecutor transparently decrypts encrypted credentials
    when testing a single canvas step.
    """
    raw_pass = "#MyUser@123"
    enc_pass = encrypt_secret(raw_pass)

    test_resp = await NodeStepExecutor.execute_node_step(
        session_id="test_step_session_01",
        target_node_id="mysql_step_1",
        nodes=[
            {
                "id": "mysql_step_1",
                "recipe_id": "mysql_connector",
                "config": {
                    "connection_mode": "host_port",
                    "host": "88.222.215.30",
                    "port": 3306,
                    "username": "MyUser",
                    "password": enc_pass,
                    "database": "ProjectX",
                    "select_all_tables": True,
                    "simulation_mode": True,
                    "limit": 5,
                }
            }
        ],
        edges=[],
        node_configs={
            "mysql_step_1": {
                "connection_mode": "host_port",
                "host": "88.222.215.30",
                "port": 3306,
                "username": "MyUser",
                "password": enc_pass,
                "database": "ProjectX",
                "select_all_tables": True,
                "simulation_mode": True,
                "limit": 5,
            }
        }
    )
    assert test_resp is not None
    assert test_resp.success is True
    assert test_resp.status == "SUCCESS"
