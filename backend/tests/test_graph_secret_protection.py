"""
Credentials in stored workflow graphs are never written in plaintext: nested settings, both node
shapes, execution snapshots built from a run's (decrypted) graph, and every write path through
the ORM hooks on Workflow / WorkflowVersion / WorkflowExecution.
"""
import uuid

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

import backend.app.infrastructure.database.tenant_guard  # noqa: F401  (registers the tenant filter)
from backend.app.core.crypto import (
    MASK_PLACEHOLDER,
    decrypt_credential,
    decrypt_node_config_secrets,
    encrypt_credential,
    encrypt_node_config_secrets,
    is_encrypted_token,
    is_secret_config_key,
    protect_execution_secrets,
    protect_graph_secrets,
)
from backend.app.core.tenant_context import set_current_tenant_id
from backend.app.infrastructure.database.session import Base
from backend.app.workflows.models import Workflow, WorkflowExecution, WorkflowVersion
from backend.app.workflows.router import mask_workflow_configs_for_frontend, sanitize_vaulted_node_configs


# ── Key names ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("key", ["password", "smtp_password", "api_key", "aws_secret_access_key",
                                 "refresh_token", "bot_token", "PROD_DB_PASSWORD", "webhook_url"])
def test_secret_keys(key):
    assert is_secret_config_key(key)


@pytest.mark.parametrize("key", ["max_tokens", "tokenizer", "token_url", "api_key_header",
                                 "private_key_path", "username", "host"])
def test_non_secret_keys(key):
    assert not is_secret_config_key(key)


# ── Nested encryption mirrors runtime decryption ──────────────────────────

def test_nested_secrets_encrypted_and_decrypted_back():
    cfg = {
        "host": "db.local",
        "password": "top-pass",
        "fallback_values": {"PROD_DB_PASSWORD": "nested-pass", "REGION": "eu"},
        "connections": [{"name": "a", "api_key": "list-key"}],
        "max_tokens": "512",
    }
    enc = encrypt_node_config_secrets(cfg)
    assert cfg["password"] == "top-pass"                       # input not mutated
    assert is_encrypted_token(enc["password"])
    assert is_encrypted_token(enc["fallback_values"]["PROD_DB_PASSWORD"])
    assert is_encrypted_token(enc["connections"][0]["api_key"])
    assert enc["fallback_values"]["REGION"] == "eu"
    assert enc["max_tokens"] == "512"
    assert decrypt_node_config_secrets(enc) == cfg


def test_already_encrypted_values_are_kept_and_masks_dropped():
    token = encrypt_credential("s3cret")
    enc = encrypt_node_config_secrets({"password": token, "inner": {"api_key": "••••••••"}})
    assert enc["password"] == token
    assert enc["inner"] == {}


# ── Graph protection ──────────────────────────────────────────────────────

def test_protect_graph_handles_both_node_shapes_and_wrapped_configs():
    nodes = [
        {"id": "a", "config": {"api_key": "flat-key"}},
        {"id": "b", "data": {"label": "B", "config": {"password": "data-pass"}}},
    ]
    configs = {
        "a": {"api_key": "flat-key"},
        "c": {"recipe_id": "telegram", "config": {"bot_token": "wrapped-token"}},
    }
    new_nodes, new_configs = protect_graph_secrets(nodes, configs)
    assert is_encrypted_token(new_nodes[0]["config"]["api_key"])
    assert is_encrypted_token(new_nodes[1]["data"]["config"]["password"])
    assert new_nodes[1]["data"]["label"] == "B"
    assert is_encrypted_token(new_configs["a"]["api_key"])
    assert is_encrypted_token(new_configs["c"]["config"]["bot_token"])
    assert nodes[0]["config"]["api_key"] == "flat-key"        # inputs not mutated
    assert decrypt_credential(new_configs["c"]["config"]["bot_token"]) == "wrapped-token"


def test_protect_graph_drops_secrets_of_vaulted_nodes():
    # what a run's graph looks like after preload_workflow_credentials injected the vault secret
    _, configs = protect_graph_secrets(None, {
        "pg": {"credential_id": "cred-1", "host": "h", "password": "from-vault"},
        "wrapped": {"config": {"credential_id": "cred-2", "api_key": "from-vault"}},
    })
    assert configs["pg"] == {"credential_id": "cred-1", "host": "h"}
    assert configs["wrapped"] == {"config": {"credential_id": "cred-2"}}


def test_protect_execution_only_touches_graph_snapshot():
    ex = {
        "execution_id": "e1",
        "snapshot_node_configs": {"n": {"config": {"password": "p"}}},
        "snapshot_nodes": [{"id": "n", "config": {"password": "p"}}],
        "node_outputs": {"n": {"share_token": "shr_public"}},
    }
    safe = protect_execution_secrets(ex)
    assert is_encrypted_token(safe["snapshot_node_configs"]["n"]["config"]["password"])
    assert is_encrypted_token(safe["snapshot_nodes"][0]["config"]["password"])
    assert safe["node_outputs"] == ex["node_outputs"]          # run results are left alone
    assert "nodes" not in safe and "node_configs" not in safe
    assert protect_execution_secrets(None) is None


# ── Frontend masking and saving masked values back ────────────────────────

def test_masking_covers_nested_and_flat_nodes_and_save_restores_them():
    stored_nodes = [{"id": "a", "config": {"host": "h", "api_key": encrypt_credential("k1")}}]
    stored_configs = {"a": {"host": "h", "fallback_values": {"DB_PASSWORD": encrypt_credential("k2")}}}

    masked_nodes, masked_configs = mask_workflow_configs_for_frontend(stored_nodes, stored_configs)
    assert masked_nodes[0]["config"]["api_key"] == MASK_PLACEHOLDER
    assert masked_configs["a"]["fallback_values"]["DB_PASSWORD"] == MASK_PLACEHOLDER
    assert masked_nodes[0]["config"]["host"] == "h"

    class _Wf:
        nodes = stored_nodes
        node_configs = stored_configs

    # the UI sends the masked graph back unchanged: stored secrets must survive the save
    clean_nodes, clean_configs = sanitize_vaulted_node_configs(masked_nodes, masked_configs, existing_wf=_Wf())
    assert decrypt_credential(clean_nodes[0]["config"]["api_key"]) == "k1"
    assert decrypt_credential(clean_configs["a"]["fallback_values"]["DB_PASSWORD"]) == "k2"


# ── ORM hooks: every write path is covered ────────────────────────────────

@pytest_asyncio.fixture
async def db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)()
    set_current_tenant_id(1)
    yield session
    await session.close()
    await engine.dispose()
    set_current_tenant_id(None)


async def _fresh(db, model, row_id):
    db.expunge_all()
    return (await db.execute(select(model).where(model.id == row_id))).scalar_one()


@pytest.mark.asyncio
async def test_hooks_encrypt_on_insert_and_update(db):
    wf_id = str(uuid.uuid4())
    db.add(Workflow(
        id=wf_id, name="wf",
        nodes=[{"id": "n", "config": {"password": "plain-1"}}],
        edges=[],
        node_configs={"n": {"password": "plain-1"}},
        last_execution={"execution_id": "e0", "snapshot_node_configs": {"n": {"config": {"api_key": "plain-2"}}}},
    ))
    await db.commit()
    wf = await _fresh(db, Workflow, wf_id)
    assert is_encrypted_token(wf.nodes[0]["config"]["password"])
    assert is_encrypted_token(wf.node_configs["n"]["password"])
    assert is_encrypted_token(wf.last_execution["snapshot_node_configs"]["n"]["config"]["api_key"])

    # an update that writes plaintext (e.g. a code path that skipped sanitizing) is encrypted too
    wf.node_configs = {"n": {"nested": {"secret": "plain-3"}}}
    await db.commit()
    wf = await _fresh(db, Workflow, wf_id)
    assert decrypt_credential(wf.node_configs["n"]["nested"]["secret"]) == "plain-3"

    # history entry built from a run's decrypted graph, as the scheduler and /execute do
    ex_id = str(uuid.uuid4())
    db.add(WorkflowExecution(
        id=ex_id, workflow_id=wf_id, version_number=1,
        snapshot_nodes=[{"id": "n", "data": {"config": {"smtp_password": "plain-4"}}}],
        snapshot_edges=[],
        snapshot_node_configs={"n": {"credential_id": "c1", "password": "from-vault"}},
    ))
    db.add(WorkflowVersion(
        workflow_id=wf_id, version_number=1, name="wf", definition_hash="x",
        nodes=[], edges=[], node_configs={"n": {"api_key": "plain-5"}},
    ))
    await db.commit()
    ex = await _fresh(db, WorkflowExecution, ex_id)
    assert is_encrypted_token(ex.snapshot_nodes[0]["data"]["config"]["smtp_password"])
    assert ex.snapshot_node_configs == {"n": {"credential_id": "c1"}}
    ver = (await db.execute(select(WorkflowVersion).where(WorkflowVersion.workflow_id == wf_id))).scalar_one()
    assert is_encrypted_token(ver.node_configs["n"]["api_key"])
