"""
Per-request tenant context.

The authenticated caller's tenant_id is stored in a ContextVar as soon as the
Bearer token is validated (see core/security.get_current_user). Everything that
must be tenant-isolated reads it from here:

  * the database guard (infrastructure/database/tenant_guard.py) automatically
    adds "tenant_id = <caller's tenant>" to every ORM query on Workflow,
    WorkflowExecution and Dataset, and stamps tenant_id on new rows;
  * the in-memory job manager only returns jobs / models of the caller's tenant;
  * the executors inject the tenant into node configs (csv_loader, dataset_join)
    and keep their temporary cache files in a per-tenant folder.

The tenant is ALWAYS taken from the token, never from a path, query or body.
"""
from contextvars import ContextVar
from typing import Optional

_current_tenant_id: ContextVar[Optional[int]] = ContextVar("current_tenant_id", default=None)


def set_current_tenant_id(tenant_id: Optional[int]) -> None:
    _current_tenant_id.set(int(tenant_id) if tenant_id is not None else None)


def get_current_tenant_id() -> Optional[int]:
    return _current_tenant_id.get()


def tenant_cache_dir() -> str:
    """Folder name used to keep local temp caches apart per tenant, e.g. 't3'."""
    tid = get_current_tenant_id()
    return f"t{tid}" if tid is not None else "t_none"


def with_tenant_config(config: Optional[dict]) -> dict:
    """
    Returns a copy of a node config with the CURRENT tenant injected as tenant_id.
    Any tenant_id the client put in the config is overwritten (or removed when
    there is no tenant context), so a workflow can never act as another tenant.
    """
    cfg = dict(config or {})
    tid = get_current_tenant_id()
    if tid is not None:
        cfg["tenant_id"] = tid
    else:
        cfg.pop("tenant_id", None)
    return cfg