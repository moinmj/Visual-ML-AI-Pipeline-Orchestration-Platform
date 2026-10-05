"""
Database-level tenant isolation for Workflow, WorkflowExecution and Dataset.

Installed once at import time (imported from main.py). Two SQLAlchemy hooks:

1. do_orm_execute: every ORM SELECT / UPDATE / DELETE that touches one of the
   tenant-owned models gets "tenant_id = <caller's tenant>" added automatically,
   so another tenant's rows behave exactly as if they did not exist (404).
   With no tenant in context the filter matches NOTHING (fail closed).
   Internal code that genuinely needs to look across tenants (only: checking
   whether a client-supplied ID is already taken) must opt out explicitly with
   .execution_options(skip_tenant_filter=True).

2. before_flush: new rows get the caller's tenant_id stamped on them, and any
   attempt to write a row that belongs to a different tenant raises.

Raw SQL (text()) and Core connections are not affected - e.g. startup
migrations in session.init_db.
"""
from sqlalchemy import event
from sqlalchemy.orm import Session, with_loader_criteria

from backend.app.core.tenant_context import get_current_tenant_id
from backend.app.workflows.models import Workflow, WorkflowExecution
from backend.app.datasets.models import Dataset

TENANT_SCOPED_MODELS = (Workflow, WorkflowExecution, Dataset)
_NO_TENANT = -1  # matches no row: used when there is no tenant in context


@event.listens_for(Session, "do_orm_execute")
def _add_tenant_filter(state):
    if state.execution_options.get("skip_tenant_filter"):
        return
    if not (state.is_select or state.is_update or state.is_delete):
        return
    tid = get_current_tenant_id()
    if tid is None:
        tid = _NO_TENANT
    options = [
        with_loader_criteria(model, lambda cls: cls.tenant_id == tid, include_aliases=True)
        for model in TENANT_SCOPED_MODELS
    ]
    state.statement = state.statement.options(*options)


@event.listens_for(Session, "before_flush")
def _stamp_and_check_tenant(session, flush_context, instances):
    tid = get_current_tenant_id()
    if tid is not None:
        try:
            tid = int(tid)
        except (ValueError, TypeError):
            pass
    for obj in list(session.new) + list(session.dirty) + list(session.deleted):
        if not isinstance(obj, TENANT_SCOPED_MODELS):
            continue
        if obj.tenant_id is None and tid is not None:
            obj.tenant_id = tid
        if tid is not None and obj.tenant_id is not None:
            obj_tid = obj.tenant_id
            try:
                obj_tid = int(obj_tid)
            except (ValueError, TypeError):
                pass
            if obj_tid != tid:
                raise PermissionError(
                    f"Refusing to write {type(obj).__name__} '{getattr(obj, 'id', '?')}' "
                    f"of tenant {obj.tenant_id} from tenant {tid}'s request"
                )