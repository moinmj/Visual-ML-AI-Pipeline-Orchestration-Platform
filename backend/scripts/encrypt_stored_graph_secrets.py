"""
One-off: encrypt credentials that were stored in plaintext before every write path was covered.

Rewrites, for all tenants, the graph columns of
    workflows            nodes, node_configs, last_execution (its graph snapshot only)
    workflow_versions    nodes, node_configs
    workflow_executions  snapshot_nodes, snapshot_node_configs
with the same protect_graph_secrets / protect_execution_secrets the ORM hooks apply on every
write (backend/app/workflows/models.py). Run results (metrics, reports, logs) are not touched,
and updated_at is left as it is.

Dry run by default: prints how many rows would change, never a secret value.

    python -m backend.scripts.encrypt_stored_graph_secrets            # dry run
    python -m backend.scripts.encrypt_stored_graph_secrets --apply    # write

Uses DATABASE_URL and the encryption key (SECRET_KEY) from the app settings, so run it with the
same .env as the API: values encrypted with another key could not be decrypted at run time.
"""
import argparse
import asyncio
from typing import Any, Callable, Dict, List, Tuple

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import create_async_engine

from backend.app.core.config import settings
from backend.app.core.crypto import protect_execution_secrets, protect_graph_secrets
from backend.app.workflows.models import Workflow, WorkflowExecution, WorkflowVersion


def _graph(nodes_col: str, configs_col: str) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    def fix(row: Dict[str, Any]) -> Dict[str, Any]:
        nodes, configs = protect_graph_secrets(row[nodes_col], row[configs_col])
        return {nodes_col: nodes, configs_col: configs}
    return fix


def _workflow(row: Dict[str, Any]) -> Dict[str, Any]:
    out = _graph("nodes", "node_configs")(row)
    out["last_execution"] = protect_execution_secrets(row["last_execution"])
    return out


# (table, columns read, fixer)
TARGETS: List[Tuple[Any, Tuple[str, ...], Callable]] = [
    (Workflow.__table__, ("nodes", "node_configs", "last_execution"), _workflow),
    (WorkflowVersion.__table__, ("nodes", "node_configs"), _graph("nodes", "node_configs")),
    (WorkflowExecution.__table__, ("snapshot_nodes", "snapshot_node_configs"),
     _graph("snapshot_nodes", "snapshot_node_configs")),
]


async def run(apply: bool) -> None:
    # Core statements on the tables (not ORM): the tenant filter doesn't apply, so all tenants
    # are covered, and the ORM hooks don't run, so the fix is applied explicitly here.
    engine = create_async_engine(str(settings.DATABASE_URL))
    try:
        async with engine.begin() as conn:
            for table, cols, fix in TARGETS:
                rows = (await conn.execute(select(table.c.id, *(table.c[c] for c in cols)))).mappings().all()
                changed_rows = 0
                changed_cols: Dict[str, int] = {c: 0 for c in cols}
                for row in rows:
                    new = fix(dict(row))
                    diff = {c: v for c, v in new.items() if v != row[c]}
                    if not diff:
                        continue
                    changed_rows += 1
                    for c in diff:
                        changed_cols[c] += 1
                    if apply:
                        values = dict(diff)
                        if "updated_at" in table.c:     # keep it: the column's onupdate would set it to now
                            values["updated_at"] = table.c.updated_at
                        await conn.execute(update(table).where(table.c.id == row["id"]).values(**values))
                per_col = ", ".join(f"{c}: {n}" for c, n in changed_cols.items())
                print(f"{table.name}: {len(rows)} rows, {changed_rows} "
                      f"{'updated' if apply else 'would change'} ({per_col})")
            if not apply:
                await conn.rollback()
                print("Dry run: nothing was written. Re-run with --apply to write.")
            else:
                print("Done: changes committed.")
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="write the changes (default: dry run)")
    asyncio.run(run(parser.parse_args().apply))


if __name__ == "__main__":
    main()
