import pandas as pd
from typing import Dict, Any, Optional
from backend.app.recipes.base.recipe import BaseRecipe


class CSVLoaderRecipe(BaseRecipe):
    recipe_id = "csv_loader"
    name = "Data Ingestion"
    version = "1.0.0"
    category = "ingestion"
    description = "Loads a tabular dataset from storage into the pipeline as a DataFrame."
    input_types = []
    output_types = ["dataframe"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "title": "Dataset ID",
                    "input_type": "dataset_select",
                    "description": "ID of the dataset previously uploaded to the platform."
                },
                "delimiter": {
                    "type": "string",
                    "title": "Delimiter",
                    "default": ",",
                    "enum": [",", ";", "\t", "|"],
                    "input_type": "dropdown"
                }
            },
            "required": ["dataset_id"]
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        # 1. Check if DataFrame was already passed from an upstream node
        if "dataframe" in inputs and inputs["dataframe"] is not None:
            return {"dataframe": inputs["dataframe"]}

        # 2. Check if inline records were provided in config
        if "dataframe_records" in config and isinstance(config["dataframe_records"], list) and config["dataframe_records"]:
            return {"dataframe": pd.DataFrame(config["dataframe_records"])}
        if "dataframe" in config and isinstance(config["dataframe"], pd.DataFrame):
            return {"dataframe": config["dataframe"]}

        storage_path = config.get("storage_path") or config.get("file_path")
        dataset_id = config.get("dataset_id")
        delimiter = config.get("delimiter", ",")

        # Tenant of the workflow run (injected by the executor). When present, only
        # that tenant's datasets/Ozone folder can be read.
        tenant_id = config.get("tenant_id")
        tenant_id = int(tenant_id) if tenant_id is not None else None
        dataset_tenant_id = None

        # Only Ozone paths are accepted; a local filesystem path in the node
        # config could otherwise read any file on the server.
        if storage_path and not str(storage_path).startswith("s3://"):
            raise RuntimeError(
                f"Only Ozone datasets (s3://...) can be loaded; got '{storage_path}'. "
                "Use a dataset_id or a path returned by the Datasets API."
            )

        # 3. Resolve dataset_id to storage_path (from the datasets table) if not directly provided
        if dataset_id and not storage_path:
            from backend.app.core.config import settings

            # Look up the dataset record in the platform database (SQLite or PostgreSQL)
            try:
                from sqlalchemy import create_engine, text
                from sqlalchemy.pool import NullPool
                sync_url = (
                    settings.DATABASE_URL
                    .replace("+aiosqlite", "")
                    .replace("+asyncpg", "")
                )
                if sync_url.startswith(("postgres://", "postgresql://")):
                    sync_url = "postgresql+psycopg2://" + sync_url.split("://", 1)[1]
                eng = create_engine(sync_url, poolclass=NullPool)
                try:
                    with eng.connect() as conn:
                        if tenant_id is not None:
                            row = conn.execute(
                                text("SELECT storage_path, tenant_id FROM datasets WHERE id = :id AND tenant_id = :tid LIMIT 1"),
                                {"id": str(dataset_id), "tid": tenant_id},
                            ).fetchone()
                        else:
                            row = conn.execute(
                                text("SELECT storage_path, tenant_id FROM datasets WHERE id = :id LIMIT 1"),
                                {"id": str(dataset_id)},
                            ).fetchone()
                        if row and row[0]:
                            storage_path = row[0]
                            dataset_tenant_id = row[1]
                finally:
                    eng.dispose()
            except Exception:
                pass

        # 4. Load from storage_path
        if storage_path:
            try:
                from backend.app.infrastructure.storage.storage_manager import storage_manager
                read_tenant = tenant_id if tenant_id is not None else dataset_tenant_id
                df = storage_manager.read_dataframe(storage_path, read_tenant)
            except Exception as e:
                raise RuntimeError(f"Could not load uploaded dataset '{dataset_id or storage_path}': {str(e)}")
        elif context and isinstance(context, dict) and "dataframe" in context and context["dataframe"] is not None:
            df = context["dataframe"]
        elif dataset_id:
            # If dataset_id was explicitly supplied but file could not be found, raise clear error instead of silent mock fallback
            raise RuntimeError(f"Dataset with ID '{dataset_id}' was not found in storage or database. Please ensure the file was uploaded successfully.")
        else:
            # Fallback to sample demo dataset only when NO dataset_id was requested
            import numpy as np
            np.random.seed(42)
            n = 300
            df = pd.DataFrame({
                "Age": np.random.randint(18, 70, size=n).astype(float),
                "Salary": np.random.normal(55000, 15000, size=n),
                "Experience": np.random.randint(1, 20, size=n),
                "Churn": np.random.choice([0, 1], size=n, p=[0.7, 0.3])
            })

        return {"dataframe": df}