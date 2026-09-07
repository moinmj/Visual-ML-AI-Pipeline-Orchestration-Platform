import pandas as pd
from typing import Dict, Any, Optional
from backend.app.recipes.base.recipe import BaseRecipe


class CSVLoaderRecipe(BaseRecipe):
    recipe_id = "csv_loader"
    name = "CSV Dataset Loader"
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
                    "description": "ID of the dataset previously uploaded to the platform."
                },
                "delimiter": {
                    "type": "string",
                    "title": "Delimiter",
                    "default": ",",
                    "enum": [",", ";", "\t", "|"]
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

        # 3. Resolve dataset_id to storage_path if not directly provided
        if dataset_id and not storage_path:
            import os
            from backend.app.core.config import settings

            storage_root = settings.LOCAL_STORAGE_DIR
            if os.path.exists(storage_root):
                for root, _, files in os.walk(storage_root):
                    for f in files:
                        if dataset_id in f:
                            storage_path = os.path.relpath(os.path.join(root, f), storage_root).replace("\\", "/")
                            break
                    if storage_path:
                        break

            # Fallback: Query SQLite database for dataset record
            if not storage_path:
                try:
                    import sqlite3
                    db_file = settings.DATABASE_URL.replace("sqlite+aiosqlite:///", "").replace("sqlite:///", "")
                    if os.path.exists(db_file):
                        conn = sqlite3.connect(db_file)
                        cur = conn.cursor()
                        cur.execute("SELECT storage_path FROM datasets WHERE id = ? LIMIT 1", (str(dataset_id),))
                        row = cur.fetchone()
                        if row and row[0]:
                            storage_path = row[0]
                        conn.close()
                except Exception:
                    pass

        # 4. Load from storage_path
        if storage_path:
            try:
                from backend.app.infrastructure.storage.storage_manager import storage_manager
                df = storage_manager.read_dataframe(storage_path)
            except Exception as e:
                import os
                if os.path.exists(storage_path):
                    if storage_path.endswith(".csv"):
                        df = pd.read_csv(storage_path, delimiter=delimiter)
                    elif storage_path.endswith(".parquet"):
                        df = pd.read_parquet(storage_path)
                    elif storage_path.endswith((".xlsx", ".xls")):
                        df = pd.read_excel(storage_path)
                    elif storage_path.endswith(".json"):
                        df = pd.read_json(storage_path)
                    else:
                        df = pd.read_csv(storage_path)
                else:
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
