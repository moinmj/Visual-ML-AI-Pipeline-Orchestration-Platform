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
        if "dataframe" in inputs and inputs["dataframe"] is not None:
            return {"dataframe": inputs["dataframe"]}

        storage_path = config.get("storage_path")
        dataset_id = config.get("dataset_id")
        
        if storage_path:
            try:
                from backend.app.infrastructure.storage.storage_manager import storage_manager
                df = storage_manager.read_dataframe(storage_path)
            except Exception:
                df = pd.read_csv(storage_path)
        elif dataset_id and context and hasattr(context, "get_dataset_path"):
            path = context.get_dataset_path(dataset_id)
            try:
                from backend.app.infrastructure.storage.storage_manager import storage_manager
                df = storage_manager.read_dataframe(path)
            except Exception:
                df = pd.read_csv(path)
        elif context and isinstance(context, dict) and "dataframe" in context:
            df = context["dataframe"]
        else:
            # Fallback to sample demo dataset if no dataset was selected
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
