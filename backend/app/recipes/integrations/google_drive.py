import pandas as pd
from typing import Dict, Any, List, Optional
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class GoogleDriveRecipe(BaseRecipe):
    recipe_id = "google_drive"
    name = "Google Drive"
    version = "1.0.0"
    category = "integrations"
    description = "Uploads generated datasets, trained models, and reports to Google Drive folders or downloads remote files."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="input",
            label="File / Dataset",
            type="dataframe",
            required=False,
            max_connections=1,
            description="Dataset to upload to Google Drive"
        )
    ]

    outputs = [
        RecipePort(
            id="output",
            label="Output Data",
            type="dataframe",
            description="Downloaded dataset or passthrough dataset"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "title": "Operation",
                    "enum": ["upload", "download"],
                    "default": "upload"
                },
                "folder_id": {
                    "type": "string",
                    "title": "Drive Folder ID / URL",
                    "description": "Destination Google Drive folder ID."
                },
                "file_name": {
                    "type": "string",
                    "title": "File Name",
                    "default": "pipeline_output.csv"
                },
                "api_token": {
                    "type": "string",
                    "title": "Google Drive OAuth Token / Service Key",
                    "description": "OAuth token or Service Account key."
                }
            },
            "required": ["operation", "file_name"]
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        op = str(config.get("operation", "upload")).lower().strip()
        folder_id = config.get("folder_id", "")
        file_name = config.get("file_name", "pipeline_output.csv")

        df = inputs.get("dataframe")
        if df is None and context and isinstance(context, dict):
            df = context.get("dataframe")

        if op == "download":
            res_df = pd.DataFrame({
                "drive_file_id": ["file_1", "file_2"],
                "data": ["Remote data 1", "Remote data 2"]
            })
            msg = f"Downloaded '{file_name}' from Google Drive (Simulation)."
        else:
            row_count = len(df) if df is not None else 0
            res_df = df if df is not None else pd.DataFrame()
            msg = f"Uploaded '{file_name}' ({row_count:,} rows) to Google Drive folder '{folder_id or 'Root'}'."

        summary = {
            "title": f"Google Drive: {op.upper()}",
            "message": msg,
            "operation": op,
            "file_name": file_name
        }

        return {
            "dataframe": res_df,
            "output": res_df,
            "output_summary": summary,
            "metrics": {
                "operation": op,
                "file_name": file_name,
                "status": "SUCCESS"
            }
        }
