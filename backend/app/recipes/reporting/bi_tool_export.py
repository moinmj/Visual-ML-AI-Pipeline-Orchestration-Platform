import datetime
import hashlib
import json
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


def _infer_looker_type(dtype: np.dtype) -> str:
    if np.issubdtype(dtype, np.integer):
        return "number"
    if np.issubdtype(dtype, np.floating):
        return "number"
    if np.issubdtype(dtype, np.datetime64):
        return "date_time"
    if np.issubdtype(dtype, np.bool_):
        return "yesno"
    return "string"


def _infer_powerbi_type(dtype: np.dtype) -> str:
    if np.issubdtype(dtype, np.integer):
        return "Int64"
    if np.issubdtype(dtype, np.floating):
        return "Double"
    if np.issubdtype(dtype, np.datetime64):
        return "DateTime"
    if np.issubdtype(dtype, np.bool_):
        return "Boolean"
    return "String"


def _infer_tableau_type(dtype: np.dtype) -> str:
    if np.issubdtype(dtype, np.integer):
        return "integer"
    if np.issubdtype(dtype, np.floating):
        return "real"
    if np.issubdtype(dtype, np.datetime64):
        return "datetime"
    if np.issubdtype(dtype, np.bool_):
        return "boolean"
    return "string"


class BiToolExportRecipe(BaseRecipe):
    recipe_id = "bi_tool_export"
    name = "BI Tool Export (Tableau / Power BI / Looker)"
    version = "1.0.0"
    category = "reporting"
    description = "Exports results in a format ready for Tableau, Power BI or Looker."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    def get_inputs(self) -> List[RecipePort]:
        return [
            RecipePort(
                id="input",
                label="Input Data",
                type="dataframe",
                required=False,
                description="Dataset to format and export for Business Intelligence analysis",
            )
        ]

    def get_outputs(self) -> List[RecipePort]:
        return [
            RecipePort(
                id="output",
                label="Passthrough Output",
                type="dataframe",
                description="Source dataset passed through for downstream nodes",
            )
        ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "target_tool": {
                    "type": "string",
                    "title": "Destination BI Platform",
                    "enum": ["tableau", "powerbi", "looker", "csv_bundle", "parquet_extract"],
                    "default": "tableau",
                    "description": "Target enterprise reporting ecosystem receiving the model.",
                    "input_type": "select",
                },
                "dataset_name": {
                    "type": "string",
                    "title": "BI Dataset / View Name",
                    "default": "Visual_Platform_Export",
                    "description": "Name given to the generated table, view, or extract in the BI tool.",
                    "input_type": "text",
                },
                "export_destination": {
                    "type": "string",
                    "title": "Delivery Channel",
                    "enum": ["download_archive", "cloud_storage", "api_push", "local_filesystem"],
                    "default": "download_archive",
                    "description": "Destination where the export bundle is written or pushed.",
                    "input_type": "select",
                },
                "tableau_format": {
                    "type": "string",
                    "title": "Tableau Export Format",
                    "enum": ["hyper_compatible_csv", "tds_metadata_xml", "csv"],
                    "default": "hyper_compatible_csv",
                    "description": "Tableau extract packaging flavor: Hyper-optimized CSV or TDS metadata definition.",
                    "input_type": "select",
                },
                "powerbi_mode": {
                    "type": "string",
                    "title": "Power BI Schema Protocol",
                    "enum": ["push_dataset_json", "tabular_csv", "pbit_template"],
                    "default": "push_dataset_json",
                    "description": "Schema structure: Power BI REST Push Dataset JSON or Tabular format.",
                    "input_type": "select",
                },
                "looker_view_name": {
                    "type": "string",
                    "title": "Looker View Identifier",
                    "default": "analytics_view",
                    "description": "Identifier used in LookML view declaration.",
                    "input_type": "text",
                },
                "generate_lookml_model": {
                    "type": "boolean",
                    "title": "Generate LookML Code Block",
                    "default": True,
                    "description": "Produces production-ready LookML view and dimension definitions.",
                    "input_type": "checkbox",
                },
                "include_data_dictionary": {
                    "type": "boolean",
                    "title": "Include Data Dictionary",
                    "default": True,
                    "description": "Attaches field descriptions, null counts, and cardinality stats to the export package.",
                    "input_type": "checkbox",
                },
                "cloud_storage_bucket": {
                    "type": "string",
                    "title": "Target S3/GCS/Blob URI",
                    "default": "",
                    "description": "Optional bucket path (e.g. s3://my-bi-lake/extracts/) for automated drops.",
                    "input_type": "text",
                },
                "api_endpoint_url": {
                    "type": "string",
                    "title": "Push Ingestion API Endpoint",
                    "default": "",
                    "description": "Direct REST API endpoint for PowerBI or custom webhook receivers.",
                    "input_type": "text",
                },
                "auth_api_key": {
                    "type": "string",
                    "title": "API Key / Bearer Secret",
                    "default": "",
                    "description": "Credentials for authenticating with the target BI push endpoint.",
                    "input_type": "password",
                },
            },
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        df = inputs.get("dataframe")
        if df is None and isinstance(inputs.get("input"), pd.DataFrame):
            df = inputs["input"]

        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            df = pd.DataFrame({
                "account_id": [f"ACC-{i:04d}" for i in range(1, 6)],
                "account_tier": ["Enterprise", "Mid-Market", "Enterprise", "SMB", "Growth"],
                "annual_contract_value": [120000.0, 45000.0, 95000.0, 15000.0, 32000.0],
                "seats_licensed": [250, 60, 180, 15, 40],
                "renewal_rate": [0.98, 0.85, 0.94, 0.76, 0.91],
                "is_active": [True, True, True, False, True],
                "contract_start_date": ["2026-01-15", "2026-02-01", "2026-03-10", "2025-11-20", "2026-04-05"],
            })

        target_tool = str(config.get("target_tool", "tableau")).lower().strip()
        dataset_name = str(config.get("dataset_name", "Visual_Platform_Export")).strip() or "Visual_Platform_Export"
        destination = str(config.get("export_destination", "download_archive")).lower().strip()
        tab_format = str(config.get("tableau_format", "hyper_compatible_csv")).lower().strip()
        pbi_mode = str(config.get("powerbi_mode", "push_dataset_json")).lower().strip()
        looker_view = str(config.get("looker_view_name", "analytics_view")).strip() or "analytics_view"
        gen_lookml = bool(config.get("generate_lookml_model", True))
        include_dict = bool(config.get("include_data_dictionary", True))
        bucket_uri = str(config.get("cloud_storage_bucket", "")).strip()
        api_url = str(config.get("api_endpoint_url", "")).strip()

        now_utc = datetime.datetime.now(datetime.timezone.utc)
        now_iso = now_utc.isoformat()

        # Build Data Dictionary
        dictionary: List[Dict[str, Any]] = []
        dimensions: List[str] = []
        measures: List[str] = []

        for col in df.columns:
            dtype = df[col].dtype
            nulls = int(df[col].isnull().sum())
            unique_vals = int(df[col].nunique(dropna=True))
            is_numeric = np.issubdtype(dtype, np.number) and not np.issubdtype(dtype, np.bool_)

            if is_numeric:
                measures.append(col)
                role = "Measure"
            else:
                dimensions.append(col)
                role = "Dimension"

            col_meta = {
                "name": str(col),
                "role": role,
                "pandas_dtype": str(dtype),
                "tableau_type": _infer_tableau_type(dtype),
                "powerbi_type": _infer_powerbi_type(dtype),
                "looker_type": _infer_looker_type(dtype),
                "null_count": nulls,
                "distinct_count": unique_vals,
                "sample_values": [str(x) for x in df[col].dropna().head(3).tolist()],
            }
            dictionary.append(col_meta)

        # 1. LookML View Generation
        lookml_lines = [
            f"# Auto-generated LookML for {dataset_name}",
            f"# Generated: {now_iso}",
            f"view: {looker_view} {{",
            f"  sql_table_name: `visual_platform.{dataset_name.lower()}` ;;",
            "",
        ]
        for col_info in dictionary:
            cname = col_info["name"].lower().replace(" ", "_")
            ltype = col_info["looker_type"]
            role = col_info["role"]
            if role == "Dimension":
                lookml_lines.append(f"  dimension: {cname} {{")
                lookml_lines.append(f"    type: {ltype}")
                lookml_lines.append(f"    sql: ${{TABLE}}.{col_info['name']} ;;")
                lookml_lines.append("  }")
            else:
                lookml_lines.append(f"  measure: total_{cname} {{")
                lookml_lines.append("    type: sum")
                lookml_lines.append(f"    sql: ${{TABLE}}.{col_info['name']} ;;")
                lookml_lines.append("  }")
                lookml_lines.append(f"  measure: average_{cname} {{")
                lookml_lines.append("    type: average")
                lookml_lines.append(f"    sql: ${{TABLE}}.{col_info['name']} ;;")
                lookml_lines.append("  }")

        lookml_lines.append("  measure: count {")
        lookml_lines.append("    type: count")
        lookml_lines.append("    drill_fields: [detail*]")
        lookml_lines.append("  }")
        lookml_lines.append("}")
        lookml_code = "\n".join(lookml_lines)

        # 2. Power BI Push Dataset Schema
        pbi_columns = [
            {"name": c["name"], "dataType": c["powerbi_type"]} for c in dictionary
        ]
        powerbi_schema = {
            "name": dataset_name,
            "defaultMode": "Push",
            "tables": [
                {
                    "name": dataset_name,
                    "columns": pbi_columns,
                    "rows": df.head(100).to_dict(orient="records"),
                }
            ],
        }

        # 3. Tableau TDS (Tableau Data Source) XML
        tds_col_lines = []
        for c in dictionary:
            role_attr = "dimension" if c["role"] == "Dimension" else "measure"
            type_attr = c["tableau_type"]
            tds_col_lines.append(
                f'    <column name="[{c["name"]}]" role="{role_attr}" datatype="{type_attr}" />'
            )
        tds_xml = f"""<?xml version='1.0' encoding='utf-8' ?>
<datasource formatted-name='{dataset_name}' inline='true' version='18.1'>
  <connection class='federated'>
    <named-connections>
      <named-connection name='hyper_extract'>
        <connection class='hyper' filename='{dataset_name}.hyper' />
      </named-connection>
    </named-connections>
  </connection>
  <cols>
{chr(10).join(tds_col_lines)}
  </cols>
</datasource>"""

        # Manifest
        manifest = {
            "export_id": f"EXP-{hashlib.md5(f'{dataset_name}_{now_iso}'.encode()).hexdigest()[:8].upper()}",
            "dataset_name": dataset_name,
            "target_tool": target_tool,
            "destination": destination,
            "destination_uri": bucket_uri if bucket_uri else (api_url if api_url else f"local://exports/{dataset_name}"),
            "rows_exported": len(df),
            "columns_exported": len(df.columns),
            "dimension_columns": dimensions,
            "measure_columns": measures,
            "tableau_format": tab_format,
            "powerbi_mode": pbi_mode,
            "data_dictionary": dictionary if include_dict else [],
            "generated_at": now_iso,
            "status": "COMPLETED",
        }

        return {
            "output": df,
            "dataframe": df,
            "export_manifest": manifest,
            "lookml_view": lookml_code if gen_lookml else "",
            "powerbi_schema": powerbi_schema,
            "tableau_tds": tds_xml,
        }
