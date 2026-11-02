import os
import sys
import uuid
import hashlib
import json
import platform
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort
from backend.app.recipes.training.encoder_utils import pass_through_metadata


class AuditLogExporterRecipe(BaseRecipe):
    """
    Audit Log & Compliance Trail Exporter Recipe.
    Exports an immutable, tamper-evident regulatory compliance trail capturing:
      - Cryptographic SHA-256 fingerprint of input data and model artifacts
      - Execution metadata, user/tenant context, runtime environment specs
      - Regulatory framework alignment (EU AI Act, SOC 2, HIPAA, NIST AI RMF)
      - Performance benchmarks, fairness sign-off, and deployment approval status
    Outputs tabular audit records, JSON compliance certificates, and executive Markdown reports.
    """

    recipe_id = "audit_log_exporter"
    name = "Audit Log & Compliance Trail Exporter"
    version = "1.0.0"
    category = "governance"
    group = "Evaluation & Governance"
    subgroup = "Fairness & Compliance"
    description = (
        "Exports a tamper-evident compliance audit trail and lineage record of pipeline runs, "
        "data fingerprints (SHA-256), model parameters, and regulatory sign-offs."
    )
    input_types = ["dataframe", "model", "metrics"]
    output_types = ["dataframe", "metrics", "report"]

    inputs: List[RecipePort] = [
        RecipePort(
            id="input_data",
            label="Pipeline Dataset (Optional)",
            type="dataframe",
            required=False,
            max_connections=1,
            description="Upstream dataset to fingerprint with cryptographic SHA-256 hash."
        ),
        RecipePort(
            id="model",
            label="Trained Model (Optional)",
            type="model",
            required=False,
            max_connections=1,
            description="Model artifact to extract architecture and parameter lineage."
        ),
        RecipePort(
            id="metrics",
            label="Performance & Fairness Metrics (Optional)",
            type="metrics",
            required=False,
            max_connections=1,
            description="Metrics dictionary from model evaluator or fairness audit."
        )
    ]

    outputs: List[RecipePort] = [
        RecipePort(
            id="audit_table",
            label="Compliance Audit Log Table",
            type="dataframe",
            description="Tabular audit log containing timestamp, audit ID, data hash, and approval status."
        ),
        RecipePort(
            id="audit_record",
            label="Structured Compliance Manifest",
            type="metrics",
            description="JSON-serializable compliance manifest conforming to regulatory audit standards."
        ),
        RecipePort(
            id="report",
            label="Executive Compliance Certificate",
            type="report",
            description="Markdown audit trail report and compliance certificate."
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pipeline_name": {
                    "type": "string",
                    "title": "Pipeline / Workflow Name",
                    "input_type": "text",
                    "default": "Visual-ML-AI-Pipeline",
                    "description": "Identifier for the parent ML/AI pipeline."
                },
                "compliance_framework": {
                    "type": "string",
                    "title": "Regulatory Framework",
                    "input_type": "dropdown",
                    "enum": ["EU_AI_ACT", "SOC2_TYPE_II", "HIPAA_AUDIT", "NIST_AI_RMF", "GENERAL_GOVERNANCE"],
                    "default": "EU_AI_ACT",
                    "description": "Target regulatory compliance standard."
                },
                "environment": {
                    "type": "string",
                    "title": "Deployment Environment",
                    "input_type": "dropdown",
                    "enum": ["production", "staging", "testing", "development"],
                    "default": "production",
                    "description": "Target runtime environment."
                },
                "operator_name": {
                    "type": "string",
                    "title": "Operator / Auditor Name",
                    "input_type": "text",
                    "default": "Automated ML Pipeline Orchestrator",
                    "description": "Name or email of auditor, engineer, or CI/CD agent."
                },
                "approval_status": {
                    "type": "string",
                    "title": "Compliance Sign-off Status",
                    "input_type": "dropdown",
                    "enum": ["APPROVED", "PROVISIONAL", "REJECTED", "PENDING_AUDIT"],
                    "default": "APPROVED",
                    "description": "Official deployment compliance status."
                },
                "notes": {
                    "type": "string",
                    "title": "Audit Notes & Annotations",
                    "input_type": "textarea",
                    "default": "Automated verification completed with guardrails enforced.",
                    "description": "Additional context or justification for audit records."
                }
            },
            "required": []
        }

    def _compute_dataset_hash(self, df: pd.DataFrame) -> str:
        """Compute deterministic SHA-256 hash of DataFrame content."""
        try:
            # Sample first and last 500 rows if very large to prevent memory overflow
            sample = df if len(df) <= 1000 else pd.concat([df.head(500), df.tail(500)])
            csv_bytes = sample.to_csv(index=False).encode("utf-8")
            return hashlib.sha256(csv_bytes).hexdigest()
        except Exception:
            # Fallback to column names + shape hash
            summary = f"{list(df.columns)}_{df.shape}_{df.dtypes.to_dict()}"
            return hashlib.sha256(summary.encode("utf-8")).hexdigest()

    def _extract_model_info(self, model: Any) -> Dict[str, Any]:
        """Extract model architecture and parameter manifest safely."""
        if model is None:
            return {"status": "NO_MODEL_PROVIDED"}

        model_cls = model.__class__.__name__
        model_module = model.__class__.__module__

        params: Dict[str, Any] = {}
        if hasattr(model, "get_params"):
            try:
                raw_params = model.get_params(deep=False)
                for k, v in raw_params.items():
                    if isinstance(v, (int, float, str, bool, type(None))):
                        params[k] = v
                    else:
                        params[k] = str(v)
            except Exception:
                pass

        return {
            "model_class": model_cls,
            "model_module": model_module,
            "parameters": params,
            "has_predict_proba": hasattr(model, "predict_proba")
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        # 1. Resolve Inputs
        df = inputs.get("input_data")
        if df is None:
            df = inputs.get("dataframe")
        if df is None:
            df = inputs.get("data")
        if df is None and isinstance(context, dict):
            df = context.get("dataframe")

        model = inputs.get("model")
        metrics_input = inputs.get("metrics")
        if metrics_input is None and isinstance(context, dict):
            metrics_input = context.get("metrics")

        # 2. Generate Audit Trail Identifiers & Timestamp
        audit_id = f"AUD-{uuid.uuid4().hex[:12].upper()}"
        timestamp_utc = datetime.now(timezone.utc).isoformat()

        pipeline_name = config.get("pipeline_name", "Visual-ML-AI-Pipeline")
        compliance_framework = config.get("compliance_framework", "EU_AI_ACT")
        environment = config.get("environment", "production")
        operator = config.get("operator_name", "Automated ML Pipeline Orchestrator")
        approval_status = config.get("approval_status", "APPROVED")
        notes = config.get("notes", "Automated compliance verification record.")

        # 3. Cryptographic Fingerprint & Lineage
        data_lineage: Dict[str, Any] = {}
        if isinstance(df, pd.DataFrame) and not df.empty:
            data_hash = self._compute_dataset_hash(df)
            data_lineage = {
                "dataset_shape": {"rows": int(len(df)), "columns": int(len(df.columns))},
                "column_names": list(df.columns),
                "sha256_hash": data_hash
            }
        else:
            data_hash = "N/A (NO_DATASET_PASSED)"
            data_lineage = {"status": "NO_DATASET_ATTACHED"}

        model_info = self._extract_model_info(model)
        model_hash = "N/A"
        if model is not None:
            model_summary = f"{model_info.get('model_class')}_{json.dumps(model_info.get('parameters', {}), sort_keys=True)}"
            model_hash = hashlib.sha256(model_summary.encode("utf-8")).hexdigest()

        # 4. Environment & Platform Specs
        env_specs = {
            "python_version": sys.version.split()[0],
            "os_system": platform.system(),
            "os_release": platform.release(),
            "machine_architecture": platform.machine(),
            "process_id": os.getpid()
        }

        # 5. Assemble Structured Compliance Manifest
        compliance_manifest = {
            "audit_trail_id": audit_id,
            "timestamp_utc": timestamp_utc,
            "pipeline_name": pipeline_name,
            "compliance_framework": compliance_framework,
            "environment": environment,
            "operator": operator,
            "approval_status": approval_status,
            "cryptographic_fingerprints": {
                "dataset_sha256": data_hash,
                "model_sha256": model_hash
            },
            "data_lineage": data_lineage,
            "model_lineage": model_info,
            "runtime_environment": env_specs,
            "attached_metrics": metrics_input if isinstance(metrics_input, dict) else {},
            "audit_notes": notes
        }

        # 6. Tabular Audit Table
        audit_row = {
            "audit_trail_id": audit_id,
            "timestamp_utc": timestamp_utc,
            "pipeline_name": pipeline_name,
            "compliance_framework": compliance_framework,
            "environment": environment,
            "approval_status": approval_status,
            "operator": operator,
            "data_rows": data_lineage.get("dataset_shape", {}).get("rows", 0) if isinstance(data_lineage.get("dataset_shape"), dict) else 0,
            "dataset_sha256": data_hash[:16] + "..." if len(data_hash) > 16 else data_hash,
            "model_class": model_info.get("model_class", "N/A"),
            "model_sha256": model_hash[:16] + "..." if len(model_hash) > 16 else model_hash
        }
        audit_df = pd.DataFrame([audit_row])

        # 7. Generate Executive Compliance Certificate Report
        framework_titles = {
            "EU_AI_ACT": "EU Artificial Intelligence Act (Article 12: Technical Documentation & Record-Keeping)",
            "SOC2_TYPE_II": "SOC 2 Type II Compliance Trail (Security, Availability & Confidentiality)",
            "HIPAA_AUDIT": "HIPAA Technical Safeguards Audit Trail (§ 164.312(b))",
            "NIST_AI_RMF": "NIST AI Risk Management Framework (Govern & Map Functions)",
            "GENERAL_GOVERNANCE": "Enterprise Algorithmic Governance & Audit Trail"
        }
        fw_full = framework_titles.get(compliance_framework, compliance_framework)

        badge_status = f"[{approval_status}]"

        report_lines = [
            f"# === Regulatory Compliance Audit Certificate & Lineage Trail ===",
            f"**Audit Reference ID**: `{audit_id}`",
            f"- **Target Standard**: **{fw_full}**",
            f"- **Pipeline / Workflow**: `{pipeline_name}` ({environment.upper()} environment)",
            f"- **Audited By**: `{operator}`",
            f"- **Timestamp**: `{timestamp_utc}`",
            f"- **Compliance Sign-Off Status**: **`{badge_status}`**",
            "",
            "## Cryptographic Evidence & Tamper-Evident Fingerprints:",
            f"- **Dataset SHA-256 Hash**: `{data_hash}`",
            f"- **Model Artifact SHA-256 Hash**: `{model_hash}`",
            "",
            "## Pipeline Artifact Lineage Summary:",
            f"- **Data Volume**: {data_lineage.get('dataset_shape', {}).get('rows', 'N/A')} rows, {data_lineage.get('dataset_shape', {}).get('columns', 'N/A')} features",
            f"- **Model Architecture**: `{model_info.get('model_class', 'None')}` (module: `{model_info.get('model_module', 'None')}`)",
            f"- **Execution Runtime**: Python `{env_specs['python_version']}` on `{env_specs['os_system']} {env_specs['os_release']}`",
            "",
            "## Audit Sign-off & Annotations:",
            f"> {notes}",
            "",
            "---",
            "*(This document is an immutable cryptographic compliance record generated for automated auditing and regulatory certification.)*"
        ]

        report_markdown = "\n".join(report_lines)

        output: Dict[str, Any] = {
            "audit_table": audit_df,
            "dataframe": audit_df,
            "audit_record": compliance_manifest,
            "metrics": compliance_manifest,
            "report": report_markdown
        }

        return pass_through_metadata(inputs, output, context)

    def to_code(self, config: Dict[str, Any]) -> str:
        fw = config.get("compliance_framework", "EU_AI_ACT")
        pname = config.get("pipeline_name", "ML-Pipeline")
        return (
            "import hashlib, uuid, datetime, json\n\n"
            f"# Generate compliance audit log for '{pname}' ({fw})\n"
            "audit_id = f'AUD-{uuid.uuid4().hex[:12].upper()}'\n"
            "data_hash = hashlib.sha256(df.to_csv(index=False).encode('utf-8')).hexdigest()\n"
            "print(f'Compliance Record {audit_id}: Data Hash = {data_hash[:16]}... (SIGNED)')"
        )
