import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.governance.audit_log_exporter import AuditLogExporterRecipe


def test_audit_log_exporter_registry_and_aliases():
    """Verify AuditLogExporterRecipe is registered with correct metadata and aliases."""
    assert recipe_registry.has("audit_log_exporter") is True
    meta = recipe_registry.get_metadata("audit_log_exporter")
    assert meta.enabled is True
    assert meta.group == "Evaluation & Governance"
    assert meta.subgroup == "Fairness & Compliance"

    # Aliases
    assert recipe_registry.get("audit_log").recipe_id == "audit_log_exporter"
    assert recipe_registry.get("compliance_audit").recipe_id == "audit_log_exporter"
    assert recipe_registry.get("compliance_trail").recipe_id == "audit_log_exporter"
    assert recipe_registry.get("audit_exporter").recipe_id == "audit_log_exporter"
    assert recipe_registry.get("lineage_exporter").recipe_id == "audit_log_exporter"


def test_audit_log_generation_with_dataset_fingerprint():
    """Verify cryptographic SHA-256 dataset fingerprint and audit table generation."""
    df = pd.DataFrame({
        "account_id": [101, 102, 103],
        "credit_score": [720, 680, 750],
        "approved": [1, 0, 1]
    })

    recipe = AuditLogExporterRecipe()
    result = recipe.execute(
        inputs={"input_data": df},
        config={
            "pipeline_name": "CreditScoringProductionPipeline",
            "compliance_framework": "EU_AI_ACT",
            "environment": "production",
            "operator_name": "RiskComplianceTeam",
            "approval_status": "APPROVED"
        }
    )

    audit_table = result["audit_table"]
    manifest = result["audit_record"]
    report = result["report"]

    assert not audit_table.empty
    assert len(audit_table) == 1
    assert "audit_trail_id" in audit_table.columns
    assert audit_table.iloc[0]["audit_trail_id"].startswith("AUD-")
    assert audit_table.iloc[0]["approval_status"] == "APPROVED"
    assert audit_table.iloc[0]["data_rows"] == 3

    # Cryptographic hash verification
    dataset_hash = manifest["cryptographic_fingerprints"]["dataset_sha256"]
    assert len(dataset_hash) == 64  # SHA-256 length
    assert manifest["data_lineage"]["dataset_shape"]["rows"] == 3
    assert manifest["data_lineage"]["dataset_shape"]["columns"] == 3

    # Report verification
    assert "# === Regulatory Compliance Audit Certificate & Lineage Trail ===" in report
    assert "EU Artificial Intelligence Act" in report
    assert "[APPROVED]" in report


def test_audit_log_with_model_and_metrics():
    """Verify model lineage extraction, parameters, and attached evaluation metrics."""
    # Train simple model
    X = [[1, 2], [3, 4], [5, 6], [7, 8]]
    y = [0, 0, 1, 1]
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    metrics_input = {
        "accuracy_pct": 98.5,
        "f1_score": 0.98,
        "four_fifths_rule_compliant": True,
        "disparate_impact_ratio": 0.92
    }

    recipe = AuditLogExporterRecipe()
    result = recipe.execute(
        inputs={"model": model, "metrics": metrics_input},
        config={
            "compliance_framework": "SOC2_TYPE_II",
            "approval_status": "APPROVED",
            "notes": "Model exceeds production accuracy threshold and passes fairness checks."
        }
    )

    manifest = result["audit_record"]
    model_lineage = manifest["model_lineage"]

    assert model_lineage["model_class"] == "RandomForestClassifier"
    assert "n_estimators" in model_lineage["parameters"]
    assert model_lineage["parameters"]["n_estimators"] == 10
    assert manifest["cryptographic_fingerprints"]["model_sha256"] != "N/A"
    assert manifest["attached_metrics"]["accuracy_pct"] == 98.5
    assert manifest["attached_metrics"]["four_fifths_rule_compliant"] is True

    # Report verification
    assert "SOC 2 Type II" in result["report"]
    assert "RandomForestClassifier" in result["report"]


def test_compliance_report_markdown_generation():
    """Verify executive compliance certificate report formatting across regulatory frameworks."""
    frameworks = ["HIPAA_AUDIT", "NIST_AI_RMF", "GENERAL_GOVERNANCE"]

    recipe = AuditLogExporterRecipe()
    for fw in frameworks:
        result = recipe.execute(
            inputs={},
            config={"compliance_framework": fw, "approval_status": "PROVISIONAL"}
        )
        report = result["report"]
        assert result["audit_record"]["compliance_framework"] == fw
        assert "[PROVISIONAL]" in report
        assert "Audit Reference ID" in report


def test_audit_log_exporter_edge_cases():
    """Verify guardrails when inputs are empty or None."""
    recipe = AuditLogExporterRecipe()

    # Empty inputs: should still produce a valid system audit record
    result = recipe.execute(inputs={}, config={})
    assert "audit_trail_id" in result["audit_record"]
    assert result["audit_record"]["data_lineage"]["status"] == "NO_DATASET_ATTACHED"
    assert result["audit_record"]["model_lineage"]["status"] == "NO_MODEL_PROVIDED"
    assert not result["audit_table"].empty

    # Empty DataFrame input
    df_empty = pd.DataFrame()
    result_empty_df = recipe.execute(inputs={"dataframe": df_empty}, config={})
    assert result_empty_df["audit_record"]["data_lineage"]["status"] == "NO_DATASET_ATTACHED"


if __name__ == "__main__":
    print("Running test_audit_log_exporter_registry_and_aliases...")
    test_audit_log_exporter_registry_and_aliases()
    print("PASS: test_audit_log_exporter_registry_and_aliases")

    print("Running test_audit_log_generation_with_dataset_fingerprint...")
    test_audit_log_generation_with_dataset_fingerprint()
    print("PASS: test_audit_log_generation_with_dataset_fingerprint")

    print("Running test_audit_log_with_model_and_metrics...")
    test_audit_log_with_model_and_metrics()
    print("PASS: test_audit_log_with_model_and_metrics")

    print("Running test_compliance_report_markdown_generation...")
    test_compliance_report_markdown_generation()
    print("PASS: test_compliance_report_markdown_generation")

    print("Running test_audit_log_exporter_edge_cases...")
    test_audit_log_exporter_edge_cases()
    print("PASS: test_audit_log_exporter_edge_cases")

    print("\nALL 5 AUDIT LOG EXPORTER TESTS PASSED SUCCESSFULLY!")
