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
from sklearn.linear_model import LogisticRegression

from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.evaluation.bias_fairness_audit import BiasFairnessAuditRecipe


def test_bias_fairness_audit_registry_and_aliases():
    """Verify BiasFairnessAuditRecipe is registered with correct metadata and aliases."""
    assert recipe_registry.has("bias_fairness_audit") is True
    meta = recipe_registry.get_metadata("bias_fairness_audit")
    assert meta.enabled is True
    assert meta.group == "Evaluation & Governance"
    assert meta.subgroup == "Fairness & Compliance"

    # Aliases
    assert recipe_registry.get("fairness_audit").recipe_id == "bias_fairness_audit"
    assert recipe_registry.get("bias_audit").recipe_id == "bias_fairness_audit"
    assert recipe_registry.get("disparate_impact").recipe_id == "bias_fairness_audit"
    assert recipe_registry.get("demographic_parity").recipe_id == "bias_fairness_audit"
    assert recipe_registry.get("equal_opportunity").recipe_id == "bias_fairness_audit"


def test_four_fifths_rule_violation_detection():
    """Verify Four-Fifths Rule violation detection when disparate impact < 0.80."""
    # Demographic data:
    # Group A (Privileged): 100 people, 80 approved (80% selection rate)
    # Group B (Unprivileged): 100 people, 40 approved (40% selection rate)
    # Disparate Impact = 40% / 80% = 0.50 (< 0.80 -> Adverse Impact)
    df = pd.DataFrame({
        "gender": ["Group_A"] * 100 + ["Group_B"] * 100,
        "prediction": [1] * 80 + [0] * 20 + [1] * 40 + [0] * 60
    })

    recipe = BiasFairnessAuditRecipe()
    result = recipe.execute(
        inputs={"dataframe": df},
        config={
            "protected_attribute": "gender",
            "prediction_col": "prediction",
            "privileged_group": "Group_A",
            "disparate_impact_threshold": 0.80
        }
    )

    metrics = result["metrics"]
    report = result["report"]
    audit_table = result["audit_table"]

    assert metrics["overall_fairness_status"] == "DISPARATE_IMPACT_DETECTED"
    assert metrics["four_fifths_rule_compliant"] is False
    assert metrics["minimum_disparate_impact_ratio"] == 0.50
    assert len(metrics["warnings"]) > 0
    assert any("FOUR-FIFTHS RULE VIOLATION" in w for w in metrics["warnings"])

    # Report verification
    assert "# === Algorithmic Bias & Fairness Compliance Audit ===" in report
    assert "FAIL: ADVERSE DISPARATE IMPACT DETECTED" in report
    assert not audit_table.empty


def test_fair_model_compliance():
    """Verify compliant model status when all groups meet Four-Fifths and Demographic Parity."""
    # Both groups have 60% selection rate -> Disparate impact = 1.0
    df = pd.DataFrame({
        "ethnicity": ["Majority"] * 200 + ["Minority"] * 200,
        "hired": [1] * 120 + [0] * 80 + [1] * 115 + [0] * 85
    })

    recipe = BiasFairnessAuditRecipe()
    result = recipe.execute(
        inputs={"dataframe": df},
        config={
            "protected_attribute": "ethnicity",
            "prediction_col": "hired",
            "privileged_group": "Majority"
        }
    )

    metrics = result["metrics"]
    report = result["report"]

    assert metrics["overall_fairness_status"] == "COMPLIANT"
    assert metrics["four_fifths_rule_compliant"] is True
    assert 0.90 <= metrics["minimum_disparate_impact_ratio"] <= 1.05
    assert len(metrics["warnings"]) == 0
    assert "PASSED: COMPLIANT WITH 4/5THS RULE" in report


def test_equal_opportunity_and_average_odds_with_ground_truth():
    """Verify Equal Opportunity (TPR) and False Positive Rate differences when actual target is provided."""
    # Synthetic dataset with ground truth target and model predictions
    # Group 1: 50 positive actuals, 45 detected (TPR = 90%), 50 negative actuals, 5 false alarms (FPR = 10%)
    # Group 2: 50 positive actuals, 30 detected (TPR = 60%), 50 negative actuals, 15 false alarms (FPR = 30%)
    df = pd.DataFrame({
        "age_band": ["Young"] * 100 + ["Senior"] * 100,
        "target": [1] * 50 + [0] * 50 + [1] * 50 + [0] * 50,
        "prediction": [1] * 45 + [0] * 5 + [1] * 5 + [0] * 45 + [1] * 30 + [0] * 20 + [1] * 15 + [0] * 35
    })

    recipe = BiasFairnessAuditRecipe()
    result = recipe.execute(
        inputs={"dataframe": df},
        config={
            "protected_attribute": "age_band",
            "target_col": "target",
            "prediction_col": "prediction",
            "privileged_group": "Young"
        }
    )

    metrics = result["metrics"]
    eval_senior = metrics["pairwise_evaluations"][0]

    assert eval_senior["unprivileged_group"] == "Senior"
    # Young TPR: 45/50 = 0.90, Senior TPR: 30/50 = 0.60 -> Difference = -0.30
    assert np.isclose(eval_senior["equal_opportunity_difference"], -0.30, atol=0.01)
    # Young FPR: 5/50 = 0.10, Senior FPR: 15/50 = 0.30 -> Difference = +0.20
    assert np.isclose(eval_senior["false_positive_rate_difference"], 0.20, atol=0.01)
    # Average Odds Diff = 0.5 * (0.30 + 0.20) = 0.25
    assert np.isclose(eval_senior["average_odds_difference"], 0.25, atol=0.01)

    # Verification of report table with TPR/FPR columns
    assert "True Positive Rate (TPR)" in result["report"]
    assert "Equal Opportunity Diff" in result["report"]


def test_trained_model_prediction_input():
    """Verify BiasFairnessAudit evaluates fairness using a trained model object directly."""
    np.random.seed(42)
    # Fit simple logistic regression
    X = np.random.randn(200, 3)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = LogisticRegression()
    model.fit(X, y)

    # Create evaluation dataset without prediction column, but with feature columns and protected attribute
    df_eval = pd.DataFrame(X, columns=["feat_1", "feat_2", "feat_3"])
    df_eval["target"] = y
    df_eval["protected_group"] = ["Group_A"] * 100 + ["Group_B"] * 100

    recipe = BiasFairnessAuditRecipe()
    result = recipe.execute(
        inputs={"test_data": df_eval, "model": model},
        config={
            "protected_attribute": "protected_group",
            "target_col": "target"
        }
    )

    assert result["metrics"]["total_groups_audited"] == 2
    assert "minimum_disparate_impact_ratio" in result["metrics"]
    assert not result["audit_table"].empty


def test_bias_fairness_audit_edge_cases():
    """Verify guardrails: single group, missing protected column, string boolean conversions, and zero selection rate."""
    recipe = BiasFairnessAuditRecipe()

    # Guard 1: Missing protected attribute column
    df_missing_prot = pd.DataFrame({"prediction": [1, 0, 1], "target": [1, 0, 1]})
    with pytest.raises(ValueError, match="could not locate protected attribute column"):
        recipe.execute(inputs={"dataframe": df_missing_prot}, config={"protected_attribute": "missing_gender"})

    # Guard 2: Single demographic group rejected
    df_single_group = pd.DataFrame({
        "gender": ["Female", "Female", "Female"],
        "prediction": [1, 0, 1]
    })
    with pytest.raises(ValueError, match="requires at least 2 distinct groups"):
        recipe.execute(inputs={"dataframe": df_single_group}, config={})

    # Guard 3: Zero selection rate in unprivileged group (handles division by zero cleanly)
    df_zero_rate = pd.DataFrame({
        "group": ["Priv", "Priv", "Unpriv", "Unpriv"],
        "prediction": [1, 1, 0, 0]
    })
    res_zero = recipe.execute(inputs={"dataframe": df_zero_rate}, config={"protected_attribute": "group", "privileged_group": "Priv"})
    assert res_zero["metrics"]["minimum_disparate_impact_ratio"] == 0.0
    assert res_zero["metrics"]["four_fifths_rule_compliant"] is False

    # Guard 4: String outcomes (approved / declined) handled cleanly
    df_strings = pd.DataFrame({
        "group": ["A", "A", "B", "B"],
        "decision": ["approved", "declined", "approved", "approved"]
    })
    res_str = recipe.execute(
        inputs={"dataframe": df_strings},
        config={"protected_attribute": "group", "prediction_col": "decision", "favorable_label": "approved"}
    )
    assert res_str["metrics"]["total_groups_audited"] == 2


if __name__ == "__main__":
    print("Running test_bias_fairness_audit_registry_and_aliases...")
    test_bias_fairness_audit_registry_and_aliases()
    print("PASS: test_bias_fairness_audit_registry_and_aliases")

    print("Running test_four_fifths_rule_violation_detection...")
    test_four_fifths_rule_violation_detection()
    print("PASS: test_four_fifths_rule_violation_detection")

    print("Running test_fair_model_compliance...")
    test_fair_model_compliance()
    print("PASS: test_fair_model_compliance")

    print("Running test_equal_opportunity_and_average_odds_with_ground_truth...")
    test_equal_opportunity_and_average_odds_with_ground_truth()
    print("PASS: test_equal_opportunity_and_average_odds_with_ground_truth")

    print("Running test_trained_model_prediction_input...")
    test_trained_model_prediction_input()
    print("PASS: test_trained_model_prediction_input")

    print("Running test_bias_fairness_audit_edge_cases...")
    test_bias_fairness_audit_edge_cases()
    print("PASS: test_bias_fairness_audit_edge_cases")

    print("\nALL 6 BIAS & FAIRNESS AUDIT TESTS PASSED SUCCESSFULLY!")
