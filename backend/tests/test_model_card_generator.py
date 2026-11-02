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
from sklearn.ensemble import GradientBoostingClassifier

from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.governance.model_card_generator import ModelCardGeneratorRecipe


def test_model_card_generator_registry_and_aliases():
    """Verify ModelCardGeneratorRecipe is registered with correct metadata and aliases."""
    assert recipe_registry.has("model_card_generator") is True
    meta = recipe_registry.get_metadata("model_card_generator")
    assert meta.enabled is True
    assert meta.group == "Evaluation & Governance"
    assert meta.subgroup == "Fairness & Compliance"

    # Aliases
    assert recipe_registry.get("model_card").recipe_id == "model_card_generator"
    assert recipe_registry.get("model_cards").recipe_id == "model_card_generator"
    assert recipe_registry.get("model_documentation").recipe_id == "model_card_generator"
    assert recipe_registry.get("model_card_report").recipe_id == "model_card_generator"
    assert recipe_registry.get("mitchell_model_card").recipe_id == "model_card_generator"


def test_model_card_with_full_context():
    """Verify Model Card generation with trained model, dataset, and metrics."""
    # 1. Train model
    X = np.random.randn(150, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = GradientBoostingClassifier(n_estimators=25, random_state=42)
    model.fit(X, y)

    # 2. DataFrame
    df = pd.DataFrame(X, columns=["feature_age", "feature_income", "feature_score", "feature_ratio"])
    df["target"] = y

    # 3. Metrics
    metrics_input = {
        "accuracy": 0.945,
        "f1_score": 0.941,
        "roc_auc": 0.982,
        "disparate_impact_ratio": 0.88,
        "four_fifths_rule_passed": True
    }

    recipe = ModelCardGeneratorRecipe()
    result = recipe.execute(
        inputs={"model": model, "data": df, "metrics": metrics_input},
        config={
            "model_name": "CreditRiskScorer",
            "model_version": "2.1.0",
            "intended_use": "Automated applicant scoring for retail credit underwriting.",
            "target_users": "Credit Risk Committee and Automated Decision Engines"
        }
    )

    report = result["report"]
    card_json = result["model_card_json"]
    summary_df = result["summary_table"]

    # Verify JSON structure
    assert card_json["model_details"]["name"] == "CreditRiskScorer"
    assert card_json["model_details"]["version"] == "2.1.0"
    assert "GradientBoostingClassifier" in card_json["model_details"]["architecture"]
    assert card_json["model_details"]["parameters"]["n_estimators"] == 25
    assert card_json["data_factors"]["total_samples"] == 150
    assert card_json["data_factors"]["total_features"] == 5
    assert card_json["quantitative_analyses"]["performance_benchmarks"]["accuracy"] == 0.945

    # Verify Markdown report
    assert "# === Model Card: CreditRiskScorer (v2.1.0) ===" in report
    assert "## 1. Model Details" in report
    assert "## 2. Intended Use & Domain" in report
    assert "## 3. Data Factors & Distribution" in report
    assert "## 4. Quantitative Evaluation Benchmarks" in report
    assert "## 5. Ethical Considerations & Algorithmic Fairness" in report
    assert "## 6. Limitations & Caveats" in report
    assert "## 7. Environmental Impact & Sustainability" in report
    assert "Mitchell, M., et al." in report

    # Verify summary table
    assert not summary_df.empty
    assert len(summary_df) == 8


def test_model_card_json_schema_compliance():
    """Verify compliance with Mitchell et al. / HuggingFace Model Card schema sections."""
    recipe = ModelCardGeneratorRecipe()
    result = recipe.execute(inputs={}, config={"model_name": "FraudDetectionModel"})

    card = result["model_card_json"]
    required_sections = [
        "schema_version",
        "model_details",
        "intended_use",
        "data_factors",
        "quantitative_analyses",
        "ethical_considerations",
        "limitations",
        "environmental_impact"
    ]
    for section in required_sections:
        assert section in card, f"Missing required Model Card section '{section}'"


def test_model_card_markdown_report_formatting():
    """Verify markdown output handles nested metrics cleanly and includes note alerts."""
    metrics_nested = {
        "overall_accuracy": 0.92,
        "slices": {"group_A": 0.95, "group_B": 0.89}
    }

    recipe = ModelCardGeneratorRecipe()
    result = recipe.execute(inputs={"metrics": metrics_nested}, config={})
    report = result["report"]

    assert "| `overall_accuracy` | **`0.9200`** |" in report
    assert "| `slices_group_A` | **`0.9500`** |" in report
    assert "> [!NOTE]" in report


def test_model_card_edge_cases():
    """Verify guardrails for empty inputs, empty dataframe, and custom black-box estimator."""
    recipe = ModelCardGeneratorRecipe()

    # Edge 1: Completely empty inputs
    res_empty = recipe.execute(inputs={}, config={})
    assert res_empty["model_card_json"]["model_details"]["name"] == "Enterprise Predictive Model"
    assert res_empty["model_card_json"]["data_factors"]["status"] == "No dataset attached"

    # Edge 2: Empty dataframe
    res_empty_df = recipe.execute(inputs={"data": pd.DataFrame()}, config={})
    assert res_empty_df["model_card_json"]["data_factors"]["status"] == "No dataset attached"

    # Edge 3: Custom object without get_params
    class CustomPredictor:
        def predict(self, X):
            return np.ones(len(X))

    res_custom = recipe.execute(inputs={"model": CustomPredictor()}, config={})
    assert "CustomPredictor" in res_custom["model_card_json"]["model_details"]["architecture"]


if __name__ == "__main__":
    print("Running test_model_card_generator_registry_and_aliases...")
    test_model_card_generator_registry_and_aliases()
    print("PASS: test_model_card_generator_registry_and_aliases")

    print("Running test_model_card_with_full_context...")
    test_model_card_with_full_context()
    print("PASS: test_model_card_with_full_context")

    print("Running test_model_card_json_schema_compliance...")
    test_model_card_json_schema_compliance()
    print("PASS: test_model_card_json_schema_compliance")

    print("Running test_model_card_markdown_report_formatting...")
    test_model_card_markdown_report_formatting()
    print("PASS: test_model_card_markdown_report_formatting")

    print("Running test_model_card_edge_cases...")
    test_model_card_edge_cases()
    print("PASS: test_model_card_edge_cases")

    print("\nALL 5 MODEL CARD GENERATOR TESTS PASSED SUCCESSFULLY!")
