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
from sklearn.linear_model import LogisticRegression, Ridge

from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.evaluation.fairness_slicing_report import FairnessSlicingReportRecipe


def test_fairness_slicing_report_registry_and_aliases():
    """Verify FairnessSlicingReportRecipe is registered with correct metadata and aliases."""
    assert recipe_registry.has("fairness_slicing_report") is True
    meta = recipe_registry.get_metadata("fairness_slicing_report")
    assert meta.enabled is True
    assert meta.group == "Evaluation & Governance"
    assert meta.subgroup == "Fairness & Compliance"

    # Aliases
    assert recipe_registry.get("fairness_slicing").recipe_id == "fairness_slicing_report"
    assert recipe_registry.get("demographic_slicing").recipe_id == "fairness_slicing_report"
    assert recipe_registry.get("slice_evaluator").recipe_id == "fairness_slicing_report"
    assert recipe_registry.get("model_slicing").recipe_id == "fairness_slicing_report"
    assert recipe_registry.get("performance_slicing").recipe_id == "fairness_slicing_report"


def test_classification_slicing_with_degradation_alert():
    """Verify binary classification slicing, baseline calculation, and degradation detection."""
    # Cohort 1: 100 samples, high accuracy (90% F1)
    # Cohort 2: 100 samples, severely degraded accuracy (50% F1)
    df = pd.DataFrame({
        "gender": ["Group_High"] * 100 + ["Group_Low"] * 100,
        "target": [1] * 50 + [0] * 50 + [1] * 50 + [0] * 50,
        "prediction": [1] * 45 + [0] * 5 + [1] * 5 + [0] * 45 + [1] * 20 + [0] * 30 + [1] * 25 + [0] * 25
    })

    recipe = FairnessSlicingReportRecipe()
    result = recipe.execute(
        inputs={"dataframe": df},
        config={
            "slice_columns": "gender",
            "target_col": "target",
            "prediction_col": "prediction",
            "degradation_tolerance_pct": 15.0
        }
    )

    metrics = result["metrics"]
    report = result["report"]
    slices_table = result["slices_table"]

    assert metrics["total_slices_audited"] == 2
    assert metrics["best_performing_slice"] == "Group_High"
    assert metrics["worst_performing_slice"] == "Group_Low"
    assert metrics["severely_degraded_slices_count"] == 1
    assert len(metrics["warnings"]) > 0
    assert any("SEVERE SLICE DEGRADATION" in w for w in metrics["warnings"])

    # Verify table columns
    assert "accuracy_pct" in slices_table.columns
    assert "f1_score_pct" in slices_table.columns
    assert "f1_diff_vs_baseline" in slices_table.columns
    assert "is_severely_degraded" in slices_table.columns

    # Verify report markdown
    assert "# === Demographic Fairness Slicing Report" in report
    assert "Disaggregated Demographic Slices Leaderboard:" in report
    assert "DEGRADED" in report


def test_intersectional_slicing_multi_columns():
    """Verify multi-column intersectional slicing (e.g. 'gender,age_band')."""
    df = pd.DataFrame({
        "gender": ["F"] * 50 + ["M"] * 50 + ["F"] * 50 + ["M"] * 50,
        "age_band": ["Young"] * 100 + ["Senior"] * 100,
        "target": [1] * 100 + [0] * 100,
        "prediction": [1] * 90 + [0] * 10 + [0] * 95 + [1] * 5
    })

    recipe = FairnessSlicingReportRecipe()
    result = recipe.execute(
        inputs={"dataframe": df},
        config={
            "slice_columns": "gender, age_band",
            "target_col": "target",
            "prediction_col": "prediction"
        }
    )

    metrics = result["metrics"]
    slices_df = result["slices_table"]

    # 4 distinct intersectional cohorts: F x Young, M x Young, F x Senior, M x Senior
    assert metrics["total_slices_audited"] == 4
    slice_names = slices_df["slice_id"].tolist()
    assert "F x Young" in slice_names
    assert "M x Senior" in slice_names


def test_regression_slicing_mae_rmse():
    """Verify regression slicing with MAE, RMSE, and error increase alerts."""
    np.random.seed(42)
    # Group A: Low error (MAE ~ 1.0)
    # Group B: High error (MAE ~ 5.0)
    y_A = np.random.uniform(10, 50, 100)
    pred_A = y_A + np.random.normal(0, 1.0, 100)

    y_B = np.random.uniform(10, 50, 100)
    pred_B = y_B + np.random.normal(0, 5.0, 100)

    df = pd.DataFrame({
        "region": ["North"] * 100 + ["South"] * 100,
        "actual_sales": np.concatenate([y_A, y_B]),
        "predicted_sales": np.concatenate([pred_A, pred_B])
    })

    recipe = FairnessSlicingReportRecipe()
    result = recipe.execute(
        inputs={"dataframe": df},
        config={
            "slice_columns": "region",
            "target_col": "actual_sales",
            "prediction_col": "predicted_sales",
            "problem_type": "regression",
            "degradation_tolerance_pct": 20.0
        }
    )

    metrics = result["metrics"]
    slices_table = result["slices_table"]

    assert metrics["global_baseline"]["task_type"] == "regression"
    assert metrics["best_performing_slice"] == "North"
    assert metrics["worst_performing_slice"] == "South"
    assert metrics["severely_degraded_slices_count"] == 1

    # Table verification
    assert "mae" in slices_table.columns
    assert "rmse" in slices_table.columns
    assert "mae_increase_pct" in slices_table.columns

    # Report verification
    assert "MAE vs Base" in result["report"]


def test_trained_model_prediction_input():
    """Verify generating predictions directly from a trained model instance."""
    np.random.seed(42)
    X = np.random.randn(200, 2)
    y = (X[:, 0] > 0).astype(int)
    model = LogisticRegression()
    model.fit(X, y)

    df_test = pd.DataFrame(X, columns=["feat_1", "feat_2"])
    df_test["target"] = y
    df_test["cohort"] = ["Cohort_1"] * 100 + ["Cohort_2"] * 100

    recipe = FairnessSlicingReportRecipe()
    result = recipe.execute(
        inputs={"test_data": df_test, "model": model},
        config={"slice_columns": "cohort", "target_col": "target"}
    )

    assert result["metrics"]["total_slices_audited"] == 2
    assert "accuracy_pct" in result["slices_table"].columns


def test_fairness_slicing_edge_cases():
    """Verify guardrails: single slice, missing target column, and small sample size guardrail."""
    recipe = FairnessSlicingReportRecipe()

    # Guard 1: Missing target column
    df_no_tgt = pd.DataFrame({"gender": ["A", "B"], "prediction": [1, 0]})
    with pytest.raises(ValueError, match="could not locate target column"):
        recipe.execute(inputs={"dataframe": df_no_tgt}, config={"slice_columns": "gender", "target_col": "missing_tgt"})

    # Guard 2: Single slice rejected
    df_single_slice = pd.DataFrame({
        "gender": ["Group_X"] * 10,
        "target": [1, 0] * 5,
        "prediction": [1, 0] * 5
    })
    with pytest.raises(ValueError, match="requires at least 2 distinct slices"):
        recipe.execute(inputs={"dataframe": df_single_slice}, config={"slice_columns": "gender"})

    # Guard 3: Small slice warning triggered when N < min_slice_sample_size (e.g. N=5 < 30)
    df_small = pd.DataFrame({
        "group": ["A"] * 50 + ["B"] * 5,
        "target": [1] * 55,
        "prediction": [1] * 55
    })
    res_small = recipe.execute(inputs={"dataframe": df_small}, config={"slice_columns": "group", "min_slice_sample_size": 30})
    assert any("SMALL SLICE SAMPLE SIZE" in w for w in res_small["metrics"]["warnings"])


if __name__ == "__main__":
    print("Running test_fairness_slicing_report_registry_and_aliases...")
    test_fairness_slicing_report_registry_and_aliases()
    print("PASS: test_fairness_slicing_report_registry_and_aliases")

    print("Running test_classification_slicing_with_degradation_alert...")
    test_classification_slicing_with_degradation_alert()
    print("PASS: test_classification_slicing_with_degradation_alert")

    print("Running test_intersectional_slicing_multi_columns...")
    test_intersectional_slicing_multi_columns()
    print("PASS: test_intersectional_slicing_multi_columns")

    print("Running test_regression_slicing_mae_rmse...")
    test_regression_slicing_mae_rmse()
    print("PASS: test_regression_slicing_mae_rmse")

    print("Running test_trained_model_prediction_input...")
    test_trained_model_prediction_input()
    print("PASS: test_trained_model_prediction_input")

    print("Running test_fairness_slicing_edge_cases...")
    test_fairness_slicing_edge_cases()
    print("PASS: test_fairness_slicing_edge_cases")

    print("\nALL 6 FAIRNESS SLICING REPORT TESTS PASSED SUCCESSFULLY!")
