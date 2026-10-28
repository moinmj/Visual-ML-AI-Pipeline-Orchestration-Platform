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

from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.training.survival_model import SurvivalModelRecipe


def test_survival_model_registry_and_aliases():
    """Verify SurvivalModelRecipe is registered with correct metadata and aliases."""
    assert recipe_registry.has("survival_model") is True
    meta = recipe_registry.get_metadata("survival_model")
    assert meta.enabled is True
    assert meta.group == "Modeling"
    assert meta.subgroup == "Specialized Models"

    # Aliases
    assert recipe_registry.get("survival").recipe_id == "survival_model"
    assert recipe_registry.get("cox_ph").recipe_id == "survival_model"
    assert recipe_registry.get("cox_proportional_hazards").recipe_id == "survival_model"
    assert recipe_registry.get("kaplan_meier").recipe_id == "survival_model"
    assert recipe_registry.get("survival_analysis").recipe_id == "survival_model"
    assert recipe_registry.get("time_to_event").recipe_id == "survival_model"


def test_kaplan_meier_survival_curves():
    """Verify Kaplan-Meier curve estimation, median survival duration, and milestones."""
    np.random.seed(42)
    N = 150
    durations = np.random.exponential(scale=24.0, size=N) + 1.0  # months
    events = np.random.binomial(n=1, p=0.6, size=N)  # 60% churn, 40% censored

    df = pd.DataFrame({"tenure": durations, "churn": events})

    recipe = SurvivalModelRecipe()
    result = recipe.execute(
        inputs={"dataframe": df},
        config={
            "duration_col": "tenure",
            "event_col": "churn",
            "model_type": "kaplan_meier"
        }
    )

    metrics = result["metrics"]
    report = result["report"]

    assert result["kmf"] is not None
    assert metrics["total_observations"] == N
    assert metrics["events_observed"] > 0
    assert metrics["censored_observations"] > 0
    assert metrics["median_survival_time"] is not None
    assert len(metrics["survival_curve_milestones"]) > 0

    # Verify report formatting
    assert "# === Survival Analysis & Cohort Retention Report" in report
    assert "Kaplan-Meier Survival Curve Milestones:" in report


def test_cox_proportional_hazards_regression():
    """Verify Cox Proportional Hazards regression estimates hazard ratios and concordance index."""
    np.random.seed(42)
    N = 180
    age = np.random.normal(45, 10, N)
    charges = np.random.normal(70, 20, N)
    # Higher charges increases risk of churn (hazard)
    risk_score = 0.03 * (charges - 70) + 0.02 * (age - 45)
    baseline_hazard = np.random.exponential(scale=30.0, size=N)
    durations = baseline_hazard / np.exp(np.clip(risk_score, -2, 2)) + 0.5
    events = np.random.binomial(n=1, p=0.7, size=N)

    df = pd.DataFrame({
        "months": durations,
        "is_churn": events,
        "age": age,
        "monthly_charges": charges
    })

    recipe = SurvivalModelRecipe()
    result = recipe.execute(
        inputs={"dataframe": df},
        config={
            "duration_col": "months",
            "event_col": "is_churn",
            "model_type": "both",
            "penalizer": 0.1
        }
    )

    metrics = result["metrics"]
    report = result["report"]

    assert result["cph"] is not None
    assert metrics["concordance_index"] is not None
    assert metrics["concordance_index"] > 0.5
    assert "monthly_charges" in metrics["hazard_ratios"]
    assert "age" in metrics["hazard_ratios"]
    assert "Cox Proportional Hazards: Feature Hazard Ratios (HR):" in report


def test_stratified_kaplan_meier_cohorts():
    """Verify stratified Kaplan-Meier curves compare median survival across subscription tiers."""
    np.random.seed(42)
    N = 150
    tiers = np.random.choice(["Basic", "Pro", "Enterprise"], size=N)
    # Enterprise retains longer
    scale_map = {"Basic": 12.0, "Pro": 24.0, "Enterprise": 48.0}
    durations = [np.random.exponential(scale=scale_map[t]) + 1.0 for t in tiers]
    events = np.random.binomial(n=1, p=0.6, size=N)

    df = pd.DataFrame({
        "tenure": durations,
        "churn": events,
        "plan_tier": tiers
    })

    recipe = SurvivalModelRecipe()
    result = recipe.execute(
        inputs={"dataframe": df},
        config={
            "duration_col": "tenure",
            "event_col": "churn",
            "strata_col": "plan_tier"
        }
    )

    metrics = result["metrics"]
    report = result["report"]

    assert "stratified_median_survival" in metrics
    assert "Basic" in metrics["stratified_median_survival"]
    assert "Enterprise" in metrics["stratified_median_survival"]
    assert "Stratified Cohort Median Survival" in report


def test_string_and_boolean_event_encodings():
    """Verify string labels ('churn'/'retained') and boolean status are cleaned into 1/0."""
    df_str = pd.DataFrame({
        "time": [10, 20, 30, 40, 50, 60],
        "status": ["churn", "retained", "churn", "retained", "churn", "retained"],
        "metric": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
    })

    recipe = SurvivalModelRecipe()
    res = recipe.execute(inputs={"dataframe": df_str}, config={})
    assert res["metrics"]["events_observed"] == 3
    assert res["metrics"]["censored_observations"] == 3


def test_survival_model_edge_cases():
    """Verify guardrails: missing columns, negative durations, zero events."""
    recipe = SurvivalModelRecipe()

    # Guard 1: Missing duration column
    df_bad = pd.DataFrame({"colA": [1, 2], "colB": [3, 4]})
    with pytest.raises(ValueError, match="could not locate duration column"):
        recipe.execute(inputs={"dataframe": df_bad}, config={"duration_col": "missing_col"})

    # Guard 2: Negative durations rejected
    df_neg = pd.DataFrame({"duration": [-5, 10, 15], "event": [1, 0, 1]})
    with pytest.raises(ValueError, match="contains negative values"):
        recipe.execute(inputs={"dataframe": df_neg}, config={})

    # Guard 3: Zero events observed rejected
    df_no_events = pd.DataFrame({"duration": [5, 10, 15], "event": [0, 0, 0]})
    with pytest.raises(ValueError, match="contains 0 observed events"):
        recipe.execute(inputs={"dataframe": df_no_events}, config={})

    # Guard 4: Small number of events triggers warning
    df_small = pd.DataFrame({
        "duration": [5, 10, 15, 20, 25, 30],
        "event": [1, 1, 0, 0, 0, 0],
        "covar": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    })
    res_small = recipe.execute(inputs={"dataframe": df_small}, config={})
    assert any("FEW EVENTS" in w for w in res_small["metrics"]["warnings"])


if __name__ == "__main__":
    print("Running test_survival_model_registry_and_aliases...")
    test_survival_model_registry_and_aliases()
    print("PASS: test_survival_model_registry_and_aliases")

    print("Running test_kaplan_meier_survival_curves...")
    test_kaplan_meier_survival_curves()
    print("PASS: test_kaplan_meier_survival_curves")

    print("Running test_cox_proportional_hazards_regression...")
    test_cox_proportional_hazards_regression()
    print("PASS: test_cox_proportional_hazards_regression")

    print("Running test_stratified_kaplan_meier_cohorts...")
    test_stratified_kaplan_meier_cohorts()
    print("PASS: test_stratified_kaplan_meier_cohorts")

    print("Running test_string_and_boolean_event_encodings...")
    test_string_and_boolean_event_encodings()
    print("PASS: test_string_and_boolean_event_encodings")

    print("Running test_survival_model_edge_cases...")
    test_survival_model_edge_cases()
    print("PASS: test_survival_model_edge_cases")

    print("\nALL 6 SURVIVAL MODEL TESTS PASSED SUCCESSFULLY!")
