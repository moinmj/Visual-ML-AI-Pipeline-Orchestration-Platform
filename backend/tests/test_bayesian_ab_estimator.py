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
from backend.app.recipes.evaluation.bayesian_ab_estimator import BayesianABEstimatorRecipe


def test_bayesian_ab_estimator_registry_and_aliases():
    """Verify BayesianABEstimatorRecipe is registered with correct metadata and aliases."""
    assert recipe_registry.has("bayesian_ab_estimator") is True
    meta = recipe_registry.get_metadata("bayesian_ab_estimator")
    assert meta.enabled is True
    assert meta.group == "Evaluation & Governance"
    assert meta.subgroup == "Evaluation"

    # Aliases
    assert recipe_registry.get("bayesian_ab").recipe_id == "bayesian_ab_estimator"
    assert recipe_registry.get("bayesian_ab_test").recipe_id == "bayesian_ab_estimator"
    assert recipe_registry.get("bayesian_experimentation").recipe_id == "bayesian_ab_estimator"
    assert recipe_registry.get("expected_loss_ab").recipe_id == "bayesian_ab_estimator"
    assert recipe_registry.get("posterior_ab_estimator").recipe_id == "bayesian_ab_estimator"


def test_bayesian_binary_conversion_ab_test():
    """Verify Beta-Binomial conjugate posterior for binary conversion A/B test."""
    np.random.seed(42)
    # Control: 1000 visitors, ~100 conversions (10%)
    ctrl_convs = np.random.binomial(n=1, p=0.10, size=1000)
    # Treatment: 1000 visitors, ~140 conversions (14%)
    treat_convs = np.random.binomial(n=1, p=0.14, size=1000)

    df = pd.DataFrame({
        "variant": ["control"] * 1000 + ["treatment"] * 1000,
        "converted": np.concatenate([ctrl_convs, treat_convs])
    })

    recipe = BayesianABEstimatorRecipe()
    result = recipe.execute(
        inputs={"dataframe": df},
        config={
            "control_variant": "control",
            "metric_type": "binary_conversion",
            "loss_threshold": 0.005,
            "n_simulations": 15000
        }
    )

    metrics = result["metrics"]
    report = result["report"]
    summary_df = result["summary_table"]

    assert metrics["metric_nature"] == "binary_conversion"
    assert metrics["overall_decision"] == "DEPLOY_TREATMENT"
    assert len(metrics["treatments_evaluated"]) == 1

    eval_t = metrics["treatments_evaluated"][0]
    assert eval_t["p_treatment_superior"] > 0.95
    assert eval_t["expected_loss_choose_treatment"] < 0.005
    assert eval_t["relative_lift_median_pct"] > 15.0
    assert not summary_df.empty

    # Report verification
    assert "# === Bayesian A/B Experimentation Decision Report" in report
    assert "DEPLOY TREATMENT VARIANT" in report
    assert "P(Treatment > Control)" in report


def test_bayesian_continuous_revenue_ab_test():
    """Verify Gaussian posterior sampling for continuous revenue A/B test."""
    np.random.seed(42)
    ctrl_rev = np.random.normal(loc=50.0, scale=10.0, size=500)
    treat_rev = np.random.normal(loc=56.0, scale=10.0, size=500)

    df = pd.DataFrame({
        "arm": ["A_Control"] * 500 + ["B_Treatment"] * 500,
        "revenue": np.concatenate([ctrl_rev, treat_rev])
    })

    recipe = BayesianABEstimatorRecipe()
    result = recipe.execute(
        inputs={"dataframe": df},
        config={
            "variant_col": "arm",
            "outcome_col": "revenue",
            "control_variant": "A_Control",
            "metric_type": "continuous_revenue",
            "loss_threshold": 0.05
        }
    )

    metrics = result["metrics"]
    assert metrics["metric_nature"] == "continuous_revenue"
    assert metrics["overall_decision"] == "DEPLOY_TREATMENT"
    eval_t = metrics["treatments_evaluated"][0]
    assert eval_t["p_treatment_superior"] > 0.99
    assert eval_t["relative_lift_median_pct"] > 5.0


def test_multi_variant_abn_testing():
    """Verify multi-variant (A/B/C) testing compares multiple treatment arms against control."""
    np.random.seed(42)
    c_out = np.random.binomial(1, 0.10, 400)
    t1_out = np.random.binomial(1, 0.11, 400)
    t2_out = np.random.binomial(1, 0.18, 400)

    df = pd.DataFrame({
        "variant": ["ctrl"] * 400 + ["treat_1"] * 400 + ["treat_2"] * 400,
        "converted": np.concatenate([c_out, t1_out, t2_out])
    })

    recipe = BayesianABEstimatorRecipe()
    result = recipe.execute(
        inputs={"dataframe": df},
        config={"control_variant": "ctrl", "loss_threshold": 0.01}
    )

    evals = result["metrics"]["treatments_evaluated"]
    assert len(evals) == 2
    # treat_2 had 18% conversion rate vs 10% control -> should have higher P(superior)
    t2_eval = next(e for e in evals if e["treatment_variant"] == "treat_2")
    assert t2_eval["p_treatment_superior"] > 0.98


def test_retain_control_and_continue_experiment_decisions():
    """Verify inferior treatment triggers RETAIN_CONTROL and small sample triggers CONTINUE_EXPERIMENT."""
    recipe = BayesianABEstimatorRecipe()

    # 1. Inferior treatment (treatment is significantly worse: 5% vs 20%)
    df_inferior = pd.DataFrame({
        "variant": ["ctrl"] * 500 + ["worse_treat"] * 500,
        "converted": [1] * 100 + [0] * 400 + [1] * 25 + [0] * 475
    })
    res_inf = recipe.execute(inputs={"dataframe": df_inferior}, config={"control_variant": "ctrl"})
    assert res_inf["metrics"]["overall_decision"] == "RETAIN_CONTROL"

    # 2. Tiny sample (10 visitors per arm) -> insufficient certainty
    df_tiny = pd.DataFrame({
        "variant": ["ctrl"] * 10 + ["treat"] * 10,
        "converted": [1, 0, 1, 0, 0, 0, 0, 0, 0, 0] + [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    })
    res_tiny = recipe.execute(inputs={"dataframe": df_tiny}, config={"loss_threshold": 0.001})
    assert res_tiny["metrics"]["overall_decision"] == "CONTINUE_EXPERIMENT"


def test_bayesian_ab_estimator_edge_cases():
    """Verify guardrails: missing columns, single variant, string boolean outcomes."""
    recipe = BayesianABEstimatorRecipe()

    # Guard 1: Missing variant column
    df_no_var = pd.DataFrame({"converted": [1, 0, 1]})
    with pytest.raises(ValueError, match="could not locate variant column"):
        recipe.execute(inputs={"dataframe": df_no_var}, config={"variant_col": "group_missing"})

    # Guard 2: Single variant rejected
    df_single = pd.DataFrame({"variant": ["A", "A", "A"], "converted": [1, 0, 1]})
    with pytest.raises(ValueError, match="requires at least 2 distinct variants"):
        recipe.execute(inputs={"dataframe": df_single}, config={})

    # Guard 3: Boolean string outcomes converted cleanly
    df_str = pd.DataFrame({
        "variant": ["ctrl", "ctrl", "treat", "treat", "treat", "treat"],
        "converted": ["true", "false", "true", "true", "true", "false"]
    })
    res_str = recipe.execute(inputs={"dataframe": df_str}, config={})
    assert res_str["metrics"]["variant_summaries"][0]["conversions"] == 1


if __name__ == "__main__":
    print("Running test_bayesian_ab_estimator_registry_and_aliases...")
    test_bayesian_ab_estimator_registry_and_aliases()
    print("PASS: test_bayesian_ab_estimator_registry_and_aliases")

    print("Running test_bayesian_binary_conversion_ab_test...")
    test_bayesian_binary_conversion_ab_test()
    print("PASS: test_bayesian_binary_conversion_ab_test")

    print("Running test_bayesian_continuous_revenue_ab_test...")
    test_bayesian_continuous_revenue_ab_test()
    print("PASS: test_bayesian_continuous_revenue_ab_test")

    print("Running test_multi_variant_abn_testing...")
    test_multi_variant_abn_testing()
    print("PASS: test_multi_variant_abn_testing")

    print("Running test_retain_control_and_continue_experiment_decisions...")
    test_retain_control_and_continue_experiment_decisions()
    print("PASS: test_retain_control_and_continue_experiment_decisions")

    print("Running test_bayesian_ab_estimator_edge_cases...")
    test_bayesian_ab_estimator_edge_cases()
    print("PASS: test_bayesian_ab_estimator_edge_cases")

    print("\nALL 6 BAYESIAN A/B ESTIMATOR TESTS PASSED SUCCESSFULLY!")
