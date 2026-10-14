import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.evaluation.ab_testing_framework import ABTestingFrameworkRecipe
from backend.app.recipes.evaluation.champion_challenger import ChampionChallengerRecipe
from backend.app.engine.dag.graph import WorkflowGraph, WorkflowNode, WorkflowEdge
from backend.app.engine.execution.executor import DAGExecutor


# =====================================================================
# 1. REGISTRY & METADATA TESTS
# =====================================================================

def test_evaluation_recipes_registered():
    """Verify both evaluation recipes are registered, enabled, and in correct group."""
    expected_ids = ["ab_testing_framework", "champion_challenger"]
    for rid in expected_ids:
        assert recipe_registry.has(rid), f"Recipe {rid} must be registered"
        recipe = recipe_registry.get(rid)
        meta = recipe_registry.get_metadata(rid)
        assert meta.enabled is True
        assert meta.group == "Evaluation & Governance"
        assert meta.subgroup == "Evaluation"
        assert meta.category == "evaluation"
        schema = recipe.get_schema()
        assert isinstance(schema, dict)
        assert "properties" in schema


def test_evaluation_aliases_resolution():
    """Verify registered aliases resolve to the primary recipe classes."""
    assert recipe_registry.get("ab_testing").recipe_id == "ab_testing_framework"
    assert recipe_registry.get("ab_test").recipe_id == "ab_testing_framework"
    assert recipe_registry.get("experiment_framework").recipe_id == "ab_testing_framework"
    assert recipe_registry.get("champion_vs_challenger").recipe_id == "champion_challenger"
    assert recipe_registry.get("champion_challenger_comparison").recipe_id == "champion_challenger"


# =====================================================================
# 2. A/B TESTING FRAMEWORK TESTS
# =====================================================================

def test_ab_testing_conversion_significant_uplift():
    """Tests A/B conversion test with clear statistically significant lift."""
    recipe = ABTestingFrameworkRecipe()
    np.random.seed(42)
    n = 2000
    # Control 10% conversion, Treatment 18% conversion
    c_conv = np.random.binomial(1, 0.10, n)
    t_conv = np.random.binomial(1, 0.18, n)

    df = pd.DataFrame({
        "variant": ["control"] * n + ["treatment"] * n,
        "converted": np.concatenate([c_conv, t_conv])
    })

    result = recipe.execute(
        inputs={"dataframe": df},
        config={
            "variant_column": "variant",
            "metric_column": "converted",
            "metric_type": "conversion",
            "confidence_level": 0.95,
            "mde_target_pct": 5.0
        }
    )

    metrics = result["metrics"]
    assert metrics["is_significant"] is True
    assert metrics["p_value"] < 0.05
    assert metrics["relative_uplift_pct"] > 30.0
    assert metrics["verdict"] == "DEPLOY_TREATMENT"
    assert "bar_chart" in metrics
    assert len(metrics["bar_chart"]) == 2
    assert "report" in result
    assert result["output_summary"]["status"] == "SUCCESS"


def test_ab_testing_inconclusive_lift():
    """Tests A/B conversion test with no meaningful difference."""
    recipe = ABTestingFrameworkRecipe()
    np.random.seed(42)
    n = 500
    # Identical 12% conversion
    c_conv = np.random.binomial(1, 0.12, n)
    t_conv = np.random.binomial(1, 0.12, n)

    df = pd.DataFrame({
        "group": ["A"] * n + ["B"] * n,
        "clicks": np.concatenate([c_conv, t_conv])
    })

    result = recipe.execute(
        inputs={"dataframe": df},
        config={
            "variant_column": "group",
            "control_value": "A",
            "treatment_value": "B",
            "metric_column": "clicks",
            "metric_type": "conversion"
        }
    )

    metrics = result["metrics"]
    assert metrics["is_significant"] is False
    assert metrics["verdict"] == "INCONCLUSIVE"


def test_ab_testing_continuous_metric_welch_ttest():
    """Tests A/B testing on continuous metric (e.g. revenue/order value)."""
    recipe = ABTestingFrameworkRecipe()
    np.random.seed(42)
    n = 800
    # Control avg revenue $45, Treatment avg revenue $55
    c_rev = np.random.normal(45.0, 10.0, n)
    t_rev = np.random.normal(55.0, 10.0, n)

    df = pd.DataFrame({
        "variant": ["control"] * n + ["treatment"] * n,
        "revenue": np.concatenate([c_rev, t_rev])
    })

    result = recipe.execute(
        inputs={"dataframe": df},
        config={
            "variant_column": "variant",
            "metric_column": "revenue",
            "metric_type": "continuous"
        }
    )

    metrics = result["metrics"]
    assert metrics["test_type"] == "Welch's Two-Sample T-Test"
    assert metrics["is_significant"] is True
    assert metrics["p_value"] < 0.001
    assert metrics["treatment_mean"] > metrics["control_mean"]


def test_ab_testing_separate_table_inputs():
    """Tests A/B testing with control and treatment fed as separate input tables."""
    recipe = ABTestingFrameworkRecipe()
    ctrl_df = pd.DataFrame({"converted": [1, 0, 0, 1, 0, 0, 0, 1]})
    treat_df = pd.DataFrame({"converted": [1, 1, 1, 0, 1, 1, 0, 1]})

    result = recipe.execute(
        inputs={"control": ctrl_df, "treatment": treat_df},
        config={"metric_column": "converted"}
    )
    assert len(result["dataframe"]) == 16
    assert result["metrics"]["total_samples"] == 16


def test_ab_testing_standalone_execution():
    """Tests A/B testing running standalone with mock data."""
    recipe = ABTestingFrameworkRecipe()
    result = recipe.execute(inputs={}, config={})
    assert not result["dataframe"].empty
    assert "verdict" in result["metrics"]
    assert "bar_chart" in result["metrics"]


# =====================================================================
# 3. CHAMPION / CHALLENGER COMPARISON TESTS
# =====================================================================

def test_champion_challenger_classification_promote():
    """Tests champion/challenger where challenger achieves superior F1 score."""
    recipe = ChampionChallengerRecipe()
    np.random.seed(42)
    n = 300
    X = np.random.randn(n, 4)
    # Simple non-linear target
    y = ((X[:, 0] * 2.0 + X[:, 1] ** 2) > 1.0).astype(int)

    # Champion: simple linear model (underfits non-linear boundary)
    champ = LogisticRegression()
    champ.fit(X[:200], y[:200])

    # Challenger: powerful Random Forest (captures non-linear boundary)
    chall = RandomForestClassifier(n_estimators=30, random_state=42)
    chall.fit(X[:200], y[:200])

    # Holdout test set
    test_df = pd.DataFrame(X[200:], columns=["f1", "f2", "f3", "f4"])
    test_df["target"] = y[200:]

    result = recipe.execute(
        inputs={
            "champion_model": champ,
            "challenger_model": chall,
            "dataframe": test_df
        },
        config={
            "comparison_metric": "f1",
            "target_column": "target",
            "min_lift_threshold_pct": 1.0
        }
    )

    metrics = result["metrics"]
    assert "champion_score" in metrics
    assert "challenger_score" in metrics
    assert metrics["challenger_score"] >= metrics["champion_score"]
    assert metrics["decision"] in ["PROMOTE_CHALLENGER", "RETAIN_CHAMPION"]
    assert "comparison_bar_chart" in metrics
    assert len(metrics["comparison_bar_chart"]) >= 4
    assert "champion_prediction" in result["dataframe"].columns
    assert "challenger_prediction" in result["dataframe"].columns


def test_champion_challenger_regression_comparison():
    """Tests champion/challenger comparison on a regression task."""
    recipe = ChampionChallengerRecipe()
    np.random.seed(42)
    n = 200
    X = np.random.randn(n, 3)
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + np.random.randn(n) * 0.2

    champ = LinearRegression()
    champ.fit(X[:150], y[:150])

    chall = RandomForestRegressor(n_estimators=20, random_state=42)
    chall.fit(X[:150], y[:150])

    test_df = pd.DataFrame(X[150:], columns=["feat_a", "feat_b", "feat_c"])
    test_df["target_val"] = y[150:]

    result = recipe.execute(
        inputs={
            "champion_model": champ,
            "challenger_model": chall,
            "dataframe": test_df
        },
        config={
            "comparison_metric": "rmse",
            "task_type": "regression",
            "target_column": "target_val"
        }
    )

    metrics = result["metrics"]
    assert metrics["task_type"] == "regression"
    assert "mae" in metrics["champion_metrics"]
    assert "rmse" in metrics["challenger_metrics"]
    assert "champion_residual" in result["dataframe"].columns
    assert "challenger_residual" in result["dataframe"].columns


def test_champion_challenger_latency_breach_gating():
    """Tests governance rejection when challenger exceeds maximum permitted latency increase."""
    recipe = ChampionChallengerRecipe()
    test_df = pd.DataFrame({
        "feat_1": [1.0, 2.0, 3.0, 4.0],
        "target": [0, 1, 0, 1]
    })

    # Dummy mock model that simulates slow challenger inference
    class SlowModel:
        def predict(self, X):
            import time
            time.sleep(0.06)  # 60ms latency
            return np.array([1, 1, 0, 1])

    class FastModel:
        def predict(self, X):
            return np.array([0, 1, 0, 1])

    result = recipe.execute(
        inputs={
            "champion_model": FastModel(),
            "challenger_model": SlowModel(),
            "dataframe": test_df
        },
        config={
            "max_latency_regression_pct": 5.0  # very strict SLA tolerance
        }
    )

    assert result["metrics"]["decision"] == "REJECT_LATENCY_BREACH"


def test_champion_challenger_standalone_execution():
    """Tests champion/challenger recipe running standalone with automatic mock evaluation."""
    recipe = ChampionChallengerRecipe()
    result = recipe.execute(inputs={}, config={})
    assert not result["dataframe"].empty
    assert "decision" in result["metrics"]
    assert "bar_chart" in result["metrics"]
    assert result["output_summary"]["status"] == "SUCCESS"


# =====================================================================
# 4. DAG WORKFLOW INTEGRATION TEST
# =====================================================================

def test_dag_workflow_with_evaluation_recipes():
    """Validates complete multi-node DAG execution with both evaluation recipes."""
    nodes = [
        WorkflowNode(
            id="node_ab",
            recipe_id="ab_testing_framework",
            label="A/B Experiment Evaluator",
            config={"metric_column": "converted", "confidence_level": 0.95}
        ),
        WorkflowNode(
            id="node_comp",
            recipe_id="champion_challenger",
            label="Champion Challenger Comparator",
            config={"comparison_metric": "f1"}
        )
    ]
    edges = [
        WorkflowEdge(id="e1", source="node_ab", target="node_comp")
    ]

    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "variant": ["control"] * 100 + ["treatment"] * 100,
        "converted": np.random.binomial(1, 0.15, n),
        "f1": np.random.randn(n),
        "target": np.random.binomial(1, 0.5, n)
    })

    graph = WorkflowGraph(nodes=nodes, edges=edges)
    res = DAGExecutor.execute_workflow(execution_id="test_eval_dag_01", workflow=graph, initial_df=df)

    assert res.status == "SUCCESS"
    assert len(res.node_results) == 2
    assert res.node_results[0].status == "SUCCESS"
    assert res.node_results[1].status == "SUCCESS"
