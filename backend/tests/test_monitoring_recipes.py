import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.monitoring import (
    ModelServingEndpointRecipe,
    ModelDriftMonitorRecipe,
    DataDriftDetectorRecipe,
    RollbackTriggerRecipe,
    CostLatencyMonitorRecipe,
    CostAttributionReportRecipe,
)


@pytest.fixture
def sample_classification_df():
    np.random.seed(42)
    n = 200
    x1 = np.random.normal(10, 2, n)
    x2 = np.random.normal(50, 10, n)
    x3 = np.random.uniform(1, 5, n)
    target = ((x1 * 0.4 + x2 * 0.05 + x3 * 0.2) > 7.0).astype(int)
    return pd.DataFrame({
        "feature_a": x1,
        "feature_b": x2,
        "feature_c": x3,
        "target": target
    })


@pytest.fixture
def trained_rf_model(sample_classification_df):
    df = sample_classification_df
    X = df[["feature_a", "feature_b", "feature_c"]]
    y = df["target"]
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model


class TestRecipeRegistryAndMetadata:
    def test_all_six_recipes_registered(self):
        expected_ids = [
            "model_serving_endpoint",
            "model_drift_monitor",
            "data_drift_detector",
            "rollback_trigger",
            "cost_latency_monitor",
            "cost_attribution_report",
        ]
        for rid in expected_ids:
            assert recipe_registry.has(rid), f"Recipe {rid} should be registered in registry"
            recipe = recipe_registry.get(rid)
            assert recipe is not None
            meta = recipe_registry.get_metadata(rid)
            assert meta.enabled is True
            assert meta.group == "Evaluation & Governance"
            assert meta.subgroup == "Deployment & Monitoring"
            assert meta.category == "monitoring"
            assert meta.parameters_schema is not None
            assert len(meta.inputs) >= 1
            assert len(meta.outputs) >= 1

    def test_aliases_resolution(self):
        aliases_map = {
            "drift_monitoring_operator": "model_drift_monitor",
            "drift_monitor": "model_drift_monitor",
            "performance_drift_operator": "model_drift_monitor",
            "model_serving": "model_serving_endpoint",
            "serving_endpoint": "model_serving_endpoint",
            "data_drift": "data_drift_detector",
            "input_drift_detector": "data_drift_detector",
            "cost_latency": "cost_latency_monitor",
            "cost_attribution": "cost_attribution_report",
        }
        for alias, target in aliases_map.items():
            resolved = recipe_registry.get(alias)
            assert resolved.recipe_id == target


class TestModelServingEndpoint:
    def test_serving_endpoint_with_model_and_df(self, sample_classification_df, trained_rf_model):
        recipe = ModelServingEndpointRecipe()
        config = {
            "endpoint_name": "credit-risk-scoring",
            "environment": "production",
            "traffic_percentage": 100,
            "autoscaling_min_replicas": 2,
            "autoscaling_max_replicas": 10
        }
        inputs = {
            "model": trained_rf_model,
            "dataframe": sample_classification_df,
            "feature_names": ["feature_a", "feature_b", "feature_c"]
        }
        result = recipe.execute(inputs, config)
        
        assert "governance_record" in result
        rec = result["governance_record"]
        assert rec["endpoint_name"] == "credit-risk-scoring"
        assert rec["status"] == "ACTIVE"
        assert rec["environment"] == "production"
        assert rec["autoscaling"]["min_replicas"] == 2
        assert rec["autoscaling"]["max_replicas"] == 10
        assert rec["predict_route"] == "/v1/models/credit-risk-scoring/predict"
        assert rec["health_route"] == "/v1/models/credit-risk-scoring/health"

        assert "dataframe" in result
        assert isinstance(result["dataframe"], pd.DataFrame)
        assert len(result["dataframe"]) > 5

        assert "metrics" in result
        assert result["metrics"]["status"] == "ACTIVE"
        assert result["metrics"]["warmup_latency_ms"] > 0

    def test_serving_endpoint_standalone_unparented(self, sample_classification_df):
        recipe = ModelServingEndpointRecipe()
        result = recipe.execute({"dataframe": sample_classification_df}, {})
        assert result["governance_record"]["status"] == "ACTIVE"
        assert isinstance(result["dataframe"], pd.DataFrame)

    def test_serving_config_validation(self):
        recipe = ModelServingEndpointRecipe()
        errors = recipe.validate_config({"autoscaling_min_replicas": 10, "autoscaling_max_replicas": 2})
        assert len(errors) == 1
        assert "cannot exceed" in errors[0]


class TestModelDriftMonitor:
    def test_model_drift_monitor_healthy(self, sample_classification_df):
        recipe = ModelDriftMonitorRecipe()
        df = sample_classification_df.copy()
        # Add high quality predictions matching target closely
        df["prediction"] = df["target"].copy()
        
        result = recipe.execute({"dataframe": df}, {
            "target_column": "target",
            "prediction_column": "prediction",
            "metric": "f1_score",
            "baseline_threshold": 0.90,
            "degradation_tolerance": 0.15
        })

        assert result["metrics"]["status"] == "HEALTHY"
        assert result["metrics"]["is_degraded"] is False
        assert result["metrics"]["alert_triggered"] is False
        assert isinstance(result["dataframe"], pd.DataFrame)
        assert len(result["dataframe"]) == 4

    def test_model_drift_monitor_degraded(self, sample_classification_df):
        recipe = ModelDriftMonitorRecipe()
        df = sample_classification_df.copy()
        # Make second half of predictions random / inverted to simulate performance degradation
        preds = df["target"].values.copy()
        half = len(preds) // 2
        preds[half:] = 1 - preds[half:]
        df["prediction"] = preds

        result = recipe.execute({"dataframe": df}, {
            "target_column": "target",
            "prediction_column": "prediction",
            "metric": "accuracy",
            "baseline_threshold": 0.90,
            "degradation_tolerance": 0.05
        })

        assert result["metrics"]["is_degraded"] is True
        assert result["metrics"]["alert_triggered"] is True
        assert result["metrics"]["status"] in ["CRITICAL_DRIFT", "WARNING"]
        assert result["report"]["is_degraded"] is True


class TestDataDriftDetector:
    def test_data_drift_psi_stable(self, sample_classification_df):
        recipe = DataDriftDetectorRecipe()
        # Split dataframe into baseline and identical distribution
        result = recipe.execute({"dataframe": sample_classification_df}, {
            "detection_method": "Population Stability Index (PSI)",
            "drift_threshold": 0.15,
            "baseline_split_ratio": 0.5
        })

        assert "dataframe" in result
        drift_df = result["dataframe"]
        assert len(drift_df) >= 3
        assert "feature" in drift_df.columns
        assert "drift_score" in drift_df.columns
        assert "status" in drift_df.columns
        assert result["metrics"]["total_features_monitored"] >= 3

    def test_data_drift_psi_shifted(self, sample_classification_df):
        recipe = DataDriftDetectorRecipe()
        df1 = sample_classification_df.copy()
        df2 = sample_classification_df.copy()
        # Heavily shift feature_a and feature_b in df2
        df2["feature_a"] = df2["feature_a"] + 50.0
        df2["feature_b"] = df2["feature_b"] * 5.0

        result = recipe.execute({"left_dataframe": df1, "right_dataframe": df2}, {
            "detection_method": "Population Stability Index (PSI)",
            "drift_threshold": 0.10,
            "dataset_drift_ratio_threshold": 0.25
        })

        assert result["metrics"]["dataset_drift_detected"] is True
        assert result["metrics"]["drifted_features_count"] >= 2
        assert result["metrics"]["status"] == "DATASET_DRIFT_ALERT"

    def test_data_drift_ks_test(self, sample_classification_df):
        recipe = DataDriftDetectorRecipe()
        result = recipe.execute({"dataframe": sample_classification_df}, {
            "detection_method": "Kolmogorov-Smirnov (KS-Test)",
            "drift_threshold": 0.05
        })
        assert "dataframe" in result
        assert any(row["method"] == "KS-Test" for _, row in result["dataframe"].iterrows())


class TestRollbackTrigger:
    def test_rollback_not_triggered_when_stable(self):
        recipe = RollbackTriggerRecipe()
        inputs = {
            "metrics": {
                "f1_score": 0.89,
                "dataset_drift_detected": False,
                "sla_met": True
            }
        }
        config = {
            "target_model_name": "Fraud_Detection_V2",
            "current_version": "v2",
            "fallback_version": "v1",
            "rollback_condition": "Metrics Threshold Failure",
            "metric_to_check": "f1_score",
            "threshold_operator": "less_than",
            "threshold_value": 0.70
        }
        result = recipe.execute(inputs, config)

        assert result["metrics"]["rollback_triggered"] is False
        assert result["metrics"]["active_version"] == "v2"
        assert result["governance_record"]["action_taken"] == "NO_ACTION_HEALTHY"
        assert result["governance_record"]["traffic_split"]["v2"] == 100
        assert result["governance_record"]["traffic_split"]["v1"] == 0

    def test_rollback_triggered_on_metric_drop(self):
        recipe = RollbackTriggerRecipe()
        inputs = {
            "metrics": {
                "f1_score": 0.62,
                "dataset_drift_detected": False
            }
        }
        config = {
            "target_model_name": "Fraud_Detection_V2",
            "current_version": "v2",
            "fallback_version": "v1",
            "rollback_condition": "Metrics Threshold Failure",
            "metric_to_check": "f1_score",
            "threshold_operator": "less_than",
            "threshold_value": 0.75,
            "automatic_traffic_shift": True
        }
        result = recipe.execute(inputs, config)

        assert result["metrics"]["rollback_triggered"] is True
        assert result["metrics"]["active_version"] == "v1"
        assert result["governance_record"]["action_taken"] == "ROLLED_BACK_TO_V1"
        assert result["governance_record"]["traffic_split"]["v1"] == 100
        assert result["governance_record"]["traffic_split"]["v2"] == 0
        assert result["governance_record"]["incident_id"].startswith("INC-")

    def test_rollback_triggered_on_data_drift(self):
        recipe = RollbackTriggerRecipe()
        inputs = {
            "metrics": {
                "dataset_drift_detected": True
            }
        }
        config = {
            "rollback_condition": "Data Drift Detected",
            "current_version": "v2",
            "fallback_version": "v1"
        }
        result = recipe.execute(inputs, config)
        assert result["metrics"]["rollback_triggered"] is True
        assert result["metrics"]["active_version"] == "v1"


class TestCostLatencyMonitor:
    def test_cost_latency_with_model_and_data(self, sample_classification_df, trained_rf_model):
        recipe = CostLatencyMonitorRecipe()
        config = {
            "latency_sla_ms": 100.0,
            "cost_per_hour_usd": 0.08,
            "estimated_monthly_requests": 2000000,
            "batch_sizes_to_benchmark": "1, 10, 50"
        }
        inputs = {
            "model": trained_rf_model,
            "dataframe": sample_classification_df
        }
        result = recipe.execute(inputs, config)

        assert "metrics" in result
        m = result["metrics"]
        assert m["latency_p50_ms"] > 0
        assert m["latency_p95_ms"] > 0
        assert m["throughput_rps"] > 0
        assert m["projected_monthly_cost_usd"] > 0
        assert m["sla_met"] is True

        assert "dataframe" in result
        df = result["dataframe"]
        assert len(df) == 3
        assert "batch_size" in df.columns
        assert "p95_latency_ms" in df.columns
        assert "throughput_rps" in df.columns
        assert "sla_status" in df.columns

    def test_cost_latency_sla_breach(self, sample_classification_df, trained_rf_model):
        recipe = CostLatencyMonitorRecipe()
        # Set impossible SLA of 0.0001 ms
        config = {
            "latency_sla_ms": 0.0001
        }
        result = recipe.execute({"model": trained_rf_model, "dataframe": sample_classification_df}, config)
        assert result["metrics"]["sla_met"] is False
        assert result["metrics"]["status"] == "SLA_BREACH"


class TestCostAttributionReport:
    def test_cost_attribution_synthetic_ledger(self):
        recipe = CostAttributionReportRecipe()
        config = {
            "monthly_budget_usd": 6000.0,
            "include_idle_capacity_cost": True,
            "time_period": "Current Month (MTD)"
        }
        result = recipe.execute({}, config)

        assert "report" in result
        rep = result["report"]
        assert "total_platform_spend" in rep
        assert "monthly_budget" in rep
        assert "financial_health" in rep

        assert "metrics" in result
        assert result["metrics"]["total_spend_usd"] > 1000.0
        assert result["metrics"]["budget_utilization_pct"] > 0
        assert "top_spending_department" in result["metrics"]

        assert "dataframe" in result
        df = result["dataframe"]
        assert len(df) >= 5
        assert "Department" in df.columns
        assert "Pipeline" in df.columns
        assert "Total_Cost" in df.columns
        assert "Budget_Status" in df.columns

    def test_cost_attribution_custom_dataframe(self):
        recipe = CostAttributionReportRecipe()
        custom_df = pd.DataFrame([
            {"team": "NLP Core", "cost": 1200.0},
            {"team": "Vision AI", "cost": 2400.0},
            {"team": "Analytics", "cost": 600.0}
        ])
        result = recipe.execute({"dataframe": custom_df}, {"monthly_budget_usd": 5000.0})
        assert result["metrics"]["total_spend_usd"] > 4000.0
        assert isinstance(result["dataframe"], pd.DataFrame)


class TestDAGExecutionMonitoring:
    def test_dag_pipeline_with_monitoring_nodes(self, sample_classification_df):
        from backend.app.engine.dag.graph import WorkflowGraph, WorkflowNode, WorkflowEdge
        from backend.app.engine.execution.executor import DAGExecutor

        nodes = [
            WorkflowNode(id="node_scaler", recipe_id="feature_scaler", config={"method": "standard"}),
            WorkflowNode(id="node_serving", recipe_id="model_serving_endpoint", config={"endpoint_name": "prod-scoring"}),
            WorkflowNode(id="node_latency", recipe_id="cost_latency_monitor", config={"latency_sla_ms": 100.0}),
            WorkflowNode(id="node_data_drift", recipe_id="data_drift_detector", config={"detection_method": "Population Stability Index (PSI)"}),
            WorkflowNode(id="node_perf_drift", recipe_id="model_drift_monitor", config={"metric": "f1_score"}),
            WorkflowNode(id="node_rollback", recipe_id="rollback_trigger", config={"rollback_condition": "Metrics Threshold Failure"}),
            WorkflowNode(id="node_finops", recipe_id="cost_attribution_report", config={"monthly_budget_usd": 10000.0})
        ]
        edges = [
            WorkflowEdge(source="node_scaler", target="node_serving"),
            WorkflowEdge(source="node_serving", target="node_latency"),
            WorkflowEdge(source="node_scaler", target="node_data_drift"),
            WorkflowEdge(source="node_scaler", target="node_perf_drift"),
            WorkflowEdge(source="node_perf_drift", target="node_rollback"),
            WorkflowEdge(source="node_rollback", target="node_finops")
        ]
        graph = WorkflowGraph(nodes=nodes, edges=edges)
        res = DAGExecutor.execute_workflow(
            execution_id="test_monitoring_dag_execution",
            workflow=graph,
            initial_df=sample_classification_df
        )

        assert res.status == "SUCCESS"
        assert len(res.node_results) == 7
        for nr in res.node_results:
            assert nr.status == "SUCCESS", f"Node {nr.node_id} ({nr.recipe_id}) failed: {nr.error_message}"


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", __file__]))

