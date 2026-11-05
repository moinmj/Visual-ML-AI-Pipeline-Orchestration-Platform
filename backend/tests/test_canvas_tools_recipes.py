"""
Unit Tests for Canvas Tools Recipe Suite (10 Recipes).

Tests registration, aliases, schema contracts, execution correctness,
and comprehensive edge-case handling for:
  1. canvas_note
  2. canvas_group
  3. canvas_comments
  4. chart_visualization
  5. version_snapshot
  6. template_macro_library
  7. pipeline_diff_viewer
  8. readonly_pipeline_snapshot
  9. secrets_manager_block
 10. parameter_sweep_runner
"""

import os
import json
import pytest
import pandas as pd
import numpy as np

from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.base.planned_recipes import PLANNED_RECIPES
from backend.app.recipes.canvas import (
    CanvasNoteRecipe,
    CanvasGroupRecipe,
    CanvasCommentsRecipe,
    ChartVisualizationRecipe,
    VersionSnapshotRecipe,
    TemplateMacroLibraryRecipe,
    PipelineDiffViewerRecipe,
    ReadonlyPipelineSnapshotRecipe,
    SecretsManagerBlockRecipe,
    ParameterSweepRunnerRecipe,
)


CANVAS_RECIPE_IDS = [
    "canvas_note",
    "canvas_group",
    "canvas_comments",
    "chart_visualization",
    "version_snapshot",
    "template_macro_library",
    "pipeline_diff_viewer",
    "readonly_pipeline_snapshot",
    "secrets_manager_block",
    "parameter_sweep_runner",
]


# =====================================================================
# 1. Registration, Aliases, and Planned Recipes Removal Tests
# =====================================================================

def test_canvas_tools_registration():
    for rid in CANVAS_RECIPE_IDS:
        recipe = recipe_registry.get(rid)
        assert recipe is not None, f"Recipe {rid} was not found in registry"
        assert recipe.recipe_id == rid
        assert recipe.group == "Canvas Tools"
        assert recipe.category == "canvas"
        # Must be enabled (not in planned_recipes)
        assert rid not in PLANNED_RECIPES, f"Recipe {rid} still in planned_recipes"


def test_canvas_tools_aliases():
    assert recipe_registry.get("note") == recipe_registry.get("canvas_note")
    assert recipe_registry.get("group") == recipe_registry.get("canvas_group")
    assert recipe_registry.get("comments") == recipe_registry.get("canvas_comments")
    assert recipe_registry.get("chart") == recipe_registry.get("chart_visualization")
    assert recipe_registry.get("snapshot") == recipe_registry.get("version_snapshot")
    assert recipe_registry.get("macro_library") == recipe_registry.get("template_macro_library")
    assert recipe_registry.get("diff_viewer") == recipe_registry.get("pipeline_diff_viewer")
    assert recipe_registry.get("readonly_snapshot") == recipe_registry.get("readonly_pipeline_snapshot")
    assert recipe_registry.get("secrets_manager") == recipe_registry.get("secrets_manager_block")
    assert recipe_registry.get("parameter_sweep") == recipe_registry.get("parameter_sweep_runner")


def test_canvas_tools_schema_contracts():
    for rid in CANVAS_RECIPE_IDS:
        recipe = recipe_registry.get(rid)
        schema = recipe.get_schema()
        assert "properties" in schema, f"Schema for {rid} missing properties"
        props = schema["properties"]
        for prop_name, prop_meta in props.items():
            assert "input_type" in prop_meta, f"Property '{prop_name}' in {rid} missing 'input_type'"
            assert "description" in prop_meta, f"Property '{prop_name}' in {rid} missing 'description'"


# =====================================================================
# 2. Canvas Note Recipe Tests & Edge Cases
# =====================================================================

def test_canvas_note_execution_and_edge_cases():
    recipe = recipe_registry.get("canvas_note")
    df_sample = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})

    # Standard run with passthrough dataframe
    res = recipe.execute({"dataframe": df_sample}, {
        "title": "Model SLA Note",
        "content": "Latencies must remain < 50ms at p99.",
        "color_theme": "emerald",  # will fallback to yellow or accepted enum
        "tags": "sla, latency, prod"
    })
    assert res["title"] == "Model SLA Note"
    assert "Latencies must remain" in res["content"]
    assert res["metrics"]["word_count"] > 0
    assert "report" in res and "markdown" in res["report"]
    assert "dataframe" in res
    pd.testing.assert_frame_equal(res["dataframe"], df_sample)

    # Edge Case: empty title, empty content, missing inputs
    res_empty = recipe.execute({}, {"title": "", "content": "", "tags": []})
    assert res_empty["title"] == "Pipeline Documentation Note"
    assert res_empty["content"] == "No content provided."
    assert res_empty["metrics"]["word_count"] > 0


# =====================================================================
# 3. Canvas Group Recipe Tests & Edge Cases
# =====================================================================

def test_canvas_group_execution_and_edge_cases():
    recipe = recipe_registry.get("canvas_group")
    df_sample = pd.DataFrame({"val": [10, 20]})

    context_mock = {
        "nodes": [
            {"id": "node_imputer"},
            {"id": "node_scaler"}
        ]
    }

    # Group with existing and missing node IDs
    res = recipe.execute({"input": df_sample}, {
        "group_name": "Preprocessing Stage",
        "node_ids": "node_imputer, node_scaler, node_nonexistent, node_imputer",  # test deduplication
        "color": "blue",
        "border_style": "dashed",
        "collapsed": True,
        "execution_order_priority": 2
    }, context=context_mock)

    assert res["group_name"] == "Preprocessing Stage"
    assert len(res["enclosed_node_ids"]) == 3  # deduplicated 4 -> 3
    assert "node_imputer" in res["verified_in_context"]
    assert "node_scaler" in res["verified_in_context"]
    assert "node_nonexistent" in res["unverified_in_context"]
    assert res["metrics"]["enclosed_nodes_count"] == 3
    assert res["metrics"]["verified_nodes_count"] == 2
    assert "dataframe" in res


# =====================================================================
# 4. Canvas Comments Recipe Tests & Edge Cases
# =====================================================================

def test_canvas_comments_execution_and_edge_cases():
    recipe = recipe_registry.get("canvas_comments")

    res = recipe.execute({}, {
        "target_node_id": "xgboost_trainer_1",
        "author": "Dr. Lead ML",
        "comment_text": "Check validation split for potential target leakage.",
        "status": "In Review",
        "priority": "Blocker",
        "tags": ["leak_check", "audit"]
    })

    assert res["target_node_id"] == "xgboost_trainer_1"
    assert res["author"] == "Dr. Lead ML"
    assert res["status"] == "In Review"
    assert res["priority"] == "Blocker"
    assert res["metrics"]["is_blocker"] == 1
    assert res["metrics"]["is_resolved"] == 0
    assert "Thread #" in res["report"]["markdown"]

    # Edge Case: Resolved comment
    res_resolved = recipe.execute({}, {"status": "Resolved", "priority": "Low"})
    assert res_resolved["metrics"]["is_resolved"] == 1
    assert res_resolved["metrics"]["is_blocker"] == 0


# =====================================================================
# 5. Chart Visualization Recipe Tests & Edge Cases
# =====================================================================

def test_chart_visualization_execution():
    recipe = recipe_registry.get("chart_visualization")
    df = pd.DataFrame({
        "Category": ["Electronics", "Clothing", "Home", "Electronics", "Clothing"],
        "Sales": [150.0, 80.0, 120.0, 200.0, 95.0],
        "Region": ["North", "South", "East", "West", "North"]
    })

    # 1. Bar Chart with aggregation
    res_bar = recipe.execute({"dataframe": df}, {
        "chart_type": "bar",
        "x_column": "Category",
        "y_column": "Sales",
        "agg_func": "mean",
        "title": "Average Sales by Category"
    })
    assert res_bar["chart_type"] == "bar"
    assert "figure_json" in res_bar
    assert len(res_bar["figure_json"]["data"]) == 1
    assert res_bar["metrics"]["aggregation_applied"] == 1
    pd.testing.assert_frame_equal(res_bar["dataframe"], df)

    # 2. Histogram with single column
    res_hist = recipe.execute({"dataframe": df}, {
        "chart_type": "histogram",
        "x_column": "Sales"
    })
    assert res_hist["chart_type"] == "histogram"
    assert res_hist["figure_json"]["data"][0]["type"] == "histogram"

    # 3. Scatter plot
    res_scatter = recipe.execute({"dataframe": df}, {
        "chart_type": "scatter",
        "x_column": "Category",
        "y_column": "Sales"
    })
    assert res_scatter["chart_type"] == "scatter"

    # 4. Box plot and Pie chart
    res_pie = recipe.execute({"dataframe": df}, {
        "chart_type": "pie",
        "x_column": "Category"
    })
    assert res_pie["chart_type"] == "pie"


def test_chart_visualization_edge_cases():
    recipe = recipe_registry.get("chart_visualization")

    # Edge Case 1: Empty DataFrame
    empty_df = pd.DataFrame()
    res_empty = recipe.execute({"dataframe": empty_df}, {"chart_type": "bar"})
    assert res_empty["output_summary"]["status"] == "EMPTY_DATA"
    assert res_empty["metrics"]["rows_evaluated"] == 0

    # Edge Case 2: Non-existent x_column and y_column fallback
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    res_fallback = recipe.execute({"dataframe": df}, {
        "x_column": "NON_EXISTENT_X",
        "y_column": "NON_EXISTENT_Y"
    })
    assert res_fallback["x_column"] in df.columns
    assert res_fallback["y_column"] in df.columns

    # Edge Case 3: Downsampling threshold
    large_df = pd.DataFrame({"x": range(500), "y": np.random.randn(500)})
    res_downsampled = recipe.execute({"dataframe": large_df}, {
        "x_column": "x",
        "y_column": "y",
        "max_points": 100
    })
    assert res_downsampled["metrics"]["downsampled"] == 1
    assert res_downsampled["metrics"]["points_rendered"] == 100


# =====================================================================
# 6. Version Snapshot Recipe Tests & Edge Cases
# =====================================================================

def test_version_snapshot_execution_and_fingerprinting():
    recipe = recipe_registry.get("version_snapshot")
    df = pd.DataFrame({"feature1": [1, 2], "target": [0, 1]})

    context_mock = {
        "nodes": [{"id": "n1", "recipe_id": "csv_loader"}],
        "edges": []
    }

    res = recipe.execute({"dataframe": df}, {
        "snapshot_name": "Milestone 1 Release",
        "tag": "v1.0.0",
        "include_data_fingerprint": True
    }, context=context_mock)

    assert res["snapshot_name"] == "Milestone 1 Release"
    assert res["tag"] == "v1.0.0"
    assert len(res["fingerprint"]) == 64  # SHA-256 hex length
    assert res["metrics"]["data_rows_fingerprinted"] == 2
    assert res["metrics"]["nodes_captured"] == 1
    assert "SHA-256 Fingerprint" in res["report"]["markdown"]
    pd.testing.assert_frame_equal(res["dataframe"], df)


# =====================================================================
# 7. Template Macro Library Recipe Tests & Edge Cases
# =====================================================================

def test_template_macro_library_execution():
    recipe = recipe_registry.get("template_macro_library")

    res = recipe.execute({}, {
        "template_name": "Tabular Classification Baseline",
        "target_pipeline_type": "tabular_classification",
        "parameter_overrides": json.dumps({
            "xgboost_trainer": {"n_estimators": 250, "max_depth": 8}
        })
    })

    assert res["template_name"] == "Tabular Classification Baseline"
    assert len(res["sub_dag_nodes"]) >= 5
    # Check that override was applied to the xgboost node
    xgb_node = next(n for n in res["sub_dag_nodes"] if n["recipe_id"] == "xgboost_trainer")
    assert xgb_node["default_config"]["n_estimators"] == 250
    assert xgb_node["default_config"]["max_depth"] == 8
    assert res["metrics"]["has_parameter_error"] == 0

    # Edge Case: Malformed JSON override
    res_bad_json = recipe.execute({}, {
        "parameter_overrides": "{invalid_json: 123"
    })
    assert res_bad_json["metrics"]["has_parameter_error"] == 1
    assert len(res_bad_json["sub_dag_nodes"]) > 0


# =====================================================================
# 8. Pipeline Diff Viewer Recipe Tests & Edge Cases
# =====================================================================

def test_pipeline_diff_viewer_execution():
    recipe = recipe_registry.get("pipeline_diff_viewer")

    base_pipe = {
        "nodes": [
            {"id": "node_impute", "recipe_id": "missing_value_imputer", "config": {"strategy": "mean"}},
            {"id": "node_train", "recipe_id": "random_forest_trainer", "config": {"n_estimators": 100}}
        ],
        "edges": [{"source": "node_impute", "target": "node_train"}]
    }

    cand_pipe = {
        "nodes": [
            {"id": "node_impute", "recipe_id": "missing_value_imputer", "config": {"strategy": "median"}},  # changed
            {"id": "node_scale", "recipe_id": "feature_scaler", "config": {"method": "standard"}},           # added
            {"id": "node_train", "recipe_id": "random_forest_trainer", "config": {"n_estimators": 200}}     # changed
        ],
        "edges": [
            {"source": "node_impute", "target": "node_scale"},
            {"source": "node_scale", "target": "node_train"}
        ]
    }

    res = recipe.execute({}, {
        "source_version_label": "v1.0 (Baseline)",
        "target_version_label": "v2.0 (Candidate)",
        "source_pipeline_json": json.dumps(base_pipe),
        "target_pipeline_json": json.dumps(cand_pipe)
    })

    assert res["diff_summary"]["nodes_added"] == 1
    assert "node_scale" in res["added_nodes"]
    assert res["diff_summary"]["nodes_removed"] == 0
    assert res["diff_summary"]["parameters_modified"] == 2  # strategy + n_estimators
    assert res["diff_summary"]["is_identical"] == 0

    # Edge Case: Identical pipelines
    res_identical = recipe.execute({}, {
        "source_pipeline_json": json.dumps(base_pipe),
        "target_pipeline_json": json.dumps(base_pipe)
    })
    assert res_identical["diff_summary"]["is_identical"] == 1
    assert "Identical Architectures" in res_identical["report"]["markdown"]


# =====================================================================
# 9. Readonly Pipeline Snapshot Recipe Tests & Edge Cases
# =====================================================================

def test_readonly_pipeline_snapshot_credential_redaction():
    recipe = recipe_registry.get("readonly_pipeline_snapshot")

    context_mock = {
        "nodes": [
            {"id": "db_node", "config": {"database_url": "postgresql://usr:MySecretPass123@db:5432/crm"}},
            {"id": "api_node", "config": {"api_key": "sk_live_998877665544332211"}},
            {"id": "trainer_node", "config": {"n_estimators": 100}}
        ]
    }

    res = recipe.execute({}, {
        "snapshot_title": "Executive Audit View",
        "mask_sensitive_credentials": True,
        "expiration_days": 14
    }, context=context_mock)

    assert "share_url" in res
    assert "viewer/share_" in res["share_url"]
    assert res["metrics"]["redacted_keys"] >= 1

    # Verify secrets are masked with asterisks in sanitized manifest
    sanitized = res["sanitized_pipeline_manifest"]
    api_cfg = next(n["config"] for n in sanitized["nodes"] if n["id"] == "api_node")
    assert api_cfg["api_key"] == "********"


# =====================================================================
# 10. Secrets Manager Block Recipe Tests & Edge Cases
# =====================================================================

def test_secrets_manager_block_execution_and_masking():
    recipe = recipe_registry.get("secrets_manager_block")

    # Set mock environment variables
    os.environ["TEST_SECRET_API_KEY"] = "super-secret-key-value-12345"
    os.environ["TEST_DB_PASSWORD"] = "my_secure_db_pass"

    res = recipe.execute({}, {
        "secret_keys": "TEST_SECRET_API_KEY, TEST_DB_PASSWORD, MISSING_VARIABLE_XYZ",
        "mask_in_logs": True,
        "allow_missing_keys": True
    })

    assert res["metrics"]["keys_requested"] == 3
    assert res["metrics"]["keys_resolved"] == 2
    assert res["metrics"]["keys_missing"] == 1
    assert "MISSING_VARIABLE_XYZ" in res["missing_keys"]

    # Verify injected secrets contains the real value for downstream usage
    assert res["_injected_secrets"]["TEST_SECRET_API_KEY"] == "super-secret-key-value-12345"

    # Verify masked preview does NOT expose the full plaintext secret
    api_entry = next(m for m in res["masked_manifest"] if m["key"] == "TEST_SECRET_API_KEY")
    assert "super-secret-key-value-12345" not in api_entry["preview"]
    assert "****" in api_entry["preview"]

    # Edge Case: Strict missing keys raises ValueError
    with pytest.raises(ValueError, match="Secrets validation failed"):
        recipe.execute({}, {
            "secret_keys": "DEFINITELY_NON_EXISTENT_KEY_ABC_123",
            "allow_missing_keys": False
        })


# =====================================================================
# 11. Parameter Sweep Runner Recipe Tests & Edge Cases
# =====================================================================

def test_parameter_sweep_runner_execution_and_strategies():
    recipe = recipe_registry.get("parameter_sweep_runner")

    # 1. Linear range strategy
    res_linear = recipe.execute({}, {
        "target_node_id": "xgboost_trainer_1",
        "parameter_name": "learning_rate",
        "sweep_strategy": "linear_range",
        "min_value": 0.01,
        "max_value": 0.2,
        "num_trials": 5,
        "optimization_metric": "f1_score"
    })
    assert len(res_linear["trials"]) == 5
    assert res_linear["metrics"]["total_trials"] == 5
    assert res_linear["best_trial"]["rank"] == 1
    assert "trial_results_table" in res_linear
    assert isinstance(res_linear["trial_results_table"], pd.DataFrame)

    # 2. Log scale strategy
    res_log = recipe.execute({}, {
        "parameter_name": "alpha",
        "sweep_strategy": "log_scale",
        "min_value": 0.001,
        "max_value": 1.0,
        "num_trials": 4
    })
    assert len(res_log["trials"]) == 4

    # 3. Discrete list strategy
    res_discrete = recipe.execute({}, {
        "parameter_name": "n_estimators",
        "sweep_strategy": "discrete_list",
        "discrete_values": "50, 100, 200, 500"
    })
    assert len(res_discrete["trials"]) == 4

    # Edge Case: Inverted bounds auto-correction (min > max)
    res_inverted = recipe.execute({}, {
        "sweep_strategy": "linear_range",
        "min_value": 0.5,
        "max_value": 0.1,
        "num_trials": 3
    })
    assert len(res_inverted["trials"]) == 3
    assert res_inverted["parameter_values"] == [0.1, 0.3, 0.5]
    assert all(0.1 <= t["value"] <= 0.5 for t in res_inverted["trials"])
