import pytest
import pandas as pd
from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.flow_control.if_condition import IfConditionRecipe
from backend.app.recipes.flow_control.row_filter import RowFilterRecipe
from backend.app.recipes.flow_control.switch_node import SwitchRecipe
from backend.app.recipes.flow_control.merge_node import MergeDatasetsRecipe
from backend.app.recipes.flow_control.loop_node import LoopBatchRecipe
from backend.app.recipes.flow_control.delay_node import DelayRecipe
from backend.app.recipes.integrations.slack_notifier import SlackRecipe
from backend.app.recipes.integrations.discord_notifier import DiscordRecipe
from backend.app.recipes.integrations.telegram_notifier import TelegramRecipe
from backend.app.recipes.integrations.gmail_notifier import GmailRecipe
from backend.app.recipes.integrations.google_sheets import GoogleSheetsRecipe
from backend.app.recipes.integrations.google_drive import GoogleDriveRecipe
from backend.app.recipes.integrations.openweathermap import OpenWeatherMapRecipe
from backend.app.engine.dag.graph import WorkflowGraph, WorkflowNode, WorkflowEdge
from backend.app.engine.execution.executor import DAGExecutor


def test_registry_has_all_new_flow_control_and_integrations():
    expected_ids = [
        "if_condition", "row_filter", "switch", "merge", "loop", "delay",
        "slack", "discord", "telegram", "gmail", "google_sheets", "google_drive", "openweathermap"
    ]
    for r_id in expected_ids:
        assert recipe_registry.has(r_id), f"Missing recipe '{r_id}' in recipe_registry"
        rec = recipe_registry.get(r_id)
        meta = rec.to_metadata()
        assert len(meta.outputs) > 0, f"Recipe '{r_id}' has no output ports defined"

    # Test aliases
    assert recipe_registry.get("if_else").recipe_id == "if_condition"
    assert recipe_registry.get("filter").recipe_id == "row_filter"
    assert recipe_registry.get("wait").recipe_id == "delay"
    assert recipe_registry.get("email").recipe_id == "gmail"
    assert recipe_registry.get("weather").recipe_id == "openweathermap"


def test_if_condition_recipe_branching():
    recipe = IfConditionRecipe()
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "status": ["ACTIVE", "CHURNED", "ACTIVE", "ACTIVE", "CHURNED"],
        "amount": [150.0, 50.0, 300.0, 80.0, 10.0]
    })
    config = {
        "combine_with": "AND",
        "conditions": [
            {"column": "status", "operator": "==", "value": "ACTIVE"},
            {"column": "amount", "operator": ">=", "value": "100"}
        ]
    }
    res = recipe.execute(inputs={"dataframe": df}, config=config)
    df_true = res["true"]
    df_false = res["false"]

    # IDs 1 and 3 match status==ACTIVE and amount>=100
    assert len(df_true) == 2
    assert set(df_true["id"]) == {1, 3}
    assert len(df_false) == 3
    assert set(df_false["id"]) == {2, 4, 5}
    assert res["metrics"]["true_rows"] == 2
    assert res["metrics"]["false_rows"] == 3


def test_row_filter_recipe():
    recipe = RowFilterRecipe()
    df = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie", "David"],
        "age": [25, 17, 30, 15]
    })
    # Keep adults (age >= 18)
    config = {
        "action": "keep",
        "conditions": [{"column": "age", "operator": ">=", "value": "18"}]
    }
    res = recipe.execute(inputs={"dataframe": df}, config=config)
    df_out = res["dataframe"]
    assert len(df_out) == 2
    assert set(df_out["name"]) == {"Alice", "Charlie"}


def test_switch_recipe_multi_branching():
    recipe = SwitchRecipe()
    df = pd.DataFrame({
        "tier": ["VIP", "Standard", "Free", "VIP", "Unknown"],
        "score": [95, 70, 40, 99, 10]
    })
    config = {
        "case_1_conditions": [{"column": "tier", "operator": "==", "value": "VIP"}],
        "case_2_conditions": [{"column": "tier", "operator": "==", "value": "Standard"}],
        "case_3_conditions": [{"column": "tier", "operator": "==", "value": "Free"}]
    }
    res = recipe.execute(inputs={"dataframe": df}, config=config)
    assert len(res["case_1"]) == 2  # VIP
    assert len(res["case_2"]) == 1  # Standard
    assert len(res["case_3"]) == 1  # Free
    assert len(res["default"]) == 1  # Unknown


def test_merge_union_recipe():
    recipe = MergeDatasetsRecipe()
    df1 = pd.DataFrame({"id": [1, 2], "val": ["A", "B"]})
    df2 = pd.DataFrame({"id": [3, 4], "val": ["C", "D"]})

    res = recipe.execute(
        inputs={"left_dataframe": df1, "right_dataframe": df2},
        config={"mode": "union"}
    )
    df_merged = res["dataframe"]
    assert len(df_merged) == 4
    assert list(df_merged["id"]) == [1, 2, 3, 4]


def test_delay_recipe():
    recipe = DelayRecipe()
    df = pd.DataFrame({"a": [1, 2]})
    res = recipe.execute(inputs={"dataframe": df}, config={"duration": 0.05, "unit": "seconds"})
    assert "delayed_seconds" in res["metrics"]
    assert res["metrics"]["delayed_seconds"] >= 0.05
    assert len(res["dataframe"]) == 2


def test_loop_batch_recipe():
    recipe = LoopBatchRecipe()
    df = pd.DataFrame({"id": list(range(250))})
    res = recipe.execute(inputs={"dataframe": df}, config={"batch_size": 50, "max_rows": 200})
    assert len(res["batch_output"]) == 50
    assert len(res["completed"]) == 200


def test_slack_recipe_simulation():
    recipe = SlackRecipe()
    df = pd.DataFrame({"col": [1, 2, 3]})
    res = recipe.execute(inputs={"dataframe": df}, config={"message": "Test alert"})
    assert res["metrics"]["delivery_status"] == "SIMULATED"
    assert "Slack Notification" in res["output_summary"]["title"]


def test_discord_recipe_simulation():
    recipe = DiscordRecipe()
    df = pd.DataFrame({"col": [1, 2, 3]})
    res = recipe.execute(inputs={"dataframe": df}, config={"content": "Discord update"})
    assert res["metrics"]["delivery_status"] == "SIMULATED"


def test_telegram_recipe_simulation():
    recipe = TelegramRecipe()
    df = pd.DataFrame({"col": [1, 2, 3]})
    res = recipe.execute(inputs={"dataframe": df}, config={"message": "Telegram message"})
    assert res["metrics"]["delivery_status"] == "SIMULATED"


def test_gmail_recipe_simulation():
    recipe = GmailRecipe()
    df = pd.DataFrame({"col": [1, 2, 3]})
    res = recipe.execute(inputs={"dataframe": df}, config={"to_email": "test@test.com", "subject": "Test", "body": "Body"})
    assert res["metrics"]["delivery_status"] == "SIMULATED"


def test_google_sheets_recipe_simulation():
    recipe = GoogleSheetsRecipe()
    res = recipe.execute(inputs={}, config={"operation": "read"})
    assert len(res["dataframe"]) > 0
    assert "Google Sheets" in res["output_summary"]["title"]


def test_google_drive_recipe_simulation():
    recipe = GoogleDriveRecipe()
    df = pd.DataFrame({"col": [1, 2, 3]})
    res = recipe.execute(inputs={"dataframe": df}, config={"operation": "upload", "file_name": "out.csv"})
    assert res["metrics"]["status"] == "SUCCESS"


def test_openweathermap_recipe():
    recipe = OpenWeatherMapRecipe()
    res = recipe.execute(inputs={}, config={"city": "London"})
    df = res["dataframe"]
    assert len(df) == 1
    assert "temperature" in df.columns
    assert "humidity" in df.columns
    assert df["city"].iloc[0] == "London"


def test_dag_executor_with_if_condition_and_handle_routing():
    # Ingestion -> IF Condition (split true / false) -> True branch to Scaler, False branch to Missing Imputer
    df_data = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "category": ["A", "B", "A", "B"],
        "val": [10.0, 20.0, 30.0, 40.0]
    })

    node_loader = WorkflowNode(id="n_load", recipe_id="csv_loader", config={"dataframe": df_data})
    node_if = WorkflowNode(id="n_if", recipe_id="if_condition", config={
        "conditions": [{"column": "category", "operator": "==", "value": "A"}]
    })
    node_scaler = WorkflowNode(id="n_scaler", recipe_id="feature_scaler", config={"columns": ["val"]})
    node_imputer = WorkflowNode(id="n_imputer", recipe_id="missing_value_imputer", config={"strategy": "median"})

    # Edges with source_handle routing: "true" to scaler, "false" to imputer
    edge1 = WorkflowEdge(source="n_load", target="n_if")
    edge_true = WorkflowEdge(source="n_if", target="n_scaler", source_handle="true")
    edge_false = WorkflowEdge(source="n_if", target="n_imputer", source_handle="false")

    graph = WorkflowGraph(
        nodes=[node_loader, node_if, node_scaler, node_imputer],
        edges=[edge1, edge_true, edge_false]
    )

    res = DAGExecutor.execute_workflow(execution_id="test_if_dag", workflow=graph)
    assert res.status == "SUCCESS"

    results_map = {nr.node_id: nr for nr in res.node_results}
    # True branch (Category == A) has 2 rows
    assert results_map["n_scaler"].output_summary["dataframe"]["shape"][0] == 2
    # False branch (Category == B) has 2 rows
    assert results_map["n_imputer"].output_summary["dataframe"]["shape"][0] == 2
