import pytest
import pandas as pd
from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.integrations.currency_exchange_api import CurrencyExchangeApiRecipe
from backend.app.engine.dag.graph import WorkflowGraph, WorkflowNode, WorkflowEdge
from backend.app.engine.execution.executor import DAGExecutor


def test_currency_exchange_api_registration():
    """Verifies that CurrencyExchangeApiRecipe is registered and active in the catalog."""
    recipe = recipe_registry.get("currency_exchange_api")
    assert recipe is not None
    assert recipe.recipe_id == "currency_exchange_api"
    assert recipe.name == "Currency Exchange Rate API"
    assert recipe.category == "integrations"
    assert recipe.group == "Apps & Integrations"
    assert recipe.subgroup == "Market & Reference Data"
    assert "currency exchange rates" in recipe.description.lower()

    # Check alias
    forex_recipe = recipe_registry.get("forex")
    assert forex_recipe is not None
    assert forex_recipe.recipe_id == "currency_exchange_api"

    meta = recipe_registry.get_metadata("currency_exchange_api")
    assert meta.enabled is True
    assert meta.recipe_id == "currency_exchange_api"
    assert meta.group == "Apps & Integrations"
    assert meta.subgroup == "Market & Reference Data"


def test_currency_exchange_api_schema():
    """Verifies parameter schema and input_type frontend properties, ensuring no guardrail triggers."""
    recipe = CurrencyExchangeApiRecipe()
    schema = recipe.get_schema()
    props = schema["properties"]

    assert "base_currency" in props
    assert props["base_currency"]["input_type"] == "dropdown"
    assert "target_currencies" in props
    assert props["target_currencies"]["input_type"] == "text"
    assert "operation_mode" in props
    assert props["operation_mode"]["input_type"] == "dropdown"
    assert "Latest Exchange Rates" in props["operation_mode"]["enum"]
    assert "Historical Rate Series" in props["operation_mode"]["enum"]
    assert "Convert Upstream Amounts" in props["operation_mode"]["enum"]
    assert "time_period" in props
    assert props["time_period"]["input_type"] == "dropdown"
    assert "amount_field" in props
    assert props["amount_field"]["input_type"] == "column_select"
    assert "source_currency_field" in props
    assert props["source_currency_field"]["input_type"] == "column_select"
    assert "data_source" in props
    assert props["data_source"]["input_type"] == "dropdown"

    # CRITICAL: Verify none of the optional property keys contain 'column', 'metric', or 'target'
    # to avoid triggering frontend pre-flight guardrails when left blank.
    for key in props:
        assert "column" not in key.lower(), f"Key '{key}' contains 'column' which triggers frontend preflight validation errors"
        assert "metric" not in key.lower(), f"Key '{key}' contains 'metric'"
        if key != "target_currencies":
            assert "target" not in key.lower(), f"Key '{key}' contains 'target'"


def test_currency_exchange_api_latest_rates():
    """Verifies latest currency exchange rates retrieval."""
    recipe = CurrencyExchangeApiRecipe()
    config = {
        "base_currency": "USD",
        "target_currencies": "EUR, GBP, JPY, CAD",
        "operation_mode": "Latest Exchange Rates",
        "data_source": "Simulated Central Bank FX Data"
    }
    result = recipe.execute(inputs={}, config=config)

    assert "dataframe" in result
    df = result["dataframe"]
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    assert "base_currency" in df.columns
    assert "quote_currency" in df.columns
    assert "currency_pair" in df.columns
    assert "rate" in df.columns
    assert "inverse_rate" in df.columns
    assert "last_updated" in df.columns

    quotes = set(df["quote_currency"].unique())
    assert quotes == {"EUR", "GBP", "JPY", "CAD"}
    assert all(df["base_currency"] == "USD")
    assert all(df["rate"] > 0)


def test_currency_exchange_api_historical_series():
    """Verifies historical daily exchange rate time series with moving averages."""
    recipe = CurrencyExchangeApiRecipe()
    config = {
        "base_currency": "USD",
        "target_currencies": "EUR",
        "operation_mode": "Historical Rate Series",
        "historical_days": 30,
        "data_source": "Simulated Central Bank FX Data"
    }
    result = recipe.execute(inputs={}, config=config)

    assert "dataframe" in result
    df = result["dataframe"]
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 30
    assert "date" in df.columns
    assert "currency_pair" in df.columns
    assert "rate" in df.columns
    assert "high" in df.columns
    assert "low" in df.columns
    assert "daily_change_pct" in df.columns
    assert "sma_20" in df.columns
    assert "volatility_20d" in df.columns


def test_currency_exchange_api_convert_upstream_amounts():
    """Verifies converting financial amounts in an upstream table using FX rates."""
    recipe = CurrencyExchangeApiRecipe()
    upstream_df = pd.DataFrame({
        "invoice_id": ["INV-001", "INV-002", "INV-003"],
        "total_cost": [100.0, 250.50, 50.0],
        "invoice_curr": ["EUR", "GBP", "JPY"]
    })

    # Test converting to USD
    config = {
        "operation_mode": "Convert Upstream Amounts",
        "amount_field": "total_cost",
        "source_currency_field": "invoice_curr",
        "base_currency": "USD",
        "data_source": "Simulated Central Bank FX Data"
    }
    result = recipe.execute(inputs={"dataframe": upstream_df}, config=config)

    assert "dataframe" in result
    df = result["dataframe"]
    assert len(df) == 3
    assert "converted_amount" in df.columns
    assert "fx_rate_applied" in df.columns
    assert "fx_base_currency" in df.columns
    assert "fx_target_currency" in df.columns
    assert all(df["converted_amount"] > 0)


def test_currency_exchange_api_dag_pipeline_execution():
    """Verifies that currency_exchange_api executes cleanly in an end-to-end DAG workflow."""
    nodes = [
        WorkflowNode(
            id="node_fx",
            recipe_id="currency_exchange_api",
            config={
                "base_currency": "USD",
                "target_currencies": "EUR, GBP, JPY, AUD, CAD, CHF, INR",
                "operation_mode": "Latest Exchange Rates",
                "data_source": "Simulated Central Bank FX Data"
            }
        ),
        WorkflowNode(
            id="node_limit",
            recipe_id="limit",
            config={"rows": 5}
        )
    ]
    edges = [
        WorkflowEdge(
            source="node_fx",
            target="node_limit",
            source_handle="dataframe",
            target_handle="dataframe"
        )
    ]
    workflow = WorkflowGraph(nodes=nodes, edges=edges)
    exec_res = DAGExecutor.execute_workflow(execution_id="test_fx_pipeline", workflow=workflow)

    assert exec_res.status == "SUCCESS"
    assert len(exec_res.node_results) == 2
    fx_res = [nr for nr in exec_res.node_results if nr.node_id == "node_fx"][0]
    limit_res = [nr for nr in exec_res.node_results if nr.node_id == "node_limit"][0]
    assert fx_res.status == "SUCCESS"
    assert limit_res.status == "SUCCESS"
