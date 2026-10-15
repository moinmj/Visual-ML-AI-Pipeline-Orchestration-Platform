import pytest
import pandas as pd
from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.integrations.stock_market_api import StockMarketApiRecipe
from backend.app.engine.dag.graph import WorkflowGraph, WorkflowNode, WorkflowEdge
from backend.app.engine.execution.executor import DAGExecutor


def test_stock_market_api_registration():
    """Verifies that StockMarketApiRecipe is registered and active in the catalog."""
    recipe = recipe_registry.get("stock_market_api")
    assert recipe is not None
    assert recipe.recipe_id == "stock_market_api"
    assert recipe.name == "Stock / Finance Market Data API"
    assert recipe.category == "integrations"
    assert recipe.group == "Apps & Integrations"
    assert recipe.subgroup == "Market & Reference Data"
    assert "stock prices" in recipe.description.lower()

    meta = recipe_registry.get_metadata("stock_market_api")
    assert meta.enabled is True
    assert meta.recipe_id == "stock_market_api"
    assert meta.group == "Apps & Integrations"
    assert meta.subgroup == "Market & Reference Data"


def test_stock_market_api_schema():
    """Verifies parameter schema and input_type frontend properties."""
    recipe = StockMarketApiRecipe()
    schema = recipe.get_schema()
    props = schema["properties"]

    assert "symbols" in props
    assert props["symbols"]["input_type"] == "text"
    assert "data_type" in props
    assert props["data_type"]["input_type"] == "dropdown"
    assert "Historical OHLCV" in props["data_type"]["enum"]
    assert "Latest Real-Time Quote" in props["data_type"]["enum"]
    assert "time_period" in props
    assert props["time_period"]["input_type"] == "dropdown"
    assert "interval" in props
    assert "data_source" in props
    assert "symbol_field" in props
    assert props["symbol_field"]["input_type"] == "column_select"
    assert "include_indicators" in props
    assert props["include_indicators"]["input_type"] == "checkbox"


def test_stock_market_api_historical_ohlcv_generation():
    """Verifies historical OHLCV data retrieval with technical indicators."""
    recipe = StockMarketApiRecipe()
    config = {
        "symbols": "AAPL, MSFT",
        "data_type": "Historical OHLCV",
        "time_period": "1mo",
        "interval": "1d",
        "data_source": "Simulated Realistic Market Data",
        "include_indicators": True
    }
    result = recipe.execute(inputs={}, config=config)

    assert "dataframe" in result
    df = result["dataframe"]
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "date" in df.columns
    assert "symbol" in df.columns
    assert "open" in df.columns
    assert "high" in df.columns
    assert "low" in df.columns
    assert "close" in df.columns
    assert "volume" in df.columns
    assert "daily_return_pct" in df.columns
    assert "sma_20" in df.columns

    symbols_found = set(df["symbol"].unique())
    assert "AAPL" in symbols_found
    assert "MSFT" in symbols_found


def test_stock_market_api_realtime_quotes():
    """Verifies latest real-time market quote retrieval."""
    recipe = StockMarketApiRecipe()
    config = {
        "symbols": "NVDA, TSLA, SPY",
        "data_type": "Latest Real-Time Quote"
    }
    result = recipe.execute(inputs={}, config=config)

    assert "dataframe" in result
    df = result["dataframe"]
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert set(df["symbol"].unique()) == {"NVDA", "TSLA", "SPY"}
    assert "price" in df.columns
    assert "change" in df.columns
    assert "change_percent" in df.columns
    assert "volume_24h" in df.columns


def test_stock_market_api_company_overview():
    """Verifies company overview and fundamental indicators."""
    recipe = StockMarketApiRecipe()
    config = {
        "symbols": "GOOGL, AMZN",
        "data_type": "Company Overview / Fundamentals"
    }
    result = recipe.execute(inputs={}, config=config)

    df = result["dataframe"]
    assert len(df) == 2
    assert "company_name" in df.columns
    assert "sector" in df.columns
    assert "pe_ratio" in df.columns
    assert "52_week_high" in df.columns


def test_stock_market_api_upstream_symbol_extraction():
    """Verifies extracting ticker symbols from an upstream DataFrame column."""
    recipe = StockMarketApiRecipe()
    upstream_df = pd.DataFrame({
        "ticker_code": ["META", "AAPL"],
        "priority": [1, 2]
    })
    config = {
        "symbol_field": "ticker_code",
        "data_type": "Latest Real-Time Quote"
    }
    result = recipe.execute(inputs={"dataframe": upstream_df}, config=config)

    df = result["dataframe"]
    assert len(df) == 2
    assert set(df["symbol"].unique()) == {"META", "AAPL"}


def test_stock_market_api_dag_pipeline_execution():
    """Verifies that stock_market_api executes end-to-end in a DAG workflow with downstream prep nodes."""
    nodes = [
        WorkflowNode(
            id="node_market",
            recipe_id="stock_market_api",
            config={
                "symbols": "AAPL, MSFT",
                "data_type": "Historical OHLCV",
                "time_period": "1mo",
                "data_source": "Simulated Realistic Market Data"
            }
        ),
        WorkflowNode(
            id="node_limit",
            recipe_id="limit",
            config={"rows": 10}
        )
    ]
    edges = [
        WorkflowEdge(
            source="node_market",
            target="node_limit",
            source_handle="dataframe",
            target_handle="dataframe"
        )
    ]
    workflow = WorkflowGraph(nodes=nodes, edges=edges)
    exec_res = DAGExecutor.execute_workflow(execution_id="test_stock_pipeline", workflow=workflow)

    assert exec_res.status == "SUCCESS"
    assert len(exec_res.node_results) == 2
    market_res = [nr for nr in exec_res.node_results if nr.node_id == "node_market"][0]
    limit_res = [nr for nr in exec_res.node_results if nr.node_id == "node_limit"][0]
    assert market_res.status == "SUCCESS"
    assert limit_res.status == "SUCCESS"
