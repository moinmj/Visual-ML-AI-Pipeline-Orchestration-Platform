import pytest
import pandas as pd
from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.integrations.news_api import NewsApiRecipe
from backend.app.engine.dag.graph import WorkflowGraph, WorkflowNode, WorkflowEdge
from backend.app.engine.execution.executor import DAGExecutor


def test_news_api_registration():
    """Verifies that NewsApiRecipe is registered and active in the catalog."""
    recipe = recipe_registry.get("news_api")
    assert recipe is not None
    assert recipe.recipe_id == "news_api"
    assert recipe.name == "News API Feed"
    assert recipe.category == "integrations"
    assert recipe.group == "Apps & Integrations"
    assert recipe.subgroup == "Market & Reference Data"
    assert "news articles" in recipe.description.lower()

    # Check aliases
    news_alias = recipe_registry.get("news")
    assert news_alias is not None
    assert news_alias.recipe_id == "news_api"

    headlines_alias = recipe_registry.get("headlines")
    assert headlines_alias is not None
    assert headlines_alias.recipe_id == "news_api"

    meta = recipe_registry.get_metadata("news_api")
    assert meta.enabled is True
    assert meta.recipe_id == "news_api"
    assert meta.group == "Apps & Integrations"
    assert meta.subgroup == "Market & Reference Data"


def test_news_api_schema():
    """Verifies parameter schema and input_type frontend properties, ensuring no guardrail triggers."""
    recipe = NewsApiRecipe()
    schema = recipe.get_schema()
    props = schema["properties"]

    assert "search_query" in props
    assert props["search_query"]["input_type"] == "text"
    assert "category" in props
    assert props["category"]["input_type"] == "dropdown"
    assert "technology" in props["category"]["enum"]
    assert "business" in props["category"]["enum"]
    assert "finance" in props["category"]["enum"]
    assert "max_articles" in props
    assert props["max_articles"]["input_type"] == "number"
    assert "data_provider" in props
    assert props["data_provider"]["input_type"] == "dropdown"
    assert "query_field" in props
    assert props["query_field"]["input_type"] == "column_select"
    assert "api_key" in props
    assert props["api_key"]["input_type"] == "text"

    # CRITICAL: Verify none of the optional property keys contain 'column', 'metric', or 'target'
    # to avoid triggering frontend pre-flight guardrails when left blank.
    for key in props:
        assert "column" not in key.lower(), f"Key '{key}' contains 'column' which triggers frontend preflight validation errors"
        assert "metric" not in key.lower(), f"Key '{key}' contains 'metric'"
        assert "target" not in key.lower(), f"Key '{key}' contains 'target'"


def test_news_api_simulated_execution():
    """Verifies simulated realistic news article generation."""
    recipe = NewsApiRecipe()
    config = {
        "search_query": "Artificial Intelligence, Quantum",
        "category": "technology",
        "max_articles": 6,
        "data_provider": "Simulated Realistic News Feed"
    }
    result = recipe.execute(inputs={}, config=config)

    assert "dataframe" in result
    df = result["dataframe"]
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "title" in df.columns
    assert "description" in df.columns
    assert "content" in df.columns
    assert "author" in df.columns
    assert "source_name" in df.columns
    assert "url" in df.columns
    assert "published_at" in df.columns
    assert "category" in df.columns
    assert "keyword" in df.columns

    assert "output_summary" in result
    summary = result["output_summary"]
    assert summary["article_count"] > 0
    assert len(summary["preview"]) > 0


def test_news_api_live_execution():
    """Verifies live or resilient fallback article fetching."""
    recipe = NewsApiRecipe()
    config = {
        "search_query": "AI",
        "category": "technology",
        "max_articles": 5,
        "data_provider": "Live News / Public Aggregators"
    }
    result = recipe.execute(inputs={}, config=config)

    assert "dataframe" in result
    df = result["dataframe"]
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df) > 0
    assert "title" in df.columns
    assert "source_name" in df.columns


def test_news_api_upstream_keywords():
    """Verifies extracting search queries from upstream table column."""
    recipe = NewsApiRecipe()
    upstream_df = pd.DataFrame({
        "topic_name": ["Semiconductors", "Cybersecurity"],
        "priority": [1, 2]
    })
    config = {
        "query_field": "topic_name",
        "category": "technology",
        "max_articles": 3,
        "data_provider": "Simulated Realistic News Feed"
    }
    result = recipe.execute(inputs={"dataframe": upstream_df}, config=config)

    assert "dataframe" in result
    df = result["dataframe"]
    assert not df.empty
    keywords_found = set(df["keyword"].unique())
    assert "Semiconductors" in keywords_found or "Cybersecurity" in keywords_found


def test_news_api_dag_pipeline_execution():
    """Verifies that news_api executes cleanly in an end-to-end DAG workflow."""
    nodes = [
        WorkflowNode(
            id="node_news",
            recipe_id="news_api",
            config={
                "search_query": "Global Markets",
                "category": "business",
                "max_articles": 8,
                "data_provider": "Simulated Realistic News Feed"
            }
        ),
        WorkflowNode(
            id="node_limit",
            recipe_id="limit",
            config={"rows": 4}
        )
    ]
    edges = [
        WorkflowEdge(
            source="node_news",
            target="node_limit",
            source_handle="dataframe",
            target_handle="dataframe"
        )
    ]
    workflow = WorkflowGraph(nodes=nodes, edges=edges)
    exec_res = DAGExecutor.execute_workflow(execution_id="test_news_pipeline", workflow=workflow)

    assert exec_res.status == "SUCCESS"
    assert len(exec_res.node_results) == 2
    news_res = [nr for nr in exec_res.node_results if nr.node_id == "node_news"][0]
    limit_res = [nr for nr in exec_res.node_results if nr.node_id == "node_limit"][0]
    assert news_res.status == "SUCCESS"
    assert limit_res.status == "SUCCESS"
