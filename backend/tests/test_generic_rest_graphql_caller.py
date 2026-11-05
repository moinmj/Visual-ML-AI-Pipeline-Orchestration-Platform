import pytest
import pandas as pd
from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.integrations.rest_graphql_caller import GenericRestGraphQLCallerRecipe


def test_generic_rest_graphql_caller_registration():
    recipe = recipe_registry.get("rest_graphql_caller")
    assert recipe is not None
    assert isinstance(recipe, GenericRestGraphQLCallerRecipe)
    assert recipe.recipe_id == "rest_graphql_caller"
    assert recipe.name == "Generic REST/GraphQL API Caller"
    assert recipe.category == "integrations"
    assert recipe.group == "Apps & Integrations"
    assert recipe.subgroup == "Generic"

    # Aliases
    assert recipe_registry.get("generic_api") == recipe
    assert recipe_registry.get("rest_api") == recipe
    assert recipe_registry.get("graphql_api") == recipe
    assert recipe_registry.get("rest_graphql") == recipe


def test_generic_rest_graphql_caller_schema():
    recipe = recipe_registry.get("rest_graphql_caller")
    schema = recipe.get_schema()
    assert "properties" in schema
    props = schema["properties"]
    
    assert "api_type" in props
    assert "method" in props
    assert "endpoint_url" in props
    assert "auth_type" in props
    assert "response_data_path" in props
    assert "simulation_mode" in props

    for k, v in props.items():
        assert "input_type" in v, f"Property {k} missing input_type"
        assert "description" in v, f"Property {k} missing description"


def test_generic_rest_graphql_caller_rest_execution():
    recipe = recipe_registry.get("rest_graphql_caller")
    config = {
        "api_type": "REST",
        "method": "GET",
        "endpoint_url": "https://jsonplaceholder.typicode.com/posts",
        "response_data_path": "",
        "record_limit": 10,
        "simulation_mode": True
    }
    result = recipe.execute({}, config)
    assert "dataframe" in result
    assert "output" in result
    assert "output_summary" in result
    assert "metrics" in result

    df = result["dataframe"]
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10
    assert "id" in df.columns
    assert "title" in df.columns
    assert "status" in df.columns
    assert result["metrics"]["api_type"] == "REST"


def test_generic_rest_graphql_caller_graphql_execution():
    recipe = recipe_registry.get("rest_graphql_caller")
    config = {
        "api_type": "GraphQL",
        "endpoint_url": "https://api.mock.graphql/v1",
        "graphql_query": "query { items { id name status } }",
        "record_limit": 15,
        "simulation_mode": True
    }
    result = recipe.execute({}, config)
    assert "dataframe" in result
    df = result["dataframe"]
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 15
    assert "id" in df.columns
    assert "node_type" in df.columns
    assert result["metrics"]["api_type"] == "GRAPHQL"


def test_generic_rest_graphql_caller_to_code():
    recipe = recipe_registry.get("rest_graphql_caller")
    code_rest = recipe.to_code({"api_type": "REST", "endpoint_url": "https://api.example.com/v1", "method": "GET"})
    assert "requests.get" in code_rest
    assert "pd.json_normalize" in code_rest

    code_gql = recipe.to_code({"api_type": "GraphQL", "endpoint_url": "https://api.example.com/graphql", "graphql_query": "{ users { id } }"})
    assert "requests.post" in code_gql
    assert "query" in code_gql

