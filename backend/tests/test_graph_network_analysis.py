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
import networkx as nx

from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.training.graph_network_analysis import GraphNetworkAnalysisRecipe


def test_graph_network_analysis_registry_and_aliases():
    """Verify GraphNetworkAnalysisRecipe is registered with correct metadata and aliases."""
    assert recipe_registry.has("graph_network_analysis") is True
    meta = recipe_registry.get_metadata("graph_network_analysis")
    assert meta.enabled is True
    assert meta.group == "Modeling"
    assert meta.subgroup == "Specialized Models"

    # Aliases
    assert recipe_registry.get("graph_analysis").recipe_id == "graph_network_analysis"
    assert recipe_registry.get("network_analysis").recipe_id == "graph_network_analysis"
    assert recipe_registry.get("networkx").recipe_id == "graph_network_analysis"
    assert recipe_registry.get("pagerank").recipe_id == "graph_network_analysis"
    assert recipe_registry.get("community_detection").recipe_id == "graph_network_analysis"
    assert recipe_registry.get("graph_centrality").recipe_id == "graph_network_analysis"
    assert recipe_registry.get("graph_analytics").recipe_id == "graph_network_analysis"


def test_undirected_graph_centrality_and_louvain_communities():
    """Verify PageRank, betweenness, closeness, and Louvain community detection on network."""
    # Synthetic network with 2 distinct dense clusters connected by a bridge node (Node_E)
    edges = [
        # Cluster 1: A, B, C, D
        {"src": "Node_A", "dst": "Node_B"},
        {"src": "Node_A", "dst": "Node_C"},
        {"src": "Node_B", "dst": "Node_C"},
        {"src": "Node_C", "dst": "Node_D"},
        # Bridge to Node_E
        {"src": "Node_D", "dst": "Node_E"},
        # Bridge from Node_E to Cluster 2: F, G, H, I
        {"src": "Node_E", "dst": "Node_F"},
        {"src": "Node_F", "dst": "Node_G"},
        {"src": "Node_F", "dst": "Node_H"},
        {"src": "Node_G", "dst": "Node_H"},
        {"src": "Node_H", "dst": "Node_I"},
    ]
    df = pd.DataFrame(edges)

    recipe = GraphNetworkAnalysisRecipe()
    result = recipe.execute(
        inputs={"edges": df},
        config={
            "source_col": "src",
            "target_col": "dst",
            "is_directed": False,
            "community_algorithm": "louvain",
            "top_k_influencers": 3
        }
    )

    G = result["graph"]
    nodes_df = result["nodes"]
    metrics = result["metrics"]
    report = result["report"]

    assert isinstance(G, nx.Graph)
    assert not nodes_df.empty
    assert "pagerank" in nodes_df.columns
    assert "betweenness_centrality" in nodes_df.columns
    assert "closeness_centrality" in nodes_df.columns
    assert "community_id" in nodes_df.columns

    # Verify Node_E has high betweenness centrality (bridge between clusters)
    e_betweenness = nodes_df[nodes_df["node_id"] == "Node_E"]["betweenness_centrality"].iloc[0]
    assert e_betweenness > 0.3

    assert metrics["total_nodes"] == 9
    assert metrics["total_edges"] == 10
    assert metrics["communities_detected"] >= 2
    assert len(metrics["top_pagerank_hubs"]) == 3
    assert len(metrics["top_betweenness_bridges"]) == 3

    # Report verification
    assert "# === Graph & Network Analysis Report" in report
    assert "Top 3 Network Hubs (Ranked by PageRank Influence):" in report
    assert "Top 3 Information Brokers & Bridges" in report


def test_directed_graph_with_weights():
    """Verify directed DiGraph construction with edge weights and weighted PageRank."""
    edges = [
        {"source": "Account_A", "target": "Account_B", "amount": 5000.0},
        {"source": "Account_B", "target": "Account_C", "amount": 4800.0},
        {"source": "Account_C", "target": "Account_D", "amount": 4500.0},
        {"source": "Account_D", "target": "Account_A", "amount": 4200.0},  # Circular flow
        {"source": "Account_E", "target": "Account_A", "amount": 10000.0}, # Inflow source
    ]
    df = pd.DataFrame(edges)

    recipe = GraphNetworkAnalysisRecipe()
    result = recipe.execute(
        inputs={"dataframe": df},
        config={
            "is_directed": True,
            "weight_col": "amount",
            "top_k_influencers": 2
        }
    )

    G = result["graph"]
    metrics = result["metrics"]

    assert isinstance(G, nx.DiGraph)
    assert metrics["network_type"] == "Directed (DiGraph)"
    assert metrics["total_nodes"] == 5
    assert metrics["total_edges"] == 5
    # Account_A has highest inflow -> highest PageRank
    assert metrics["top_pagerank_hubs"][0]["node_id"] == "Account_A"


def test_greedy_modularity_community_detection():
    """Verify greedy modularity community detection runs and assigns clusters."""
    # 2 disjoint triangles connected by a single edge
    edges = [
        {"source": "1", "target": "2"},
        {"source": "2", "target": "3"},
        {"source": "3", "target": "1"},
        {"source": "3", "target": "4"},
        {"source": "4", "target": "5"},
        {"source": "5", "target": "6"},
        {"source": "6", "target": "4"},
    ]
    df = pd.DataFrame(edges)

    recipe = GraphNetworkAnalysisRecipe()
    result = recipe.execute(
        inputs={"dataframe": df},
        config={"community_algorithm": "greedy_modularity"}
    )

    metrics = result["metrics"]
    assert metrics["communities_detected"] >= 2
    assert metrics["modularity_score"] is not None
    assert metrics["modularity_score"] > 0.0


def test_disconnected_graph_components():
    """Verify disconnected subgraphs (components > 1) compute metrics without errors."""
    edges = [
        # Component 1
        {"source": "U1", "target": "U2"},
        {"source": "U2", "target": "U3"},
        # Component 2
        {"source": "U4", "target": "U5"},
    ]
    df = pd.DataFrame(edges)

    recipe = GraphNetworkAnalysisRecipe()
    result = recipe.execute(inputs={"dataframe": df}, config={})
    assert result["metrics"]["connected_components"] == 2
    assert result["metrics"]["total_nodes"] == 5


def test_graph_network_analysis_edge_cases():
    """Verify guardrails: missing columns, single node, empty dataframe."""
    recipe = GraphNetworkAnalysisRecipe()

    # Guard 1: Missing source column
    df_no_src = pd.DataFrame({"target": ["A", "B"]})
    with pytest.raises(ValueError, match="could not locate source column"):
        recipe.execute(inputs={"dataframe": df_no_src}, config={"source_col": "missing_src"})

    # Guard 2: Fewer than 2 distinct nodes
    df_self_loop = pd.DataFrame({"source": ["A"], "target": ["A"]})
    with pytest.raises(ValueError, match="requires at least 2 distinct nodes"):
        recipe.execute(inputs={"dataframe": df_self_loop}, config={})

    # Guard 3: Empty table
    df_empty = pd.DataFrame(columns=["source", "target"])
    with pytest.raises(ValueError, match="expects a non-empty pandas DataFrame"):
        recipe.execute(inputs={"dataframe": df_empty}, config={})


if __name__ == "__main__":
    print("Running test_graph_network_analysis_registry_and_aliases...")
    test_graph_network_analysis_registry_and_aliases()
    print("PASS: test_graph_network_analysis_registry_and_aliases")

    print("Running test_undirected_graph_centrality_and_louvain_communities...")
    test_undirected_graph_centrality_and_louvain_communities()
    print("PASS: test_undirected_graph_centrality_and_louvain_communities")

    print("Running test_directed_graph_with_weights...")
    test_directed_graph_with_weights()
    print("PASS: test_directed_graph_with_weights")

    print("Running test_greedy_modularity_community_detection...")
    test_greedy_modularity_community_detection()
    print("PASS: test_greedy_modularity_community_detection")

    print("Running test_disconnected_graph_components...")
    test_disconnected_graph_components()
    print("PASS: test_disconnected_graph_components")

    print("Running test_graph_network_analysis_edge_cases...")
    test_graph_network_analysis_edge_cases()
    print("PASS: test_graph_network_analysis_edge_cases")

    print("\nALL 6 GRAPH NETWORK ANALYSIS TESTS PASSED SUCCESSFULLY!")
