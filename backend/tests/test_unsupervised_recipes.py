import pytest
import pandas as pd
import numpy as np

from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.unsupervised.clustering import ClusteringRecipe
from backend.app.recipes.unsupervised.pca import PCARecipe
from backend.app.recipes.unsupervised.umap import UMAPRecipe
from backend.app.recipes.unsupervised.tsne import TSNERecipe
from backend.app.recipes.unsupervised.association_rules import AssociationRuleMiningRecipe


# =====================================================================
# 1. REGISTRY & METADATA TESTS
# =====================================================================

def test_recipes_registered():
    """Verify all 5 unsupervised recipes are registered and enabled."""
    expected_ids = ["clustering", "pca", "umap", "tsne", "association_rule_mining"]
    for rid in expected_ids:
        assert recipe_registry.has(rid), f"Recipe {rid} should be registered in recipe_registry"
        recipe = recipe_registry.get(rid)
        meta = recipe_registry.get_metadata(rid)
        assert meta.enabled is True
        assert meta.group == "Modeling"
        assert meta.subgroup == "Unsupervised & Dimensionality Reduction"
        schema = recipe.get_schema()
        assert isinstance(schema, dict)
        assert "properties" in schema


# =====================================================================
# 2. CLUSTERING RECIPE TESTS
# =====================================================================

def test_clustering_kmeans_execution():
    """Tests KMeans clustering with standard numeric DataFrame."""
    recipe = ClusteringRecipe()
    df = pd.DataFrame({
        "feature_a": [1.0, 1.2, 0.9, 8.0, 8.2, 7.9],
        "feature_b": [2.0, 2.1, 1.9, 9.0, 8.9, 9.1],
        "label": ["A", "A", "A", "B", "B", "B"]
    })
    result = recipe.execute(inputs={"dataframe": df}, config={"algorithm": "KMeans", "n_clusters": 2})
    out_df = result["dataframe"]
    assert "cluster" in out_df.columns
    assert len(out_df) == 6
    assert result["model"] is not None
    assert result["metrics"]["n_clusters_found"] == 2
    assert "silhouette_score" in result["metrics"]
    assert "scatter_preview" in result["metrics"]
    assert "cluster_sizes" in result["metrics"]
    assert len(result["metrics"]["cluster_sizes"]) == 2
    assert "scatter_plot" in result["metrics"]
    assert "centroids" in result["metrics"]["scatter_plot"]
    assert len(result["metrics"]["scatter_plot"]["centroids"]) == 2
    assert "cluster_sizes" in result["output_summary"]
    assert "bar_chart" in result["output_summary"]
    assert "scatter_plot" in result["output_summary"]


def test_clustering_dbscan_and_agglomerative():
    """Tests DBSCAN and Agglomerative clustering algorithms."""
    recipe = ClusteringRecipe()
    df = pd.DataFrame({
        "x": [0.0, 0.1, 0.2, 10.0, 10.1, 10.2, 50.0],
        "y": [0.0, 0.1, -0.1, 10.0, 9.9, 10.1, 50.0]
    })
    # DBSCAN
    res_db = recipe.execute(inputs={"dataframe": df}, config={"algorithm": "DBSCAN", "eps": 1.0, "min_samples": 2})
    assert "cluster" in res_db["dataframe"].columns
    assert res_db["metrics"]["algorithm"] == "DBSCAN"

    # Agglomerative
    res_agg = recipe.execute(inputs={"dataframe": df}, config={"algorithm": "Agglomerative", "n_clusters": 2})
    assert "cluster" in res_agg["dataframe"].columns
    assert res_agg["metrics"]["algorithm"] == "Agglomerative"


def test_clustering_edge_cases():
    """Tests clustering with missing values, empty DF, and small samples."""
    recipe = ClusteringRecipe()

    # Empty DF
    empty_res = recipe.execute(inputs={"dataframe": pd.DataFrame()}, config={})
    assert empty_res["dataframe"].empty

    # Missing values and clamp n_clusters > n_samples
    df_missing = pd.DataFrame({
        "a": [1.0, np.nan, 3.0],
        "b": [np.nan, 2.0, 3.0]
    })
    res_clamped = recipe.execute(inputs={"dataframe": df_missing}, config={"n_clusters": 10})
    assert "cluster" in res_clamped["dataframe"].columns
    assert len(res_clamped["dataframe"]) == 3


# =====================================================================
# 3. PCA RECIPE TESTS
# =====================================================================

def test_pca_execution():
    """Tests standard PCA reduction and metrics."""
    recipe = PCARecipe()
    df = pd.DataFrame({
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "x2": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        "x3": [0.5, 0.8, 1.2, 1.5, 2.0, 2.3],
        "category": ["cat1", "cat2", "cat1", "cat2", "cat1", "cat2"]
    })
    result = recipe.execute(
        inputs={"dataframe": df},
        config={"n_components": 2, "prefix": "PCA_", "append_components": True}
    )
    out_df = result["dataframe"]
    assert "PCA_1" in out_df.columns
    assert "PCA_2" in out_df.columns
    assert "x1" in out_df.columns  # Appended alongside
    assert len(out_df) == 6
    assert result["metrics"]["n_components"] == 2
    assert "explained_variance_ratio" in result["metrics"]
    assert "feature_loadings" in result["metrics"]
    assert len(result["metrics"]["explained_variance_ratio"]) == 2
    assert "explained_variance_ratio" in result["output_summary"]
    assert "total_variance_explained" in result["output_summary"]
    assert "bar_chart" in result["output_summary"]
    assert "scatter_plot" in result["output_summary"]


def test_pca_edge_cases():
    """Tests PCA with n_components exceeding features, and non-append mode."""
    recipe = PCARecipe()
    df = pd.DataFrame({"val": [10.0, 20.0, 30.0]})
    # Request 5 components on 1 feature
    res = recipe.execute(
        inputs={"dataframe": df},
        config={"n_components": 5, "append_components": False}
    )
    out_df = res["dataframe"]
    assert "PCA_1" in out_df.columns
    assert "val" not in out_df.columns  # Replaced
    assert res["metrics"]["n_components"] == 1  # Clamped to 1 feature


# =====================================================================
# 4. UMAP RECIPE TESTS
# =====================================================================

def test_umap_execution():
    """Tests UMAP embedding (native or manifold fallback)."""
    recipe = UMAPRecipe()
    df = pd.DataFrame({
        "f1": [1.0, 2.0, 1.5, 8.0, 9.0, 8.5],
        "f2": [2.0, 1.0, 1.8, 7.0, 8.0, 7.5],
        "f3": [0.1, 0.2, 0.15, 5.0, 5.2, 5.1]
    })
    result = recipe.execute(
        inputs={"dataframe": df},
        config={"n_components": 2, "n_neighbors": 3, "prefix": "UMAP_"}
    )
    out_df = result["dataframe"]
    assert "UMAP_1" in out_df.columns
    assert "UMAP_2" in out_df.columns
    assert len(out_df) == 6
    assert result["metrics"]["n_components"] == 2
    assert "engine" in result["metrics"]
    assert "scatter_plot" in result["metrics"]
    assert "points" in result["metrics"]["scatter_plot"]
    assert "scatter_plot" in result["output_summary"]


def test_umap_edge_cases():
    """Tests UMAP with empty DF and single column."""
    recipe = UMAPRecipe()
    empty_res = recipe.execute(inputs={"dataframe": pd.DataFrame()}, config={})
    assert empty_res["dataframe"].empty

    # Auto-detection of columns
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "text": ["a", "b", "c"]})
    res = recipe.execute(inputs={"dataframe": df}, config={"n_components": 1})
    assert "UMAP_1" in res["dataframe"].columns


# =====================================================================
# 5. t-SNE RECIPE TESTS
# =====================================================================

def test_tsne_execution():
    """Tests t-SNE embedding with automatic perplexity scaling."""
    recipe = TSNERecipe()
    # Small dataset (6 rows) where default perplexity=30 must auto-clamp
    df = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 10.0, 11.0, 12.0],
        "b": [2.0, 3.0, 4.0, 12.0, 13.0, 14.0],
        "c": [0.5, 0.6, 0.7, 5.0, 5.1, 5.2]
    })
    result = recipe.execute(
        inputs={"dataframe": df},
        config={"n_components": 2, "perplexity": 30.0, "max_iter": 300, "prefix": "tSNE_"}
    )
    out_df = result["dataframe"]
    assert "tSNE_1" in out_df.columns
    assert "tSNE_2" in out_df.columns
    assert len(out_df) == 6
    # Clamped to <= (6-1)/3 = 1.66
    assert result["metrics"]["perplexity_used"] < 30.0
    assert "scatter_preview" in result["metrics"]
    assert "scatter_plot" in result["metrics"]
    assert "points" in result["metrics"]["scatter_plot"]
    assert "scatter_plot" in result["output_summary"]


def test_tsne_edge_cases():
    """Tests t-SNE with missing values and empty dataframe."""
    recipe = TSNERecipe()
    empty_res = recipe.execute(inputs={"dataframe": pd.DataFrame()}, config={})
    assert empty_res["dataframe"].empty

    df_nan = pd.DataFrame({
        "col1": [1.0, np.nan, 3.0, 4.0, 5.0],
        "col2": [np.nan, 2.0, 3.0, 4.0, 5.0]
    })
    res = recipe.execute(inputs={"dataframe": df_nan}, config={"n_components": 2})
    assert "tSNE_1" in res["dataframe"].columns


# =====================================================================
# 6. ASSOCIATION RULE MINING RECIPE TESTS
# =====================================================================

def test_association_rules_delimited():
    """Tests Association Rule Mining with comma-separated transaction strings."""
    recipe = AssociationRuleMiningRecipe()
    df = pd.DataFrame({
        "transaction_id": [1, 2, 3, 4, 5],
        "items": [
            "Milk, Bread, Butter",
            "Bread, Butter",
            "Milk, Bread",
            "Milk, Butter",
            "Beer, Diapers"
        ]
    })
    result = recipe.execute(
        inputs={"dataframe": df},
        config={
            "input_format": "Delimited Item String",
            "item_column": "items",
            "min_support": 0.2,
            "min_confidence": 0.3,
            "min_lift": 1.0
        }
    )
    rules_df = result["dataframe"]
    assert not rules_df.empty
    assert "antecedents" in rules_df.columns
    assert "consequents" in rules_df.columns
    assert "support" in rules_df.columns
    assert "confidence" in rules_df.columns
    assert "lift" in rules_df.columns
    assert result["metrics"]["rules_count"] > 0
    assert result["metrics"]["total_transactions"] == 5


def test_association_rules_one_hot_matrix():
    """Tests Association Rule Mining with boolean / one-hot matrix."""
    recipe = AssociationRuleMiningRecipe()
    df = pd.DataFrame({
        "Milk": [1, 0, 1, 1, 0],
        "Bread": [1, 1, 1, 0, 0],
        "Butter": [1, 1, 0, 1, 0],
        "Beer": [0, 0, 0, 0, 1]
    })
    result = recipe.execute(
        inputs={"dataframe": df},
        config={
            "input_format": "One-Hot / Boolean Matrix",
            "min_support": 0.2,
            "min_confidence": 0.3,
            "min_lift": 1.0
        }
    )
    rules_df = result["dataframe"]
    assert not rules_df.empty
    assert "lift" in rules_df.columns


def test_association_rules_no_rules_found():
    """Tests threshold behavior when thresholds are too strict."""
    recipe = AssociationRuleMiningRecipe()
    df = pd.DataFrame({"items": ["Apple", "Banana", "Cherry"]})
    result = recipe.execute(
        inputs={"dataframe": df},
        config={"min_support": 0.9, "min_confidence": 0.9}
    )
    rules_df = result["dataframe"]
    assert rules_df.empty
    assert result["output_summary"]["status"] == "NO_RULES_FOUND"
    assert "hint" in result["output_summary"]


# =====================================================================
# 7. END-TO-END DAG PIPELINE TEST
# =====================================================================

def test_dag_execution_with_unsupervised_recipes():
    """Verify unsupervised nodes chain together cleanly in the DAG execution engine."""
    from backend.app.engine.dag.graph import WorkflowGraph, WorkflowNode, WorkflowEdge
    from backend.app.engine.execution.executor import DAGExecutor

    nodes = [
        WorkflowNode(id="pca_node", recipe_id="pca", config={"n_components": 2, "prefix": "comp_"}),
        WorkflowNode(id="cluster_node", recipe_id="clustering", config={"n_clusters": 2, "algorithm": "KMeans"})
    ]
    edges = [
        WorkflowEdge(source="pca_node", target="cluster_node")
    ]
    graph = WorkflowGraph(nodes=nodes, edges=edges)
    df = pd.DataFrame({
        "a": [1.0, 1.1, 1.2, 8.0, 8.1, 8.2],
        "b": [2.0, 2.1, 2.2, 9.0, 9.1, 9.2],
        "c": [3.0, 3.1, 3.2, 7.0, 7.1, 7.2]
    })
    res = DAGExecutor.execute_workflow(
        execution_id="test_unsupervised_dag",
        workflow=graph,
        initial_df=df
    )
    assert res.status == "SUCCESS"
    assert len(res.node_results) == 2
    assert res.node_results[0].status == "SUCCESS"
    assert res.node_results[1].status == "SUCCESS"

