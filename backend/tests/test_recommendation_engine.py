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

from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.training.recommendation_engine import RecommendationEngineRecipe


def test_recommendation_engine_registry_and_aliases():
    """Verify RecommendationEngineRecipe is registered with correct metadata and aliases."""
    assert recipe_registry.has("recommendation_engine") is True
    meta = recipe_registry.get_metadata("recommendation_engine")
    assert meta.enabled is True
    assert meta.group == "Modeling"
    assert meta.subgroup == "Specialized Models"

    # Aliases
    assert recipe_registry.get("recommender").recipe_id == "recommendation_engine"
    assert recipe_registry.get("recommendations").recipe_id == "recommendation_engine"
    assert recipe_registry.get("collaborative_filtering").recipe_id == "recommendation_engine"
    assert recipe_registry.get("matrix_factorization").recipe_id == "recommendation_engine"
    assert recipe_registry.get("content_based_recommender").recipe_id == "recommendation_engine"


def test_svd_matrix_factorization_recommendations():
    """Verify SVD matrix factorization produces personalized Top-K recommendations and item similarity."""
    np.random.seed(42)
    users = [f"user_{i}" for i in range(1, 41)]
    items = [f"item_{j}" for j in range(1, 21)]

    # Generate synthetic interactions
    rows = []
    for u in users:
        # Each user interacts with 3 to 8 random items
        n_it = np.random.randint(3, 9)
        chosen_items = np.random.choice(items, size=n_it, replace=False)
        for it in chosen_items:
            rating = np.random.randint(1, 6)
            rows.append({"customer": u, "product": it, "stars": rating})

    df = pd.DataFrame(rows)

    recipe = RecommendationEngineRecipe()
    result = recipe.execute(
        inputs={"dataframe": df},
        config={
            "algorithm": "matrix_factorization_svd",
            "user_col": "customer",
            "item_col": "product",
            "rating_col": "stars",
            "n_factors": 6,
            "top_k": 3
        }
    )

    recommender = result["model"]
    recs_df = result["recommendations"]
    metrics = result["metrics"]
    report = result["report"]

    assert hasattr(recommender, "recommend")
    assert hasattr(recommender, "predict")
    assert hasattr(recommender, "get_similar_items")
    assert not recs_df.empty
    assert "recommendation_score" in recs_df.columns
    assert metrics["algorithm"] == "matrix_factorization_svd"
    assert metrics["catalog_coverage_pct"] > 0.0
    assert metrics["rmse"] is not None
    assert metrics["rmse"] >= 0.0

    # User recommendation check
    user_recs = recommender.recommend("user_1", top_k=3, exclude_known=True)
    assert len(user_recs) <= 3
    assert all("item_id" in r and "score" in r for r in user_recs)

    # Similar items check
    sims = recommender.get_similar_items("item_1", top_k=3)
    assert len(sims) <= 3

    # Report verification
    assert "# === Recommendation Engine Report" in report
    assert "Sample Top-3 Recommendations:" in report


def test_item_collaborative_filtering():
    """Verify Item-Item Collaborative Filtering computes cosine similarity matrix and recommendations."""
    df = pd.DataFrame([
        {"user": "Alice", "movie": "Matrix", "score": 5},
        {"user": "Alice", "movie": "Inception", "score": 4},
        {"user": "Bob", "movie": "Matrix", "score": 5},
        {"user": "Bob", "movie": "Avatar", "score": 3},
        {"user": "Charlie", "movie": "Inception", "score": 5},
        {"user": "Charlie", "movie": "Interstellar", "score": 5},
        {"user": "David", "movie": "Interstellar", "score": 4},
        {"user": "David", "movie": "Matrix", "score": 4}
    ])

    recipe = RecommendationEngineRecipe()
    result = recipe.execute(
        inputs={"dataframe": df},
        config={
            "algorithm": "item_collaborative",
            "user_col": "user",
            "item_col": "movie",
            "rating_col": "score",
            "top_k": 2
        }
    )

    recommender = result["model"]
    assert recommender.item_sim_matrix is not None
    alice_recs = recommender.recommend("Alice", top_k=2, exclude_known=True)
    assert len(alice_recs) > 0
    # Alice shouldn't be recommended Matrix or Inception (known)
    rec_titles = [r["item_id"] for r in alice_recs]
    assert "Matrix" not in rec_titles
    assert "Inception" not in rec_titles


def test_implicit_feedback_without_ratings():
    """Verify table with only user and item columns works seamlessly with implicit 1.0 weights."""
    df = pd.DataFrame([
        {"uid": "U1", "iid": "P1"},
        {"uid": "U1", "iid": "P2"},
        {"uid": "U2", "iid": "P2"},
        {"uid": "U2", "iid": "P3"},
        {"uid": "U3", "iid": "P1"},
        {"uid": "U3", "iid": "P3"}
    ])

    recipe = RecommendationEngineRecipe()
    result = recipe.execute(inputs={"dataframe": df}, config={"top_k": 2})
    assert result["metrics"]["total_interactions"] == 6
    assert result["recommendations"] is not None


def test_cold_start_fallback():
    """Verify unseen/new user receives global popularity recommendations without crashing."""
    df = pd.DataFrame([
        {"user": "U1", "item": "I1"},
        {"user": "U1", "item": "I2"},
        {"user": "U2", "item": "I2"},
        {"user": "U2", "item": "I3"}
    ])

    recipe = RecommendationEngineRecipe()
    result = recipe.execute(inputs={"dataframe": df}, config={})
    recommender = result["model"]

    cold_recs = recommender.recommend("unseen_user_999", top_k=2)
    assert len(cold_recs) == 2
    assert all("item_id" in r for r in cold_recs)


def test_recommendation_engine_edge_cases():
    """Verify guardrails: missing columns, single user, single item, adaptive factors."""
    recipe = RecommendationEngineRecipe()

    # Guard 1: Missing user column
    df_no_user = pd.DataFrame({"item_id": ["I1", "I2"], "score": [5, 4]})
    with pytest.raises(ValueError, match="could not locate user column"):
        recipe.execute(inputs={"dataframe": df_no_user}, config={"user_col": "custom_user"})

    # Guard 2: Single user rejected
    df_single_u = pd.DataFrame({"user_id": ["U1", "U1"], "item_id": ["I1", "I2"]})
    with pytest.raises(ValueError, match="requires at least 2 distinct users"):
        recipe.execute(inputs={"dataframe": df_single_u}, config={})

    # Guard 3: Single item rejected
    df_single_i = pd.DataFrame({"user_id": ["U1", "U2"], "item_id": ["I1", "I1"]})
    with pytest.raises(ValueError, match="requires at least 2 distinct items"):
        recipe.execute(inputs={"dataframe": df_single_i}, config={})

    # Guard 4: High factor count automatically adjusted to matrix rank
    df_small = pd.DataFrame([
        {"user_id": "U1", "item_id": "I1"},
        {"user_id": "U1", "item_id": "I2"},
        {"user_id": "U2", "item_id": "I2"},
        {"user_id": "U2", "item_id": "I3"},
        {"user_id": "U3", "item_id": "I1"}
    ])
    res = recipe.execute(inputs={"dataframe": df_small}, config={"n_factors": 20})
    assert any("ADAPTIVE FACTORS" in w for w in res["metrics"]["warnings"])
    assert res["model"] is not None


if __name__ == "__main__":
    print("Running test_recommendation_engine_registry_and_aliases...")
    test_recommendation_engine_registry_and_aliases()
    print("PASS: test_recommendation_engine_registry_and_aliases")

    print("Running test_svd_matrix_factorization_recommendations...")
    test_svd_matrix_factorization_recommendations()
    print("PASS: test_svd_matrix_factorization_recommendations")

    print("Running test_item_collaborative_filtering...")
    test_item_collaborative_filtering()
    print("PASS: test_item_collaborative_filtering")

    print("Running test_implicit_feedback_without_ratings...")
    test_implicit_feedback_without_ratings()
    print("PASS: test_implicit_feedback_without_ratings")

    print("Running test_cold_start_fallback...")
    test_cold_start_fallback()
    print("PASS: test_cold_start_fallback")

    print("Running test_recommendation_engine_edge_cases...")
    test_recommendation_engine_edge_cases()
    print("PASS: test_recommendation_engine_edge_cases")

    print("\nALL 6 RECOMMENDATION ENGINE TESTS PASSED SUCCESSFULLY!")
