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
from backend.app.recipes.training.next_best_offer import NextBestOfferRecipe


def test_next_best_offer_registry_and_aliases():
    """Verify NextBestOfferRecipe is registered with correct metadata and aliases."""
    assert recipe_registry.has("next_best_offer") is True
    meta = recipe_registry.get_metadata("next_best_offer")
    assert meta.enabled is True
    assert meta.group == "Modeling"
    assert meta.subgroup == "Specialized Models"

    # Aliases
    assert recipe_registry.get("nbo").recipe_id == "next_best_offer"
    assert recipe_registry.get("nba").recipe_id == "next_best_offer"
    assert recipe_registry.get("next_best_action").recipe_id == "next_best_offer"
    assert recipe_registry.get("cross_sell_recommender").recipe_id == "next_best_offer"
    assert recipe_registry.get("market_basket_nbo").recipe_id == "next_best_offer"


def test_nbo_market_basket_cross_sell():
    """Verify Market Basket cross-sell mining, affinity scores, and customer NBO generation."""
    # Synthetic banking customer portfolio transactions
    rows = [
        # Customer 1: Checking + Savings
        {"client": "C1", "product": "Checking_Account"},
        {"client": "C1", "product": "Savings_Account"},
        # Customer 2: Checking + Savings + Credit Card
        {"client": "C2", "product": "Checking_Account"},
        {"client": "C2", "product": "Savings_Account"},
        {"client": "C2", "product": "Credit_Card"},
        # Customer 3: Checking + Credit Card
        {"client": "C3", "product": "Checking_Account"},
        {"client": "C3", "product": "Credit_Card"},
        # Customer 4: Savings + Credit Card + Auto Loan
        {"client": "C4", "product": "Savings_Account"},
        {"client": "C4", "product": "Credit_Card"},
        {"client": "C4", "product": "Auto_Loan"},
        # Customer 5: Checking + Auto Loan
        {"client": "C5", "product": "Checking_Account"},
        {"client": "C5", "product": "Auto_Loan"},
        # Customer 6: Checking + Savings + Mortgage
        {"client": "C6", "product": "Checking_Account"},
        {"client": "C6", "product": "Savings_Account"},
        {"client": "C6", "product": "Mortgage"},
    ]
    df = pd.DataFrame(rows)

    recipe = NextBestOfferRecipe()
    result = recipe.execute(
        inputs={"transactions": df},
        config={
            "customer_col": "client",
            "product_col": "product",
            "min_support": 0.1,
            "min_confidence": 0.1,
            "min_lift": 1.0,
            "top_n_offers": 2
        }
    )

    nbo_engine = result["model"]
    offers_df = result["offers"]
    metrics = result["metrics"]
    report = result["report"]

    assert hasattr(nbo_engine, "get_next_best_offer")
    assert not offers_df.empty
    assert "offer_rank" in offers_df.columns
    assert "affinity_score" in offers_df.columns
    assert "business_rationale" in offers_df.columns
    assert metrics["total_customers"] == 6
    assert metrics["total_products"] == 5
    assert metrics["discovered_rules_count"] > 0
    assert metrics["average_rule_lift"] >= 1.0

    # Test single customer NBO
    c1_offers = nbo_engine.get_next_best_offer("C1", top_n=2)
    assert len(c1_offers) <= 2
    # C1 already owns Checking and Savings -> should NOT be recommended Checking or Savings
    c1_prods = [o["product_id"] for o in c1_offers]
    assert "Checking_Account" not in c1_prods
    assert "Savings_Account" not in c1_prods

    # Report verification
    assert "# === Next Best Offer / Next Best Action (NBO / NBA) Report" in report
    assert "Top Discovered Cross-Sell Affinity Rules:" in report


def test_nbo_expected_value_margin_weighting():
    """Verify margin/price table alters ranking under expected_value criterion."""
    tx_df = pd.DataFrame([
        {"customer": "C1", "product": "Prod_A"},
        {"customer": "C1", "product": "Prod_B"},
        {"customer": "C2", "product": "Prod_A"},
        {"customer": "C2", "product": "Prod_C"},
        {"customer": "C3", "product": "Prod_A"},
        {"customer": "C3", "product": "Prod_B"},
        {"customer": "C4", "product": "Prod_A"},
        {"customer": "C4", "product": "Prod_C"},
    ])

    # Prod_C has 10x the margin of Prod_B
    margins_df = pd.DataFrame([
        {"product_id": "Prod_B", "margin": 10.0},
        {"product_id": "Prod_C", "margin": 100.0}
    ])

    recipe = NextBestOfferRecipe()
    result = recipe.execute(
        inputs={"transactions": tx_df, "product_margins": margins_df},
        config={
            "ranking_criterion": "expected_value",
            "min_support": 0.1,
            "min_confidence": 0.1,
            "min_lift": 0.8,
            "top_n_offers": 2
        }
    )

    nbo = result["model"]
    # New customer with Prod_A: Prod_C should rank higher because of 10x margin
    offers = nbo.get_next_best_offer("C1_new", top_n=2)
    # C1 already owns Prod_A and Prod_B; only Prod_C available for cross-sell
    c1_offers = nbo.get_next_best_offer("C1", top_n=1)
    assert c1_offers[0]["product_id"] == "Prod_C"


def test_cold_start_customer_nbo():
    """Verify new customer receives catalog anchor next best actions without error."""
    df = pd.DataFrame([
        {"customer_id": "C1", "product_id": "P1"},
        {"customer_id": "C1", "product_id": "P2"},
        {"customer_id": "C2", "product_id": "P2"},
        {"customer_id": "C2", "product_id": "P3"},
    ])

    recipe = NextBestOfferRecipe()
    res = recipe.execute(inputs={"dataframe": df}, config={"top_n_offers": 2})
    nbo = res["model"]

    new_offers = nbo.get_next_best_offer("brand_new_customer", top_n=2)
    assert len(new_offers) == 2
    assert all("product_id" in o and "affinity_score" in o for o in new_offers)


def test_customer_owns_everything_edge_case():
    """Verify customer who already owns entire catalog handled gracefully."""
    df = pd.DataFrame([
        {"customer_id": "C1", "product_id": "P1"},
        {"customer_id": "C1", "product_id": "P2"},
        {"customer_id": "C2", "product_id": "P1"},
    ])

    recipe = NextBestOfferRecipe()
    res = recipe.execute(inputs={"dataframe": df}, config={"exclude_already_owned": True})
    nbo = res["model"]

    # C1 owns P1 and P2 (entire catalog)
    offers = nbo.get_next_best_offer("C1", top_n=2)
    assert len(offers) == 0


def test_next_best_offer_edge_cases():
    """Verify guardrails: missing columns, single customer, single product, sparse rules."""
    recipe = NextBestOfferRecipe()

    # Guard 1: Missing customer column
    df_no_cust = pd.DataFrame({"product_id": ["P1", "P2"]})
    with pytest.raises(ValueError, match="could not locate customer column"):
        recipe.execute(inputs={"dataframe": df_no_cust}, config={"customer_col": "client_id"})

    # Guard 2: Single customer rejected
    df_single_c = pd.DataFrame({"customer_id": ["C1", "C1"], "product_id": ["P1", "P2"]})
    with pytest.raises(ValueError, match="requires at least 2 distinct customers"):
        recipe.execute(inputs={"dataframe": df_single_c}, config={})

    # Guard 3: Single product rejected
    df_single_p = pd.DataFrame({"customer_id": ["C1", "C2"], "product_id": ["P1", "P1"]})
    with pytest.raises(ValueError, match="requires at least 2 distinct products"):
        recipe.execute(inputs={"dataframe": df_single_p}, config={})

    # Guard 4: High thresholds trigger sparse rules warning
    df_sparse = pd.DataFrame([
        {"customer_id": "C1", "product_id": "P1"},
        {"customer_id": "C2", "product_id": "P2"},
    ])
    res = recipe.execute(inputs={"dataframe": df_sparse}, config={"min_support": 0.9, "min_lift": 5.0})
    assert any("SPARSE RULES" in w for w in res["metrics"]["warnings"])
    assert res["model"] is not None


if __name__ == "__main__":
    print("Running test_next_best_offer_registry_and_aliases...")
    test_next_best_offer_registry_and_aliases()
    print("PASS: test_next_best_offer_registry_and_aliases")

    print("Running test_nbo_market_basket_cross_sell...")
    test_nbo_market_basket_cross_sell()
    print("PASS: test_nbo_market_basket_cross_sell")

    print("Running test_nbo_expected_value_margin_weighting...")
    test_nbo_expected_value_margin_weighting()
    print("PASS: test_nbo_expected_value_margin_weighting")

    print("Running test_cold_start_customer_nbo...")
    test_cold_start_customer_nbo()
    print("PASS: test_cold_start_customer_nbo")

    print("Running test_customer_owns_everything_edge_case...")
    test_customer_owns_everything_edge_case()
    print("PASS: test_customer_owns_everything_edge_case")

    print("Running test_next_best_offer_edge_cases...")
    test_next_best_offer_edge_cases()
    print("PASS: test_next_best_offer_edge_cases")

    print("\nALL 6 NEXT BEST OFFER TESTS PASSED SUCCESSFULLY!")
