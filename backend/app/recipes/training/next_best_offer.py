import math
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union, Set

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort
from backend.app.recipes.training.encoder_utils import pass_through_metadata


class FittedNBOEngine:
    """
    Fitted Next Best Offer (NBO) / Next Best Action (NBA) Recommender Engine.
    Provides get_next_best_offer(customer_id, top_n) with business explanations.
    """

    def __init__(
        self,
        customer_baskets: Dict[Any, Set[Any]],
        product_support: Dict[Any, float],
        rules: Dict[Any, List[Dict[str, Any]]],
        popular_items: List[Tuple[Any, float]],
        product_margins: Optional[Dict[Any, float]] = None,
        ranking_criterion: str = "affinity_lift",
        exclude_already_owned: bool = True
    ):
        self.customer_baskets = customer_baskets
        self.product_support = product_support
        self.rules = rules  # Map: antecedent_item -> list of rules {consequent, confidence, lift, support}
        self.popular_items = popular_items
        self.product_margins = product_margins or {}
        self.ranking_criterion = ranking_criterion
        self.exclude_already_owned = exclude_already_owned

    def get_next_best_offer(self, customer_id: Any, top_n: int = 3) -> List[Dict[str, Any]]:
        owned = self.customer_baskets.get(customer_id, set())
        candidate_scores: Dict[Any, float] = defaultdict(float)
        candidate_rationales: Dict[Any, List[str]] = defaultdict(list)

        if owned:
            for owned_item in owned:
                if owned_item in self.rules:
                    for rule in self.rules[owned_item]:
                        target = rule["consequent"]
                        if self.exclude_already_owned and target in owned:
                            continue

                        conf = rule["confidence"]
                        lift = rule["lift"]
                        margin_factor = self.product_margins.get(target, 1.0)

                        if self.ranking_criterion == "expected_value":
                            score_increment = conf * lift * margin_factor
                        elif self.ranking_criterion == "confidence":
                            score_increment = conf
                        elif self.ranking_criterion == "support":
                            score_increment = rule["support"]
                        else:  # affinity_lift
                            score_increment = conf * math.log2(1.0 + lift) * margin_factor

                        candidate_scores[target] += score_increment
                        candidate_rationales[target].append(
                            f"Frequently adopted alongside owned '{owned_item}' (Lift: {lift:.2f}x, Conf: {conf*100:.1f}%)"
                        )

        # Fallback to globally popular items for cold-start or empty candidates
        if len(candidate_scores) < top_n:
            for item, supp in self.popular_items:
                if self.exclude_already_owned and item in owned:
                    continue
                if item not in candidate_scores:
                    base_score = supp * self.product_margins.get(item, 1.0) * 0.1
                    candidate_scores[item] = base_score
                    candidate_rationales[item].append("Top-performing catalog anchor recommendation")
                if len(candidate_scores) >= top_n * 2:
                    break

        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

        offers: List[Dict[str, Any]] = []
        for rank, (prod, score) in enumerate(sorted_candidates, start=1):
            offers.append({
                "rank": rank,
                "product_id": prod,
                "affinity_score": float(round(score, 4)),
                "business_rationale": "; ".join(candidate_rationales.get(prod, ["High catalog popularity"])[:2])
            })

        return offers


class NextBestOfferRecipe(BaseRecipe):
    """
    Next Best Offer / Action (NBO / NBA) Recommender Recipe.
    Determines the optimal cross-sell and up-sell offer per customer using Market Basket
    affinity, association rule lift, and profit margin expected value ranking.
    """

    recipe_id = "next_best_offer"
    name = "Next Best Offer / Action (NBO / NBA Recommender)"
    version = "1.0.0"
    category = "training"
    group = "Modeling"
    subgroup = "Specialized Models"
    description = (
        "Determines the optimal Next Best Offer (NBO) or Next Best Action (NBA) per customer using "
        "Market Basket cross-sell affinity, rule confidence, and expected value ranking."
    )
    input_types = ["dataframe"]
    output_types = ["model", "dataframe", "metrics", "report"]

    inputs: List[RecipePort] = [
        RecipePort(
            id="transactions",
            label="Customer Interactions / Orders Table",
            type="dataframe",
            required=True,
            max_connections=1,
            description="Table with customer_id/order_id and product_id/offer_id."
        ),
        RecipePort(
            id="product_margins",
            label="Optional Product Margin/Value Table",
            type="dataframe",
            required=False,
            max_connections=1,
            description="Optional table mapping product_id to price, profit margin, or strategic business weight."
        )
    ]

    outputs: List[RecipePort] = [
        RecipePort(
            id="model",
            label="Trained NBO Engine",
            type="model",
            description="Trained NBO Engine exposing get_next_best_offer(customer_id, top_n)."
        ),
        RecipePort(
            id="offers",
            label="Customer Next Best Offers Table",
            type="dataframe",
            description="Table mapping each customer to their Rank #1, #2, #3 Next Best Offers with affinity scores and rationale."
        ),
        RecipePort(
            id="metrics",
            label="NBO Engine Metrics",
            type="metrics",
            description="Discovered cross-sell rules count, average lift, and catalog recommendation distribution."
        ),
        RecipePort(
            id="report",
            label="Executive Summary Report",
            type="report",
            description="Markdown report containing top cross-sell rules, offer distribution, and expected value lift."
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "customer_col": {
                    "type": "string",
                    "title": "Customer / Order Identifier Column",
                    "input_type": "text",
                    "default": "customer_id",
                    "description": "Identifier grouping products into a customer portfolio or purchase transaction."
                },
                "product_col": {
                    "type": "string",
                    "title": "Product / Offer Column",
                    "input_type": "text",
                    "default": "product_id"
                },
                "ranking_criterion": {
                    "type": "string",
                    "title": "Offer Ranking Criterion",
                    "input_type": "dropdown",
                    "enum": [
                        "affinity_lift",
                        "expected_value",
                        "confidence",
                        "support"
                    ],
                    "default": "affinity_lift",
                    "description": "Metric used to prioritize next offers. 'expected_value' multiplies affinity by margin/price."
                },
                "top_n_offers": {
                    "type": "integer",
                    "title": "Offers per Customer",
                    "input_type": "slider",
                    "default": 3,
                    "minimum": 1,
                    "maximum": 10
                },
                "min_support": {
                    "type": "number",
                    "title": "Minimum Rule Support",
                    "input_type": "number",
                    "default": 0.01,
                    "description": "Minimum fraction of customers adopting both products."
                },
                "min_confidence": {
                    "type": "number",
                    "title": "Minimum Rule Confidence",
                    "input_type": "number",
                    "default": 0.05,
                    "description": "Conditional probability P(Offer | Current Products)."
                },
                "min_lift": {
                    "type": "number",
                    "title": "Minimum Cross-Sell Lift",
                    "input_type": "number",
                    "default": 1.05,
                    "description": "Lift ratio > 1.0 indicates positive complementary affinity."
                },
                "exclude_already_owned": {
                    "type": "boolean",
                    "title": "Exclude Already Owned Products",
                    "input_type": "toggle",
                    "default": True,
                    "description": "Suppresses offers for products the customer currently holds."
                }
            },
            "required": []
        }

    def _auto_detect_columns(self, df: pd.DataFrame, cust_cfg: str, prod_cfg: str) -> Tuple[str, str]:
        cols_lower = {c.lower(): c for c in df.columns}

        # Resolve customer
        cust_col = ""
        if cust_cfg in df.columns:
            cust_col = cust_cfg
        elif cust_cfg.lower() in cols_lower:
            cust_col = cols_lower[cust_cfg.lower()]
        else:
            candidates = ["customer_id", "customer", "user_id", "user", "client_id", "account_id", "order_id", "basket_id", "uid"]
            for cand in candidates:
                if cand in cols_lower:
                    cust_col = cols_lower[cand]
                    break

        # Resolve product
        prod_col = ""
        if prod_cfg in df.columns:
            prod_col = prod_cfg
        elif prod_cfg.lower() in cols_lower:
            prod_col = cols_lower[prod_cfg.lower()]
        else:
            candidates = ["product_id", "product", "offer_id", "offer", "item_id", "item", "sku", "service", "action", "iid"]
            for cand in candidates:
                if cand in cols_lower:
                    prod_col = cols_lower[cand]
                    break

        return cust_col, prod_col

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        # 1. Resolve Dataframe
        df = inputs.get("dataframe")
        if df is None:
            df = inputs.get("transactions")
        if df is None:
            df = inputs.get("data")
        if df is None and isinstance(context, dict):
            df = context.get("dataframe")
            if df is None:
                df = context.get("transactions")

        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("NextBestOffer expects a non-empty pandas DataFrame in inputs.")

        df = df.copy()

        # 2. Detect & Validate Columns
        cust_cfg = config.get("customer_col", "customer_id")
        prod_cfg = config.get("product_col", "product_id")
        cust_col, prod_col = self._auto_detect_columns(df, cust_cfg, prod_cfg)

        if not cust_col or cust_col not in df.columns:
            raise ValueError(
                f"NextBestOffer could not locate customer column '{cust_cfg}'. Available columns: {list(df.columns)}"
            )
        if not prod_col or prod_col not in df.columns:
            raise ValueError(
                f"NextBestOffer could not locate product/offer column '{prod_cfg}'. Available columns: {list(df.columns)}"
            )

        df = df.dropna(subset=[cust_col, prod_col])

        unique_customers = df[cust_col].unique()
        unique_products = df[prod_col].unique()

        if len(unique_customers) < 2:
            raise ValueError("NextBestOffer requires at least 2 distinct customers in the interactions table.")
        if len(unique_products) < 2:
            raise ValueError("NextBestOffer requires at least 2 distinct products in the interactions table.")

        N_cust = len(unique_customers)
        N_prod = len(unique_products)

        # 3. Extract Optional Margin / Value Table
        margin_table = inputs.get("product_margins")
        product_margins: Dict[Any, float] = {}
        if isinstance(margin_table, pd.DataFrame) and not margin_table.empty:
            # Detect product and margin columns
            m_p_col, _ = self._auto_detect_columns(margin_table, "", "product_id")
            val_candidates = ["margin", "profit", "price", "value", "weight", "revenue"]
            m_v_col = next((c for c in margin_table.columns if any(vc in c.lower() for vc in val_candidates)), None)
            if m_p_col and m_v_col:
                for _, r in margin_table[[m_p_col, m_v_col]].iterrows():
                    try:
                        product_margins[r[m_p_col]] = float(r[m_v_col])
                    except Exception:
                        pass

        # 4. Construct Customer Portfolios & Association Statistics
        customer_baskets: Dict[Any, Set[Any]] = df.groupby(cust_col)[prod_col].apply(set).to_dict()

        # Item support
        item_counts = Counter()
        pair_counts = Counter()

        for basket in customer_baskets.values():
            for it in basket:
                item_counts[it] += 1
            basket_list = sorted(list(basket))
            for i in range(len(basket_list)):
                for j in range(len(basket_list)):
                    if i != j:
                        pair_counts[(basket_list[i], basket_list[j])] += 1

        product_support = {it: count / N_cust for it, count in item_counts.items()}
        popular_items = sorted(product_support.items(), key=lambda x: x[1], reverse=True)

        min_supp = float(config.get("min_support", 0.01))
        min_conf = float(config.get("min_confidence", 0.05))
        min_lift = float(config.get("min_lift", 1.05))
        ranking_criterion = config.get("ranking_criterion", "affinity_lift")
        top_n = int(config.get("top_n_offers", 3))
        exclude_already_owned = bool(config.get("exclude_already_owned", True))

        # 5. Mine Association Rules A -> B
        rules_by_antecedent: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
        all_rules_flat: List[Dict[str, Any]] = []

        for (item_a, item_b), pair_count in pair_counts.items():
            pair_supp = pair_count / N_cust
            if pair_supp < min_supp:
                continue

            supp_a = product_support[item_a]
            supp_b = product_support[item_b]
            conf = pair_supp / supp_a
            lift = conf / max(0.0001, supp_b)

            if conf >= min_conf and lift >= min_lift:
                rule_obj = {
                    "antecedent": item_a,
                    "consequent": item_b,
                    "support": float(round(pair_supp, 4)),
                    "confidence": float(round(conf, 4)),
                    "lift": float(round(lift, 4))
                }
                rules_by_antecedent[item_a].append(rule_obj)
                all_rules_flat.append(rule_obj)

        warnings: List[str] = []
        if not all_rules_flat:
            warnings.append(
                f"[INFO: SPARSE RULES] No association rules met min_support={min_supp} & min_lift={min_lift}. "
                "Engine automatically activated popularity-weighted fallback for Next Best Offers."
            )

        # 6. Instantiate Trained NBO Engine
        nbo_engine = FittedNBOEngine(
            customer_baskets=customer_baskets,
            product_support=product_support,
            rules=rules_by_antecedent,
            popular_items=popular_items,
            product_margins=product_margins,
            ranking_criterion=ranking_criterion,
            exclude_already_owned=exclude_already_owned
        )

        # 7. Generate Next Best Offers Dataframe across Customer Base
        offers_rows: List[Dict[str, Any]] = []
        offered_product_counts = Counter()

        for cust in unique_customers:
            cust_offers = nbo_engine.get_next_best_offer(cust, top_n=top_n)
            for rec in cust_offers:
                p_id = rec["product_id"]
                offered_product_counts[p_id] += 1
                offers_rows.append({
                    cust_col: cust,
                    "offer_rank": rec["rank"],
                    prod_col: p_id,
                    "affinity_score": rec["affinity_score"],
                    "business_rationale": rec["business_rationale"]
                })

        offers_df = pd.DataFrame(offers_rows)

        # 8. Compute Summary Metrics
        all_rules_flat.sort(key=lambda x: x["lift"], reverse=True)
        avg_lift = float(round(float(np.mean([r["lift"] for r in all_rules_flat])), 4)) if all_rules_flat else 1.0
        n_rules = len(all_rules_flat)
        coverage_pct = round((len(offered_product_counts) / max(1, N_prod)) * 100.0, 2)

        metrics = {
            "total_customers": N_cust,
            "total_products": N_prod,
            "discovered_rules_count": n_rules,
            "average_rule_lift": avg_lift,
            "catalog_offer_coverage_pct": coverage_pct,
            "top_n_offers_per_customer": top_n,
            "ranking_criterion": ranking_criterion,
            "warnings": warnings
        }

        # 9. Format Executive Report
        report_lines = [
            "# === Next Best Offer / Next Best Action (NBO / NBA) Report ===",
            f"- **Customer Base Analyzed**: {N_cust:,} customers",
            f"- **Product Catalog Scope**: {N_prod:,} products / services",
            f"- **Cross-Sell Rules Mined**: {n_rules} affinity relationships (Avg Lift: **{avg_lift:.2f}x**)",
            f"- **Offer Coverage**: **{coverage_pct:.1f}%** of product catalog active in NBO recommendations",
            "",
            "## Top Discovered Cross-Sell Affinity Rules:"
        ]

        if all_rules_flat:
            report_lines.append("| Current Owned Product | Next Best Offer | Support | Confidence | Cross-Sell Lift |")
            report_lines.append("| :--- | :--- | :--- | :--- | :--- |")
            for r in all_rules_flat[:6]:
                report_lines.append(
                    f"| `{r['antecedent']}` | **`{r['consequent']}`** | {r['support']*100:.1f}% | {r['confidence']*100:.1f}% | **{r['lift']:.2f}x** |"
                )
        else:
            report_lines.append("- *(No strong association rules above threshold; fallback popularity ranking used)*")

        report_lines.append("")
        report_lines.append(f"## Most Frequently Surfaced Next Best Offers (Top-1 Share):")
        report_lines.append("| Product / Service | Times Pitched | Share of Recommendations |")
        report_lines.append("| :--- | :--- | :--- |")
        top_pitches = offers_df[offers_df["offer_rank"] == 1][prod_col].value_counts().head(5)
        for prod, cnt in top_pitches.items():
            pct = (cnt / max(1, N_cust)) * 100.0
            report_lines.append(f"| `{prod}` | {cnt} customers | **{pct:.1f}%** |")

        if warnings:
            report_lines.append("")
            report_lines.append("## Guardrail Alerts:")
            for w in warnings:
                report_lines.append(f"- {w}")

        report_markdown = "\n".join(report_lines)

        output: Dict[str, Any] = {
            "model": nbo_engine,
            "nbo_engine": nbo_engine,
            "offers": offers_df,
            "dataframe": offers_df,
            "metrics": metrics,
            "report": report_markdown
        }

        return pass_through_metadata(inputs, output, context)

    def to_code(self, config: Dict[str, Any]) -> str:
        c_col = config.get("customer_col", "customer_id")
        p_col = config.get("product_col", "product_id")
        top_n = config.get("top_n_offers", 3)
        return (
            "from collections import Counter\n"
            f"# 1. Group customer transactions into active product portfolios\n"
            f"portfolios = df.groupby('{c_col}')['{p_col}'].apply(set).to_dict()\n\n"
            "# 2. Calculate Market Basket affinity & association rules\n"
            "# 3. Rank Next Best Offers by cross-sell lift\n"
            f"print('Computed Next Best Offers with top_n={top_n}')"
        )
