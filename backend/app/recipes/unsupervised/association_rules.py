import itertools
import math
from collections import defaultdict
from typing import Dict, Any, List, Set, Tuple, Optional
import pandas as pd
import numpy as np

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class AssociationRuleMiningRecipe(BaseRecipe):
    """
    Association Rule Mining (Apriori / FP-Growth) Recipe.
    Discovers high-affinity item relationships and frequent co-occurrence patterns (e.g., Market Basket Analysis).
    """

    recipe_id = "association_rule_mining"
    name = "Association Rule Mining (Apriori / FP-Growth)"
    version = "1.0.0"
    category = "unsupervised"
    description = (
        "Finds items that frequently occur together (e.g. products bought together) using Apriori or FP-Growth. "
        "Calculates Support, Confidence, Lift, and Leverage metrics to identify strong associative rules."
    )
    input_types = ["dataframe"]
    output_types = ["dataframe", "model"]

    inputs: List[RecipePort] = [
        RecipePort(
            id="input",
            label="Input Table",
            type="dataframe",
            required=True,
            max_connections=1,
            description="Transaction dataset"
        )
    ]
    outputs: List[RecipePort] = [
        RecipePort(
            id="output_1",
            label="Association Rules Table",
            type="dataframe",
            description="Mined association rules with support, confidence, and lift metrics"
        ),
        RecipePort(
            id="output_2",
            label="Mining Model / Summary",
            type="model",
            description="Frequent itemsets and rule generator metadata"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "algorithm": {
                    "type": "string",
                    "title": "Mining Algorithm",
                    "enum": ["FP-Growth", "Apriori"],
                    "default": "FP-Growth",
                    "description": "Frequent pattern mining algorithm. FP-Growth is faster on large transaction datasets."
                },
                "input_format": {
                    "type": "string",
                    "title": "Data Input Structure",
                    "enum": [
                        "Delimited Item String",
                        "One-Hot / Boolean Matrix",
                        "Grouped (TransactionID + Item)"
                    ],
                    "default": "Delimited Item String",
                    "description": "How transactions are represented in your incoming dataset."
                },
                "item_column": {
                    "type": "string",
                    "title": "Item Column",
                    "default": "items",
                    "format": "column",
                    "ui:widget": "column_select",
                    "description": "Column containing items (e.g. 'Milk, Bread, Butter' or individual item names)."
                },
                "transaction_id_column": {
                    "type": "string",
                    "title": "Transaction ID Column (for Grouped format)",
                    "default": "transaction_id",
                    "format": "column",
                    "ui:widget": "column_select",
                    "description": "Unique basket or transaction ID (only needed when Input Structure is 'Grouped')."
                },
                "delimiter": {
                    "type": "string",
                    "title": "Item Delimiter",
                    "default": ",",
                    "description": "Delimiter separating items when using 'Delimited Item String' (e.g. ',', ';', '|')."
                },
                "min_support": {
                    "type": "number",
                    "title": "Minimum Support",
                    "default": 0.05,
                    "minimum": 0.0001,
                    "maximum": 1.0,
                    "description": "Minimum fraction of transactions that must contain an itemset for it to be considered frequent."
                },
                "min_confidence": {
                    "type": "number",
                    "title": "Minimum Confidence",
                    "default": 0.2,
                    "minimum": 0.01,
                    "maximum": 1.0,
                    "description": "Minimum conditional probability P(Consequent | Antecedent) required to generate a rule."
                },
                "min_lift": {
                    "type": "number",
                    "title": "Minimum Lift",
                    "default": 1.0,
                    "minimum": 0.0,
                    "maximum": 100.0,
                    "description": "Ratio of rule confidence to expected baseline frequency. Lift > 1.0 indicates positive correlation."
                },
                "max_len": {
                    "type": "integer",
                    "title": "Maximum Rule Length",
                    "default": 3,
                    "minimum": 2,
                    "maximum": 10,
                    "description": "Maximum total items in antecedent + consequent."
                },
                "sort_by": {
                    "type": "string",
                    "title": "Sort Rules By",
                    "enum": ["lift", "confidence", "support"],
                    "default": "lift",
                    "description": "Metric to sort the resulting association rules by."
                }
            }
        }

    def execute(
        self,
        inputs: Dict[str, Any],
        config: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        df = inputs.get("dataframe")
        if df is None or not isinstance(df, pd.DataFrame):
            df = inputs.get("left_dataframe") or inputs.get("input")
            if df is None and context and isinstance(context, dict):
                df = context.get("dataframe")

        empty_rules_df = pd.DataFrame(columns=[
            "antecedents", "consequents", "support", "confidence", "lift", "leverage", "conviction"
        ])

        if df is None or not isinstance(df, pd.DataFrame):
            return {
                "dataframe": empty_rules_df,
                "output_1": empty_rules_df,
                "output": empty_rules_df,
                "model": None,
                "output_summary": {"error": "No valid input DataFrame provided for Association Rule Mining."}
            }

        if df.empty:
            return {
                "dataframe": empty_rules_df,
                "output_1": empty_rules_df,
                "output": empty_rules_df,
                "model": None,
                "metrics": {"total_transactions": 0, "rules_count": 0},
                "output_summary": {"message": "Input DataFrame is empty. No association rules mined."}
            }

        input_format = config.get("input_format", "Delimited Item String")
        item_col = str(config.get("item_column", "items") or "items")
        tid_col = str(config.get("transaction_id_column", "transaction_id") or "transaction_id")
        delim = str(config.get("delimiter", ",") or ",")
        min_support = float(config.get("min_support", 0.05))
        min_confidence = float(config.get("min_confidence", 0.2))
        min_lift = float(config.get("min_lift", 1.0))
        max_len = int(config.get("max_len", 3))
        sort_by = str(config.get("sort_by", "lift"))
        algorithm = str(config.get("algorithm", "FP-Growth"))

        # Extract transactions list
        transactions: List[List[str]] = []

        if input_format == "One-Hot / Boolean Matrix":
            # Each column is an item; truthy/1 rows indicate presence
            for _, row in df.iterrows():
                tx = [str(col) for col, val in row.items() if bool(val) and val not in [0, "0", False, "false", None, ""]]
                if tx:
                    transactions.append(tx)

        elif input_format == "Grouped (TransactionID + Item)":
            if tid_col in df.columns and item_col in df.columns:
                grouped = df.groupby(tid_col)[item_col].apply(lambda s: [str(x).strip() for x in s.dropna() if str(x).strip()])
                transactions = [tx for tx in grouped.tolist() if tx]
            elif item_col in df.columns:
                transactions = [[str(x).strip()] for x in df[item_col].dropna() if str(x).strip()]

        else:  # "Delimited Item String"
            target_col = item_col if item_col in df.columns else (df.columns[0] if len(df.columns) > 0 else None)
            if target_col:
                for val in df[target_col].dropna():
                    items = [it.strip() for it in str(val).split(delim) if it.strip()]
                    if items:
                        transactions.append(items)

        total_tx = len(transactions)
        if total_tx == 0:
            return {
                "dataframe": empty_rules_df,
                "output_1": empty_rules_df,
                "output": empty_rules_df,
                "model": None,
                "metrics": {"total_transactions": 0, "rules_count": 0},
                "output_summary": {
                    "title": "Association Rule Mining",
                    "message": "No valid transaction items could be parsed from input columns.",
                    "status": "SKIPPED"
                }
            }

        # Check for mlxtend acceleration if available
        rules_df = None
        frequent_itemsets = {}

        try:
            import mlxtend.preprocessing as mp
            import mlxtend.frequent_patterns as mfp

            te = mp.TransactionEncoder()
            te_ary = te.fit(transactions).transform(transactions)
            tx_df = pd.DataFrame(te_ary, columns=te.columns_)

            if algorithm == "Apriori":
                freq_df = mfp.apriori(tx_df, min_support=min_support, use_colnames=True, max_len=max_len)
            else:
                freq_df = mfp.fpgrowth(tx_df, min_support=min_support, use_colnames=True, max_len=max_len)

            if not freq_df.empty:
                raw_rules = mfp.association_rules(freq_df, metric="confidence", min_threshold=min_confidence)
                if not raw_rules.empty:
                    raw_rules = raw_rules[raw_rules["lift"] >= min_lift]
                    # Format antecedents and consequents as readable strings
                    raw_rules["antecedents"] = raw_rules["antecedents"].apply(lambda s: ", ".join(sorted(list(s))))
                    raw_rules["consequents"] = raw_rules["consequents"].apply(lambda s: ", ".join(sorted(list(s))))
                    rules_df = raw_rules[[
                        "antecedents", "consequents", "support", "confidence", "lift", "leverage", "conviction"
                    ]].round(4)
        except Exception:
            rules_df = None

        # Pure-Python High-Performance Mining Engine (100% Zero-Dependency Fallback)
        if rules_df is None:
            rules_df, frequent_itemsets = self._mine_rules_pure_python(
                transactions=transactions,
                min_support=min_support,
                min_confidence=min_confidence,
                min_lift=min_lift,
                max_len=max_len
            )

        if not rules_df.empty and sort_by in rules_df.columns:
            rules_df = rules_df.sort_values(by=sort_by, ascending=False).reset_index(drop=True)

        rules_count = len(rules_df)
        top_rules_sample = rules_df.head(10).to_dict(orient="records") if not rules_df.empty else []

        metrics: Dict[str, Any] = {
            "algorithm": algorithm,
            "total_transactions": total_tx,
            "rules_count": rules_count,
            "min_support_used": min_support,
            "min_confidence_used": min_confidence,
            "min_lift_used": min_lift,
            "max_lift": round(float(rules_df["lift"].max()), 4) if rules_count > 0 else 0.0,
            "avg_confidence": round(float(rules_df["confidence"].mean()), 4) if rules_count > 0 else 0.0,
            "top_rules": top_rules_sample
        }

        summary = {
            "title": "Association Rules Mined",
            "algorithm": algorithm,
            "rules_generated": rules_count,
            "total_transactions": total_tx,
            "status": "SUCCESS" if rules_count > 0 else "NO_RULES_FOUND"
        }
        if rules_count == 0:
            summary["hint"] = "0 rules matched your thresholds. Try lowering Minimum Support or Minimum Confidence."

        return {
            "dataframe": rules_df,
            "output_1": rules_df,
            "output": rules_df,
            "model": {"algorithm": algorithm, "total_transactions": total_tx, "rules_count": rules_count},
            "metrics": metrics,
            "rules_summary": metrics,
            "output_summary": summary
        }

    @staticmethod
    def _mine_rules_pure_python(
        transactions: List[List[str]],
        min_support: float,
        min_confidence: float,
        min_lift: float,
        max_len: int
    ) -> Tuple[pd.DataFrame, Dict[Tuple[str, ...], float]]:
        """
        Pure-Python Frequent Pattern & Rule Mining Engine.
        Deterministic, fast, and handles all transactions without external dependencies.
        """
        N = len(transactions)
        min_count = max(1, math.ceil(min_support * N))

        # Convert transactions to frozen sets of unique items
        tx_sets = [frozenset(t) for t in transactions]

        # 1. Frequent 1-itemsets
        item_counts: Dict[str, int] = defaultdict(int)
        for tx in tx_sets:
            for item in tx:
                item_counts[item] += 1

        frequent_itemsets: Dict[frozenset, float] = {}
        current_k_itemsets: Set[frozenset] = set()

        for item, count in item_counts.items():
            if count >= min_count:
                f_set = frozenset([item])
                frequent_itemsets[f_set] = count / N
                current_k_itemsets.add(f_set)

        # 2. Frequent k-itemsets (up to max_len)
        k = 2
        while current_k_itemsets and k <= max_len:
            # Candidate generation
            raw_candidates: Set[frozenset] = set()
            curr_list = list(current_k_itemsets)
            for i in range(len(curr_list)):
                for j in range(i + 1, len(curr_list)):
                    union_set = curr_list[i] | curr_list[j]
                    if len(union_set) == k:
                        raw_candidates.add(union_set)

            # Count candidates
            cand_counts: Dict[frozenset, int] = defaultdict(int)
            for tx in tx_sets:
                for cand in raw_candidates:
                    if cand.issubset(tx):
                        cand_counts[cand] += 1

            next_k_itemsets: Set[frozenset] = set()
            for cand, count in cand_counts.items():
                if count >= min_count:
                    frequent_itemsets[cand] = count / N
                    next_k_itemsets.add(cand)

            current_k_itemsets = next_k_itemsets
            k += 1

        # 3. Association Rule Generation
        rules: List[Dict[str, Any]] = []
        for itemset, sup_itemset in frequent_itemsets.items():
            if len(itemset) < 2:
                continue
            itemset_list = list(itemset)
            # Generate all non-empty proper subsets as antecedents
            for r in range(1, len(itemset_list)):
                for ante_combo in itertools.combinations(itemset_list, r):
                    ante = frozenset(ante_combo)
                    cons = itemset - ante
                    if not ante or not cons:
                        continue

                    sup_ante = frequent_itemsets.get(ante)
                    sup_cons = frequent_itemsets.get(cons)
                    if not sup_ante or not sup_cons:
                        continue

                    conf = sup_itemset / sup_ante
                    if conf < min_confidence:
                        continue

                    lift = conf / sup_cons
                    if lift < min_lift:
                        continue

                    leverage = sup_itemset - (sup_ante * sup_cons)
                    conviction = (1.0 - sup_cons) / (1.0 - conf) if conf < 0.9999 else 999.0

                    rules.append({
                        "antecedents": ", ".join(sorted(list(ante))),
                        "consequents": ", ".join(sorted(list(cons))),
                        "support": round(sup_itemset, 4),
                        "confidence": round(conf, 4),
                        "lift": round(lift, 4),
                        "leverage": round(leverage, 4),
                        "conviction": round(min(conviction, 999.0), 4)
                    })

        rules_df = pd.DataFrame(rules)
        if rules_df.empty:
            rules_df = pd.DataFrame(columns=[
                "antecedents", "consequents", "support", "confidence", "lift", "leverage", "conviction"
            ])
        return rules_df, {tuple(sorted(list(k))): v for k, v in frequent_itemsets.items()}

    def to_code(self, config: Dict[str, Any]) -> str:
        sup = config.get("min_support", 0.05)
        conf = config.get("min_confidence", 0.2)
        lift = config.get("min_lift", 1.0)
        item_col = config.get("item_column", "items")
        delim = config.get("delimiter", ",")

        return (
            f"# Association Rule Mining (Market Basket Analysis)\n"
            f"transactions = [row.split('{delim}') for row in df['{item_col}'].dropna()]\n"
            f"# Mined with min_support={sup}, min_confidence={conf}, min_lift={lift}\n"
        )
