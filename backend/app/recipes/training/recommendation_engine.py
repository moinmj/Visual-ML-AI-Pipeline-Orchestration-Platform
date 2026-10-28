import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort
from backend.app.recipes.training.encoder_utils import pass_through_metadata


class FittedRecommender:
    """
    Wrapper model exposing unified recommend() and get_similar_items() APIs.
    """

    def __init__(
        self,
        algorithm: str,
        user_to_idx: Dict[Any, int],
        idx_to_user: Dict[int, Any],
        item_to_idx: Dict[Any, int],
        idx_to_item: Dict[int, Any],
        user_factors: Optional[np.ndarray] = None,
        item_factors: Optional[np.ndarray] = None,
        item_sim_matrix: Optional[np.ndarray] = None,
        user_known_items: Optional[Dict[Any, set]] = None,
        global_mean: float = 0.0
    ):
        self.algorithm = algorithm
        self.user_to_idx = user_to_idx
        self.idx_to_user = idx_to_user
        self.item_to_idx = item_to_idx
        self.idx_to_item = idx_to_item
        self.user_factors = user_factors
        self.item_factors = item_factors
        self.item_sim_matrix = item_sim_matrix
        self.user_known_items = user_known_items or {}
        self.global_mean = global_mean

    def predict(self, user_id: Any, item_id: Any) -> float:
        if self.algorithm == "matrix_factorization_svd" and self.user_factors is not None and self.item_factors is not None:
            if user_id in self.user_to_idx and item_id in self.item_to_idx:
                u_idx = self.user_to_idx[user_id]
                i_idx = self.item_to_idx[item_id]
                return float(np.dot(self.user_factors[u_idx], self.item_factors[i_idx]))
        return self.global_mean

    def recommend(self, user_id: Any, top_k: int = 5, exclude_known: bool = True) -> List[Dict[str, Any]]:
        if user_id not in self.user_to_idx:
            # Cold-start fallback: return globally popular items
            return [{"item_id": self.idx_to_item[i], "score": round(self.global_mean, 4)} for i in range(min(top_k, len(self.idx_to_item)))]

        u_idx = self.user_to_idx[user_id]
        known = self.user_known_items.get(user_id, set()) if exclude_known else set()

        if self.algorithm == "matrix_factorization_svd" and self.user_factors is not None and self.item_factors is not None:
            scores = np.dot(self.user_factors[u_idx], self.item_factors.T)
        elif self.item_sim_matrix is not None:
            # Collaborative filtering: sum similarity to user's interacted items
            known_indices = [self.item_to_idx[it] for it in known if it in self.item_to_idx]
            if known_indices:
                scores = np.sum(self.item_sim_matrix[known_indices, :], axis=0)
            else:
                scores = np.zeros(len(self.idx_to_item))
        else:
            scores = np.zeros(len(self.idx_to_item))

        # Rank items
        ranked_indices = np.argsort(scores)[::-1]
        recs: List[Dict[str, Any]] = []
        for idx in ranked_indices:
            item_id = self.idx_to_item[idx]
            if exclude_known and item_id in known:
                continue
            recs.append({
                "item_id": item_id,
                "score": float(round(float(scores[idx]), 4))
            })
            if len(recs) >= top_k:
                break

        return recs

    def get_similar_items(self, item_id: Any, top_k: int = 5) -> List[Dict[str, Any]]:
        if item_id not in self.item_to_idx:
            return []

        i_idx = self.item_to_idx[item_id]
        if self.item_sim_matrix is not None:
            sims = self.item_sim_matrix[i_idx].copy()
            sims[i_idx] = -float("inf")  # exclude self
            ranked = np.argsort(sims)[::-1][:top_k]
            return [{"item_id": self.idx_to_item[idx], "similarity": float(round(float(sims[idx]), 4))} for idx in ranked]
        elif self.item_factors is not None:
            item_vec = self.item_factors[i_idx].reshape(1, -1)
            sims = cosine_similarity(item_vec, self.item_factors)[0]
            sims[i_idx] = -float("inf")
            ranked = np.argsort(sims)[::-1][:top_k]
            return [{"item_id": self.idx_to_item[idx], "similarity": float(round(float(sims[idx]), 4))} for idx in ranked]

        return []


class RecommendationEngineRecipe(BaseRecipe):
    """
    Recommendation Engine Recipe (Collaborative Filtering & Matrix Factorization).
    Generates personalized Top-K recommendations and item similarity maps using
    Latent Factor Matrix Factorization (Truncated SVD) and Item-Item Collaborative Filtering.
    """

    recipe_id = "recommendation_engine"
    name = "Recommendation Engine (Collaborative & Content-Based)"
    version = "1.0.0"
    category = "training"
    group = "Modeling"
    subgroup = "Specialized Models"
    description = (
        "Generates personalized Top-K recommendations and item similarity maps using "
        "Matrix Factorization (SVD) and Item-Item Collaborative Filtering."
    )
    input_types = ["dataframe"]
    output_types = ["model", "dataframe", "metrics", "report"]

    inputs: List[RecipePort] = [
        RecipePort(
            id="data",
            label="User-Item Interaction Table",
            type="dataframe",
            required=True,
            max_connections=1,
            description="Table containing user identifiers, item identifiers, and optional interaction ratings/frequencies."
        )
    ]

    outputs: List[RecipePort] = [
        RecipePort(
            id="model",
            label="Fitted Recommender Model",
            type="model",
            description="Trained Recommender instance with recommend(user_id, top_k) and predict(user_id, item_id)."
        ),
        RecipePort(
            id="recommendations",
            label="Top-K Recommendations Table",
            type="dataframe",
            description="Dataframe of recommended items per user with confidence scores."
        ),
        RecipePort(
            id="metrics",
            label="Recommender Metrics",
            type="metrics",
            description="Catalog coverage percentage, matrix sparsity ratio, and RMSE rating error."
        ),
        RecipePort(
            id="report",
            label="Executive Summary Report",
            type="report",
            description="Markdown report containing catalog coverage, top recommended items, and cohort insights."
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "algorithm": {
                    "type": "string",
                    "title": "Recommendation Algorithm",
                    "input_type": "dropdown",
                    "enum": [
                        "matrix_factorization_svd",
                        "item_collaborative"
                    ],
                    "default": "matrix_factorization_svd",
                    "description": "Matrix Factorization (SVD) learns latent preference vectors; Item Collaborative calculates direct item-item cosine similarities."
                },
                "user_col": {
                    "type": "string",
                    "title": "User Identifier Column",
                    "input_type": "text",
                    "default": "user_id"
                },
                "item_col": {
                    "type": "string",
                    "title": "Item Identifier Column",
                    "input_type": "text",
                    "default": "item_id"
                },
                "rating_col": {
                    "type": "string",
                    "title": "Optional Rating / Weight Column",
                    "input_type": "text",
                    "default": "",
                    "description": "Explicit numeric rating or interaction count. If left blank, implicit 1.0 interaction weight is used."
                },
                "n_factors": {
                    "type": "integer",
                    "title": "Latent Dimensions (Factors)",
                    "input_type": "slider",
                    "default": 10,
                    "minimum": 2,
                    "maximum": 50
                },
                "top_k": {
                    "type": "integer",
                    "title": "Top-K Recommendations per User",
                    "input_type": "slider",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 20
                }
            },
            "required": []
        }

    def _auto_detect_columns(self, df: pd.DataFrame, user_cfg: str, item_cfg: str, rating_cfg: str) -> Tuple[str, str, str]:
        cols_lower = {c.lower(): c for c in df.columns}

        # Resolve user
        user_col = ""
        if user_cfg in df.columns:
            user_col = user_cfg
        elif user_cfg.lower() in cols_lower:
            user_col = cols_lower[user_cfg.lower()]
        else:
            candidates = ["user_id", "user", "customer_id", "customer", "client_id", "account_id", "uid"]
            for cand in candidates:
                if cand in cols_lower:
                    user_col = cols_lower[cand]
                    break

        # Resolve item
        item_col = ""
        if item_cfg in df.columns:
            item_col = item_cfg
        elif item_cfg.lower() in cols_lower:
            item_col = cols_lower[item_cfg.lower()]
        else:
            candidates = ["item_id", "item", "product_id", "product", "sku", "movie_id", "content_id", "asin", "iid"]
            for cand in candidates:
                if cand in cols_lower:
                    item_col = cols_lower[cand]
                    break

        # Resolve rating
        rating_col = ""
        if rating_cfg and rating_cfg in df.columns:
            rating_col = rating_cfg
        elif rating_cfg and rating_cfg.lower() in cols_lower:
            rating_col = cols_lower[rating_cfg.lower()]
        else:
            candidates = ["rating", "score", "weight", "quantity", "interaction", "count"]
            for cand in candidates:
                if cand in cols_lower:
                    rating_col = cols_lower[cand]
                    break

        return user_col, item_col, rating_col

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        # 1. Resolve Dataframe
        df = inputs.get("dataframe")
        if df is None:
            df = inputs.get("data")
        if df is None and isinstance(context, dict):
            df = context.get("dataframe")
            if df is None:
                df = context.get("data")

        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("RecommendationEngine expects a non-empty pandas DataFrame in inputs.")

        df = df.copy()

        # 2. Detect & Validate Columns
        user_cfg = config.get("user_col", "user_id")
        item_cfg = config.get("item_col", "item_id")
        rating_cfg = config.get("rating_col", "")

        user_col, item_col, rating_col = self._auto_detect_columns(df, user_cfg, item_cfg, rating_cfg)

        if not user_col or user_col not in df.columns:
            raise ValueError(
                f"RecommendationEngine could not locate user column '{user_cfg}'. Available columns: {list(df.columns)}"
            )
        if not item_col or item_col not in df.columns:
            raise ValueError(
                f"RecommendationEngine could not locate item column '{item_cfg}'. Available columns: {list(df.columns)}"
            )

        # Drop missing user or item rows
        df = df.dropna(subset=[user_col, item_col])

        # 3. Process Ratings / Interactions
        if rating_col and rating_col in df.columns:
            try:
                df["_val"] = pd.to_numeric(df[rating_col]).fillna(1.0)
            except Exception:
                df["_val"] = 1.0
        else:
            df["_val"] = 1.0

        # Unique users & items
        unique_users = df[user_col].unique()
        unique_items = df[item_col].unique()

        if len(unique_users) < 2:
            raise ValueError("RecommendationEngine requires at least 2 distinct users in the interaction table.")
        if len(unique_items) < 2:
            raise ValueError("RecommendationEngine requires at least 2 distinct items in the interaction table.")

        user_to_idx = {u: idx for idx, u in enumerate(unique_users)}
        idx_to_user = {idx: u for idx, u in enumerate(unique_users)}
        item_to_idx = {it: idx for idx, it in enumerate(unique_items)}
        idx_to_item = {idx: it for idx, it in enumerate(unique_items)}

        # Build user interaction history
        user_known: Dict[Any, set] = {}
        for _, row in df[[user_col, item_col]].iterrows():
            u = row[user_col]
            it = row[item_col]
            if u not in user_known:
                user_known[u] = set()
            user_known[u].add(it)

        # 4. Construct Sparse User-Item Matrix
        row_indices = df[user_col].map(user_to_idx).to_numpy()
        col_indices = df[item_col].map(item_to_idx).to_numpy()
        values = df["_val"].to_numpy()

        n_u = len(unique_users)
        n_i = len(unique_items)
        R_sparse = csr_matrix((values, (row_indices, col_indices)), shape=(n_u, n_i))

        # Sparsity calculation
        total_possible = n_u * n_i
        observed = len(df)
        sparsity_pct = round((1.0 - (observed / max(1, total_possible))) * 100.0, 2)
        global_mean = float(np.mean(values))

        algorithm = config.get("algorithm", "matrix_factorization_svd")
        n_factors_cfg = int(config.get("n_factors", 10))
        top_k = int(config.get("top_k", 5))

        warnings: List[str] = []
        user_factors = None
        item_factors = None
        item_sim_matrix = None
        rmse = None

        # 5. Fit Chosen Recommender Algorithm
        if algorithm == "matrix_factorization_svd":
            # Latent Truncated SVD Matrix Factorization
            max_possible_factors = min(n_u, n_i) - 1
            effective_factors = max(2, min(n_factors_cfg, max_possible_factors))
            if effective_factors < n_factors_cfg:
                warnings.append(
                    f"[INFO: ADAPTIVE FACTORS] Requested {n_factors_cfg} latent factors exceeds matrix rank. "
                    f"Adjusted to {effective_factors} factors."
                )

            svd = TruncatedSVD(n_components=effective_factors, random_state=42)
            user_factors = svd.fit_transform(R_sparse)
            item_factors = svd.components_.T  # Shape: (n_items, n_components)

            # Reconstruct predictions to calculate in-sample RMSE
            R_pred = np.dot(user_factors, item_factors.T)
            pred_vals = R_pred[row_indices, col_indices]
            rmse = float(round(math.sqrt(mean_squared_error(values, pred_vals)), 4))

        else:
            # Item-Item Cosine Similarity Collaborative Filtering
            # Transpose so rows are items, columns are users
            item_user_matrix = R_sparse.T.tocsr()
            item_sim_matrix = cosine_similarity(item_user_matrix, dense_output=True)

        recommender = FittedRecommender(
            algorithm=algorithm,
            user_to_idx=user_to_idx,
            idx_to_user=idx_to_user,
            item_to_idx=item_to_idx,
            idx_to_item=idx_to_item,
            user_factors=user_factors,
            item_factors=item_factors,
            item_sim_matrix=item_sim_matrix,
            user_known_items=user_known,
            global_mean=global_mean
        )

        # 6. Generate Batch Top-K Recommendations Dataframe
        rec_rows: List[Dict[str, Any]] = []
        recommended_catalog_items = set()

        for u in unique_users:
            top_items = recommender.recommend(u, top_k=top_k, exclude_known=True)
            for rank, rec in enumerate(top_items, start=1):
                recommended_catalog_items.add(rec["item_id"])
                rec_rows.append({
                    user_col: u,
                    "rank": rank,
                    item_col: rec["item_id"],
                    "recommendation_score": rec["score"]
                })

        recs_df = pd.DataFrame(rec_rows)

        # Catalog Coverage
        catalog_coverage_pct = round((len(recommended_catalog_items) / max(1, n_i)) * 100.0, 2)

        metrics = {
            "algorithm": algorithm,
            "total_users": n_u,
            "total_items": n_i,
            "total_interactions": observed,
            "matrix_sparsity_pct": sparsity_pct,
            "catalog_coverage_pct": catalog_coverage_pct,
            "rmse": rmse,
            "top_k": top_k,
            "recommendations_generated": len(recs_df),
            "warnings": warnings
        }

        # 7. Format Executive Report
        report_lines = [
            f"# === Recommendation Engine Report ({algorithm.replace('_', ' ').title()}) ===",
            f"- **User Cohort**: {n_u} distinct users",
            f"- **Catalog Size**: {n_i} distinct items",
            f"- **Observed Interactions**: {observed:,} (Matrix Sparsity: **{sparsity_pct:.1f}%**)",
            f"- **Catalog Coverage**: **{catalog_coverage_pct:.1f}%** of items surfaced across Top-{top_k} lists",
        ]
        if rmse is not None:
            report_lines.append(f"- **Rating Prediction RMSE**: **{rmse:.4f}**")

        report_lines.append("")
        report_lines.append(f"## Sample Top-{top_k} Recommendations:")
        report_lines.append(f"| User ID | Rank | Recommended Item | Score |")
        report_lines.append(f"| :--- | :--- | :--- | :--- |")
        for _, r in recs_df.head(10).iterrows():
            report_lines.append(f"| `{r[user_col]}` | #{r['rank']} | **{r[item_col]}** | `{r['recommendation_score']:.4f}` |")

        # Top Recommended Items overall
        top_item_freq = recs_df[item_col].value_counts().head(5)
        report_lines.append("")
        report_lines.append("## Most Frequently Recommended Items Across Cohort:")
        report_lines.append("| Item ID | Times Recommended | Share of Recommendations |")
        report_lines.append("| :--- | :--- | :--- |")
        for it, count in top_item_freq.items():
            pct = (count / max(1, len(recs_df))) * 100.0
            report_lines.append(f"| `{it}` | {count} users | **{pct:.1f}%** |")

        if warnings:
            report_lines.append("")
            report_lines.append("## Guardrail Alerts:")
            for w in warnings:
                report_lines.append(f"- {w}")

        report_markdown = "\n".join(report_lines)

        output: Dict[str, Any] = {
            "model": recommender,
            "recommender": recommender,
            "recommendations": recs_df,
            "dataframe": recs_df,
            "metrics": metrics,
            "report": report_markdown
        }

        return pass_through_metadata(inputs, output, context)

    def to_code(self, config: Dict[str, Any]) -> str:
        u_col = config.get("user_col", "user_id")
        i_col = config.get("item_col", "item_id")
        k = config.get("top_k", 5)
        return (
            "from scipy.sparse import csr_matrix\n"
            "from sklearn.decomposition import TruncatedSVD\n"
            "import numpy as np\n\n"
            f"# 1. Construct user-item interaction matrix\n"
            f"user_ids = df['{u_col}'].astype('category').cat.codes\n"
            f"item_ids = df['{i_col}'].astype('category').cat.codes\n"
            "R = csr_matrix((np.ones(len(df)), (user_ids, item_ids)))\n\n"
            "# 2. Learn latent preference vectors via SVD\n"
            "svd = TruncatedSVD(n_components=10, random_state=42)\n"
            "user_factors = svd.fit_transform(R)\n"
            "item_factors = svd.components_.T\n"
            f"print('Trained recommender with top_k={k}')"
        )
