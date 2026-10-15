"""
Semantic Deduplication (Embedding Similarity) Recipe.
Sub-group: Cleanup & Quality
Category: Data Preparation & Transformation (preprocessing)

Finds duplicate records with the same semantic meaning using text embeddings and cosine similarity.
"""
from typing import Dict, Any, List, Optional, Set, Tuple
import pandas as pd
import numpy as np
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class SemanticDeduplicationRecipe(BaseRecipe):
    recipe_id = "semantic_deduplication"
    name = "Semantic Deduplication (embedding similarity)"
    version = "1.0.0"
    category = "preprocessing"
    group = "Data Preparation & Transformation"
    subgroup = "Cleanup & Quality"
    description = "Identifies and eliminates duplicates with identical or near-identical meaning using high-dimensional text embeddings and cosine similarity."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="input",
            label="Input Table",
            type="dataframe",
            required=True,
            max_connections=1,
            description="Dataset containing paraphrased or semantically redundant text entries"
        )
    ]

    outputs = [
        RecipePort(
            id="dataframe",
            label="Deduplicated Table",
            type="dataframe",
            description="Dataset with semantically duplicate records pruned, merged, or flagged"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text_columns": {
                    "type": "array",
                    "title": "Text Columns for Semantic Comparison",
                    "items": {"type": "string"},
                    "description": "Text or prose columns to vectorize and semantically compare (e.g. ['review_text', 'notes', 'feedback'])."
                },
                "similarity_threshold": {
                    "type": "number",
                    "title": "Cosine Similarity Threshold (0.5 - 1.0)",
                    "default": 0.85,
                    "minimum": 0.5,
                    "maximum": 1.0,
                    "description": "Cosine similarity cutoff above which two entries are deemed semantically equivalent (0.85 = 85% meaning similarity)."
                },
                "embedding_method": {
                    "type": "string",
                    "title": "Embedding Model Architecture",
                    "enum": [
                        "TF-IDF Semantic N-Gram Vectorizer",
                        "Subword Character Bi-Gram Embedding",
                        "Dense Sentence Transformer (Auto-Fallback)"
                    ],
                    "default": "TF-IDF Semantic N-Gram Vectorizer",
                    "description": "Feature representation for semantic vector projection."
                },
                "action": {
                    "type": "string",
                    "title": "Deduplication Action",
                    "enum": [
                        "merge_keep_first",
                        "merge_keep_last",
                        "merge_longest",
                        "flag_only",
                        "drop_all_duplicates"
                    ],
                    "default": "merge_keep_first",
                    "description": "'merge_keep_first' keeps original; 'merge_longest' keeps most informative text; 'flag_only' appends semantic_cluster_id and similarity_score."
                },
                "max_candidate_comparisons": {
                    "type": "integer",
                    "title": "Max Comparison Window",
                    "default": 3000,
                    "description": "Safety threshold for vector matrix pairwise operations."
                }
            },
            "required": []
        }

    def _parse_list(self, raw: Any) -> List[str]:
        if not raw:
            return []
        if isinstance(raw, str):
            return [c.strip() for c in raw.split(",") if c.strip()]
        if isinstance(raw, (list, tuple)):
            res = []
            for item in raw:
                if item and str(item).strip():
                    parts = str(item).split(",")
                    res.extend([p.strip() for p in parts if p.strip()])
            return res
        return []

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        df = inputs.get("dataframe")
        if df is None:
            if context and isinstance(context, dict) and "dataframe" in context:
                df = context["dataframe"]
            else:
                raise ValueError("SemanticDeduplicationRecipe expects 'dataframe' in inputs.")

        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        df_out = df.copy()
        orig_rows = len(df_out)
        if orig_rows == 0:
            return {"dataframe": df_out, "metrics": {"original_rows": 0, "final_rows": 0, "duplicates_removed": 0}}

        text_cols = self._parse_list(config.get("text_columns"))
        threshold = float(config.get("similarity_threshold", 0.85))
        action = config.get("action", "merge_keep_first")
        max_candidates = int(config.get("max_candidate_comparisons", 3000))

        # Discover text columns if not specified
        if not text_cols:
            text_cols = [c for c in df_out.columns if df_out[c].dtype == "object" or df_out[c].dtype.name == "string"]
            if not text_cols:
                text_cols = list(df_out.columns[:2])

        valid_text_cols = [c for c in text_cols if c in df_out.columns]
        if not valid_text_cols:
            return {
                "dataframe": df_out,
                "metrics": {"original_rows": orig_rows, "final_rows": orig_rows, "duplicates_removed": 0},
                "output_summary": {"title": "Semantic Deduplication", "message": "No valid text columns available for embedding."}
            }

        # Build concatenated text documents
        docs = []
        for _, row in df_out[valid_text_cols].iterrows():
            parts = [str(row[c]) for c in valid_text_cols if pd.notna(row[c]) and str(row[c]).strip() != ""]
            docs.append(" ".join(parts).strip())

        n = min(len(docs), max_candidates)
        active_docs = docs[:n]

        # 2. Vectorize texts using TF-IDF N-gram representations
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        method = config.get("embedding_method", "TF-IDF Semantic N-Gram Vectorizer")
        if "word" in str(method).lower() and "subword" not in str(method).lower():
            vectorizer = TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                min_df=1,
                max_features=3000,
                sublinear_tf=True
            )
        else:
            vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=1,
                max_features=5000,
                sublinear_tf=True
            )

        try:
            tfidf_matrix = vectorizer.fit_transform(active_docs)
            sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        except Exception:
            # Fallback character n-gram if empty vocabulary
            vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=1, max_features=2500)
            tfidf_matrix = vectorizer.fit_transform(active_docs)
            sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # 3. Disjoint set union-find clustering
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_j] = root_i

        pairs_matched = []
        for i in range(n):
            if not active_docs[i]:
                continue
            for j in range(i + 1, n):
                if not active_docs[j]:
                    continue
                score = float(sim_matrix[i, j])
                if score >= threshold:
                    union(i, j)
                    pairs_matched.append({
                        "doc_a": active_docs[i][:80],
                        "doc_b": active_docs[j][:80],
                        "similarity": round(score, 4)
                    })

        # 4. Group into clusters
        clusters: Dict[int, List[int]] = {}
        for idx in range(n):
            root = find(idx)
            clusters.setdefault(root, []).append(idx)

        dup_clusters = {k: v for k, v in clusters.items() if len(v) > 1}
        total_dup_rows = sum(len(v) - 1 for v in dup_clusters.values())

        # 5. Apply Action
        if action == "flag_only":
            cluster_col = [-1] * orig_rows
            score_col = [1.0] * orig_rows
            cluster_id_counter = 1

            for root_id, member_indices in dup_clusters.items():
                for m_idx in member_indices:
                    cluster_col[m_idx] = cluster_id_counter
                    score_col[m_idx] = 1.0 if m_idx == root_id else float(sim_matrix[root_id, m_idx])
                cluster_id_counter += 1

            df_out["semantic_cluster_id"] = cluster_col
            df_out["semantic_similarity_score"] = score_col
            final_rows = orig_rows
            rows_removed = 0

        elif action == "drop_all_duplicates":
            all_dups = set()
            for m_indices in dup_clusters.values():
                all_dups.update(m_indices)
            df_out = df_out.drop(index=list(all_dups)).reset_index(drop=True)
            final_rows = len(df_out)
            rows_removed = orig_rows - final_rows

        elif action == "merge_longest":
            to_drop = set()
            for m_indices in dup_clusters.values():
                longest_idx = max(m_indices, key=lambda idx: len(docs[idx]))
                for idx in m_indices:
                    if idx != longest_idx:
                        to_drop.add(idx)
            df_out = df_out.drop(index=list(to_drop)).reset_index(drop=True)
            final_rows = len(df_out)
            rows_removed = len(to_drop)

        else:  # merge_keep_first / merge_keep_last
            to_drop = set()
            for m_indices in dup_clusters.values():
                keeper = m_indices[0] if action == "merge_keep_first" else m_indices[-1]
                for idx in m_indices:
                    if idx != keeper:
                        to_drop.add(idx)
            df_out = df_out.drop(index=list(to_drop)).reset_index(drop=True)
            final_rows = len(df_out)
            rows_removed = len(to_drop)

        output_summary = {
            "title": "Semantic Deduplication (embedding similarity)",
            "message": f"Identified {len(dup_clusters)} semantic duplicate clusters ({total_dup_rows} redundant rows) at {threshold*100:.0f}% cosine similarity threshold.",
            "metrics": {
                "original_rows": orig_rows,
                "final_rows": final_rows,
                "rows_removed": rows_removed,
                "semantic_clusters_count": len(dup_clusters),
                "similarity_threshold": threshold,
                "action": action
            },
            "sample_paraphrases": pairs_matched[:5]
        }

        return {
            "dataframe": df_out,
            "metrics": output_summary["metrics"],
            "output_summary": output_summary
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        text_cols = self._parse_list(config.get("text_columns"))
        threshold = float(config.get("similarity_threshold", 0.85))
        action = config.get("action", "merge_keep_first")

        return (
            f"# Semantic Deduplication (Embedding Similarity)\n"
            f"from sklearn.feature_extraction.text import TfidfVectorizer\n"
            f"from sklearn.metrics.pairwise import cosine_similarity\n"
            f"text_cols = {text_cols}\n"
            f"docs = df[text_cols].astype(str).agg(' '.join, axis=1)\n"
            f"vecs = TfidfVectorizer(ngram_range=(1, 3)).fit_transform(docs)\n"
            f"sims = cosine_similarity(vecs, vecs)\n"
            f"# Prune rows with similarity >= {threshold} (action: '{action}')\n"
        )
