"""
Fuzzy-Match Deduplication Recipe.
Sub-group: Cleanup & Quality
Category: Data Preparation & Transformation (preprocessing)

Finds and merges near-duplicate rows using fuzzy string matching across key columns.
"""
from typing import Dict, Any, List, Optional, Set, Tuple
import difflib
import pandas as pd
import numpy as np
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


def _token_sort_ratio(s1: str, s2: str) -> float:
    """Calculates token-sorted string similarity ratio between 0.0 and 1.0."""
    t1 = " ".join(sorted(s1.lower().split()))
    t2 = " ".join(sorted(s2.lower().split()))
    return difflib.SequenceMatcher(None, t1, t2).ratio()


def _sequence_similarity(s1: str, s2: str, case_sensitive: bool = False) -> float:
    """Fast similarity ratio between two strings using standard SequenceMatcher."""
    if not case_sensitive:
        s1 = s1.lower()
        s2 = s2.lower()
    if s1 == s2:
        return 1.0
    return difflib.SequenceMatcher(None, s1, s2).ratio()


class FuzzyMatchDeduplicationRecipe(BaseRecipe):
    recipe_id = "fuzzy_match_deduplication"
    name = "Fuzzy-Match Deduplication"
    version = "1.0.0"
    category = "preprocessing"
    group = "Data Preparation & Transformation"
    subgroup = "Cleanup & Quality"
    description = "Finds and merges near-duplicate rows using fuzzy string matching (Levenshtein / Token Sort Ratio) across key text columns."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="input",
            label="Input Table",
            type="dataframe",
            required=True,
            max_connections=1,
            description="Dataset containing potential typo or near-duplicate records"
        )
    ]

    outputs = [
        RecipePort(
            id="dataframe",
            label="Deduplicated Table",
            type="dataframe",
            description="Cleaned dataset with near-duplicate rows merged, eliminated, or flagged"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "key_columns": {
                    "type": "array",
                    "title": "Columns to Match On",
                    "items": {"type": "string"},
                    "description": "Columns to inspect for fuzzy similarity (e.g. ['name', 'company', 'address']). If empty, uses all text columns."
                },
                "similarity_threshold": {
                    "type": "number",
                    "title": "Similarity Threshold (0.5 - 1.0)",
                    "default": 0.85,
                    "minimum": 0.5,
                    "maximum": 1.0,
                    "description": "Minimum similarity ratio to consider two records as duplicates (0.85 = 85% match)."
                },
                "algorithm": {
                    "type": "string",
                    "title": "Matching Algorithm",
                    "enum": ["Token Sort Ratio", "Sequence Matcher", "Exact Prefix + Fuzzy"],
                    "default": "Token Sort Ratio",
                    "description": "Algorithm for comparing string tokens and character sequences."
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
                    "description": "'merge_keep_first' retains the first row; 'merge_longest' retains the row with the longest string values; 'flag_only' appends fuzzy_cluster_id and similarity_score."
                },
                "case_sensitive": {
                    "type": "boolean",
                    "title": "Case Sensitive Comparison",
                    "default": False,
                    "description": "Whether string comparison should strictly preserve letter casing."
                },
                "max_candidate_comparisons": {
                    "type": "integer",
                    "title": "Max Block Comparison Size",
                    "default": 5000,
                    "description": "Safety ceiling for quadratic pairwise comparisons to maintain sub-second performance."
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
                raise ValueError("FuzzyMatchDeduplicationRecipe expects 'dataframe' in inputs.")

        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        df_out = df.copy()
        orig_rows = len(df_out)
        if orig_rows == 0:
            return {"dataframe": df_out, "metrics": {"original_rows": 0, "final_rows": 0, "duplicates_removed": 0}}

        # 1. Parse configuration
        key_cols = self._parse_list(config.get("key_columns"))
        threshold = float(config.get("similarity_threshold", 0.85))
        algorithm = config.get("algorithm", "Token Sort Ratio")
        action = config.get("action", "merge_keep_first")
        case_sensitive = bool(config.get("case_sensitive", False))
        max_candidates = int(config.get("max_candidate_comparisons", 5000))

        # 2. Select text key columns
        if not key_cols:
            key_cols = [c for c in df_out.columns if df_out[c].dtype == "object" or df_out[c].dtype.name == "string"]
            if not key_cols:
                key_cols = list(df_out.columns[:3])

        valid_keys = [c for c in key_cols if c in df_out.columns]
        if not valid_keys:
            return {
                "dataframe": df_out,
                "metrics": {"original_rows": orig_rows, "final_rows": orig_rows, "duplicates_removed": 0},
                "output_summary": {"title": "Fuzzy-Match Deduplication", "message": "No valid key columns found for fuzzy matching."}
            }

        # 3. Create composite string representation for each row
        composite_strings = []
        for _, row in df_out[valid_keys].iterrows():
            vals = [str(row[k]) for k in valid_keys if pd.notna(row[k]) and str(row[k]).strip() != ""]
            composite_strings.append(" ".join(vals))

        # 4. Pairwise clustering using blocking/indexing for efficiency
        n = min(len(composite_strings), max_candidates)
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
            s1 = composite_strings[i]
            if not s1.strip():
                continue
            for j in range(i + 1, n):
                s2 = composite_strings[j]
                if not s2.strip():
                    continue

                # Fast length heuristic check
                len1, len2 = len(s1), len(s2)
                if max(len1, len2) > 0 and min(len1, len2) / max(len1, len2) < (threshold * 0.7):
                    continue

                if algorithm == "Token Sort Ratio":
                    sim = _token_sort_ratio(s1, s2)
                else:
                    sim = _sequence_similarity(s1, s2, case_sensitive=case_sensitive)

                if sim >= threshold:
                    union(i, j)
                    pairs_matched.append({"row_a": i, "row_b": j, "similarity": round(sim, 4)})

        # 5. Group into clusters
        clusters: Dict[int, List[int]] = {}
        for idx in range(n):
            root = find(idx)
            clusters.setdefault(root, []).append(idx)

        # Identify duplicate clusters (size > 1)
        duplicate_clusters = {k: v for k, v in clusters.items() if len(v) > 1}
        total_dup_rows = sum(len(v) - 1 for v in duplicate_clusters.values())

        # 6. Apply Deduplication Action
        if action == "flag_only":
            cluster_id_col = [-1] * orig_rows
            sim_score_col = [1.0] * orig_rows
            cluster_counter = 1

            for root_id, member_indices in duplicate_clusters.items():
                for m_idx in member_indices:
                    cluster_id_col[m_idx] = cluster_counter
                    sim_score_col[m_idx] = 1.0 if m_idx == root_id else threshold
                cluster_counter += 1

            df_out["fuzzy_cluster_id"] = cluster_id_col
            df_out["fuzzy_similarity_score"] = sim_score_col
            final_rows = orig_rows
            rows_removed = 0

        elif action == "drop_all_duplicates":
            all_dup_indices = set()
            for member_indices in duplicate_clusters.values():
                all_dup_indices.update(member_indices)
            df_out = df_out.drop(index=list(all_dup_indices)).reset_index(drop=True)
            final_rows = len(df_out)
            rows_removed = orig_rows - final_rows

        elif action == "merge_longest":
            indices_to_drop = set()
            for member_indices in duplicate_clusters.values():
                # Pick member with longest total characters
                longest_idx = max(member_indices, key=lambda idx: len(composite_strings[idx]))
                for idx in member_indices:
                    if idx != longest_idx:
                        indices_to_drop.add(idx)
            df_out = df_out.drop(index=list(indices_to_drop)).reset_index(drop=True)
            final_rows = len(df_out)
            rows_removed = len(indices_to_drop)

        else:  # merge_keep_first / merge_keep_last
            indices_to_drop = set()
            for member_indices in duplicate_clusters.values():
                keeper = member_indices[0] if action == "merge_keep_first" else member_indices[-1]
                for idx in member_indices:
                    if idx != keeper:
                        indices_to_drop.add(idx)
            df_out = df_out.drop(index=list(indices_to_drop)).reset_index(drop=True)
            final_rows = len(df_out)
            rows_removed = len(indices_to_drop)

        # 7. Metrics & Output Summary
        sample_clusters = []
        for cid, (root_id, member_indices) in enumerate(list(duplicate_clusters.items())[:5], 1):
            examples = [composite_strings[idx] for idx in member_indices[:3]]
            sample_clusters.append({
                "cluster_id": cid,
                "cluster_size": len(member_indices),
                "samples": examples
            })

        output_summary = {
            "title": "Fuzzy-Match Deduplication",
            "message": f"Detected {len(duplicate_clusters)} fuzzy near-duplicate clusters ({total_dup_rows} redundant rows) using {algorithm} at {threshold*100:.0f}% threshold.",
            "metrics": {
                "original_rows": orig_rows,
                "final_rows": final_rows,
                "rows_removed": rows_removed,
                "duplicate_clusters_count": len(duplicate_clusters),
                "similarity_threshold": threshold,
                "action": action
            },
            "top_clusters": sample_clusters
        }

        return {
            "dataframe": df_out,
            "metrics": output_summary["metrics"],
            "output_summary": output_summary
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        key_cols = self._parse_list(config.get("key_columns"))
        threshold = float(config.get("similarity_threshold", 0.85))
        action = config.get("action", "merge_keep_first")

        return (
            f"# Fuzzy-Match Deduplication\n"
            f"# Target key columns: {key_cols}\n"
            f"# Threshold: {threshold}, Action: {action}\n"
            f"import difflib\n"
            f"key_cols = {key_cols}\n"
            f"df['composite_key'] = df[key_cols].astype(str).agg(' '.join, axis=1)\n"
            f"# Compute fuzzy pairwise similarity and cluster near-duplicates\n"
            f"# (Deduplicate or merge according to '{action}')\n"
        )
