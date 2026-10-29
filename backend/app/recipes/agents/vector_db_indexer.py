"""
Vector DB Indexer & Query recipe (Agents & LLM Orchestration -> RAG & Retrieval).

Indexes dense vector embeddings into an in-memory vector store and executes
top-K semantic similarity queries with metric customization and metadata filtering.
Supports sharing indices across DAG nodes via execution context.
"""

import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from backend.app.recipes.base.recipe import BaseRecipe
from backend.app.recipes.agents.embedding_generator import _deterministic_local_embedding


def _parse_vector(raw_val: Any, target_dim: Optional[int] = None) -> Optional[np.ndarray]:
    """Robustly parses a vector from list, numpy array, JSON string, or CSV string."""
    if raw_val is None or (isinstance(raw_val, float) and np.isnan(raw_val)):
        return None
    if isinstance(raw_val, (list, tuple)):
        try:
            return np.array([float(x) for x in raw_val], dtype=np.float32)
        except Exception:
            return None
    if isinstance(raw_val, np.ndarray):
        return raw_val.astype(np.float32)
    if isinstance(raw_val, str):
        val = raw_val.strip()
        if val.startswith("[") and val.endswith("]"):
            try:
                parsed = json.loads(val)
                return np.array([float(x) for x in parsed], dtype=np.float32)
            except Exception:
                pass
        # Try comma-separated
        if "," in val:
            try:
                parts = [float(x.strip()) for x in val.split(",") if x.strip()]
                return np.array(parts, dtype=np.float32)
            except Exception:
                pass
    return None


class VectorStore:
    """In-memory vectorized search engine for DAG execution context."""

    def __init__(self, store_id: str, metric: str = "cosine"):
        self.store_id = store_id
        self.metric = metric
        self.doc_ids: List[str] = []
        self.doc_texts: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.matrix: Optional[np.ndarray] = None
        self.dim: int = 0

    def build(self, doc_ids: List[str], doc_texts: List[str], metadatas: List[Dict[str, Any]], vectors: np.ndarray):
        self.doc_ids = doc_ids
        self.doc_texts = doc_texts
        self.metadatas = metadatas
        self.matrix = vectors
        self.dim = int(vectors.shape[1]) if vectors.ndim == 2 and vectors.shape[0] > 0 else 0

    def query(
        self,
        query_vec: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        if self.matrix is None or len(self.matrix) == 0 or query_vec is None:
            return []

        # Dimension alignment
        q = query_vec.copy()
        if len(q) != self.dim:
            if len(q) < self.dim:
                q = np.pad(q, (0, self.dim - len(q)))
            else:
                q = q[:self.dim]

        # Calculate scores
        if self.metric == "dot_product":
            scores = np.dot(self.matrix, q)
        elif self.metric == "euclidean":
            diffs = self.matrix - q
            dist = np.linalg.norm(diffs, axis=1)
            scores = 1.0 / (1.0 + dist)
        else:  # cosine
            q_norm = np.linalg.norm(q)
            q_norm = 1.0 if q_norm == 0 else q_norm
            norms = np.linalg.norm(self.matrix, axis=1)
            norms = np.where(norms == 0, 1.0, norms)
            scores = np.dot(self.matrix, q) / (norms * q_norm)

        # Apply metadata filtering
        results = []
        for i, score in enumerate(scores):
            sc = float(score)
            if sc < threshold:
                continue

            meta = self.metadatas[i]
            if metadata_filter:
                match = True
                for k, expected_v in metadata_filter.items():
                    actual_v = meta.get(k)
                    if isinstance(expected_v, list):
                        if actual_v not in expected_v:
                            match = False
                            break
                    elif actual_v != expected_v:
                        match = False
                        break
                if not match:
                    continue

            results.append({
                "id": self.doc_ids[i],
                "text": self.doc_texts[i],
                "similarity_score": round(sc, 5),
                "metadata": meta
            })

        # Sort descending by similarity
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:top_k]


class VectorDbIndexerRecipe(BaseRecipe):
    recipe_id = "vector_db_indexer"
    name = "Vector DB Indexer & Query"
    version = "1.0.0"
    category = "agents"
    media_type = "agents"
    description = "Indexes embedding vectors into high-performance in-memory vector store and executes top-K semantic similarity queries with metadata filtering."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "title": "Operation Mode",
                    "enum": ["index_and_search", "index_only", "search_only"],
                    "default": "index_and_search",
                    "description": "'index_and_search' indexes input and runs query; 'index_only' prepares store in context; 'search_only' queries previously stored index."
                },
                "vector_column": {
                    "type": "string",
                    "title": "Vector Column",
                    "default": "embedding",
                    "description": "Column holding embedding vectors (list, array, JSON string, or CSV string)."
                },
                "id_column": {
                    "type": "string",
                    "title": "Document ID Column",
                    "default": "id",
                    "description": "Unique identifier column for each document or chunk. Auto-generated if absent."
                },
                "text_column": {
                    "type": "string",
                    "title": "Document Text Column",
                    "default": "text",
                    "description": "Column containing original text snippet."
                },
                "query_text": {
                    "type": "string",
                    "title": "Search Query Text",
                    "default": "",
                    "description": "Text query to search against the indexed documents."
                },
                "top_k": {
                    "type": "integer",
                    "title": "Top-K Results",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Maximum number of most relevant documents to retrieve."
                },
                "metric": {
                    "type": "string",
                    "title": "Similarity Metric",
                    "enum": ["cosine", "dot_product", "euclidean"],
                    "default": "cosine",
                    "description": "Distance metric for comparing vectors."
                },
                "similarity_threshold": {
                    "type": "number",
                    "title": "Similarity Threshold",
                    "default": 0.0,
                    "minimum": -1.0,
                    "maximum": 1.0,
                    "description": "Minimum similarity score required to include a document in results."
                },
                "metadata_filter": {
                    "type": "object",
                    "title": "Metadata Filter",
                    "default": {},
                    "description": "Key-value filter criteria to match against document metadata (e.g. {'status': 'active'})."
                },
                "store_id": {
                    "type": "string",
                    "title": "Vector Store ID",
                    "default": "default_vector_store",
                    "description": "Namespace key used to persist index inside workflow execution context."
                }
            },
            "required": ["mode"]
        }

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors = []
        mode = config.get("mode", "index_and_search")
        if mode not in ["index_and_search", "index_only", "search_only"]:
            errors.append(f"Invalid mode '{mode}'. Must be 'index_and_search', 'index_only', or 'search_only'.")
        top_k = config.get("top_k", 5)
        if not isinstance(top_k, int) or top_k < 1:
            errors.append("top_k must be a positive integer >= 1.")
        return errors

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        start_time = time.time()
        
        # 1. Resolve Input DataFrame
        raw_input = None
        for k in ["dataframe", "input", "df"]:
            if k in inputs and inputs[k] is not None:
                raw_input = inputs[k]
                break

        if isinstance(raw_input, pd.DataFrame):
            df = raw_input.copy()
        elif isinstance(raw_input, list):
            df = pd.DataFrame(raw_input)
        elif isinstance(raw_input, dict) and "records" in raw_input:
            df = pd.DataFrame(raw_input["records"])
        elif isinstance(raw_input, dict):
            df = pd.DataFrame([raw_input])
        else:
            df = pd.DataFrame()


        mode = str(config.get("mode") or "index_and_search").strip().lower()
        vector_col = str(config.get("vector_column") or "embedding").strip()
        id_col = str(config.get("id_column") or "id").strip()
        text_col = str(config.get("text_column") or "text").strip()
        query_text = str(config.get("query_text") or "").strip()
        top_k = int(config.get("top_k") or 5)
        metric = str(config.get("metric") or "cosine").strip().lower()
        threshold = float(config.get("similarity_threshold") or 0.0)
        metadata_filter = config.get("metadata_filter") or {}
        store_id = str(config.get("store_id") or "default_vector_store").strip()

        # Context store container
        context_dict = context if (context is not None and isinstance(context, dict)) else {}
        if "vector_stores" not in context_dict:
            context_dict["vector_stores"] = {}

        # 2. Handle 'search_only' Mode
        if mode == "search_only":
            store: Optional[VectorStore] = context_dict["vector_stores"].get(store_id)
            if not store:
                # Return empty dataframe with search schema
                out_df = pd.DataFrame(columns=["rank", "id", "text", "similarity_score"])
                return {
                    "dataframe": out_df,
                    "store_id": store_id,
                    "results_count": 0,
                    "summary": {"error": f"Vector store '{store_id}' not found in context.", "results_count": 0}
                }
            # Embed query text
            query_vec = np.array(_deterministic_local_embedding(query_text, dim=store.dim, normalize=True), dtype=np.float32)
            results = store.query(query_vec, top_k=top_k, threshold=threshold, metadata_filter=metadata_filter)
            records = []
            for rank, r in enumerate(results, start=1):
                rec = {"rank": rank, "id": r["id"], "text": r["text"], "similarity_score": r["similarity_score"]}
                rec.update(r["metadata"])
                records.append(rec)
            out_df = pd.DataFrame(records) if records else pd.DataFrame(columns=["rank", "id", "text", "similarity_score"])
            return {
                "dataframe": out_df,
                "store_id": store_id,
                "results_count": len(out_df),
                "summary": {"mode": mode, "results_count": len(out_df), "top_k": top_k}
            }

        # 3. Handle Empty DataFrame in Index Modes
        if df.empty:
            out_df = pd.DataFrame(columns=["rank", "id", "text", "similarity_score"])
            return {
                "dataframe": out_df,
                "store_id": store_id,
                "indexed_count": 0,
                "results_count": 0,
                "summary": {"total_rows": 0, "indexed_count": 0, "results_count": 0}
            }

        # 4. Resolve Columns
        if id_col not in df.columns:
            df[id_col] = [f"doc_{i+1}" for i in range(len(df))]

        if text_col not in df.columns:
            # Find first text column
            str_cols = [c for c in df.columns if c != id_col and (df[c].dtype == object or str(df[c].dtype).startswith("str"))]
            text_col = str_cols[0] if str_cols else id_col

        # Parse Vectors or auto-generate if missing
        vectors: List[Optional[np.ndarray]] = []
        if vector_col in df.columns:
            for val in df[vector_col]:
                vectors.append(_parse_vector(val))
        else:
            # Look for emb_0..emb_N numeric columns
            emb_cols = [c for c in df.columns if c.startswith("emb_") and c[4:].isdigit()]
            if emb_cols:
                emb_cols.sort(key=lambda x: int(x[4:]))
                mat = df[emb_cols].values.astype(np.float32)
                vectors = [mat[i] for i in range(len(df))]
            else:
                # Fallback: Auto-generate embeddings using local fast embedder
                for text in df[text_col]:
                    vec = _deterministic_local_embedding(str(text or ""), dim=128, normalize=True)
                    vectors.append(np.array(vec, dtype=np.float32))

        # Filter valid indexed vectors
        valid_indices = [i for i, v in enumerate(vectors) if v is not None and len(v) > 0]
        if not valid_indices:
            out_df = pd.DataFrame(columns=["rank", "id", "text", "similarity_score"])
            return {
                "dataframe": out_df,
                "store_id": store_id,
                "indexed_count": 0,
                "results_count": 0,
                "summary": {"error": "No valid embedding vectors could be parsed from input.", "indexed_count": 0}
            }

        first_dim = len(vectors[valid_indices[0]])
        filtered_vectors = np.array([vectors[i] for i in valid_indices], dtype=np.float32)
        filtered_ids = [str(df.iloc[i][id_col]) for i in valid_indices]
        filtered_texts = [str(df.iloc[i][text_col]) for i in valid_indices]

        # Extract metadata dictionaries
        metadata_cols = [c for c in df.columns if c not in [vector_col, id_col, text_col]]
        metadatas = [df.iloc[i][metadata_cols].to_dict() for i in valid_indices]

        # 5. Build In-Memory Vector Store
        store = VectorStore(store_id=store_id, metric=metric)
        store.build(filtered_ids, filtered_texts, metadatas, filtered_vectors)
        context_dict["vector_stores"][store_id] = store

        duration_ms = round((time.time() - start_time) * 1000, 2)

        # 6. Index Only Mode
        if mode == "index_only":
            out_df = pd.DataFrame([{
                "store_id": store_id,
                "indexed_documents": len(filtered_ids),
                "vector_dimension": first_dim,
                "metric": metric,
                "status": "ready"
            }])
            return {
                "dataframe": out_df,
                "store_id": store_id,
                "indexed_count": len(filtered_ids),
                "summary": {
                    "mode": mode,
                    "store_id": store_id,
                    "indexed_count": len(filtered_ids),
                    "dimension": first_dim,
                    "duration_ms": duration_ms
                }
            }

        # 7. Index and Search Mode
        # Determine query vector
        if not query_text and len(df) > 0:
            query_text = str(df.iloc[0][text_col])

        query_vec = np.array(_deterministic_local_embedding(query_text, dim=first_dim, normalize=True), dtype=np.float32)
        search_results = store.query(query_vec, top_k=top_k, threshold=threshold, metadata_filter=metadata_filter)

        rows = []
        for rank, r in enumerate(search_results, start=1):
            row = {
                "rank": rank,
                "id": r["id"],
                "text": r["text"],
                "similarity_score": r["similarity_score"]
            }
            row.update(r["metadata"])
            rows.append(row)

        out_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["rank", "id", "text", "similarity_score"])

        summary = {
            "mode": mode,
            "store_id": store_id,
            "query_text": query_text,
            "indexed_count": len(filtered_ids),
            "results_returned": len(out_df),
            "top_k": top_k,
            "metric": metric,
            "duration_ms": duration_ms
        }

        return {
            "dataframe": out_df,
            "store_id": store_id,
            "indexed_count": len(filtered_ids),
            "results_count": len(out_df),
            "summary": summary
        }
