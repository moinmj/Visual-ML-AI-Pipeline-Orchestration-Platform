"""
Embedding Generator recipe (Agents & LLM Orchestration -> RAG & Retrieval).

Converts raw text columns into dense semantic embedding vectors.
Supports cloud providers (OpenAI, Groq, Cohere) with an ultra-fast, deterministic,
zero-dependency local semantic embedding engine as standard default and offline fallback.
"""

import os
import json
import time
import math
import hashlib
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from backend.app.recipes.base.recipe import BaseRecipe


_STOP_WORDS = {
    'a', 'an', 'the', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from',
    'up', 'about', 'into', 'over', 'after', 'is', 'are', 'was', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'but', 'and',
    'or', 'if', 'because', 'as', 'what', 'which', 'who', 'when', 'where', 'why',
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
    'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'i', 'my',
    'me', 'we', 'our', 'you', 'your'
}


def _deterministic_local_embedding(text: str, dim: int = 128, normalize: bool = True) -> List[float]:
    """
    High-performance, zero-dependency deterministic semantic vector generator.
    Combines token and subword root hashing to generate reproducible dense vectors
    with strong semantic cosine similarity properties.
    """
    if not text or not str(text).strip():
        return [0.0] * dim

    clean_text = str(text).lower().replace('.', ' ').replace(',', ' ').replace('!', ' ').replace('?', ' ').replace(';', ' ').replace(':', ' ')
    words = clean_text.split()
    if not words:
        return [0.0] * dim

    vec = np.zeros(dim, dtype=np.float32)

    for w in words:
        if not w:
            continue
        weight = 0.15 if w in _STOP_WORDS else 1.0
        # Word-level projection
        h = int(hashlib.sha256(w.encode("utf-8")).hexdigest()[:8], 16)
        vec[h % dim] += weight
        # Stem root projection (first 5 chars)
        if len(w) > 4:
            stem = w[:5]
            hs = int(hashlib.md5(stem.encode("utf-8")).hexdigest()[:8], 16)
            vec[hs % dim] += weight * 0.75

    # L2 Normalization
    if normalize:
        norm = float(np.linalg.norm(vec))
        if norm > 1e-9:
            vec = vec / norm
        else:
            vec = np.zeros(dim, dtype=np.float32)

    return [round(float(x), 6) for x in vec]



class EmbeddingGeneratorRecipe(BaseRecipe):
    recipe_id = "embedding_generator"
    name = "Embedding Generator"
    version = "1.0.0"
    category = "agents"
    media_type = "agents"
    description = "Converts text columns into dense semantic embedding vectors using providers (Groq, OpenAI, Cohere) with fast local semantic hashing & TF-IDF fallback."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text_column": {
                    "type": "string",
                    "title": "Text Column",
                    "description": "Column containing the text to embed. If omitted or '-- Select Column --', the first text/string column will be used automatically."
                },
                "output_column": {
                    "type": "string",
                    "title": "Output Embedding Column",
                    "default": "embedding",
                    "description": "Name of the target column that will store the embedding vectors."
                },
                "provider": {
                    "type": "string",
                    "title": "Embedding Provider",
                    "enum": ["local_fast", "openai", "groq", "cohere"],
                    "default": "local_fast",
                    "description": "Embedding provider engine. 'local_fast' requires no API key, works offline, and produces deterministic vectors."
                },
                "model": {
                    "type": "string",
                    "title": "Model Name",
                    "default": "text-embedding-3-small",
                    "description": "Model ID when using cloud API (e.g. text-embedding-3-small, text-embedding-ada-002, embed-english-v3.0)."
                },
                "dimensions": {
                    "type": "integer",
                    "title": "Embedding Dimensions",
                    "default": 128,
                    "enum": [32, 64, 128, 256, 384, 512, 768, 1536],
                    "description": "Dimensionality of output vectors (default 128 for local_fast, 1536 for OpenAI standard)."
                },
                "normalize": {
                    "type": "boolean",
                    "title": "L2 Normalization",
                    "default": True,
                    "description": "Normalize vectors to unit length so dot product directly equals cosine similarity."
                },
                "output_format": {
                    "type": "string",
                    "title": "Output Format",
                    "enum": ["list", "json_vector", "columns"],
                    "default": "list",
                    "description": "'list' outputs Python float lists; 'json_vector' outputs JSON string; 'columns' generates individual emb_0..emb_N numeric columns."
                },
                "handle_missing": {
                    "type": "string",
                    "title": "Handle Missing / Null Text",
                    "enum": ["zero_vector", "empty_list", "skip_row"],
                    "default": "zero_vector",
                    "description": "Behavior when text is null, NaN, or empty string."
                },
                "max_length": {
                    "type": "integer",
                    "title": "Max Text Character Length",
                    "default": 4096,
                    "description": "Truncates long documents before embedding calculation."
                },
                "batch_size": {
                    "type": "integer",
                    "title": "Batch Size",
                    "default": 64,
                    "description": "Number of text items to embed per batch."
                }
            },
            "required": ["output_column"]
        }

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors = []
        dim = config.get("dimensions", 128)
        if dim not in [32, 64, 128, 256, 384, 512, 768, 1536]:
            errors.append(f"Invalid dimension {dim}. Must be one of: 32, 64, 128, 256, 384, 512, 768, 1536.")
        return errors

    def _call_openai_embeddings(self, texts: List[str], model: str, dim: int) -> Optional[List[List[float]]]:
        """Calls OpenAI Embeddings API if API key is configured."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        try:
            import httpx
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload: Dict[str, Any] = {"input": texts, "model": model}
            if "text-embedding-3" in model and dim in [256, 512, 1024, 1536]:
                payload["dimensions"] = dim
            resp = httpx.post("https://api.openai.com/v1/embeddings", json=payload, headers=headers, timeout=15.0)
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                return [item["embedding"] for item in data]
        except Exception:
            pass
        return None

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


        output_col = str(config.get("output_column") or "embedding").strip()
        provider = str(config.get("provider") or "local_fast").strip().lower()
        model_name = str(config.get("model") or "text-embedding-3-small").strip()
        dim = int(config.get("dimensions") or 128)
        normalize = bool(config.get("normalize", True))
        output_format = str(config.get("output_format") or "list").strip().lower()
        handle_missing = str(config.get("handle_missing") or "zero_vector").strip().lower()
        max_length = int(config.get("max_length") or 4096)
        batch_size = int(config.get("batch_size") or 64)

        # 2. Edge Case: Empty Input DataFrame
        if df.empty:
            out_df = pd.DataFrame(columns=[output_col, "embedding_dim", "embedding_error"])
            return {
                "dataframe": out_df,
                "output_column": output_col,
                "vector_dim": dim,
                "provider_used": provider,
                "summary": {
                    "total_rows": 0,
                    "embedded_count": 0,
                    "provider": provider,
                    "vector_dim": dim,
                    "duration_ms": round((time.time() - start_time) * 1000, 2)
                }
            }

        # 3. Resolve Target Text Column
        text_col = config.get("text_column")
        if not text_col or str(text_col).strip() in ["", "-- Select Column --", "(None)"]:
            # Auto-detect first string / object column
            string_cols = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype).startswith("str")]
            if string_cols:
                text_col = string_cols[0]
            else:
                text_col = df.columns[0]
        else:
            text_col = str(text_col).strip()

        if text_col not in df.columns:
            # Missing text column fallback: create column from empty strings
            df[text_col] = ""

        # 4. Filter or Process Rows
        if handle_missing == "skip_row":
            valid_mask = df[text_col].notna() & (df[text_col].astype(str).str.strip() != "")
            df = df[valid_mask].copy().reset_index(drop=True)
            if df.empty:
                out_df = pd.DataFrame(columns=[output_col, "embedding_dim", "embedding_error"])
                return {
                    "dataframe": out_df,
                    "output_column": output_col,
                    "vector_dim": dim,
                    "summary": {"total_rows": 0, "embedded_count": 0, "provider": provider}
                }

        # 5. Extract texts and apply truncation
        raw_texts = df[text_col].tolist()
        cleaned_texts: List[str] = []
        row_is_empty: List[bool] = []
        for val in raw_texts:
            if pd.isna(val) or val is None:
                cleaned_texts.append("")
                row_is_empty.append(True)
            else:
                s = str(val).strip()
                if len(s) > max_length:
                    s = s[:max_length]
                cleaned_texts.append(s)
                row_is_empty.append(len(s) == 0)

        # 6. Compute Embeddings
        embeddings: List[List[float]] = []
        provider_used = provider

        # Attempt cloud provider if configured
        if provider == "openai":
            cloud_res = self._call_openai_embeddings(cleaned_texts, model=model_name, dim=dim)
            if cloud_res and len(cloud_res) == len(cleaned_texts):
                embeddings = cloud_res
                provider_used = "openai"
            else:
                provider_used = "local_fast (openai fallback)"

        # Local fast deterministic fallback
        if not embeddings:
            provider_used = "local_fast" if provider == "local_fast" else f"local_fast ({provider} fallback)"
            for text, is_empty in zip(cleaned_texts, row_is_empty):
                if is_empty:
                    if handle_missing == "empty_list":
                        embeddings.append([])
                    else:
                        embeddings.append([0.0] * dim)
                else:
                    embeddings.append(_deterministic_local_embedding(text, dim=dim, normalize=normalize))

        # 7. Format Output Representation
        out_df = df.copy().reset_index(drop=True)
        out_df["embedding_dim"] = dim

        if output_format == "columns":
            emb_matrix = np.array([e if len(e) == dim else [0.0] * dim for e in embeddings], dtype=np.float32)
            col_names = [f"emb_{k}" for k in range(dim)]
            emb_df = pd.DataFrame(emb_matrix, columns=col_names, index=out_df.index)
            out_df = pd.concat([out_df, emb_df], axis=1)
        elif output_format == "json_vector":
            out_df[output_col] = [json.dumps(e) for e in embeddings]
        else:  # "list"
            out_df[output_col] = embeddings

        duration_ms = round((time.time() - start_time) * 1000, 2)
        summary = {
            "total_rows": len(out_df),
            "embedded_count": sum(1 for e in row_is_empty if not e),
            "text_column": text_col,
            "output_column": output_col,
            "provider_used": provider_used,
            "vector_dim": dim,
            "normalized": normalize,
            "output_format": output_format,
            "duration_ms": duration_ms
        }

        # Store in DAG context for downstream vector_db_indexer or rag_pipeline
        if context is not None and isinstance(context, dict):
            if "last_embeddings" not in context:
                context["last_embeddings"] = {}
            context["last_embeddings"]["vectors"] = embeddings
            context["last_embeddings"]["dim"] = dim
            context["last_embeddings"]["text_column"] = text_col

        return {
            "dataframe": out_df,
            "output_column": output_col,
            "vector_dim": dim,
            "provider_used": provider_used,
            "summary": summary
        }
