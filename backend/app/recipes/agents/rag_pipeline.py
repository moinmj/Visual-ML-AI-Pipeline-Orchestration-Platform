"""
RAG Retrieval Pipeline recipe (Agents & LLM Orchestration -> RAG & Retrieval).

Coordinates end-to-end Retrieval-Augmented Generation:
1. Context Retrieval (semantic similarity search over knowledge base documents)
2. Grounded Context Assembly (with citation metadata & scores)
3. LLM Answer Generation (Groq / OpenAI API with deterministic grounded offline fallback)
"""

import os
import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from backend.app.recipes.base.recipe import BaseRecipe
from backend.app.recipes.agents.embedding_generator import _deterministic_local_embedding
from backend.app.recipes.agents.vector_db_indexer import VectorStore


def _synthesize_offline_grounded_answer(query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Synthesizes a strictly grounded, deterministic extractive response from retrieved
    knowledge base chunks when running offline or without active LLM API credentials.
    """
    if not retrieved_chunks:
        return "I could not find any relevant information in the provided knowledge base to answer your question."

    top_chunk = retrieved_chunks[0]
    doc_id = top_chunk.get("id", "Doc-1")
    text = top_chunk.get("text", "").strip()
    score = top_chunk.get("similarity_score", 0.0)

    # Extract the most salient sentences
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 10]
    relevant_excerpt = ". ".join(sentences[:2]) + "." if sentences else text[:250]

    citations = ", ".join([f"[{c.get('id', f'Doc-{i+1}')} (Relevance: {c.get('similarity_score', 0):.2f})]" for i, c in enumerate(retrieved_chunks)])

    return (
        f"Based on the retrieved enterprise documentation {citations}:\n\n"
        f"{relevant_excerpt}\n\n"
        f"[Source Reference: {doc_id} | Retrieval Confidence: {score:.1%}]"
    )


class RagPipelineRecipe(BaseRecipe):
    recipe_id = "rag_pipeline"
    name = "RAG Retrieval Pipeline"
    version = "1.0.0"
    category = "agents"
    media_type = "agents"
    description = "Retrieves relevant context from a knowledge base and synthesizes grounded answers with an LLM."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "title": "Static Query",
                    "default": "",
                    "description": "Direct question to answer. If blank, questions will be read from the input dataframe query_column."
                },
                "query_column": {
                    "type": "string",
                    "title": "Query Column Name",
                    "default": "query",
                    "description": "Column containing questions if dataframe contains multiple query rows."
                },
                "context_column": {
                    "type": "string",
                    "title": "Knowledge Base Text Column",
                    "default": "text",
                    "description": "Column in knowledge base containing document text chunks."
                },
                "id_column": {
                    "type": "string",
                    "title": "Document ID Column",
                    "default": "id",
                    "description": "Column identifying document/chunk ID."
                },
                "top_k": {
                    "type": "integer",
                    "title": "Top-K Retrieved Chunks",
                    "default": 3,
                    "minimum": 1,
                    "maximum": 20,
                    "description": "Number of most relevant context chunks to inject into the LLM prompt."
                },
                "similarity_threshold": {
                    "type": "number",
                    "title": "Min Similarity Threshold",
                    "default": 0.05,
                    "minimum": -1.0,
                    "maximum": 1.0,
                    "description": "Minimum similarity required to consider a chunk relevant."
                },
                "system_prompt": {
                    "type": "string",
                    "title": "System Prompt",
                    "default": "You are a helpful, enterprise-grade AI assistant. Answer the user question accurately and concisely using ONLY the provided context. If the context does not contain enough information, state that clearly.",
                    "description": "Instruction for the LLM generator."
                },
                "llm_provider": {
                    "type": "string",
                    "title": "LLM Provider",
                    "enum": ["groq", "openai", "offline_grounded"],
                    "default": "groq",
                    "description": "LLM generation provider. Falls back to offline_grounded if credentials are absent."
                },
                "llm_model": {
                    "type": "string",
                    "title": "Model ID",
                    "default": "llama-3.3-70b-versatile",
                    "description": "Model ID (e.g. llama-3.3-70b-versatile on Groq, gpt-4o-mini on OpenAI)."
                },
                "temperature": {
                    "type": "number",
                    "title": "Temperature",
                    "default": 0.1,
                    "minimum": 0.0,
                    "maximum": 2.0,
                    "description": "Sampling temperature for generation."
                },
                "include_citations": {
                    "type": "boolean",
                    "title": "Include Citations",
                    "default": True,
                    "description": "Append source document IDs and relevance scores to output."
                },
                "store_id": {
                    "type": "string",
                    "title": "Vector Store ID",
                    "default": "default_vector_store",
                    "description": "ID of pre-indexed vector store in context, if already prepared."
                }
            },
            "required": ["top_k"]
        }

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors = []
        top_k = config.get("top_k", 3)
        if not isinstance(top_k, int) or top_k < 1:
            errors.append("top_k must be a positive integer >= 1.")
        return errors

    def _call_groq_or_openai(
        self,
        system_prompt: str,
        user_prompt: str,
        provider: str,
        model: str,
        temperature: float
    ) -> Optional[str]:
        """Calls Groq or OpenAI chat completions API if credentials exist."""
        try:
            import httpx
            if provider == "groq" and os.getenv("GROQ_API_KEY"):
                resp = httpx.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}", "Content-Type": "application/json"},
                    json={
                        "model": model or "llama-3.3-70b-versatile",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "temperature": temperature,
                        "max_tokens": 512
                    },
                    timeout=15.0
                )
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"]

            elif provider == "openai" and os.getenv("OPENAI_API_KEY"):
                resp = httpx.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}", "Content-Type": "application/json"},
                    json={
                        "model": model or "gpt-4o-mini",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "temperature": temperature,
                        "max_tokens": 512
                    },
                    timeout=15.0
                )
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"]
        except Exception:
            pass
        return None

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        start_time = time.time()
        
        # 1. Resolve Inputs
        raw_input = None
        for k in ["dataframe", "corpus", "input", "df"]:
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


        static_query = str(config.get("query") or "").strip()
        query_col = str(config.get("query_column") or "query").strip()
        context_col = str(config.get("context_column") or "text").strip()
        id_col = str(config.get("id_column") or "id").strip()
        top_k = int(config.get("top_k") or 3)
        threshold = float(config.get("similarity_threshold") or 0.05)
        system_prompt = str(config.get("system_prompt") or "You are an accurate enterprise AI assistant. Answer using only the provided context.")
        llm_provider = str(config.get("llm_provider") or "groq").strip().lower()
        llm_model = str(config.get("llm_model") or "llama-3.3-70b-versatile").strip()
        temperature = float(config.get("temperature") or 0.1)
        include_citations = bool(config.get("include_citations", True))
        store_id = str(config.get("store_id") or "default_vector_store").strip()

        context_dict = context if (context is not None and isinstance(context, dict)) else {}

        # 2. Edge Case: Empty Input
        if df.empty and not static_query:
            out_df = pd.DataFrame(columns=["query", "generated_answer", "sources", "retrieved_chunks_count"])
            return {
                "dataframe": out_df,
                "summary": {"total_queries": 0, "status": "empty"}
            }

        # 3. Retrieve or Build In-Memory Corpus
        # Check if an existing vector store exists in context
        existing_store: Optional[VectorStore] = context_dict.get("vector_stores", {}).get(store_id)
        
        if existing_store and existing_store.matrix is not None and len(existing_store.matrix) > 0:
            store = existing_store
        else:
            # Build store from current dataframe chunks
            if id_col not in df.columns:
                df[id_col] = [f"doc_{i+1}" for i in range(len(df))]
            if context_col not in df.columns:
                # Find first text column
                str_cols = [c for c in df.columns if c != id_col and (df[c].dtype == object or str(df[c].dtype).startswith("str"))]
                context_col = str_cols[0] if str_cols else id_col

            doc_ids = [str(x) for x in df[id_col]]
            doc_texts = [str(x) for x in df[context_col]]
            metadatas = df.to_dict("records")
            
            # Check for pre-computed vectors
            if "embedding" in df.columns and len(df) > 0 and isinstance(df["embedding"].iloc[0], (list, np.ndarray)):
                vectors = np.array([np.array(v, dtype=np.float32) for v in df["embedding"]], dtype=np.float32)
            else:
                vectors = np.array([_deterministic_local_embedding(t, dim=128, normalize=True) for t in doc_texts], dtype=np.float32)

            store = VectorStore(store_id=store_id, metric="cosine")
            store.build(doc_ids, doc_texts, metadatas, vectors)
            if "vector_stores" not in context_dict:
                context_dict["vector_stores"] = {}
            context_dict["vector_stores"][store_id] = store

        # 4. Resolve Queries to Answer
        query_list: List[str] = []
        if static_query:
            query_list = [static_query]
        elif query_col in df.columns:
            query_list = [str(q).strip() for q in df[query_col] if str(q).strip()]
        else:
            # Default to first row text as inquiry
            query_list = [str(df.iloc[0][context_col])] if len(df) > 0 else ["What is this document about?"]

        # 5. Execute RAG Retrieval & Generation per Query
        out_rows = []
        for q in query_list:
            q_vec = np.array(_deterministic_local_embedding(q, dim=store.dim or 128, normalize=True), dtype=np.float32)
            retrieved = store.query(q_vec, top_k=top_k, threshold=threshold)

            # Format Context Snippets
            context_blocks = []
            sources_meta = []
            for item in retrieved:
                doc_ref = item.get("id", "Unknown")
                sc = item.get("similarity_score", 0.0)
                txt = item.get("text", "")
                context_blocks.append(f"--- Document Source [{doc_ref}] (Relevance Score: {sc:.2f}) ---\n{txt}")
                sources_meta.append({"id": doc_ref, "similarity_score": sc})

            formatted_context = "\n\n".join(context_blocks) if context_blocks else "No relevant documents found."
            user_prompt = f"Context Information:\n{formatted_context}\n\nQuestion: {q}\n\nAnswer:"

            # LLM Generation with fallback
            generated_ans = None
            provider_used = "offline_grounded"
            if llm_provider in ["groq", "openai"]:
                api_ans = self._call_groq_or_openai(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    provider=llm_provider,
                    model=llm_model,
                    temperature=temperature
                )
                if api_ans:
                    generated_ans = api_ans
                    provider_used = llm_provider

            if not generated_ans:
                generated_ans = _synthesize_offline_grounded_answer(q, retrieved)
                provider_used = "offline_grounded"

            top1_score = retrieved[0]["similarity_score"] if retrieved else 0.0

            out_rows.append({
                "query": q,
                "generated_answer": generated_ans,
                "retrieved_chunks_count": len(retrieved),
                "retrieval_score_top1": top1_score,
                "sources": json.dumps(sources_meta),
                "retrieved_context": formatted_context,
                "llm_provider_used": provider_used
            })

        out_df = pd.DataFrame(out_rows)
        duration_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "dataframe": out_df,
            "queries_processed": len(out_df),
            "summary": {
                "queries_processed": len(out_df),
                "top_k": top_k,
                "llm_provider_used": out_rows[0]["llm_provider_used"] if out_rows else "none",
                "duration_ms": duration_ms
            }
        }
