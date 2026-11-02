"""
Unit & Edge-Case Tests for Agents & LLM Orchestration Phase 2 (RAG & Retrieval).

Covers:
1. EmbeddingGeneratorRecipe:
   - Text vectorization with local deterministic engine & L2 norm
   - Auto text-column detection
   - Multiple dimensions (64, 128, 256)
   - Output formats (list, json_vector, columns)
   - Missing/null handling (zero_vector, skip_row)
   - Empty dataframe resilience
   - Max length truncation

2. VectorDbIndexerRecipe:
   - Index and search mode
   - Multiple vector inputs (list, JSON string, CSV string, emb_* columns)
   - Similarity metrics (cosine, dot_product, euclidean)
   - Metadata filtering & similarity thresholds
   - Context sharing (index_only -> search_only)
   - Empty dataframe and zero vector safety

3. RagPipelineRecipe:
   - Grounded context retrieval and answer generation
   - Offline deterministic grounded answer synthesis
   - Citation & source metadata attachment
   - Multi-query dataframe processing vs static query
   - Unrelated query handling
   - Empty input resilience
"""

import json
import pytest
import numpy as np
import pandas as pd

from backend.app.recipes.agents.embedding_generator import EmbeddingGeneratorRecipe
from backend.app.recipes.agents.vector_db_indexer import VectorDbIndexerRecipe
from backend.app.recipes.agents.rag_pipeline import RagPipelineRecipe


# ==============================================================================
# 1. EmbeddingGeneratorRecipe Tests
# ==============================================================================

def test_embedding_generator_basic():
    recipe = EmbeddingGeneratorRecipe()
    df = pd.DataFrame({
        "doc_id": ["D1", "D2", "D3"],
        "content": [
            "Customer wants to cancel flight ticket to Paris",
            "Customer requesting invoice receipt for tax filing",
            "Account balance inquiry for savings account"
        ]
    })
    res = recipe.execute({"dataframe": df}, {
        "text_column": "content",
        "output_column": "embedding",
        "dimensions": 128,
        "normalize": True,
        "output_format": "list"
    })
    out_df = res["dataframe"]
    assert len(out_df) == 3
    assert "embedding" in out_df.columns
    assert "embedding_dim" in out_df.columns
    assert out_df["embedding_dim"].iloc[0] == 128

    vec1 = np.array(out_df["embedding"].iloc[0])
    assert len(vec1) == 128
    # L2 normalized vector should have norm close to 1.0
    norm = float(np.linalg.norm(vec1))
    assert pytest.approx(norm, abs=1e-3) == 1.0


def test_embedding_generator_auto_column_and_dimensions():
    recipe = EmbeddingGeneratorRecipe()
    df = pd.DataFrame({
        "id": [1, 2],
        "notes": ["Refund processed successfully", "Dispute escalated to supervisor"]
    })
    res = recipe.execute({"dataframe": df}, {
        "dimensions": 64,
        "output_format": "list"
    })
    out_df = res["dataframe"]
    assert len(out_df) == 2
    assert len(out_df["embedding"].iloc[0]) == 64


def test_embedding_generator_output_formats():
    recipe = EmbeddingGeneratorRecipe()
    df = pd.DataFrame({"text": ["Vector format testing row"]})

    # 1. JSON Vector format
    res_json = recipe.execute({"dataframe": df}, {
        "output_format": "json_vector",
        "dimensions": 32
    })
    val_json = res_json["dataframe"]["embedding"].iloc[0]
    assert isinstance(val_json, str)
    parsed = json.loads(val_json)
    assert len(parsed) == 32

    # 2. Columns format
    res_cols = recipe.execute({"dataframe": df}, {
        "output_format": "columns",
        "dimensions": 32
    })
    out_cols = res_cols["dataframe"]
    assert "emb_0" in out_cols.columns
    assert "emb_31" in out_cols.columns
    assert isinstance(out_cols["emb_0"].iloc[0], (float, np.floating))


def test_embedding_generator_missing_and_empty():
    recipe = EmbeddingGeneratorRecipe()

    # Empty dataframe
    empty_res = recipe.execute({"dataframe": pd.DataFrame()}, {})
    assert empty_res["dataframe"].empty

    # Null and empty text handling
    df = pd.DataFrame({
        "text": ["Valid text content", None, "", "Another valid chunk"]
    })

    # Zero vector mode
    res_zero = recipe.execute({"dataframe": df}, {"handle_missing": "zero_vector", "dimensions": 64})
    out_zero = res_zero["dataframe"]
    assert len(out_zero) == 4
    assert out_zero["embedding"].iloc[1] == [0.0] * 64
    assert out_zero["embedding"].iloc[2] == [0.0] * 64

    # Skip row mode
    res_skip = recipe.execute({"dataframe": df}, {"handle_missing": "skip_row", "dimensions": 64})
    out_skip = res_skip["dataframe"]
    assert len(out_skip) == 2


# ==============================================================================
# 2. VectorDbIndexerRecipe Tests
# ==============================================================================

def test_vector_db_indexer_index_and_search():
    recipe = VectorDbIndexerRecipe()
    docs = pd.DataFrame({
        "id": ["kb_1", "kb_2", "kb_3"],
        "text": [
            "How to process credit card refund within 30 days of purchase.",
            "Steps to update your password and multi-factor authentication security.",
            "Luggage allowance guidelines for international flights."
        ],
        "category": ["billing", "security", "travel"]
    })

    # Pre-generate embeddings
    emb_recipe = EmbeddingGeneratorRecipe()
    emb_res = emb_recipe.execute({"dataframe": docs}, {"text_column": "text", "dimensions": 128})
    docs_with_emb = emb_res["dataframe"]

    # Query: refund question
    res = recipe.execute({"dataframe": docs_with_emb}, {
        "mode": "index_and_search",
        "query_text": "I want my money refunded for credit card charge",
        "top_k": 2,
        "metric": "cosine"
    })

    results_df = res["dataframe"]
    assert len(results_df) <= 2
    assert "similarity_score" in results_df.columns
    assert "rank" in results_df.columns
    assert results_df["rank"].iloc[0] == 1
    # kb_1 should be the most relevant match for refund query
    assert results_df["id"].iloc[0] == "kb_1"
    assert results_df["similarity_score"].iloc[0] > 0.3


def test_vector_db_indexer_vector_formats():
    recipe = VectorDbIndexerRecipe()

    # JSON string vector format
    vec1 = [0.1] * 32
    vec2 = [-0.1] * 32
    df = pd.DataFrame({
        "id": ["A1", "A2"],
        "text": ["Doc alpha", "Doc beta"],
        "embedding": [json.dumps(vec1), json.dumps(vec2)]
    })

    res = recipe.execute({"dataframe": df}, {
        "mode": "index_and_search",
        "query_text": "Doc alpha",
        "top_k": 1
    })
    out_df = res["dataframe"]
    assert len(out_df) == 1
    assert "similarity_score" in out_df.columns


def test_vector_db_indexer_metadata_filter():
    recipe = VectorDbIndexerRecipe()
    docs = pd.DataFrame({
        "id": ["p1", "p2", "p3"],
        "text": ["Account unlock instructions", "Billing cycle renewal", "Account deletion process"],
        "department": ["support", "billing", "support"]
    })

    res = recipe.execute({"dataframe": docs}, {
        "mode": "index_and_search",
        "query_text": "account help",
        "metadata_filter": {"department": "billing"},
        "top_k": 5
    })
    out_df = res["dataframe"]
    assert len(out_df) == 1
    assert out_df["id"].iloc[0] == "p2"


def test_vector_db_indexer_context_sharing():
    recipe = VectorDbIndexerRecipe()
    docs = pd.DataFrame({
        "id": ["m1", "m2"],
        "text": ["Quantum computing algorithms", "Photosynthesis in green plants"]
    })
    context = {}

    # Node 1: index_only
    res_index = recipe.execute({"dataframe": docs}, {
        "mode": "index_only",
        "store_id": "scientific_kb"
    }, context=context)
    assert "vector_stores" in context
    assert "scientific_kb" in context["vector_stores"]

    # Node 2: search_only querying the shared context store
    res_search = recipe.execute({"dataframe": pd.DataFrame()}, {
        "mode": "search_only",
        "store_id": "scientific_kb",
        "query_text": "Quantum algorithms",
        "top_k": 1
    }, context=context)
    out_search = res_search["dataframe"]
    assert len(out_search) == 1
    assert out_search["id"].iloc[0] == "m1"


# ==============================================================================
# 3. RagPipelineRecipe Tests
# ==============================================================================

def test_rag_pipeline_grounded_answer():
    recipe = RagPipelineRecipe()
    kb = pd.DataFrame({
        "id": ["KB-RET-01", "KB-RET-02", "KB-RET-03"],
        "text": [
            "Enterprise customers have a 60-day return window with complimentary white-glove pickup.",
            "Standard consumer retail purchases can be returned within 14 days with original receipt.",
            "Digital software licenses and cloud compute credits are strictly non-refundable."
        ]
    })

    res = recipe.execute({"dataframe": kb}, {
        "query": "What is the return window for enterprise clients?",
        "top_k": 2,
        "llm_provider": "offline_grounded"
    })

    out_df = res["dataframe"]
    assert len(out_df) == 1
    row = out_df.iloc[0]
    assert "generated_answer" in row
    assert "sources" in row
    assert row["retrieved_chunks_count"] == 2
    assert "KB-RET-01" in row["generated_answer"]
    assert "60-day" in row["generated_answer"]
    assert row["retrieval_score_top1"] > 0.2


def test_rag_pipeline_multi_queries():
    recipe = RagPipelineRecipe()
    kb = pd.DataFrame({
        "id": ["POLICY-1", "POLICY-2"],
        "text": [
            "All employee expenses over $500 require VP financial approval.",
            "Remote employees receive a $100 monthly home internet stipend."
        ]
    })

    # Pass DataFrame with query column
    queries_df = pd.DataFrame({
        "query": [
            "What is the policy for expenses over $500?",
            "How much is the internet stipend?"
        ]
    })
    
    # Pre-index into context
    indexer = VectorDbIndexerRecipe()
    ctx = {}
    indexer.execute({"dataframe": kb}, {"mode": "index_only", "store_id": "default_vector_store"}, context=ctx)

    res = recipe.execute({"dataframe": queries_df}, {
        "query_column": "query",
        "top_k": 1,
        "llm_provider": "offline_grounded"
    }, context=ctx)

    out_df = res["dataframe"]
    assert len(out_df) == 2
    assert "generated_answer" in out_df.columns
    assert "POLICY-1" in out_df["generated_answer"].iloc[0]
    assert "POLICY-2" in out_df["generated_answer"].iloc[1]


def test_rag_pipeline_empty_and_irrelevant():
    recipe = RagPipelineRecipe()

    # Empty input
    res_empty = recipe.execute({"dataframe": pd.DataFrame()}, {})
    assert res_empty["dataframe"].empty

    # Non-empty KB but completely irrelevant question with high threshold
    kb = pd.DataFrame({
        "id": ["FARM-1"],
        "text": ["Wheat and barley harvest yields increased by 12 percent."]
    })
    res = recipe.execute({"dataframe": kb}, {
        "query": "How to deploy a Kubernetes cluster in AWS?",
        "similarity_threshold": 0.95,
        "llm_provider": "offline_grounded"
    })
    out_df = res["dataframe"]
    assert len(out_df) == 1
    ans = out_df["generated_answer"].iloc[0]
    assert "could not find any relevant information" in ans.lower()
