"""
Document Q&A Agent Recipe.
(Agents & LLM Orchestration -> Core Agents)

Answers questions over multi-page documents, PDFs, manuals, and reports
with grounded citations and section/page references.
"""

import re
import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from backend.app.recipes.base.recipe import BaseRecipe
from backend.app.recipes.agents.embedding_generator import _deterministic_local_embedding


def _chunk_document(doc_text: str, chunk_size: int = 400, overlap: int = 50) -> List[Dict[str, Any]]:
    """Splits long document text into indexed chunks with overlap."""
    clean = str(doc_text).strip()
    if not clean:
        return []

    words = clean.split()
    chunks = []
    step = max(10, chunk_size - overlap)

    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        section_idx = (i // step) + 1
        chunks.append({
            "section": f"Section {section_idx}",
            "text": chunk_text
        })
    return chunks


def _answer_over_document(
    doc_text: str,
    query: str,
    top_k: int = 2
) -> Tuple[str, List[Dict[str, Any]], float, int]:
    """
    Retrieves most relevant document sections and synthesizes a grounded answer.
    Returns (answer, references, confidence, total_chunks).
    """
    clean_q = str(query).strip()
    clean_doc = str(doc_text).strip()

    if not clean_doc:
        return "No document text available to answer this question.", [], 0.0, 0
    if not clean_q:
        return f"Document summary: {clean_doc[:250]}...", [{"section": "Section 1", "score": 1.0}], 1.0, 1

    chunks = _chunk_document(clean_doc, chunk_size=300, overlap=50)
    if not chunks:
        return "Document is empty.", [], 0.0, 0

    # Score each chunk against the query
    q_vec = np.array(_deterministic_local_embedding(clean_q, dim=64, normalize=True), dtype=np.float32)
    scored = []
    for c in chunks:
        c_vec = np.array(_deterministic_local_embedding(c["text"], dim=64, normalize=True), dtype=np.float32)
        score = float(np.dot(q_vec, c_vec))
        scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_matches = scored[:top_k]

    refs = []
    excerpts = []
    top_score = top_matches[0][0] if top_matches else 0.0

    for sc, c in top_matches:
        refs.append({"section": c["section"], "relevance": round(sc, 3)})
        excerpts.append(f"[{c['section']}]: {c['text'][:150]}...")

    confidence = max(0.20, min(round(top_score * 1.5, 2), 0.98))
    answer = (
        f"Based on {', '.join([r['section'] for r in refs])} of the document:\n\n"
        + "\n".join(excerpts)
        + f"\n\n[Confidence: {confidence:.0%}]"
    )

    return answer, refs, confidence, len(chunks)


class DocumentQaAgentRecipe(BaseRecipe):
    recipe_id = "document_qa_agent"
    name = "Document Q&A Agent"
    version = "1.0.0"
    category = "Agents & LLM Orchestration"
    group = "Agents & LLM Orchestration"
    subgroup = "Core Agents"
    media_type = "agents"
    description = "Answers questions over uploaded documents, manuals, and reports with page and section references."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "document_column": {
                    "type": "string",
                    "title": "Document Text Column",
                    "description": "Column holding document body text or manual contents. Auto-detected if empty."
                },
                "query_column": {
                    "type": "string",
                    "title": "Question Column",
                    "description": "Column holding questions about the document. Auto-detected if empty."
                },
                "output_column": {
                    "type": "string",
                    "title": "Answer Output Column",
                    "default": "document_answer",
                    "description": "Target column name storing the synthesized document answer."
                }
            },
            "required": ["output_column"]
        }

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        return []

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        start_time = time.time()

        raw_input = None
        for k in ["dataframe", "input", "df", "data"]:
            if k in inputs and inputs[k] is not None:
                raw_input = inputs[k]
                break

        if raw_input is None and inputs:
            q_direct = inputs.get("question") or inputs.get("query")
            d_direct = inputs.get("context") or inputs.get("document") or inputs.get("text")
            if q_direct is not None or d_direct is not None:
                df = pd.DataFrame([{
                    "query": str(q_direct or ""),
                    "document": str(d_direct or "")
                }])
            else:
                df = pd.DataFrame()
        elif isinstance(raw_input, pd.DataFrame):
            df = raw_input.copy()
        elif isinstance(raw_input, list):
            df = pd.DataFrame(raw_input)
        elif isinstance(raw_input, dict) and "records" in raw_input:
            df = pd.DataFrame(raw_input["records"])
        elif isinstance(raw_input, dict):
            df = pd.DataFrame([raw_input])
        else:
            df = pd.DataFrame([{"query": "", "document": str(raw_input or "")}])

        doc_col = config.get("document_column") or config.get("context_column")
        query_col = config.get("query_column") or config.get("question_column")
        output_col = str(config.get("output_column") or "document_answer").strip()

        # Edge case: Empty DataFrame
        if df.empty:
            out_df = pd.DataFrame(columns=[output_col, "qa_answer", "qa_confidence", "page_section_references", "confidence_score", "total_document_chunks"])
            return {
                "dataframe": out_df,
                "status": "success",
                "answer": "",
                "confidence": 0.0,
                "summary": {"total_qa_pairs": 0, "status": "empty"}
            }

        # Resolve document column
        if not doc_col or str(doc_col).strip() in ["", "-- Select Column --", "(None)"]:
            candidates = ["document", "context", "text", "body", "content", "manual", "doc"]
            found = [c for c in candidates if c in df.columns]
            doc_col = found[0] if found else df.columns[0]
        else:
            doc_col = str(doc_col).strip()

        # Resolve query column
        if not query_col or str(query_col).strip() in ["", "-- Select Column --", "(None)"]:
            candidates = ["query", "question", "prompt", "inquiry", "q"]
            found = [c for c in candidates if c in df.columns and c != doc_col]
            query_col = found[0] if found else (df.columns[1] if len(df.columns) > 1 else doc_col)
        else:
            query_col = str(query_col).strip()

        if doc_col not in df.columns:
            df[doc_col] = ""
        if query_col not in df.columns:
            df[query_col] = "Summarize document"

        answers: List[str] = []
        references_list: List[str] = []
        confidences: List[float] = []
        total_chunks_list: List[int] = []

        for i in range(len(df)):
            row = df.iloc[i]
            d_val = str(row[doc_col]) if pd.notna(row[doc_col]) else ""
            q_val = str(row[query_col]) if pd.notna(row[query_col]) else ""

            ans, refs, conf, n_chunks = _answer_over_document(d_val, q_val, top_k=2)
            answers.append(ans)
            references_list.append(json.dumps(refs))
            confidences.append(conf)
            total_chunks_list.append(n_chunks)

        out_df = df.copy().reset_index(drop=True)
        out_df[output_col] = answers
        out_df["qa_answer"] = answers
        out_df["qa_confidence"] = confidences
        out_df["page_section_references"] = references_list
        out_df["confidence_score"] = confidences
        out_df["total_document_chunks"] = total_chunks_list

        duration_ms = round((time.time() - start_time) * 1000, 2)
        summary = {
            "total_qa_pairs": len(out_df),
            "total_chunks_evaluated": sum(total_chunks_list),
            "avg_confidence": round(float(sum(confidences) / len(confidences)), 2) if confidences else 0.0,
            "duration_ms": duration_ms
        }

        first_ans = answers[0] if answers else ""
        first_conf = confidences[0] if confidences else 0.0

        return {
            "dataframe": out_df,
            "status": "success",
            "answer": first_ans,
            "confidence": first_conf,
            "summary": summary
        }
