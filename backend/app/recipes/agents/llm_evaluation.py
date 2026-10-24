"""
LLM Evaluation & Groundedness Scorer (Agents & LLM Orchestration -> Governance & Ops).

Evaluates LLM completions across enterprise quality dimensions:
1. Groundedness (Faithfulness to retrieved context documents)
2. Relevance (Semantic coverage of user question intent)
3. Hallucination Risk (1.0 - Groundedness)
4. Conciseness (Information density and lack of token padding)
5. Overall Quality Verdict (PASS / FAIL gating)
"""

import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from backend.app.recipes.base.recipe import BaseRecipe
from backend.app.recipes.agents.embedding_generator import _deterministic_local_embedding


def _extract_content_tokens(text: str) -> List[str]:
    """Extracts non-trivial content words (3+ chars) for lexical containment verification."""
    clean = re.sub(r"[^\w\s]", " ", text.lower())
    words = [w for w in clean.split() if len(w) >= 3]
    return words


def _evaluate_single(
    query: str,
    answer: str,
    context: str,
    reference: Optional[str] = None
) -> Tuple[float, float, float, float, float]:
    """
    Computes (groundedness, relevance, hallucination_risk, conciseness, overall_score).
    """
    if not answer or not str(answer).strip():
        return 0.0, 0.0, 1.0, 0.0, 0.0

    ans_text = str(answer).strip()
    ans_tokens = _extract_content_tokens(ans_text)
    if not ans_tokens:
        return 0.0, 0.0, 1.0, 0.0, 0.0

    # 1. Groundedness Evaluation
    if context and str(context).strip():
        ctx_tokens = set(_extract_content_tokens(str(context)))
        if ctx_tokens:
            grounded_count = sum(1 for tok in ans_tokens if tok in ctx_tokens or any(tok in ct for ct in ctx_tokens))
            groundedness = round(grounded_count / len(ans_tokens), 3)
            # Add small tolerance for common English connecting phrases
            groundedness = min(1.0, round(groundedness * 1.15, 3))
        else:
            groundedness = 0.50
    else:
        # Context was not provided
        groundedness = 0.80

    # 2. Relevance Evaluation (Cosine similarity of query and answer vectors)
    if query and str(query).strip():
        q_vec = np.array(_deterministic_local_embedding(str(query), dim=64, normalize=True), dtype=np.float32)
        a_vec = np.array(_deterministic_local_embedding(ans_text, dim=64, normalize=True), dtype=np.float32)
        sim = float(np.dot(q_vec, a_vec))
        # Scale to 0.0 - 1.0
        relevance = round(max(0.10, min(sim * 1.8, 1.0)), 3)
    else:
        relevance = 0.75

    # 3. Hallucination Risk
    hallucination_risk = round(max(0.0, 1.0 - groundedness), 3)

    # 4. Conciseness Evaluation
    word_count = len(ans_text.split())
    if word_count < 5:
        conciseness = 0.40  # too short / incomplete
    elif word_count <= 80:
        conciseness = 0.95  # ideal crisp answer
    elif word_count <= 200:
        conciseness = 0.85
    elif word_count <= 400:
        conciseness = 0.70
    else:
        conciseness = max(0.30, round(1.0 - (word_count / 1000.0), 3))

    # 5. Composite Score
    overall = round(
        (0.45 * groundedness) + (0.40 * relevance) + (0.15 * conciseness),
        3
    )

    return groundedness, relevance, hallucination_risk, conciseness, overall


class LlmEvaluationRecipe(BaseRecipe):
    recipe_id = "llm_evaluation"
    name = "LLM Evaluation & Groundedness Scorer"
    version = "1.0.0"
    category = "agents"
    media_type = "agents"
    description = "Scores LLM answers for hallucination risk, groundedness against retrieved context, query relevance, and conciseness."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "answer_column": {
                    "type": "string",
                    "title": "LLM Answer Column",
                    "default": "generated_answer",
                    "description": "Column containing the generated answer to evaluate."
                },
                "context_column": {
                    "type": "string",
                    "title": "Retrieved Context Column",
                    "default": "retrieved_context",
                    "description": "Column containing the context chunks provided to the LLM."
                },
                "query_column": {
                    "type": "string",
                    "title": "User Query Column",
                    "default": "query",
                    "description": "Column containing the original user prompt or inquiry."
                },
                "pass_threshold": {
                    "type": "number",
                    "title": "Quality Pass Threshold (0.0 - 1.0)",
                    "default": 0.65,
                    "minimum": 0.1,
                    "maximum": 1.0,
                    "description": "Overall score required to receive a 'PASS' verdict."
                }
            },
            "required": ["answer_column"]
        }

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors = []
        thresh = config.get("pass_threshold", 0.65)
        if not isinstance(thresh, (int, float)) or thresh < 0.0 or thresh > 1.0:
            errors.append("pass_threshold must be between 0.0 and 1.0.")
        return errors

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        start_time = time.time()

        # 1. Resolve Input DataFrame safely without evaluating bool(df)
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

        answer_col = str(config.get("answer_column") or "generated_answer").strip()
        context_col = str(config.get("context_column") or "retrieved_context").strip()
        query_col = str(config.get("query_column") or "query").strip()
        threshold = float(config.get("pass_threshold") or 0.65)

        # 2. Edge Case: Empty DataFrame
        if df.empty:
            out_df = pd.DataFrame(columns=[
                "groundedness_score", "relevance_score", "hallucination_risk",
                "conciseness_score", "overall_quality_score", "evaluation_verdict"
            ])
            return {
                "dataframe": out_df,
                "summary": {"total_evaluated": 0, "status": "empty"}
            }

        # 3. Resolve Column Fallbacks
        if answer_col not in df.columns:
            # Candidate columns
            candidates = ["generated_answer", "answer", "response", "sanitized_content", "text"]
            found = [c for c in candidates if c in df.columns]
            answer_col = found[0] if found else df.columns[0]

        has_context = context_col in df.columns
        has_query = query_col in df.columns

        # 4. Evaluate Each Row
        groundedness_scores: List[float] = []
        relevance_scores: List[float] = []
        hallucination_risks: List[float] = []
        conciseness_scores: List[float] = []
        overall_scores: List[float] = []
        verdicts: List[str] = []

        pass_count = 0

        for i in range(len(df)):
            row = df.iloc[i]
            ans_val = str(row[answer_col]) if pd.notna(row[answer_col]) else ""
            ctx_val = str(row[context_col]) if has_context and pd.notna(row[context_col]) else ""
            qry_val = str(row[query_col]) if has_query and pd.notna(row[query_col]) else ""

            g, r, h, c, overall = _evaluate_single(query=qry_val, answer=ans_val, context=ctx_val)
            verdict = "PASS" if overall >= threshold else "FAIL"
            if verdict == "PASS":
                pass_count += 1

            groundedness_scores.append(g)
            relevance_scores.append(r)
            hallucination_risks.append(h)
            conciseness_scores.append(c)
            overall_scores.append(overall)
            verdicts.append(verdict)

        # 5. Build Output DataFrame
        out_df = df.copy().reset_index(drop=True)
        out_df["groundedness_score"] = groundedness_scores
        out_df["relevance_score"] = relevance_scores
        out_df["hallucination_risk"] = hallucination_risks
        out_df["conciseness_score"] = conciseness_scores
        out_df["overall_quality_score"] = overall_scores
        out_df["evaluation_verdict"] = verdicts

        duration_ms = round((time.time() - start_time) * 1000, 2)
        summary = {
            "total_evaluated": len(out_df),
            "passed_count": pass_count,
            "failed_count": len(out_df) - pass_count,
            "pass_rate": round(pass_count / len(out_df), 3) if len(out_df) > 0 else 0.0,
            "mean_groundedness": round(float(np.mean(groundedness_scores)), 3) if groundedness_scores else 0.0,
            "mean_relevance": round(float(np.mean(relevance_scores)), 3) if relevance_scores else 0.0,
            "mean_overall": round(float(np.mean(overall_scores)), 3) if overall_scores else 0.0,
            "duration_ms": duration_ms
        }

        return {
            "dataframe": out_df,
            "summary": summary
        }
