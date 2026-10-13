import re
import math
from collections import Counter
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Set
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort

# ----------------------------------------------------------------------
# ZERO-DEPENDENCY EXTRACTIVE SUMMARIZATION ENGINE
# Provides fast, deterministic, offline text summarization
# ----------------------------------------------------------------------

COMMON_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't",
    "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have",
    "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
    "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into",
    "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's",
    "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs",
    "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't",
    "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't",
    "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
    "yourselves"
}


def split_into_sentences(text: str) -> List[str]:
    """Splits text into meaningful sentences preserving punctuation."""
    if not text or not isinstance(text, str):
        return []
    cleaned = text.strip()
    if not cleaned:
        return []
    # Split on sentence boundaries (. ! ?) followed by whitespace or linebreaks
    raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", cleaned)
    # Filter out empty or punctuation-only strings
    valid_sentences = [s.strip() for s in raw_sentences if len(re.findall(r"\w+", s)) > 0]
    return valid_sentences if valid_sentences else [cleaned]


def extract_keywords_weights(text: str) -> Dict[str, float]:
    """Calculates frequency-based word weights normalized by maximum frequency."""
    words = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
    filtered = [w for w in words if w not in COMMON_STOPWORDS]
    if not filtered:
        return {}
    counts = Counter(filtered)
    max_c = max(counts.values()) if counts else 1
    return {w: c / max_c for w, c in counts.items()}


def summarize_extractive_tfidf(sentences: List[str], max_sentences: int = 2) -> List[str]:
    """
    Extractive sentence summarization based on word frequency salience and sentence position.
    Preserves original sentence order in the output.
    """
    if len(sentences) <= max_sentences:
        return sentences

    full_text = " ".join(sentences)
    word_weights = extract_keywords_weights(full_text)
    if not word_weights:
        return sentences[:max_sentences]

    scored_sentences: List[Tuple[float, int, str]] = []
    for idx, s in enumerate(sentences):
        tokens = [w for w in re.findall(r"\b[a-zA-Z]{2,}\b", s.lower()) if w not in COMMON_STOPWORDS]
        if not tokens:
            continue
        # Raw score from word importance
        raw_score = sum(word_weights.get(w, 0.0) for w in tokens)
        # Length normalization
        norm_score = raw_score / (math.log(len(tokens) + 2.0))
        # Position weight (lead sentences in news/reviews often summarize the point)
        pos_factor = 1.25 if idx == 0 else (1.1 if idx == 1 else 1.0)
        scored_sentences.append((norm_score * pos_factor, idx, s))

    if not scored_sentences:
        return sentences[:max_sentences]

    # Select top N scoring sentences
    scored_sentences.sort(key=lambda item: item[0], reverse=True)
    selected = scored_sentences[:max_sentences]
    # Re-order selected sentences chronologically as they appeared in original text
    selected.sort(key=lambda item: item[1])
    return [s[2] for s in selected]


def summarize_textrank(sentences: List[str], max_sentences: int = 2) -> List[str]:
    """
    Graph-based TextRank salience algorithm (PageRank over token overlap matrix).
    """
    if len(sentences) <= max_sentences:
        return sentences

    n = len(sentences)
    token_sets = [set(re.findall(r"\b[a-zA-Z]{2,}\b", s.lower())) - COMMON_STOPWORDS for s in sentences]

    # Build similarity adjacency matrix (Jaccard token overlap)
    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            union = len(token_sets[i] | token_sets[j])
            if union > 0:
                score = len(token_sets[i] & token_sets[j]) / union
                sim[i][j] = score
                sim[j][i] = score

    # Power iteration PageRank
    ranks = [1.0 / n] * n
    d = 0.85
    for _ in range(25):
        new_ranks = [0.0] * n
        for i in range(n):
            rank_sum = sum(ranks[j] * (sim[j][i] / (sum(sim[j]) or 1.0)) for j in range(n) if i != j)
            new_ranks[i] = (1 - d) / n + d * rank_sum
        ranks = new_ranks

    ranked_indices = sorted(range(n), key=lambda i: ranks[i], reverse=True)
    top_indices = sorted(ranked_indices[:max_sentences])
    return [sentences[i] for i in top_indices]


def summarize_lead(sentences: List[str], max_sentences: int = 2) -> List[str]:
    """Lead sentences summarization (first N sentences)."""
    return sentences[:max_sentences]


def summarize_key_takeaway(sentences: List[str]) -> List[str]:
    """Extracts single most representative takeaway sentence."""
    if len(sentences) <= 1:
        return sentences
    top_one = summarize_extractive_tfidf(sentences, max_sentences=1)
    return top_one if top_one else sentences[:1]


def generate_text_summary(
    text: str,
    method: str = "Extractive (TF-IDF / Frequency-Based)",
    max_sentences: int = 2,
    max_characters: int = 180
) -> str:
    """
    Generates a concise summary from input text based on the chosen method,
    sentence count, and character limit.
    """
    if not text or not isinstance(text, str):
        return "" if text is None else str(text)

    cleaned = text.strip()
    if not cleaned:
        return ""

    sentences = split_into_sentences(cleaned)
    if not sentences:
        return cleaned

    # If text is already very brief, return clean original
    if len(sentences) <= 1 and len(cleaned) <= max_characters:
        return cleaned

    m_lower = method.lower()
    if "lead" in m_lower:
        summary_sentences = summarize_lead(sentences, max_sentences)
    elif "textrank" in m_lower:
        summary_sentences = summarize_textrank(sentences, max_sentences)
    elif "takeaway" in m_lower or "headline" in m_lower:
        summary_sentences = summarize_key_takeaway(sentences)
    else:
        # Default: Extractive TF-IDF
        summary_sentences = summarize_extractive_tfidf(sentences, max_sentences)

    summary_text = " ".join(summary_sentences).strip()

    # Enforce character limit if specified
    if max_characters and max_characters > 0 and len(summary_text) > max_characters:
        # Trim gracefully at word boundary with ellipsis
        truncated = summary_text[:max_characters].rsplit(" ", 1)[0].rstrip(".,;:- ")
        summary_text = f"{truncated}..."

    return summary_text


# ----------------------------------------------------------------------
# SUMMARIZATION RECIPE CLASS
# ----------------------------------------------------------------------

class SummarizationRecipe(BaseRecipe):
    """
    NLP / Text Summarization Recipe.
    Generates short, concise summaries of long text columns using extractive
    TF-IDF, TextRank, or Lead algorithms.
    """

    recipe_id = "text_summarization"
    name = "Summarization"
    version = "1.0.0"
    category = "nlp"
    description = "Generates short summaries of long text columns."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="dataframe",
            label="Input DataFrame",
            type="dataframe",
            required=True,
            max_connections=1,
            description="Dataset containing text columns to summarize"
        )
    ]

    outputs = [
        RecipePort(
            id="dataframe",
            label="Summarized DataFrame",
            type="dataframe",
            description="Dataset containing generated summary column(s)"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "title": "Target Text Columns",
                    "input_type": "column_select",
                    "description": "Text columns to summarize. If left empty, all detected text/object columns will be summarized."
                },
                "method": {
                    "type": "select",
                    "input_type": "dropdown",
                    "title": "Summarization Method",
                    "enum": [
                        "Extractive (TF-IDF / Frequency-Based)",
                        "TextRank (Graph-Based Salience)",
                        "Lead Sentences (First N Sentences)",
                        "Key Takeaway (Single Headline)"
                    ],
                    "default": "Extractive (TF-IDF / Frequency-Based)",
                    "description": "Algorithm used to condense long text into concise summaries."
                },
                "max_sentences": {
                    "type": "integer",
                    "input_type": "number",
                    "title": "Max Sentences",
                    "default": 2,
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Maximum number of key sentences to include in each generated summary."
                },
                "max_characters": {
                    "type": "integer",
                    "input_type": "number",
                    "title": "Max Characters",
                    "default": 180,
                    "minimum": 0,
                    "maximum": 2000,
                    "description": "Maximum character length cap for the generated summary (0 for no character limit)."
                },
                "append_columns": {
                    "type": "boolean",
                    "input_type": "boolean",
                    "title": "Append Summary Columns",
                    "default": True,
                    "description": "When true, appends new summary columns alongside existing columns. When false, replaces original text columns."
                },
                "output_suffix": {
                    "type": "text",
                    "input_type": "text",
                    "title": "Output Column Suffix",
                    "default": "_summary",
                    "description": "Suffix for newly appended summary columns (e.g. '_summary')."
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
            if df is None or not isinstance(df, pd.DataFrame):
                return {
                    "dataframe": pd.DataFrame(),
                    "output_summary": {"error": "No valid input DataFrame provided for Text Summarization."}
                }

        if df.empty:
            return {
                "dataframe": df.copy(),
                "metrics": {"total_rows": 0, "summarized_columns": []},
                "output_summary": {"message": "Input DataFrame is empty. No summarization performed."}
            }

        out_df = df.copy()

        # Parse configuration
        method = str(config.get("method") or "Extractive (TF-IDF / Frequency-Based)")
        max_sentences = int(config.get("max_sentences") or 2)
        max_sentences = max(1, min(10, max_sentences))

        max_characters = int(config.get("max_characters") if config.get("max_characters") is not None else 180)
        append_cols = bool(config.get("append_columns", True))
        suffix = str(config.get("output_suffix", "_summary")).strip() or "_summary"

        # Resolve target columns
        cols_cfg = config.get("columns")
        target_cols: List[str] = []
        if isinstance(cols_cfg, list) and cols_cfg:
            target_cols = [str(c) for c in cols_cfg if str(c) in out_df.columns]
        elif isinstance(cols_cfg, str) and cols_cfg in out_df.columns:
            target_cols = [cols_cfg]
        else:
            # Auto-detect text columns (object or string dtypes)
            for col in out_df.columns:
                dtype_str = str(out_df[col].dtype).lower()
                if dtype_str in ["object", "string"] or "str" in dtype_str:
                    target_cols.append(col)

        if not target_cols:
            return {
                "dataframe": out_df,
                "metrics": {"total_rows": len(out_df), "summarized_columns": []},
                "output_summary": {
                    "title": "Text Summarization",
                    "message": "No text or string columns detected to summarize.",
                    "status": "SKIPPED"
                }
            }

        # Memoization cache for duplicate texts
        summary_cache: Dict[str, str] = {}

        def _summarize_value(val: Any) -> Any:
            if val is None or pd.isna(val):
                return val
            val_str = str(val)
            if not val_str.strip():
                return val_str
            if val_str in summary_cache:
                return summary_cache[val_str]

            sum_result = generate_text_summary(
                text=val_str,
                method=method,
                max_sentences=max_sentences,
                max_characters=max_characters
            )
            summary_cache[val_str] = sum_result
            return sum_result

        total_cells_summarized = 0
        column_summaries = {}
        all_orig_lens = []
        all_sum_lens = []

        for col in target_cols:
            series = out_df[col]
            summarized_series = series.apply(_summarize_value)
            non_empty_count = summarized_series.dropna().count()
            total_cells_summarized += int(non_empty_count)

            out_col_name = f"{col}{suffix}" if append_cols else col
            out_df[out_col_name] = summarized_series

            # Length analytics
            col_orig_lens = [len(str(v)) for v in series.dropna() if str(v).strip()]
            col_sum_lens = [len(str(v)) for v in summarized_series.dropna() if str(v).strip()]
            all_orig_lens.extend(col_orig_lens)
            all_sum_lens.extend(col_sum_lens)

            avg_orig = round(sum(col_orig_lens) / max(1, len(col_orig_lens)), 1)
            avg_sum = round(sum(col_sum_lens) / max(1, len(col_sum_lens)), 1)
            ratio_pct = round((avg_sum / max(1.0, avg_orig)) * 100.0, 1)

            sample_pairs = []
            for orig, sm in zip(series.head(3), summarized_series.head(3)):
                if pd.notna(orig) and pd.notna(sm):
                    sample_pairs.append({
                        "original": str(orig),
                        "summary": str(sm)
                    })

            column_summaries[col] = {
                "output_column": out_col_name,
                "summarized_rows": int(non_empty_count),
                "original_avg_characters": avg_orig,
                "summary_avg_characters": avg_sum,
                "compression_ratio_pct": ratio_pct,
                "samples": sample_pairs
            }

        overall_orig_avg = round(sum(all_orig_lens) / max(1, len(all_orig_lens)), 1)
        overall_sum_avg = round(sum(all_sum_lens) / max(1, len(all_sum_lens)), 1)
        overall_compression_pct = round((overall_sum_avg / max(1.0, overall_orig_avg)) * 100.0, 1)

        metrics = {
            "total_rows": len(out_df),
            "summarized_columns": target_cols,
            "method": method,
            "max_sentences": max_sentences,
            "max_characters": max_characters,
            "total_summarized_cells": total_cells_summarized,
            "average_original_characters": overall_orig_avg,
            "average_summary_characters": overall_sum_avg,
            "overall_compression_ratio_pct": overall_compression_pct,
            "column_breakdown": column_summaries
        }

        return {
            "dataframe": out_df,
            "metrics": metrics,
            "output_summary": {
                "title": "Text Summarization Completed",
                "message": (
                    f"Successfully summarized {len(target_cols)} column(s) across {len(out_df)} rows "
                    f"using {method} ({overall_compression_pct}% of original length)."
                ),
                "metrics": metrics,
                "status": "SUCCESS"
            }
        }
