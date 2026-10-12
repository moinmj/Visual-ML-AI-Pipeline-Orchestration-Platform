import re
import math
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from backend.app.recipes.base.recipe import BaseRecipe

# ----------------------------------------------------------------------
# BUILT-IN ZERO-DEPENDENCY SENTIMENT LEXICON & ENGINE
# Provides fast, deterministic, offline sentiment analysis (VADER-style)
# ----------------------------------------------------------------------

# Base valence scores from -4.0 (extremely negative) to +4.0 (extremely positive)
SENTIMENT_LEXICON: Dict[str, float] = {
    # Strong Positive (+2.5 to +4.0)
    "excellent": 3.5, "outstanding": 3.6, "exceptional": 3.7, "fantastic": 3.5,
    "wonderful": 3.4, "amazing": 3.4, "awesome": 3.4, "brilliant": 3.3,
    "perfect": 3.5, "love": 3.2, "loved": 3.2, "loving": 3.1, "delighted": 3.2,
    "delight": 3.0, "superb": 3.3, "magnificent": 3.4, "stellar": 3.2,
    "terrific": 3.2, "triumph": 3.0, "triumphant": 3.1, "flawless": 3.5,
    "impressed": 2.8, "impressive": 3.0, "enthusiastic": 2.8, "thrilled": 3.4,
    "genius": 3.0, "masterpiece": 3.5, "ideal": 3.0, "superior": 2.8,
    "miracle": 3.3, "legendary": 3.1, "breathtaking": 3.3, "fabulous": 3.2,

    # Moderate Positive (+1.2 to +2.4)
    "good": 2.0, "great": 2.5, "happy": 2.3, "satisfaction": 2.1,
    "satisfied": 2.1, "satisfying": 2.0, "pleased": 2.0, "pleasant": 2.0,
    "recommend": 2.2, "recommended": 2.2, "helpful": 2.0, "friendly": 2.1,
    "reliable": 2.2, "efficient": 2.1, "effective": 2.0, "easy": 1.8,
    "smooth": 1.8, "fast": 1.7, "clean": 1.7, "neat": 1.6, "valuable": 2.1,
    "worth": 1.7, "benefit": 1.9, "beneficial": 2.0, "profit": 2.0,
    "profitable": 2.1, "gain": 1.8, "advantage": 1.9, "success": 2.4,
    "successful": 2.4, "win": 2.3, "winner": 2.3, "winning": 2.2,
    "joy": 2.4, "enjoy": 2.2, "enjoyed": 2.2, "enjoyable": 2.1,
    "cheer": 2.0, "cheerful": 2.1, "blessed": 2.2, "glad": 1.9,
    "grateful": 2.3, "appreciate": 2.0, "appreciated": 2.0, "honored": 2.2,
    "welcome": 1.8, "welcoming": 1.9, "innovative": 2.2, "smart": 1.9,
    "bright": 1.8, "clear": 1.6, "fresh": 1.7, "positive": 2.0,
    "trust": 2.1, "trusted": 2.2, "trustworthy": 2.4, "faith": 2.0,
    "safe": 1.9, "safety": 1.9, "secure": 2.0, "security": 1.8,
    "resolved": 1.9, "solution": 1.8, "solved": 1.9, "fixed": 1.7,
    "upgrade": 1.8, "upgraded": 1.8, "bonus": 1.9, "reward": 2.0,
    "supportive": 2.0, "generous": 2.2, "kind": 2.0, "kindness": 2.1,
    "promising": 1.9, "boost": 1.8, "boosted": 1.8, "improve": 1.9,
    "improved": 2.0, "improvement": 2.0, "top": 1.9, "premier": 2.2,

    # Mild Positive (+0.5 to +1.1)
    "nice": 1.2, "fine": 1.0, "ok": 0.6, "okay": 0.6, "decent": 1.0,
    "fair": 0.8, "acceptable": 0.8, "like": 1.1, "liked": 1.1,
    "pretty": 1.0, "active": 0.8, "ready": 0.9, "neatly": 1.0,
    "capable": 1.2, "solid": 1.2, "steady": 1.0, "stable": 1.1,

    # Strong Negative (-2.5 to -4.0)
    "terrible": -3.5, "horrible": -3.5, "awful": -3.5, "worst": -3.6,
    "atrocious": -3.6, "abysmal": -3.5, "disaster": -3.4, "disastrous": -3.4,
    "catastrophe": -3.4, "catastrophic": -3.4, "hate": -3.2, "hated": -3.2,
    "hateful": -3.2, "scam": -3.5, "fraud": -3.5, "fraudulent": -3.5,
    "fake": -2.8, "corrupt": -3.2, "corruption": -3.2, "toxic": -3.1,
    "poison": -3.0, "ruin": -3.0, "ruined": -3.1, "destroy": -3.2,
    "destroyed": -3.2, "destructive": -3.1, "pathetic": -3.0, "disgrace": -3.2,
    "disgraceful": -3.3, "shameful": -3.0, "nightmare": -3.4, "outrage": -3.1,
    "outrageous": -3.2, "offensive": -2.9, "repulsive": -3.1, "disgusting": -3.3,

    # Moderate Negative (-1.2 to -2.4)
    "bad": -2.2, "poor": -2.1, "poorly": -2.1, "fail": -2.3,
    "failed": -2.4, "failure": -2.4, "failing": -2.2, "crash": -2.4,
    "crashed": -2.4, "crashing": -2.2, "broke": -2.2, "broken": -2.3,
    "damage": -2.2, "damaged": -2.3, "defect": -2.2, "defective": -2.4,
    "error": -2.0, "bug": -1.9, "buggy": -2.1, "glitch": -1.8,
    "problem": -2.0, "problems": -2.0, "issue": -1.7, "issues": -1.7,
    "trouble": -2.0, "disappointed": -2.4, "disappointing": -2.3,
    "disappointment": -2.3, "frustrated": -2.3, "frustrating": -2.3,
    "frustration": -2.2, "angry": -2.4, "anger": -2.3, "furious": -2.8,
    "annoyed": -1.9, "annoying": -2.0, "complaint": -2.0, "complaints": -2.0,
    "complain": -1.8, "complained": -1.9, "refund": -1.8, "cancel": -1.7,
    "cancelled": -1.7, "cancellation": -1.8, "delayed": -1.7, "delay": -1.6,
    "slow": -1.6, "useless": -2.5, "waste": -2.3, "wasted": -2.3,
    "loss": -2.2, "losses": -2.2, "lost": -1.9, "decline": -1.8,
    "declined": -1.8, "drop": -1.5, "dropped": -1.5, "crisis": -2.6,
    "danger": -2.3, "dangerous": -2.4, "hazard": -2.2, "risk": -1.5,
    "risky": -1.8, "threat": -2.3, "threatened": -2.2, "pain": -2.2,
    "painful": -2.3, "suffer": -2.4, "suffering": -2.4, "cheat": -2.6,
    "cheated": -2.6, "expensive": -1.5, "overpriced": -2.0, "unreliable": -2.4,
    "inefficient": -2.0, "unacceptable": -2.5, "reject": -2.0, "rejected": -2.1,

    # Mild Negative (-0.5 to -1.1)
    "hard": -0.8, "difficult": -1.1, "difficulty": -1.1, "tough": -0.9,
    "confusing": -1.1, "confused": -1.0, "complex": -0.6, "weird": -0.8,
    "odd": -0.7, "lacking": -1.1, "weak": -1.2, "weakness": -1.2,
    "doubt": -1.0, "doubtful": -1.1, "hesitant": -0.8, "boring": -1.2,

    # Emoticons / Symbols
    ":)": 2.0, ":-)": 2.0, ":D": 2.5, "xD": 2.2, ";)": 1.8,
    ":(": -2.0, ":-(": -2.0, ":'(": -2.5, "D:": -2.5, ":/": -1.2,
    "👍": 2.2, "❤️": 2.8, "🎉": 2.5, "⭐": 1.5, "🔥": 2.0,
    "👎": -2.2, "💔": -2.5, "😡": -2.8, "💩": -2.0, "⚠️": -1.5
}

# Negation words invert subsequent word valence
NEGATION_WORDS = {
    "not", "never", "no", "neither", "nor", "hardly", "scarcely",
    "barely", "doesn't", "don't", "didn't", "wasn't", "weren't",
    "isn't", "aren't", "won't", "wouldn't", "couldn't", "shouldn't",
    "can't", "cannot", "without", "rarely", "seldom"
}

# Intensifier / Booster words modify valence magnitude
BOOSTER_DICT: Dict[str, float] = {
    "very": 0.35, "extremely": 0.45, "incredibly": 0.40, "absolutely": 0.40,
    "totally": 0.35, "completely": 0.35, "highly": 0.30, "especially": 0.25,
    "super": 0.35, "ultra": 0.40, "hugely": 0.30, "deeply": 0.30,
    "remarkably": 0.35, "exceptionally": 0.40, "really": 0.25, "so": 0.20,
    "slightly": -0.25, "somewhat": -0.20, "a bit": -0.20, "hardly": -0.30,
    "barely": -0.30, "marginally": -0.20
}


def compute_vader_compound(valences: List[float], alpha: float = 15.0) -> float:
    """Computes the standard VADER hyperbolic compound score in range [-1.0, +1.0]."""
    if not valences:
        return 0.0
    sum_s = sum(valences)
    return round(sum_s / math.sqrt((sum_s * sum_s) + alpha), 4)


def analyze_text_sentiment(text: Any) -> Dict[str, float]:
    """
    Evaluates text polarity scores with negation tracking, booster scaling,
    and returns compound (-1 to +1), positive, negative, and neutral ratios.
    """
    if pd.isna(text) or text is None:
        return {"compound": 0.0, "positive": 0.0, "negative": 0.0, "neutral": 1.0}

    s = str(text).strip()
    if not s:
        return {"compound": 0.0, "positive": 0.0, "negative": 0.0, "neutral": 1.0}

    # Tokenize preserving punctuation for emoticons
    words = re.findall(r"(?:[A-Za-z]+(?:'[A-Za-z]+)?|[:;]-?[)(D/]|👍|❤️|🎉|👎|💔|😡|[!?])", s)
    if not words:
        return {"compound": 0.0, "positive": 0.0, "negative": 0.0, "neutral": 1.0}

    sentiments: List[float] = []
    pos_sum = 0.0
    neg_sum = 0.0
    neu_count = 0

    # Punctuation emphasis
    exclamation_count = min(s.count("!"), 4)
    punct_boost = exclamation_count * 0.15

    for i, token in enumerate(words):
        token_lower = token.lower()
        v = SENTIMENT_LEXICON.get(token_lower, 0.0)

        # Check emoticons as-is
        if v == 0.0 and token in SENTIMENT_LEXICON:
            v = SENTIMENT_LEXICON[token]

        if v != 0.0:
            # All-caps emphasis (e.g. "GREAT", "TERRIBLE")
            if token.isupper() and len(token) > 1:
                v += (0.5 if v > 0 else -0.5)

            # Lookback up to 3 tokens for negation or boosters
            negated = False
            booster_scalar = 0.0

            start_idx = max(0, i - 3)
            for prev_token in words[start_idx:i]:
                prev_lower = prev_token.lower()
                if prev_lower in NEGATION_WORDS:
                    negated = not negated
                if prev_lower in BOOSTER_DICT:
                    booster_scalar += BOOSTER_DICT[prev_lower]

            # Apply booster
            if v > 0:
                v += (v * booster_scalar)
            else:
                v -= (abs(v) * booster_scalar)

            # Apply exclamation boost
            if punct_boost > 0:
                v += (punct_boost if v > 0 else -punct_boost)

            # Apply negation inversion
            if negated:
                v = v * -0.74

            sentiments.append(v)
            if v > 0:
                pos_sum += v
            else:
                neg_sum += abs(v)
        else:
            neu_count += 1

    compound = compute_vader_compound(sentiments)

    # Normalize pos/neg/neu to sum to 1.0
    total_tokens = len(words)
    total_valence = pos_sum + neg_sum + neu_count
    if total_valence > 0:
        pos_ratio = round(pos_sum / total_valence, 3)
        neg_ratio = round(neg_sum / total_valence, 3)
        neu_ratio = round(max(0.0, 1.0 - (pos_ratio + neg_ratio)), 3)
    else:
        pos_ratio, neg_ratio, neu_ratio = 0.0, 0.0, 1.0

    return {
        "compound": compound,
        "positive": pos_ratio,
        "negative": neg_ratio,
        "neutral": neu_ratio
    }


# ----------------------------------------------------------------------
# SENTIMENT ANALYSIS RECIPE
# ----------------------------------------------------------------------

class SentimentAnalysisRecipe(BaseRecipe):
    """
    NLP / Text Sentiment Analysis Recipe.
    Scores text columns as positive, negative, or neutral with compound polarity confidence.
    """

    recipe_id = "sentiment_analysis"
    name = "Sentiment Analysis"
    version = "1.0.0"
    category = "nlp"
    description = "Scores text columns as positive, negative or neutral with polarity and compound confidence metrics."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "title": "Target Text Columns",
                    "description": "Text columns to score for sentiment. If left empty, all detected text/object columns will be scored.",
                    "format": "column",
                    "ui:widget": "column_select"
                },
                "output_mode": {
                    "type": "string",
                    "title": "Output Format",
                    "enum": ["Label and Score", "Label Only", "Detailed Probabilities"],
                    "default": "Label and Score",
                    "description": "'Label and Score' adds {col}_sentiment and {col}_sentiment_score (-1 to +1). 'Label Only' adds sentiment labels. 'Detailed' adds positive, negative, and neutral score breakdown."
                },
                "positive_threshold": {
                    "type": "number",
                    "title": "Positive Threshold",
                    "default": 0.05,
                    "description": "Compound score above which text is classified as positive (standard VADER threshold is 0.05)."
                },
                "negative_threshold": {
                    "type": "number",
                    "title": "Negative Threshold",
                    "default": -0.05,
                    "description": "Compound score below which text is classified as negative (standard VADER threshold is -0.05)."
                },
                "append_columns": {
                    "type": "boolean",
                    "title": "Append Output Columns",
                    "default": True,
                    "description": "When true, appends new sentiment columns alongside existing data. When false, replaces the original text column."
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
            # Accept left_dataframe or input fallback
            df = inputs.get("left_dataframe") or inputs.get("input")
            if df is None or not isinstance(df, pd.DataFrame):
                return {
                    "dataframe": pd.DataFrame(),
                    "output_summary": {"error": "No valid input DataFrame provided for Sentiment Analysis."}
                }

        if df.empty:
            return {
                "dataframe": df.copy(),
                "metrics": {"total_rows": 0, "analyzed_columns": []},
                "output_summary": {"message": "Input DataFrame is empty. No sentiment scoring performed."}
            }

        out_df = df.copy()

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
                "metrics": {"total_rows": len(out_df), "analyzed_columns": []},
                "output_summary": {
                    "title": "Sentiment Analysis",
                    "message": "No text or string columns detected to analyze.",
                    "status": "SKIPPED"
                }
            }

        # Parameters
        output_mode = config.get("output_mode", "Label and Score")
        pos_thresh = float(config.get("positive_threshold", 0.05))
        neg_thresh = float(config.get("negative_threshold", -0.05))
        append_cols = bool(config.get("append_columns", True))

        # Overall summary stats across analyzed columns
        overall_distribution = {"positive": 0, "neutral": 0, "negative": 0}
        column_summaries = {}
        all_compounds: List[float] = []

        for col in target_cols:
            series = out_df[col]
            scores = [analyze_text_sentiment(val) for val in series]

            labels: List[str] = []
            compounds: List[float] = []
            pos_scores: List[float] = []
            neg_scores: List[float] = []
            neu_scores: List[float] = []

            for sc in scores:
                comp = sc["compound"]
                compounds.append(comp)
                all_compounds.append(comp)

                if comp >= pos_thresh:
                    label = "positive"
                elif comp <= neg_thresh:
                    label = "negative"
                else:
                    label = "neutral"

                labels.append(label)
                overall_distribution[label] += 1
                pos_scores.append(sc["positive"])
                neg_scores.append(sc["negative"])
                neu_scores.append(sc["neutral"])

            # Column metrics
            col_pos = labels.count("positive")
            col_neu = labels.count("neutral")
            col_neg = labels.count("negative")
            total_items = max(1, len(labels))

            column_summaries[col] = {
                "average_score": round(sum(compounds) / total_items, 4) if compounds else 0.0,
                "distribution": {
                    "positive": col_pos,
                    "neutral": col_neu,
                    "negative": col_neg,
                    "positive_pct": round((col_pos / total_items) * 100.0, 1),
                    "neutral_pct": round((col_neu / total_items) * 100.0, 1),
                    "negative_pct": round((col_neg / total_items) * 100.0, 1)
                }
            }

            # Apply to DataFrame
            if append_cols:
                out_df[f"{col}_sentiment"] = labels
                if output_mode in ["Label and Score", "Detailed Probabilities"]:
                    out_df[f"{col}_sentiment_score"] = compounds
                if output_mode == "Detailed Probabilities":
                    out_df[f"{col}_pos_score"] = pos_scores
                    out_df[f"{col}_neg_score"] = neg_scores
                    out_df[f"{col}_neu_score"] = neu_scores
            else:
                # Replace original text column
                out_df[col] = labels
                if output_mode in ["Label and Score", "Detailed Probabilities"]:
                    out_df[f"{col}_sentiment_score"] = compounds

        total_analyzed = max(1, sum(overall_distribution.values()))
        avg_score = round(sum(all_compounds) / len(all_compounds), 4) if all_compounds else 0.0

        metrics = {
            "total_rows": len(out_df),
            "analyzed_columns": target_cols,
            "average_sentiment_score": avg_score,
            "sentiment_distribution": {
                "positive": overall_distribution["positive"],
                "neutral": overall_distribution["neutral"],
                "negative": overall_distribution["negative"],
                "positive_percentage": round((overall_distribution["positive"] / total_analyzed) * 100.0, 1),
                "neutral_percentage": round((overall_distribution["neutral"] / total_analyzed) * 100.0, 1),
                "negative_percentage": round((overall_distribution["negative"] / total_analyzed) * 100.0, 1)
            },
            "column_breakdown": column_summaries
        }

        output_summary = {
            "title": "Sentiment Analysis Completed",
            "message": f"Successfully analyzed {len(target_cols)} text column(s). Found {overall_distribution['positive']} positive, {overall_distribution['neutral']} neutral, and {overall_distribution['negative']} negative rows.",
            "metrics": metrics
        }

        return {
            "dataframe": out_df,
            "metrics": metrics,
            "output_summary": output_summary
        }
