import re
import pandas as pd
from typing import Dict, Any, List, Optional
from backend.app.recipes.base.recipe import BaseRecipe

# Built-in high-performance English stopword dictionary
DEFAULT_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once",
    "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd",
    "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where",
    "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
}

def rule_based_stem(word: str) -> str:
    """Fast, deterministic fallback Porter-style suffix stemmer."""
    if len(word) <= 3:
        return word
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s") and not word.endswith("us") and not word.endswith("is"):
        return word[:-1]
    if word.endswith("eed") and len(word) > 4:
        return word[:-1]
    if word.endswith("ed") and len(word) > 4:
        return word[:-2]
    if word.endswith("ing") and len(word) > 5:
        return word[:-3]
    if word.endswith("ly") and len(word) > 4:
        return word[:-2]
    if word.endswith("tional"):
        return word[:-4]
    return word

def rule_based_lemmatize(word: str) -> str:
    """Fast, deterministic rule-based lemmatizer for common English inflections."""
    irregulars = {
        "ran": "run", "running": "run", "runs": "run",
        "went": "go", "going": "go", "goes": "go", "gone": "go",
        "better": "good", "best": "good",
        "worse": "bad", "worst": "bad",
        "mice": "mouse", "geese": "goose", "children": "child",
        "bought": "buy", "buying": "buy", "buys": "buy",
        "seen": "see", "saw": "see", "seeing": "see", "sees": "see"
    }
    if word in irregulars:
        return irregulars[word]
    return rule_based_stem(word)


class TextPreprocessorRecipe(BaseRecipe):
    recipe_id = "text_preprocessor"
    name = "Text Preprocessor & Normalizer (NLP)"
    version = "1.0.0"
    category = "nlp"
    description = "Normalizes raw text columns using Stemming, Lemmatization, Stopwords removal, and Regex HTML/URL sanitization."
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
                    "description": "Text/String columns to preprocess. If empty, all object/string columns are processed."
                },
                "lowercase": {
                    "type": "boolean",
                    "title": "Lowercase Text",
                    "default": True
                },
                "strip_html_urls": {
                    "type": "boolean",
                    "title": "Strip HTML & URLs",
                    "default": True,
                    "description": "Removes <tags> and http://... links."
                },
                "remove_punctuation": {
                    "type": "boolean",
                    "title": "Remove Punctuation",
                    "default": True
                },
                "remove_numbers": {
                    "type": "boolean",
                    "title": "Remove Numbers",
                    "default": False
                },
                "remove_stopwords": {
                    "type": "boolean",
                    "title": "Remove Stopwords",
                    "default": True,
                    "description": "Strips common filler words (the, is, at, which...)."
                },
                "normalization": {
                    "type": "string",
                    "title": "Word Normalization",
                    "enum": ["lemmatization", "stemming", "none"],
                    "default": "lemmatization",
                    "description": "Stemming cuts word suffixes, Lemmatization maps words to dictionary roots."
                }
            }
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        df: pd.DataFrame = inputs.get("dataframe")
        if df is None:
            raise ValueError("TextPreprocessor expects 'dataframe' in inputs.")

        df = df.copy()
        target_cols = config.get("columns", [])
        do_lower = config.get("lowercase", True)
        do_strip_links = config.get("strip_html_urls", True)
        do_punct = config.get("remove_punctuation", True)
        do_numbers = config.get("remove_numbers", False)
        do_stopwords = config.get("remove_stopwords", True)
        norm_mode = config.get("normalization", "lemmatization")

        # Resolve target columns
        if isinstance(target_cols, str):
            if target_cols.strip():
                parsed = [c.strip() for c in target_cols.split(",") if c.strip() in df.columns]
                target_cols = parsed if parsed else ([target_cols] if target_cols in df.columns else [])
            else:
                target_cols = []
        elif isinstance(target_cols, (list, tuple)):
            target_cols = [c for c in target_cols if c in df.columns]
        else:
            target_cols = []

        if not target_cols:
            target_cols = [c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype) == "string"]

        if not target_cols:
            return {"dataframe": df, "metrics": {"processed_columns": []}}

        # Regex patterns
        url_pattern = re.compile(r"https?://\S+|www\.\S+")
        html_pattern = re.compile(r"<.*?>")
        punct_pattern = re.compile(r"[^\w\s]")
        number_pattern = re.compile(r"\d+")

        # Check for NLTK stemmer / lemmatizer availability
        stemmer = None
        lemmatizer = None
        try:
            from nltk.stem import PorterStemmer, WordNetLemmatizer
            stemmer = PorterStemmer()
            lemmatizer = WordNetLemmatizer()
        except Exception:
            pass

        def clean_text(text: Any) -> str:
            if pd.isna(text) or text is None:
                return ""
            s = str(text)

            if do_strip_links:
                s = url_pattern.sub(" ", s)
                s = html_pattern.sub(" ", s)

            if do_lower:
                s = s.lower()

            if do_punct:
                s = punct_pattern.sub(" ", s)

            if do_numbers:
                s = number_pattern.sub(" ", s)

            # Tokenize by whitespace
            tokens = s.split()

            if do_stopwords:
                tokens = [t for t in tokens if t not in DEFAULT_STOPWORDS]

            if norm_mode == "stemming":
                if stemmer:
                    tokens = [stemmer.stem(t) for t in tokens]
                else:
                    tokens = [rule_based_stem(t) for t in tokens]
            elif norm_mode == "lemmatization":
                if lemmatizer:
                    try:
                        tokens = [lemmatizer.lemmatize(t) for t in tokens]
                    except Exception:
                        tokens = [rule_based_lemmatize(t) for t in tokens]
                else:
                    tokens = [rule_based_lemmatize(t) for t in tokens]

            return " ".join(tokens)

        for col in target_cols:
            df[col] = df[col].apply(clean_text)

        return {
            "dataframe": df,
            "metrics": {
                "processed_columns": target_cols,
                "normalization_applied": norm_mode,
                "stopwords_removed": do_stopwords
            }
        }
