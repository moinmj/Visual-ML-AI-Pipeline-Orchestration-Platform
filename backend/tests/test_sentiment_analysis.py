import pytest
import pandas as pd
from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.nlp.sentiment_analysis import SentimentAnalysisRecipe, analyze_text_sentiment


def test_sentiment_recipe_registration():
    """Verifies that sentiment_analysis recipe is properly registered and discoverable."""
    recipe = recipe_registry.get("sentiment_analysis")
    assert recipe is not None
    assert recipe.recipe_id == "sentiment_analysis"
    assert recipe.name == "Sentiment Analysis"
    assert recipe.category == "nlp"

    schema = recipe.get_schema()
    assert "columns" in schema["properties"]
    assert "output_mode" in schema["properties"]
    assert "positive_threshold" in schema["properties"]
    assert "negative_threshold" in schema["properties"]

    meta = recipe_registry.get_metadata("sentiment_analysis")
    assert meta.group == "Data Preparation & Transformation"
    assert meta.subgroup == "NLP/Text"


def test_analyze_text_sentiment_lexicon():
    """Tests valence scoring on positive, negative, neutral, and negated phrases."""
    # Positive
    res_pos = analyze_text_sentiment("This product is amazing, fast, and I love it!")
    assert res_pos["compound"] > 0.4
    assert res_pos["positive"] > res_pos["negative"]

    # Negative
    res_neg = analyze_text_sentiment("Terrible customer service, awful experience and broken item.")
    assert res_neg["compound"] < -0.4
    assert res_neg["negative"] > res_pos["negative"]

    # Neutral
    res_neu = analyze_text_sentiment("The meeting is scheduled on Tuesday at 3 PM.")
    assert -0.05 <= res_neu["compound"] <= 0.05
    assert res_neu["neutral"] >= 0.8

    # Negation
    res_negated = analyze_text_sentiment("The service was not good and definitely not helpful.")
    assert res_negated["compound"] < 0.0

    # Emoticons and punctuation
    res_emoticon = analyze_text_sentiment("Great job! :) 👍")
    assert res_emoticon["compound"] > 0.5


def test_sentiment_analysis_execution_dataframe():
    """Tests execution of SentimentAnalysisRecipe on a DataFrame."""
    recipe = SentimentAnalysisRecipe()

    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "review": [
            "Outstanding performance and wonderful quality!",
            "Horrible defect, completely useless and broke immediately.",
            "The delivery was made to office building B.",
            "Really great, very happy with the purchase!",
            "Poor battery life, disappointed."
        ],
        "category": ["Electronics", "Electronics", "Logistics", "Home", "Electronics"]
    })

    # Test Default Mode (Label and Score)
    result = recipe.execute(
        inputs={"dataframe": df},
        config={
            "columns": ["review"],
            "output_mode": "Label and Score",
            "positive_threshold": 0.05,
            "negative_threshold": -0.05
        }
    )

    out_df = result["dataframe"]
    assert "review_sentiment" in out_df.columns
    assert "review_sentiment_score" in out_df.columns

    # Row 1 is positive
    assert out_df.loc[0, "review_sentiment"] == "positive"
    assert out_df.loc[0, "review_sentiment_score"] > 0.05

    # Row 2 is negative
    assert out_df.loc[1, "review_sentiment"] == "negative"
    assert out_df.loc[1, "review_sentiment_score"] < -0.05

    # Row 3 is neutral
    assert out_df.loc[2, "review_sentiment"] == "neutral"

    # Row 4 is positive
    assert out_df.loc[3, "review_sentiment"] == "positive"

    # Row 5 is negative
    assert out_df.loc[4, "review_sentiment"] == "negative"

    # Check metrics
    metrics = result["metrics"]
    assert metrics["total_rows"] == 5
    assert metrics["sentiment_distribution"]["positive"] >= 2
    assert metrics["sentiment_distribution"]["negative"] >= 2
    assert metrics["sentiment_distribution"]["neutral"] >= 1


def test_sentiment_analysis_auto_detect_columns():
    """Tests auto-detecting text columns when columns config is omitted."""
    recipe = SentimentAnalysisRecipe()

    df = pd.DataFrame({
        "score": [10, 20],
        "feedback": ["Excellent service!", "Awful delay."]
    })

    result = recipe.execute(
        inputs={"dataframe": df},
        config={}  # Empty config should auto-detect 'feedback'
    )

    out_df = result["dataframe"]
    assert "feedback_sentiment" in out_df.columns
    assert out_df.loc[0, "feedback_sentiment"] == "positive"
    assert out_df.loc[1, "feedback_sentiment"] == "negative"


def test_sentiment_analysis_detailed_probabilities():
    """Tests Detailed Probabilities output mode."""
    recipe = SentimentAnalysisRecipe()

    df = pd.DataFrame({
        "text": ["Very happy with this!", "Total waste of time and money."]
    })

    result = recipe.execute(
        inputs={"dataframe": df},
        config={
            "columns": ["text"],
            "output_mode": "Detailed Probabilities"
        }
    )

    out_df = result["dataframe"]
    assert "text_sentiment" in out_df.columns
    assert "text_sentiment_score" in out_df.columns
    assert "text_pos_score" in out_df.columns
    assert "text_neg_score" in out_df.columns
    assert "text_neu_score" in out_df.columns

    assert out_df.loc[0, "text_pos_score"] > 0
    assert out_df.loc[1, "text_neg_score"] > 0


def test_sentiment_analysis_replace_mode():
    """Tests replacing original column when append_columns is False."""
    recipe = SentimentAnalysisRecipe()

    df = pd.DataFrame({
        "feedback": ["Loved it!", "Hated it!"]
    })

    result = recipe.execute(
        inputs={"dataframe": df},
        config={
            "columns": ["feedback"],
            "append_columns": False
        }
    )

    out_df = result["dataframe"]
    # 'feedback' was replaced by sentiment labels
    assert list(out_df["feedback"]) == ["positive", "negative"]
