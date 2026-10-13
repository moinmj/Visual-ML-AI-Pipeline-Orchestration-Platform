import pytest
import pandas as pd
from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.nlp.summarization import (
    SummarizationRecipe,
    split_into_sentences,
    extract_keywords_weights,
    summarize_extractive_tfidf,
    summarize_textrank,
    summarize_lead,
    summarize_key_takeaway,
    generate_text_summary
)


def test_summarization_recipe_registration():
    """Verifies that text_summarization is properly registered, enabled, and aliased."""
    recipe = recipe_registry.get("text_summarization")
    assert recipe is not None
    assert recipe.recipe_id == "text_summarization"
    assert recipe.name == "Summarization"
    assert recipe.category == "nlp"

    # Test alias
    aliased_recipe = recipe_registry.get("summarization")
    assert aliased_recipe is recipe

    # Schema verification
    schema = recipe.get_schema()
    props = schema["properties"]
    assert "columns" in props
    assert "method" in props
    assert "max_sentences" in props
    assert "max_characters" in props
    assert "append_columns" in props
    assert "output_suffix" in props

    # Verify input_type is present on configs
    assert props["columns"].get("input_type") == "column_select"
    assert props["method"].get("input_type") == "dropdown"
    assert props["max_sentences"].get("input_type") == "number"
    assert props["append_columns"].get("input_type") == "boolean"

    # Metadata group and subgroup
    meta = recipe_registry.get_metadata("text_summarization")
    assert meta.group == "Data Preparation & Transformation"
    assert meta.subgroup == "NLP/Text"
    assert meta.enabled is True


def test_sentence_splitting_and_weights():
    """Tests sentence splitting and keyword weights extraction."""
    text = "First sentence here. Second sentence with words! Third sentence is awesome?"
    sentences = split_into_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "First sentence here."
    assert sentences[1] == "Second sentence with words!"
    assert sentences[2] == "Third sentence is awesome?"

    # Empty / whitespace
    assert split_into_sentences("") == []
    assert split_into_sentences(None) == []

    # Keyword weights
    weights = extract_keywords_weights("Quality quality build performance build build.")
    assert "build" in weights
    assert weights["build"] == 1.0  # max frequency normalized to 1.0


def test_extractive_tfidf_and_textrank_methods():
    """Tests extractive TF-IDF and TextRank summarization methods."""
    long_review = (
        "The customer service was remarkably friendly, helpful and responsive throughout our issue. "
        "We ordered several complex machine components last Tuesday. "
        "The components arrived safely without any physical damage or delays. "
        "Overall, this supplier provided prompt service, great communication, and excellent customer service."
    )
    sentences = split_into_sentences(long_review)
    assert len(sentences) >= 4

    # TF-IDF
    tfidf_sum = summarize_extractive_tfidf(sentences, max_sentences=2)
    assert len(tfidf_sum) == 2
    # Check that sentences maintain original relative order
    idx0 = sentences.index(tfidf_sum[0])
    idx1 = sentences.index(tfidf_sum[1])
    assert idx0 < idx1

    # TextRank
    tr_sum = summarize_textrank(sentences, max_sentences=2)
    assert len(tr_sum) == 2

    # Lead Sentences
    lead_sum = summarize_lead(sentences, max_sentences=2)
    assert len(lead_sum) == 2
    assert lead_sum == sentences[:2]

    # Key Takeaway
    takeaway = summarize_key_takeaway(sentences)
    assert len(takeaway) == 1


def test_generate_text_summary_character_cap():
    """Tests max_characters trimming with ellipsis."""
    text = (
        "First sentence discussing technology. "
        "Second sentence elaborating on advanced multi-agent systems and automated pipelines. "
        "Third sentence summarizing the conclusion."
    )
    short_summary = generate_text_summary(text, method="Lead Sentences", max_sentences=2, max_characters=45)
    assert len(short_summary) <= 48  # including ellipsis
    assert short_summary.endswith("...")


def test_summarization_execution_append_columns():
    """Tests DataFrame execution with append_columns=True."""
    recipe = SummarizationRecipe()

    df = pd.DataFrame({
        "id": [1, 2],
        "feedback": [
            (
                "The product build quality is absolutely outstanding and exceeded all our expectations! "
                "Delivery was surprisingly fast, arriving two days earlier than expected. "
                "The packaging was secure and eco-friendly. "
                "Highly recommend this product to anyone looking for premium reliability."
            ),
            (
                "Terrible experience with delayed shipping. "
                "The package was broken upon arrival. "
                "Customer support took three days to answer my ticket. "
                "I will not be ordering from this vendor again."
            )
        ]
    })

    result = recipe.execute(
        inputs={"dataframe": df},
        config={
            "columns": ["feedback"],
            "method": "Extractive (TF-IDF / Frequency-Based)",
            "max_sentences": 2,
            "max_characters": 180,
            "append_columns": True,
            "output_suffix": "_summary"
        }
    )

    out_df = result["dataframe"]
    assert "feedback_summary" in out_df.columns
    assert "feedback" in out_df.columns  # original preserved

    # Check metrics
    metrics = result["metrics"]
    assert metrics["total_rows"] == 2
    assert "feedback" in metrics["summarized_columns"]
    assert metrics["method"] == "Extractive (TF-IDF / Frequency-Based)"
    assert metrics["total_summarized_cells"] == 2
    assert result["output_summary"]["status"] == "SUCCESS"

    # Verify summary is shorter than or equal to original
    orig_len_0 = len(out_df.loc[0, "feedback"])
    sum_len_0 = len(out_df.loc[0, "feedback_summary"])
    assert sum_len_0 <= orig_len_0


def test_summarization_execution_replace_columns():
    """Tests in-place replacement when append_columns=False."""
    recipe = SummarizationRecipe()

    df = pd.DataFrame({
        "article": [
            "Alpha sentence one. Beta sentence two. Gamma sentence three. Delta sentence four."
        ]
    })

    result = recipe.execute(
        inputs={"dataframe": df},
        config={
            "columns": ["article"],
            "method": "Lead Sentences (First N Sentences)",
            "max_sentences": 2,
            "append_columns": False
        }
    )

    out_df = result["dataframe"]
    assert "article" in out_df.columns
    assert "article_summary" not in out_df.columns
    assert "Alpha sentence one. Beta sentence two." in out_df.loc[0, "article"]


def test_summarization_auto_detect_columns_and_empty():
    """Tests automatic detection of text columns and handling empty data."""
    recipe = SummarizationRecipe()

    # Empty df
    empty_res = recipe.execute(inputs={"dataframe": pd.DataFrame()}, config={})
    assert empty_res["metrics"]["total_rows"] == 0

    # Auto-detection
    mixed_df = pd.DataFrame({
        "num": [10, 20],
        "notes": [
            "First detailed note about system performance and server metrics. Second note on latency.",
            "Database connection pool exhausted during peak traffic. Reconfigured pool size to resolve."
        ]
    })

    res = recipe.execute(inputs={"dataframe": mixed_df}, config={})
    out = res["dataframe"]
    assert "notes_summary" in out.columns
    assert "num_summary" not in out.columns
