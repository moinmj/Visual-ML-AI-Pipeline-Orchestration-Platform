import pytest
import pandas as pd
from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.nlp.translation import (
    TranslationRecipe,
    translate_text_offline,
    normalize_lang_code,
    translate_text_online_google
)


def test_translation_recipe_registration():
    """Verifies that text_translation is properly registered, enabled, and aliased."""
    recipe = recipe_registry.get("text_translation")
    assert recipe is not None
    assert recipe.recipe_id == "text_translation"
    assert recipe.name == "Translation"
    assert recipe.category == "nlp"

    # Test alias
    aliased_recipe = recipe_registry.get("translation")
    assert aliased_recipe is recipe

    # Schema verification
    schema = recipe.get_schema()
    props = schema["properties"]
    assert "columns" in props
    assert "destination_language" in props
    assert "source_language" in props
    assert "engine" in props
    assert "append_columns" in props
    assert "output_suffix" in props

    # Metadata group and subgroup
    meta = recipe_registry.get_metadata("text_translation")
    assert meta.group == "Data Preparation & Transformation"
    assert meta.subgroup == "NLP/Text"
    assert meta.enabled is True


def test_normalize_lang_code():
    """Tests parsing and normalizing language strings into ISO codes."""
    assert normalize_lang_code("Spanish (es)") == "es"
    assert normalize_lang_code("French (fr)") == "fr"
    assert normalize_lang_code("de") == "de"
    assert normalize_lang_code("German") == "de"
    assert normalize_lang_code("Japanese (ja)") == "ja"
    assert normalize_lang_code("unknown_xyz") == "es"  # default fallback


def test_translate_text_offline_phrases_and_casing():
    """Tests offline dictionary translation for words, phrases, and capitalization."""
    # Greetings & phrases
    assert "hola" in translate_text_offline("hello", "es").lower()
    assert "gracias" in translate_text_offline("thank you", "es").lower()
    assert "merci" in translate_text_offline("thank you", "fr").lower()
    assert "danke" in translate_text_offline("thank you", "de").lower()
    assert "grazie" in translate_text_offline("thank you", "it").lower()
    assert "obrigado" in translate_text_offline("thank you", "pt").lower()

    # Title casing preservation
    res_title = translate_text_offline("Hello, Customer!", "es")
    assert res_title == "Hola, Cliente!"

    # UPPERCASE preservation
    res_upper = translate_text_offline("HELLO", "es")
    assert res_upper == "HOLA"

    # Complex customer feedback sentence
    sentence = "The product is broken and customer service was slow."
    translated_es = translate_text_offline(sentence, "es")
    assert "producto" in translated_es.lower()
    assert "roto" in translated_es.lower()
    assert "servicio al cliente" in translated_es.lower() or "cliente" in translated_es.lower()

    # Empty / whitespace / None handling
    assert translate_text_offline("", "es") == ""
    assert translate_text_offline(None, "es") == ""
    assert translate_text_offline("   ", "es") == "   "


def test_translation_execution_append_columns():
    """Tests DataFrame execution with append_columns=True."""
    recipe = TranslationRecipe()

    df = pd.DataFrame({
        "id": [1, 2, 3],
        "feedback": ["Excellent product and fast shipping!", "Broken item, horrible quality.", "Thank you very much."],
        "status": ["completed", "failed", "pending"]
    })

    result = recipe.execute(
        inputs={"dataframe": df},
        config={
            "columns": ["feedback", "status"],
            "target_language": "Spanish (es)",
            "engine": "Offline Dictionary (Fast & Deterministic)",
            "append_columns": True,
            "output_suffix": "_translated"
        }
    )

    out_df = result["dataframe"]
    assert "feedback_translated" in out_df.columns
    assert "status_translated" in out_df.columns
    assert "feedback" in out_df.columns  # Original preserved

    # Check content in row 0
    assert "excelente" in out_df.loc[0, "feedback_translated"].lower()
    assert "completado" in out_df.loc[0, "status_translated"].lower()

    # Check metrics
    metrics = result["metrics"]
    assert metrics["total_rows"] == 3
    assert "feedback" in metrics["translated_columns"]
    assert "status" in metrics["translated_columns"]
    assert metrics["target_language"] == "es"
    assert result["output_summary"]["status"] == "SUCCESS"


def test_translation_execution_replace_columns():
    """Tests in-place column replacement when append_columns=False."""
    recipe = TranslationRecipe()

    df = pd.DataFrame({
        "greeting": ["hello", "goodbye"]
    })

    result = recipe.execute(
        inputs={"dataframe": df},
        config={
            "columns": ["greeting"],
            "target_language": "French (fr)",
            "engine": "Offline Dictionary (Fast & Deterministic)",
            "append_columns": False
        }
    )

    out_df = result["dataframe"]
    assert "greeting" in out_df.columns
    assert "greeting_translated" not in out_df.columns
    assert "bonjour" in out_df.loc[0, "greeting"].lower()
    assert "au revoir" in out_df.loc[1, "greeting"].lower()


def test_translation_auto_detect_columns():
    """Tests auto-detection of text/string columns when columns config is omitted."""
    recipe = TranslationRecipe()

    df = pd.DataFrame({
        "numeric_val": [10, 20, 30],
        "message": ["Welcome to our service", "Thank you for the update", "Order pending"]
    })

    result = recipe.execute(
        inputs={"dataframe": df},
        config={
            "target_language": "German (de)",
            "engine": "Offline Dictionary (Fast & Deterministic)",
            "append_columns": True
        }
    )

    out_df = result["dataframe"]
    assert "message_translated" in out_df.columns
    assert "numeric_val_translated" not in out_df.columns
    assert "willkommen" in out_df.loc[0, "message_translated"].lower()


def test_translation_empty_dataframe_and_skipped():
    """Tests handling of empty DataFrames and DataFrames with no text columns."""
    recipe = TranslationRecipe()

    # 1. Empty DataFrame
    res_empty = recipe.execute(
        inputs={"dataframe": pd.DataFrame()},
        config={}
    )
    assert res_empty["dataframe"].empty
    assert res_empty["metrics"]["total_rows"] == 0

    # 2. DataFrame with no text columns
    num_df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.5, 6.2]})
    res_skipped = recipe.execute(
        inputs={"dataframe": num_df},
        config={"columns": []}
    )
    assert res_skipped["output_summary"]["status"] == "SKIPPED"
    assert res_skipped["metrics"]["translated_columns"] == []


def test_translation_mock_engine():
    """Tests mock / simulation engine for rapid synthetic labeling."""
    recipe = TranslationRecipe()
    df = pd.DataFrame({"text": ["Sample pipeline text"]})

    res = recipe.execute(
        inputs={"dataframe": df},
        config={
            "target_language": "Spanish (es)",
            "engine": "Mock / Simulation",
            "append_columns": False
        }
    )
    assert res["dataframe"].loc[0, "text"] == "[ES] Sample pipeline text"
