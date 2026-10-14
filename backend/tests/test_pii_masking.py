import pytest
import pandas as pd
from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes import register_all_recipes
from backend.app.recipes.nlp.pii_masking import (
    PIIEntityMaskingRecipe,
    mask_text_pii,
    is_luhn_valid,
)


@pytest.fixture(autouse=True)
def setup_recipes():
    register_all_recipes()


def test_pii_recipe_registration():
    recipe = recipe_registry.get("pii_entity_masking")
    assert recipe is not None
    assert recipe.recipe_id == "pii_entity_masking"
    assert recipe.name == "PII / Entity Masking"
    assert recipe.category == "nlp"

    meta = recipe_registry.get_metadata("pii_entity_masking")
    assert meta.enabled is True
    assert meta.group == "Data Preparation & Transformation"
    assert meta.subgroup == "NLP/Text"


def test_pii_recipe_schema_input_types():
    """Ensure every configuration property strictly contains 'input_type'."""
    recipe = PIIEntityMaskingRecipe()
    schema = recipe.get_schema()
    assert "properties" in schema
    props = schema["properties"]

    assert len(props) > 0
    for prop_name, prop_data in props.items():
        assert "input_type" in prop_data, f"Property '{prop_name}' is missing mandatory 'input_type'!"
        assert prop_data["input_type"] in [
            "column_select",
            "dropdown",
            "number",
            "boolean",
            "text",
            "dataset_select",
        ], f"Property '{prop_name}' has unrecognized input_type: {prop_data['input_type']}"


def test_email_and_phone_masking():
    text = "Please reach out to support@mycompany.org or call +1 (555) 234-5678 immediately."
    masked, counts = mask_text_pii(text, mask_style="Entity Tag Replacement")

    assert "[EMAIL]" in masked
    assert "support@mycompany.org" not in masked
    assert "[PHONE]" in masked
    assert "555" not in masked
    assert counts.get("EMAIL") == 1
    assert counts.get("PHONE") == 1


def test_credit_card_and_ssn_masking():
    # Valid Visa card format (using a Luhn-compliant test card)
    valid_test_card = "4532-0150-1234-5671"
    text = f"Card: {valid_test_card}, SSN: 123-45-6789."
    masked, counts = mask_text_pii(text, mask_style="Entity Tag Replacement")

    assert "[CREDIT_CARD]" in masked
    assert valid_test_card not in masked
    assert "[SSN]" in masked
    assert "123-45-6789" not in masked
    assert counts.get("CREDIT_CARD") == 1
    assert counts.get("SSN") == 1


def test_luhn_algorithm():
    # Valid Luhn cards (standard 16-digit and Wikipedia example 79927398713)
    assert is_luhn_valid("4532015012345671") is True
    # Invalid card
    assert is_luhn_valid("4532015012345679") is False
    # Short string
    assert is_luhn_valid("12345") is False


def test_ip_address_and_name_masking():
    text = "Connection from 192.168.1.105 by Dr. John Doe and Alice Smith."
    masked, counts = mask_text_pii(text, mask_style="Entity Tag Replacement")

    assert "[IP_ADDRESS]" in masked
    assert "192.168.1.105" not in masked
    assert "[NAME]" in masked
    assert "Dr. John Doe" not in masked
    assert counts.get("IP_ADDRESS") == 1
    assert counts.get("NAME") >= 1


def test_masking_styles():
    text = "Contact john.doe@example.com for info"

    # 1. Full Redaction
    redacted, _ = mask_text_pii(text, mask_style="Full Redaction", custom_token="[CONFIDENTIAL]")
    assert "[CONFIDENTIAL]" in redacted
    assert "john.doe@example.com" not in redacted

    # 2. Asterisks
    asterisked, _ = mask_text_pii(text, mask_style="Asterisks Obfuscation")
    assert "john.doe@example.com" not in asterisked
    assert "*" in asterisked

    # 3. Pseudonymization (Hash)
    hashed1, _ = mask_text_pii(text, mask_style="Pseudonymization (Hash)")
    hashed2, _ = mask_text_pii(text, mask_style="Pseudonymization (Hash)")
    assert "[EMAIL_" in hashed1
    # Hash should be deterministic
    assert hashed1 == hashed2

    # 4. Partial Masking
    card_text = "4532-0150-1234-5674"
    partial, _ = mask_text_pii(card_text, mask_style="Partial Masking (Last 4 Visible)")
    assert partial.endswith("5674")
    assert "*" in partial


def test_custom_regex_masking():
    text = "Secret token is APIKEY-98471 and employee is EMP-4321."
    masked, counts = mask_text_pii(
        text,
        mask_style="Entity Tag Replacement",
        custom_regex_str=r"(?:APIKEY|EMP)-\d{4,5}"
    )
    assert "[CUSTOM]" in masked
    assert "APIKEY-98471" not in masked
    assert "EMP-4321" not in masked
    assert counts.get("CUSTOM") == 2


def test_recipe_execution_on_dataframe():
    df = pd.DataFrame({
        "customer_notes": [
            "User email is alice@wonderland.com and phone is +1-555-432-1098.",
            "Client Mr. Robert Brown logged in from 10.0.0.1.",
            "Normal text without any identifiers."
        ],
        "department": ["Support", "Security", "General"]
    })

    recipe = PIIEntityMaskingRecipe()
    config = {
        "columns": ["customer_notes"],
        "mask_style": "Entity Tag Replacement",
        "append_columns": True,
        "output_suffix": "_masked"
    }

    result = recipe.execute({"dataframe": df}, config)
    out_df = result["dataframe"]
    metrics = result["metrics"]
    summary = result["output_summary"]

    assert "customer_notes_masked" in out_df.columns
    assert "customer_notes" in out_df.columns
    assert metrics["total_rows_processed"] == 3
    assert metrics["total_entities_masked"] >= 3
    assert "EMAIL" in metrics["entities_by_type"]
    assert "IP_ADDRESS" in metrics["entities_by_type"]

    # Verify first row
    row0_masked = out_df["customer_notes_masked"].iloc[0]
    assert "[EMAIL]" in row0_masked
    assert "alice@wonderland.com" not in row0_masked
    assert "[PHONE]" in row0_masked


def test_recipe_replace_columns_in_place():
    df = pd.DataFrame({
        "notes": ["Contact bob@mail.com"]
    })
    recipe = PIIEntityMaskingRecipe()
    result = recipe.execute({"dataframe": df}, {"append_columns": False})
    out_df = result["dataframe"]

    assert "notes_masked" not in out_df.columns
    assert "[EMAIL]" in out_df["notes"].iloc[0]


def test_empty_dataframe_and_no_target_columns():
    recipe = PIIEntityMaskingRecipe()

    # Empty df
    res_empty = recipe.execute({"dataframe": pd.DataFrame()}, {})
    assert res_empty["dataframe"].empty

    # Non-text df
    num_df = pd.DataFrame({"age": [25, 30, 35]})
    res_num = recipe.execute({"dataframe": num_df}, {})
    assert res_num["metrics"]["total_entities_masked"] == 0
