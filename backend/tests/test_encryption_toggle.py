import pandas as pd
import pytest

from backend.app.recipes.security.encryption_toggle import EncryptionToggleRecipe


def test_encryption_toggle_metadata_and_schema():
    recipe = EncryptionToggleRecipe()
    assert recipe.recipe_id == "encryption_toggle"
    assert recipe.category == "security"
    assert "report" in recipe.output_types

    schema = recipe.get_schema()
    assert schema["type"] == "object"
    props = schema["properties"]
    assert "encryption_at_rest_algorithm" in props
    assert "enforce_tls_version" in props
    assert "operation_mode" in props

    for prop_name, prop_data in props.items():
        assert "description" in prop_data, f"Property {prop_name} missing description"
        assert "input_type" in prop_data, f"Property {prop_name} missing input_type"


def test_encryption_toggle_audit_compliance():
    recipe = EncryptionToggleRecipe()
    result = recipe.execute(
        {},
        {
            "encryption_at_rest_algorithm": "AES_256_GCM",
            "enforce_tls_version": "TLS_1_3",
            "operation_mode": "audit_and_enforce",
        },
    )

    assert result["status"] == "COMPLIANT"
    report = result["report"]
    assert report["encryption_at_rest"]["fips_140_2_approved"] is True
    assert report["encryption_in_transit"]["forward_secrecy"] is True


def test_encryption_and_decryption_roundtrip():
    recipe = EncryptionToggleRecipe()
    df = pd.DataFrame({
        "patient_id": ["P-101", "P-102"],
        "ssn": ["123-45-6789", "987-65-4321"],
    })

    # Encrypt
    enc_res = recipe.execute(
        {"input": df},
        {
            "target_columns": "ssn",
            "operation_mode": "encrypt_columns",
            "secret_passphrase": "super-strong-aes-key-2026",
        },
    )
    enc_df = enc_res["dataframe"]
    assert enc_df["ssn"].iloc[0].startswith("ENC:v1:")
    assert enc_df["patient_id"].iloc[0] == "P-101"

    # Decrypt
    dec_res = recipe.execute(
        {"input": enc_df},
        {
            "target_columns": "ssn",
            "operation_mode": "decrypt_columns",
            "secret_passphrase": "super-strong-aes-key-2026",
        },
    )
    dec_df = dec_res["dataframe"]
    assert dec_df["ssn"].iloc[0] == "123-45-6789"
    assert dec_df["ssn"].iloc[1] == "987-65-4321"
