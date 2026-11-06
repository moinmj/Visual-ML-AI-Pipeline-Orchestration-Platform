"""
Unit tests for Email Attachment Extractor Recipe.
"""

import io
import json
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from backend.app.core.exceptions import ExecutionException, ValidationException
from backend.app.recipes.ingestion.email_attachment_extractor import EmailAttachmentExtractorRecipe
from backend.app.recipes.base.registry import recipe_registry


def test_email_attachment_extractor_registration():
    """Verify recipe registration and catalog metadata."""
    recipe = recipe_registry.get("email_attachment_extractor")
    assert recipe is not None
    assert recipe.recipe_id == "email_attachment_extractor"
    assert recipe.group == "Ingestion & Orchestration"
    assert recipe.subgroup == "File & Web Ingestion"


def test_email_attachment_extractor_schema_contract():
    """Verify all schema properties have input_type and description."""
    recipe = EmailAttachmentExtractorRecipe()
    schema = recipe.get_schema()
    assert schema["type"] == "object"
    props = schema["properties"]
    for field_name, spec in props.items():
        assert "input_type" in spec, f"Field '{field_name}' missing input_type"
        assert "description" in spec, f"Field '{field_name}' missing description"


def test_email_attachment_extractor_validation_errors():
    """Test parameter validation rules."""
    recipe = EmailAttachmentExtractorRecipe()

    # Missing imap_server
    with pytest.raises(ValidationException) as exc_1:
        recipe.execute(inputs={}, config={"imap_server": "", "username": "user@test.com"})
    assert "Parameter 'imap_server' is required" in str(exc_1.value)

    # Missing username
    with pytest.raises(ValidationException) as exc_2:
        recipe.execute(inputs={}, config={"imap_server": "imap.test.com", "username": ""})
    assert "Parameter 'username' is required" in str(exc_2.value)

    # Port out of bounds
    with pytest.raises(ValidationException) as exc_3:
        recipe.execute(inputs={}, config={"imap_server": "imap.test.com", "username": "u", "port": 70000})
    assert "Port must be between 1 and 65535" in str(exc_3.value)

    # Malformed mock emails JSON
    with pytest.raises(ValidationException) as exc_4:
        recipe.execute(
            inputs={},
            config={"imap_server": "imap.test.com", "username": "u", "mock_emails": "{not-valid-json"},
        )
    assert "Invalid JSON provided in 'mock_emails'" in str(exc_4.value)


def test_email_attachment_extractor_simulation_fallback():
    """Test synthetic invoice data generated when server is unreachable."""
    recipe = EmailAttachmentExtractorRecipe()
    config = {
        "imap_server": "offline.imap.example.com",
        "username": "analytics@company.com",
        "simulation_fallback": True,
        "include_metadata_columns": True,
    }
    with patch.object(recipe, "_fetch_real_emails", side_effect=ConnectionError("Server unreachable")):
        result = recipe.execute(inputs={}, config=config)
    df = result["dataframe"]
    metrics = result["metrics"]

    assert not df.empty
    assert metrics["is_simulation"] is True
    assert metrics["attachments_ingested"] > 0
    assert "_email_sender" in df.columns
    assert "_email_subject" in df.columns
    assert "_attachment_name" in df.columns
    assert "invoice_id" in df.columns


def test_email_attachment_extractor_manifest_mode():
    """Test attachment_manifest output mode returns discovered file metadata table."""
    recipe = EmailAttachmentExtractorRecipe()
    mock_emails = [
        {
            "email_id": "MSG-101",
            "sender": "vendor@partner.com",
            "subject": "Monthly Statement",
            "filename": "statement_jan.csv",
            "content": "id,amount\n1,500.00\n",
            "content_type": "text/csv",
        }
    ]
    config = {
        "imap_server": "imap.company.com",
        "username": "reports@company.com",
        "output_mode": "attachment_manifest",
        "mock_emails": json.dumps(mock_emails),
    }
    result = recipe.execute(inputs={}, config=config)
    df = result["dataframe"]
    metrics = result["metrics"]

    assert len(df) == 1
    assert df.iloc[0]["filename"] == "statement_jan.csv"
    assert df.iloc[0]["sender"] == "vendor@partner.com"
    assert "size_bytes" in df.columns
    assert metrics["output_mode"] == "attachment_manifest"


def test_email_attachment_extractor_mock_csv():
    """Test parsing CSV attachment from mock emails."""
    recipe = EmailAttachmentExtractorRecipe()
    mock_emails = [
        {
            "sender": "feed@daily.com",
            "subject": "Daily KPI",
            "filename": "kpi.csv",
            "content": "kpi,value\nchurn,0.02\nnps,68\n",
        }
    ]
    config = {
        "imap_server": "imap.test.com",
        "username": "user@test.com",
        "mock_emails": json.dumps(mock_emails),
        "include_metadata_columns": True,
    }
    result = recipe.execute(inputs={}, config=config)
    df = result["dataframe"]
    assert len(df) == 2
    assert list(df["kpi"]) == ["churn", "nps"]
    assert df.iloc[0]["_email_sender"] == "feed@daily.com"


def test_email_attachment_extractor_mock_parquet():
    """Test parsing Parquet attachment."""
    recipe = EmailAttachmentExtractorRecipe()
    raw_df = pd.DataFrame([{"account": "ACC-1", "balance": 1250.5}])
    bio = io.BytesIO()
    raw_df.to_parquet(bio, index=False)
    pq_bytes = bio.getvalue()

    with patch.object(recipe, "_fetch_real_emails", return_value=[{
        "email_id": "MSG-99",
        "sender": "finance@corp.com",
        "subject": "Weekly Accounts",
        "date": "2026-11-06T08:00:00Z",
        "filename": "accounts.parquet",
        "bytes": pq_bytes,
        "size_bytes": len(pq_bytes),
        "content_type": "application/octet-stream",
    }]):
        config = {
            "imap_server": "imap.corp.com",
            "username": "fin@corp.com",
            "include_metadata_columns": False,
        }
        result = recipe.execute(inputs={}, config=config)
        df = result["dataframe"]
        assert len(df) == 1
        assert df.iloc[0]["account"] == "ACC-1"
        assert "_email_sender" not in df.columns


def test_email_attachment_extractor_simulation_disabled_raises():
    """Test error raised when simulation fallback is disabled and IMAP fails."""
    recipe = EmailAttachmentExtractorRecipe()
    with patch.object(recipe, "_fetch_real_emails", side_effect=ConnectionRefusedError("Refused")):
        with pytest.raises(ExecutionException) as exc:
            recipe.execute(
                inputs={},
                config={"imap_server": "bad.host", "username": "u", "simulation_fallback": False},
            )
        assert "Failed connecting to IMAP server" in str(exc.value)


def test_email_attachment_extractor_pattern_matching():
    """Test matching glob patterns in attachment filename."""
    recipe = EmailAttachmentExtractorRecipe()
    assert recipe._matches_pattern("orders.csv", "*.csv;*.xlsx") is True
    assert recipe._matches_pattern("orders.xlsx", "*.csv;*.xlsx") is True
    assert recipe._matches_pattern("orders.pdf", "*.csv;*.xlsx") is False
    assert recipe._matches_pattern("anything.xyz", "*") is True
