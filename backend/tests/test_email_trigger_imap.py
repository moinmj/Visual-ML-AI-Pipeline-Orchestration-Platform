import pytest
import unittest.mock as mock
import pandas as pd
from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.triggers.email_trigger_imap import EmailTriggerIMAPRecipe, _decode_mime_header


def test_imap_recipe_registration_and_aliases():
    """Verify recipe registration and n8n alias resolution."""
    recipe = recipe_registry.get("email_trigger_imap")
    assert isinstance(recipe, EmailTriggerIMAPRecipe)
    assert recipe.recipe_id == "email_trigger_imap"
    assert recipe.name == "Email Trigger (IMAP)"
    assert recipe.category == "triggers"

    # Aliases
    assert recipe_registry.get("imap").recipe_id == "email_trigger_imap"
    assert recipe_registry.get("imap_trigger").recipe_id == "email_trigger_imap"
    assert recipe_registry.get("email_trigger").recipe_id == "email_trigger_imap"


def test_imap_schema_structure_and_n8n_tabs():
    """Verify separate tabs (Connection, Parameters, Settings) and concise types."""
    recipe = EmailTriggerIMAPRecipe()
    schema = recipe.get_schema()

    assert schema["type"] == "object"
    assert schema["tabs"] == ["Connection", "Parameters", "Settings"]

    props = schema["properties"]

    # Connection tab
    conn_props = [k for k, v in props.items() if v.get("tab") == "Connection"]
    assert "host" in conn_props
    assert "port" in conn_props
    assert "user" in conn_props
    assert "password" in conn_props
    assert "ssl_tls" in conn_props
    assert "allow_self_signed" in conn_props

    # Types are concise
    assert props["host"]["type"] == "text"
    assert props["port"]["type"] == "number"
    assert props["password"]["type"] == "password"
    assert props["ssl_tls"]["type"] == "boolean"

    # Parameters tab
    param_props = [k for k, v in props.items() if v.get("tab") == "Parameters"]
    assert "mailbox_name" in param_props
    assert "action" in param_props
    assert "format" in param_props
    assert "download_attachments" in param_props
    assert "unread_only" in param_props
    assert "limit" in param_props

    assert props["action"]["type"] == "select"
    assert props["action"]["enum"] == ["Mark as Read", "None", "Delete"]
    assert props["format"]["type"] == "select"
    assert props["format"]["enum"] == ["Simple", "Resolved", "RAW"]

    # Settings tab
    settings_props = [k for k, v in props.items() if v.get("tab") == "Settings"]
    assert "always_output_data" in settings_props
    assert "execute_once" in settings_props
    assert "retry_on_fail" in settings_props
    assert "on_error" in settings_props
    assert props["on_error"]["enum"] == ["Stop Workflow", "Continue", "Continue (using error output)"]


def test_imap_simulated_execution():
    """Verify simulation mode when credentials are not yet supplied."""
    recipe = EmailTriggerIMAPRecipe()

    # Default run
    result = recipe.execute(inputs={}, config={"mailbox_name": "INBOX", "limit": 5})
    assert "dataframe" in result
    df = result["dataframe"]
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "subject" in df.columns
    assert "from" in df.columns
    assert "body" in df.columns

    # Attachment download simulation
    result_att = recipe.execute(inputs={}, config={"mailbox_name": "INBOX", "download_attachments": True})
    df_att = result_att["dataframe"]
    assert isinstance(df_att, pd.DataFrame)
    assert "transaction_id" in df_att.columns
    assert "revenue" in df_att.columns


def test_imap_decode_mime_header():
    """Verify MIME header decoder."""
    assert _decode_mime_header("Simple Subject") == "Simple Subject"
    assert _decode_mime_header("") == ""
    # Test RFC2047 encoded
    encoded = "=?utf-8?B?VGVzdCBTdWJqZWN0?="
    assert _decode_mime_header(encoded) == "Test Subject"


def test_imap_live_mocked_execution():
    """Verify execution logic with mocked IMAP server."""
    recipe = EmailTriggerIMAPRecipe()

    mock_mail = mock.MagicMock()
    mock_mail.login.return_value = ("OK", [b"Logged in"])
    mock_mail.select.return_value = ("OK", [b"1"])
    mock_mail.search.return_value = ("OK", [b"1"])

    # Raw RFC822 message bytes
    raw_message = (
        b"From: sender@example.com\r\n"
        b"To: receiver@example.com\r\n"
        b"Subject: Sales Report\r\n"
        b"Date: Mon, 1 Jan 2026 12:00:00 +0000\r\n"
        b"Content-Type: text/plain; charset=utf-8\r\n\r\n"
        b"Here is the weekly update body."
    )
    mock_mail.fetch.return_value = ("OK", [(b"1 (RFC822 {100})", raw_message)])

    with mock.patch("imaplib.IMAP4_SSL", return_value=mock_mail):
        result = recipe.execute(
            inputs={},
            config={
                "host": "imap.example.com",
                "port": 993,
                "user": "test@example.com",
                "password": "secret_password",
                "mailbox_name": "INBOX",
                "action": "Mark as Read",
                "ssl_tls": True
            }
        )

    assert "dataframe" in result
    df = result["dataframe"]
    assert len(df) == 1
    assert df.iloc[0]["subject"] == "Sales Report"
    assert df.iloc[0]["from"] == "sender@example.com"
    assert "weekly update" in df.iloc[0]["body"]
    mock_mail.store.assert_called_with(b"1", "+FLAGS", "\\Seen")


def test_imap_error_handling_continue():
    """Verify on_error='Continue' returns an error DataFrame without crashing the pipeline."""
    recipe = EmailTriggerIMAPRecipe()

    with mock.patch("imaplib.IMAP4_SSL", side_effect=Exception("Connection timed out")):
        result = recipe.execute(
            inputs={},
            config={
                "host": "imap.fail.com",
                "user": "fail@example.com",
                "password": "pass",
                "on_error": "Continue"
            }
        )

    assert "dataframe" in result
    df = result["dataframe"]
    assert len(df) == 1
    assert "error" in df.columns
    assert "Connection timed out" in df.iloc[0]["error"]
