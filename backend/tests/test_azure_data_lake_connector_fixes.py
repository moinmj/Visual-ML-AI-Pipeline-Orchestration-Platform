"""
Unit tests verifying Azure Data Lake Storage Connector fixes:
- Root ingestion inputs (inputs == [])
- Icon metadata
- ValidationException handling on missing config/credentials
- Clean credential context resolution
- Simulation benchmark generation
- Live mock ingestion
"""

import io
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from backend.app.recipes.ingestion.azure_data_lake_connector import AzureDataLakeConnectorRecipe
from backend.app.core.exceptions import ValidationException


def test_azure_data_lake_port_definitions():
    """Verify Azure Data Lake connector is a root ingestion recipe with no input ports."""
    r = AzureDataLakeConnectorRecipe()
    inputs = r.get_inputs()
    assert inputs == [], f"Expected no input ports for Azure root ingestion, got: {inputs}"

    outputs = r.get_outputs()
    assert len(outputs) == 1
    assert outputs[0].id == "output"
    assert outputs[0].type == "dataframe"
    assert r.icon == "azure"


def test_azure_data_lake_validation_live_missing_fields():
    """Verify live mode raises ValidationException when required fields are missing."""
    r = AzureDataLakeConnectorRecipe()

    # Missing container
    with pytest.raises(ValidationException) as exc_1:
        r.execute({}, {"simulation_mode": False, "container_name": "", "blob_path": "data.csv"})
    assert "container" in str(exc_1.value).lower()

    # Missing blob path
    with pytest.raises(ValidationException) as exc_2:
        r.execute({}, {"simulation_mode": False, "container_name": "lake", "blob_path": ""})
    assert "blob" in str(exc_2.value).lower()

    # Missing credentials
    with pytest.raises(ValidationException) as exc_3:
        r.execute({}, {
            "simulation_mode": False,
            "container_name": "lake",
            "blob_path": "data.csv",
            "auth_mode": "connection_string",
            "connection_string": "",
            "account_key": "",
            "sas_token": ""
        })
    assert "credentials missing" in str(exc_3.value).lower()


def test_azure_data_lake_simulation_mode():
    """Verify simulation mode produces realistic benchmark records without Azure access."""
    r = AzureDataLakeConnectorRecipe()
    res = r.execute({}, {
        "simulation_mode": True,
        "container_name": "analytics-container",
        "blob_path": "raw/telemetry.parquet",
        "limit": 25
    })

    assert "dataframe" in res
    df = res["dataframe"]
    assert len(df) == 25
    assert "telemetry_id" in df.columns
    assert "device_family" in df.columns
    assert "storage_tier" in df.columns

    summary = res["output_summary"]
    assert summary["status"] == "SIMULATED"
    assert summary["rows_retrieved"] == 25
    assert isinstance(summary["duration_ms"], int)


def test_azure_data_lake_live_mocked_csv():
    """Verify live ingestion reads CSV blob and formats output summary correctly."""
    r = AzureDataLakeConnectorRecipe()

    sample_csv = "id,name,tier\n101,Alice,premium\n102,Bob,standard\n".encode("utf-8")

    mock_downloader = MagicMock()
    mock_downloader.readall.return_value = sample_csv

    mock_blob_client = MagicMock()
    mock_blob_client.download_blob.return_value = mock_downloader

    mock_service = MagicMock()
    mock_service.get_blob_client.return_value = mock_blob_client

    with patch("azure.storage.blob.BlobServiceClient.from_connection_string", return_value=mock_service):
        res = r.execute({}, {
            "simulation_mode": False,
            "auth_mode": "connection_string",
            "connection_string": "DefaultEndpointsProtocol=https;AccountName=test;AccountKey=fake;EndpointSuffix=core.windows.net",
            "container_name": "test-lake",
            "blob_path": "data/users.csv",
            "file_format": "csv",
            "limit": 10
        })

    df = res["dataframe"]
    assert len(df) == 2
    assert list(df.columns) == ["id", "name", "tier"]
    assert df["name"].iloc[0] == "Alice"

    summary = res["output_summary"]
    assert summary["status"] == "LIVE"
    assert summary["rows_retrieved"] == 2
    assert summary["container"] == "test-lake"


def test_azure_data_lake_credential_context_resolution():
    """Verify credentials resolved from context dict work seamlessly."""
    r = AzureDataLakeConnectorRecipe()

    sample_csv = "k,v\n1,alpha\n2,beta\n".encode("utf-8")
    mock_downloader = MagicMock()
    mock_downloader.readall.return_value = sample_csv

    mock_blob_client = MagicMock()
    mock_blob_client.download_blob.return_value = mock_downloader

    mock_service = MagicMock()
    mock_service.get_blob_client.return_value = mock_blob_client

    context = {
        "credentials": {
            "cred_azure_prod": {
                "connection_string": "DefaultEndpointsProtocol=https;AccountName=prod;AccountKey=fakeKey;EndpointSuffix=core.windows.net"
            }
        }
    }

    with patch("azure.storage.blob.BlobServiceClient.from_connection_string", return_value=mock_service):
        res = r.execute({}, {
            "simulation_mode": False,
            "auth_mode": "credential",
            "credential_id": "cred_azure_prod",
            "container_name": "prod-data",
            "blob_path": "metrics.csv",
            "file_format": "csv"
        }, context=context)

    df = res["dataframe"]
    assert len(df) == 2
    assert res["output_summary"]["status"] == "LIVE"
