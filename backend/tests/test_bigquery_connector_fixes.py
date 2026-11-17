"""
Tests verifying Google BigQuery Connector fixes:
- Root ingestion inputs (inputs == [])
- Table ref construction with cross-project dataset paths (colon & dotted syntax)
- Execution summary title updated to 'Data Warehouse Query Metrics'
- Count metrics formatted as clean integers without trailing decimals
- Resilient fallback from to_dataframe to native row mapping
- Real-time dynamic inspection (test_connection, datasets, tables, columns)
"""

import pytest
import decimal
import datetime
from unittest.mock import MagicMock, patch

from backend.app.recipes.ingestion.bigquery_connector import BigQueryConnectorRecipe


def test_bigquery_root_port_definitions():
    """Verify BigQuery is a root ingestion recipe with no input ports and dataframe output."""
    r = BigQueryConnectorRecipe()
    inputs = r.get_inputs()
    assert inputs == [], f"Expected no input ports for BigQuery root ingestion, got: {inputs}"

    outputs = r.get_outputs()
    assert len(outputs) == 1
    assert outputs[0].id == "output"
    assert outputs[0].type == "dataframe"
    assert r.icon == "google-bigquery"


def test_bigquery_build_table_ref_cross_project():
    """Verify _build_table_ref handles standard, dotted, and colon-delimited cross-project dataset paths."""
    r = BigQueryConnectorRecipe()

    # Standard 3-part
    assert r._build_table_ref("my-proj", "analytics", "users") == "`my-proj.analytics.users`"

    # Dotted cross-project dataset
    assert r._build_table_ref("my-proj", "bigquery-public-data.samples", "shakespeare") == "`bigquery-public-data.samples.shakespeare`"

    # Colon cross-project dataset (e.g. bigquery-public-data:samples)
    assert r._build_table_ref("my-proj", "bigquery-public-data:samples", "shakespeare") == "`bigquery-public-data.samples.shakespeare`"

    # Fully qualified table_id
    assert r._build_table_ref("my-proj", "", "bigquery-public-data.samples.shakespeare") == "`bigquery-public-data.samples.shakespeare`"


def test_bigquery_summary_title_and_integer_metrics():
    """Verify summary title is 'Data Warehouse Query Metrics' and all count metrics are integers."""
    r = BigQueryConnectorRecipe()
    res = r.execute({}, {
        "project_id": "analytics-prod-123",
        "dataset_id": "ga4_events",
        "table_id": "events_daily",
        "limit": 15,
        "simulation_mode": True
    })

    summary = res["output_summary"]
    metrics = res["metrics"]

    assert summary["title"] == "Data Warehouse Query Metrics"
    assert summary["status"] == "SIMULATED"
    assert isinstance(summary["rows_retrieved"], int)
    assert summary["rows_retrieved"] == 15
    assert isinstance(summary["columns_count"], int)
    assert isinstance(summary["duration_ms"], int)

    assert isinstance(metrics["rows_retrieved"], int)
    assert isinstance(metrics["columns_count"], int)
    assert isinstance(metrics["duration_ms"], int)


def test_bigquery_resilient_fallback_on_missing_db_dtypes():
    """Verify query execution falls back to native row mapping if to_dataframe fails (e.g. db-dtypes issue)."""
    r = BigQueryConnectorRecipe()

    mock_row_1 = MagicMock()
    mock_row_1.items.return_value = [("word", "king"), ("word_count", 100)]
    mock_row_2 = MagicMock()
    mock_row_2.items.return_value = [("word", "queen"), ("word_count", 85)]

    mock_query_job = MagicMock()
    # Simulate to_dataframe() failure due to missing db-dtypes or arrow issue
    mock_query_job.to_dataframe.side_effect = ValueError("Please install the 'db-dtypes' package to use this function")
    mock_query_job.result.return_value = [mock_row_1, mock_row_2]

    mock_client = MagicMock()
    mock_client.query.return_value = mock_query_job

    with patch("google.cloud.bigquery.Client", return_value=mock_client):
        res = r.execute({}, {
            "simulation_mode": False,
            "project_id": "demo-proj",
            "connection_mode": "adc",
            "read_mode": "table",
            "dataset_id": "bigquery-public-data:samples",
            "table_id": "shakespeare",
            "limit": 10
        })

    assert res["output_summary"]["status"] == "LIVE"
    assert res["output_summary"]["title"] == "Data Warehouse Query Metrics"
    assert res["metrics"]["rows_retrieved"] == 2
    df = res["dataframe"]
    assert len(df) == 2
    assert list(df["word"]) == ["king", "queen"]


def test_bigquery_inspect_datasets_includes_public():
    """Verify inspect action 'datasets' includes public datasets so empty projects never show 'No options found'."""
    r = BigQueryConnectorRecipe()

    # In simulation mode
    sim_res = r.inspect(action="datasets", config={"simulation_mode": True})
    assert sim_res["success"] is True
    assert "bigquery-public-data.samples" in sim_res["datasets"]

    # In live mode with mock client having zero local datasets
    mock_client = MagicMock()
    mock_client.list_datasets.return_value = []

    with patch("google.cloud.bigquery.Client", return_value=mock_client):
        live_res = r.inspect(action="datasets", config={"simulation_mode": False, "project_id": "empty-proj"})

    assert live_res["success"] is True
    assert "bigquery-public-data.samples" in live_res["datasets"]
    assert live_res["count"] >= 1


def test_bigquery_clean_records_serialization():
    """Verify _clean_records cleanly converts Decimals, datetimes, and bytes without loss."""
    raw = [{
        "amount": decimal.Decimal("42.50"),
        "count": decimal.Decimal("10"),
        "date": datetime.date(2026, 3, 1),
        "raw_bytes": b"sample_data"
    }]
    cleaned = BigQueryConnectorRecipe._clean_records(raw)
    assert cleaned[0]["amount"] == 42.5
    assert cleaned[0]["count"] == 10
    assert cleaned[0]["date"] == "2026-03-01"
    assert cleaned[0]["raw_bytes"] == "sample_data"
