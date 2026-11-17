import pytest
from decimal import Decimal
from datetime import time as dt_time, timedelta, datetime, date
import uuid
import pandas as pd

from backend.app.engine.execution.executor import make_json_safe, diagnose_execution_error
from backend.app.recipes.ingestion.postgresql_connector import (
    PostgreSQLConnectorRecipe,
    _clean_sql_value,
    _clean_sql_row,
    _sanitize_sql_query
)


def test_make_json_safe_decimal_and_time():
    """Verify Decimal objects become floats/ints and time objects become ISO strings, not <Object> placeholders."""
    raw_dict = {
        "price": Decimal("199.99"),
        "quantity": Decimal("50.00"),
        "zero_dec": Decimal("0.00"),
        "opening_time": dt_time(9, 30, 0),
        "closing_time": dt_time(18, 0, 0),
        "uuid_val": uuid.UUID("12345678-1234-5678-1234-567812345678"),
        "delta": timedelta(hours=2, minutes=15),
    }

    safe = make_json_safe(raw_dict)

    # Decimal assertions
    assert safe["price"] == 199.99
    assert isinstance(safe["price"], float)
    assert safe["quantity"] == 50
    assert isinstance(safe["quantity"], int)
    assert safe["zero_dec"] == 0
    assert isinstance(safe["zero_dec"], int)
    assert "<Decimal Object>" not in str(safe)

    # Time assertions
    assert safe["opening_time"] == "09:30:00"
    assert safe["closing_time"] == "18:00:00"
    assert "<time Object>" not in str(safe)

    # UUID and timedelta assertions
    assert safe["uuid_val"] == "12345678-1234-5678-1234-567812345678"
    assert "2:15:00" in safe["delta"]


def test_clean_sql_value_and_row():
    """Verify SQL cleaning functions safely convert database types."""
    row = {
        "amount": Decimal("4500.50"),
        "count": Decimal("12.00"),
        "punch_in": dt_time(8, 15, 30),
    }
    cleaned = _clean_sql_row(row)
    assert cleaned["amount"] == 4500.50
    assert isinstance(cleaned["amount"], float)
    assert cleaned["count"] == 12
    assert isinstance(cleaned["count"], int)
    assert cleaned["punch_in"] == "08:15:30"


def test_postgresql_connector_inputs_is_empty():
    """Verify PostgreSQL connector does NOT require an incoming dataset (is an ingestion source)."""
    recipe = PostgreSQLConnectorRecipe()
    meta = recipe.to_metadata()

    # Inputs list must be empty so frontend does not think it requires an upstream connection
    assert meta.inputs == []
    assert recipe.get_inputs() == []


def test_postgresql_select_all_tables_resets_column_chips():
    """Verify select_all_tables overrides column chips to '*' so cross-table column mismatches are avoided."""
    recipe = PostgreSQLConnectorRecipe()

    # Configure with specific columns (e.g. from table 1 chips) AND select_all_tables=True
    config = {
        "read_mode": "table",
        "schema_name": "public",
        "select_all_tables": True,
        "columns": ["id", "employee_name", "salary"],
        "simulation_mode": True,
    }

    result = recipe.execute(inputs={}, config=config)
    assert result is not None
    assert "dataframe" in result
    summary = result.get("output_summary", {})
    assert summary.get("title") == "Database Ingestion & Query Metrics"
    assert isinstance(summary.get("records_retrieved"), int)
    assert isinstance(summary.get("tables_count"), int)


def test_missing_column_error_classification():
    """Verify missing column errors are labeled as 'Database Column Not Found' rather than 'Database Table Not Found'."""
    recipe = PostgreSQLConnectorRecipe()

    # Simulate psycopg2 UndefinedColumn error: column "xyz" does not exist
    simulated_err = Exception('column "salary_bonus" does not exist')
    diag = diagnose_execution_error(
        recipe=recipe,
        error=simulated_err,
        inputs={},
        config={"table_name": "employees"},
        context={}
    )

    assert diag["title"] == "Database Column Not Found"
    assert "salary_bonus" in diag["message"]
    assert "select all columns (*)" in diag["suggestion"]

    # Verify table error is still classified correctly when table doesn't exist
    simulated_table_err = Exception('relation "nonexistent_table" does not exist')
    diag_table = diagnose_execution_error(
        recipe=recipe,
        error=simulated_table_err,
        inputs={},
        config={"table_name": "nonexistent_table"},
        context={}
    )
    assert diag_table["title"] == "Database Table Not Found"


def test_postgresql_query_sanitizer():
    """Verify _sanitize_sql_query strips semicolons and cleanly appends LIMIT."""
    assert _sanitize_sql_query("SELECT * FROM customers;", 50) == "SELECT * FROM customers LIMIT 50"
    assert _sanitize_sql_query("SELECT * FROM customers LIMIT 10;", 50) == "SELECT * FROM customers LIMIT 10"
    assert _sanitize_sql_query("SELECT id FROM users ;;; \n", 20) == "SELECT id FROM users LIMIT 20"


def test_postgresql_live_execution_defines_q_variable(monkeypatch):
    """Verify live table execution constructs variable 'q' and does not throw NameError: name 'q' is not defined."""
    from unittest.mock import MagicMock
    recipe = PostgreSQLConnectorRecipe()

    mock_cur = MagicMock()
    mock_cur.fetchall.return_value = [{"id": 1, "name": "Test Company"}]
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    import psycopg2
    monkeypatch.setattr(psycopg2, "connect", lambda **kwargs: mock_conn)

    # Force fallback to psycopg2 by making sqlalchemy raise
    import sqlalchemy
    monkeypatch.setattr(sqlalchemy, "create_engine", MagicMock(side_effect=Exception("force psycopg2")))

    config = {
        "read_mode": "table",
        "table_name": "companies",
        "schema_name": "public",
        "simulation_mode": False,
        "host": "localhost",
        "port": 5432,
        "database": "test_db",
        "username": "postgres",
        "password": "pwd"
    }

    result = recipe.execute(inputs={}, config=config)
    assert result is not None
    assert "dataframe" in result
    df = result["dataframe"]
    assert len(df) == 1
    # Verify mock_cur was called with a query containing 'SELECT * FROM "public"."companies"'
    executed_sql = mock_cur.execute.call_args[0][0]
    assert 'SELECT * FROM "public"."companies"' in executed_sql

