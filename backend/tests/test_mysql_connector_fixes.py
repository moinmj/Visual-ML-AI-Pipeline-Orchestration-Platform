import pytest
from decimal import Decimal
from datetime import time as dt_time, timedelta
import pandas as pd

from backend.app.engine.execution.executor import make_json_safe, diagnose_execution_error
from backend.app.recipes.ingestion.mysql_connector import (
    MySQLConnectorRecipe,
    _clean_mysql_value,
    _clean_mysql_row,
    _sanitize_mysql_query
)


def test_mysql_bit_and_binary_serialization():
    """Verify BIT(1) values b'\\x01' and b'\\x00' become booleans True and False."""
    row = {
        "is_active": b"\x01",
        "is_deleted": b"\x00",
        "small_bit": b"\x00\x05",
        "name": b"Enterprise Gear",
        "amount": Decimal("1299.99"),
        "created_time": dt_time(14, 30, 0),
    }

    cleaned = _clean_mysql_row(row)
    assert cleaned["is_active"] is True
    assert cleaned["is_deleted"] is False
    assert cleaned["small_bit"] == 5
    assert cleaned["name"] == "Enterprise Gear"
    assert cleaned["amount"] == 1299.99
    assert cleaned["created_time"] == "14:30:00"

    # Also verify global make_json_safe handles bytes properly
    safe = make_json_safe(row)
    assert safe["is_active"] is True
    assert safe["is_deleted"] is False
    assert safe["small_bit"] == 5
    assert safe["name"] == "Enterprise Gear"
    assert safe["amount"] == 1299.99
    assert safe["created_time"] == "14:30:00"


def test_mysql_sanitize_query():
    """Verify _sanitize_mysql_query strips semicolons, handles prefixes, and avoids '; LIMIT X' syntax error 1064."""
    # 1. Simple query with trailing semicolon
    q1 = "SELECT * FROM orders;"
    assert _sanitize_mysql_query(q1, 100) == "SELECT * FROM orders LIMIT 100"

    # 2. Query with multiple trailing semicolons and whitespace
    q2 = "SELECT id, name FROM users ;;;   \n"
    assert _sanitize_mysql_query(q2, 50) == "SELECT id, name FROM users LIMIT 50"

    # 3. Query with existing LIMIT and semicolon
    q3 = "SELECT * FROM orders LIMIT 25;"
    assert _sanitize_mysql_query(q3, 100) == "SELECT * FROM orders LIMIT 25"

    # 4. Multi-statement USE prefix
    q4 = "USE shop_db; SELECT * FROM products;"
    assert _sanitize_mysql_query(q4, 10) == "SELECT * FROM products LIMIT 10"


def test_mysql_connector_inputs_empty():
    """Verify MySQL connector does not require incoming dataset (is an ingestion DAG root)."""
    recipe = MySQLConnectorRecipe()
    meta = recipe.to_metadata()
    assert meta.inputs == []
    assert recipe.get_inputs() == []


def test_mysql_select_all_tables_metrics_and_tables_count():
    """Verify select_all_tables provides tables_count in summary and metrics with clean integer formatting."""
    recipe = MySQLConnectorRecipe()

    config = {
        "read_mode": "table",
        "database": "analytics_db",
        "select_all_tables": True,
        "columns": ["id", "order_date"],
        "simulation_mode": True,
    }

    result = recipe.execute(inputs={}, config=config)
    assert result is not None
    assert "dataframe" in result

    summary = result.get("output_summary", {})
    metrics = result.get("metrics", {})

    # Check title and tables_count KPI card
    assert summary.get("title") == "Database Ingestion & Query Metrics"
    assert "tables_count" in summary
    assert "tables_count" in metrics
    assert isinstance(summary["tables_count"], int)
    assert summary["tables_count"] >= 3
    assert isinstance(summary["records_retrieved"], int)
    assert isinstance(summary["columns_count"], int)
    assert isinstance(summary["duration_ms"], int)


def test_mysql_error_diagnostics():
    """Verify MySQL syntax error 1064 and column error 1054 are classified accurately."""
    recipe = MySQLConnectorRecipe()

    # Syntax error 1064
    syntax_err = Exception("(1064, \"You have an error in your SQL syntax; check the manual near '; LIMIT 5'\")")
    diag_syntax = diagnose_execution_error(
        recipe=recipe,
        error=syntax_err,
        inputs={},
        config={"read_mode": "query"},
        context={}
    )
    assert diag_syntax["title"] == "SQL Syntax Error"
    assert "semicolons" in diag_syntax["suggestion"]

    # Unknown column 1054
    col_err = Exception("(1054, \"Unknown column 'discount_rate' in 'field list'\")")
    diag_col = diagnose_execution_error(
        recipe=recipe,
        error=col_err,
        inputs={},
        config={"table_name": "orders"},
        context={}
    )
    assert diag_col["title"] == "Database Column Not Found"


def test_mysql_live_execution_table_and_query_mock(monkeypatch):
    """Verify live table and query execution construct valid SQL, execute cleanly, and deserialize fields."""
    from unittest.mock import MagicMock
    recipe = MySQLConnectorRecipe()

    mock_cur = MagicMock()
    mock_cur.fetchall.return_value = [
        {"id": 1, "name": "MySQL Product", "is_active": b"\x01", "is_deleted": b"\x00"}
    ]
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    import pymysql
    monkeypatch.setattr(pymysql, "connect", lambda **kwargs: mock_conn)

    # 1. Live Table Mode execution
    config_table = {
        "read_mode": "table",
        "table_name": "products",
        "database": "store_db",
        "simulation_mode": False,
        "host": "localhost",
        "port": 3306,
        "username": "root",
        "password": "pwd"
    }

    result = recipe.execute(inputs={}, config=config_table)
    assert result is not None
    assert "dataframe" in result
    df = result["dataframe"]
    assert len(df) == 1
    assert bool(df["is_active"].iloc[0]) is True
    assert bool(df["is_deleted"].iloc[0]) is False

    # Verify query was constructed and executed
    executed_q = mock_cur.execute.call_args[0][0]
    assert "SELECT * FROM `products`" in executed_q

    # 2. Live Custom Query Mode with trailing semicolon
    config_query = {
        "read_mode": "query",
        "query": "SELECT id, name FROM products WHERE is_active = 1;\n",
        "limit": 50,
        "database": "store_db",
        "simulation_mode": False,
        "host": "localhost",
        "port": 3306,
        "username": "root",
        "password": "pwd"
    }

    result_q = recipe.execute(inputs={}, config=config_query)
    assert result_q is not None
    assert "dataframe" in result_q
    executed_sql = mock_cur.execute.call_args[0][0]
    # Semicolon stripped and LIMIT appended cleanly
    assert executed_sql == "SELECT id, name FROM products WHERE is_active = 1 LIMIT 50"


def test_mysql_inspect_databases_without_database_name(monkeypatch):
    """Verify action='databases' connects to MySQL with database=None when database is omitted."""
    from unittest.mock import MagicMock
    recipe = MySQLConnectorRecipe()

    captured_connect_kwargs = {}

    def mock_connect(**kwargs):
        captured_connect_kwargs.update(kwargs)
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {"version": "8.0.35"}
        mock_cursor.fetchall.return_value = [{"Database": "sales"}, {"Database": "inventory"}]
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        return mock_conn

    import sys
    mock_pymysql = MagicMock()
    mock_pymysql.connect = mock_connect
    mock_pymysql.cursors.DictCursor = "DictCursor"
    monkeypatch.setitem(sys.modules, "pymysql", mock_pymysql)
    monkeypatch.setitem(sys.modules, "pymysql.cursors", mock_pymysql.cursors)

    payload_config = {
        "connection_mode": "host_port",
        "host": "88.222.215.30",
        "port": 3306,
        "username": "Mysuer",
        "password": "#MyUser@123",
        "simulation_mode": False
    }

    res = recipe.inspect(action="databases", config=payload_config)
    assert res["success"] is True
    assert res["databases"] == ["sales", "inventory"]
    assert res["count"] == 2
    # Ensure database is None, so it doesn't fail with 'Unknown database shop_db'
    assert captured_connect_kwargs.get("database") is None
    assert captured_connect_kwargs.get("host") == "88.222.215.30"
    assert captured_connect_kwargs.get("port") == 3306
    assert captured_connect_kwargs.get("user") == "Mysuer"
    assert captured_connect_kwargs.get("password") == "#MyUser@123"


