"""
Unit Test Suite for 7 Data Preparation & Transformation Recipes:
1. Formula Calculator (formula_calculator)
2. Replace Value (replace_value)
3. Regex Replace / Extract (regex_replace_extract)
4. Parse Date (parse_date)
5. Geocode / Reverse Geocode / Distance Calculator (geocode_distance_calculator)
6. Timezone Converter (timezone_converter)
7. Business-Day Calculator / Holiday Calendar (business_day_holiday_calculator)
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from backend.app.recipes.preprocessing.formula_calculator import FormulaCalculatorRecipe
from backend.app.recipes.preprocessing.replace_value import ReplaceValueRecipe
from backend.app.recipes.preprocessing.regex_replace_extract import RegexReplaceExtractRecipe
from backend.app.recipes.preprocessing.parse_date import ParseDateRecipe
from backend.app.recipes.preprocessing.geocode_distance import GeocodeDistanceCalculatorRecipe
from backend.app.recipes.preprocessing.timezone_converter import TimezoneConverterRecipe
from backend.app.recipes.preprocessing.business_day_holiday import BusinessDayHolidayCalculatorRecipe


# =========================================================================
# 1. FORMULA CALCULATOR TESTS
# =========================================================================

def test_formula_calculator_arithmetic_and_brackets():
    recipe = FormulaCalculatorRecipe()
    df = pd.DataFrame({
        "Revenue": [100.0, 250.0, 50.0],
        "Cost of Goods": [40.0, 150.0, 20.0]
    })
    config = {
        "target_column": "Net Profit",
        "formula": "[Revenue] - [Cost of Goods]"
    }
    res = recipe.execute({"dataframe": df}, config)
    out_df = res["dataframe"]
    assert "Net Profit" in out_df.columns
    assert out_df["Net Profit"].tolist() == [60.0, 100.0, 30.0]
    assert res["metrics"]["formulas_evaluated"] == 1


def test_formula_calculator_if_condition_and_functions():
    recipe = FormulaCalculatorRecipe()
    df = pd.DataFrame({
        "Score": [45, 80, 92, 30],
        "Raw Name": ["alice", "bob", "carol", "dave"]
    })
    config = {
        "formulas": [
            {"target_column": "Status", "formula": "IF([Score] >= 60, 'PASS', 'FAIL')"},
            {"target_column": "Upper Name", "formula": "UPPER([Raw Name])"}
        ]
    }
    res = recipe.execute({"dataframe": df}, config)
    out_df = res["dataframe"]
    assert out_df["Status"].tolist() == ["FAIL", "PASS", "PASS", "FAIL"]
    assert out_df["Upper Name"].tolist() == ["ALICE", "BOB", "CAROL", "DAVE"]
    assert res["metrics"]["formulas_evaluated"] == 2


def test_formula_calculator_error_handling():
    recipe = FormulaCalculatorRecipe()
    df = pd.DataFrame({"A": [10, 20]})
    # Column B does not exist
    config = {
        "target_column": "C",
        "formula": "[A] + [NonExistentCol]",
        "on_error": "raise_error"
    }
    with pytest.raises(KeyError):
        recipe.execute({"dataframe": df}, config)

    # set_null mode
    config["on_error"] = "set_null"
    res = recipe.execute({"dataframe": df}, config)
    assert res["dataframe"]["C"].isna().all()


# =========================================================================
# 2. REPLACE VALUE TESTS
# =========================================================================

def test_replace_value_exact_and_mapping():
    recipe = ReplaceValueRecipe()
    df = pd.DataFrame({
        "gender": ["M", "F", "M", "Unknown", None],
        "status": ["Y", "N", "Y", "Y", "N"]
    })
    config = {
        "columns": ["gender", "status"],
        "mapping": {"M": "Male", "F": "Female", "Y": "Active", "N": "Inactive"},
        "replace_nulls": True,
        "null_replacement_value": "Not Specified"
    }
    res = recipe.execute({"dataframe": df}, config)
    out_df = res["dataframe"]
    assert out_df["gender"].tolist() == ["Male", "Female", "Male", "Unknown", "Not Specified"]
    assert out_df["status"].tolist() == ["Active", "Inactive", "Active", "Active", "Inactive"]
    assert res["metrics"]["total_replacements"] >= 5


def test_replace_value_substring_contains():
    recipe = ReplaceValueRecipe()
    df = pd.DataFrame({
        "text": ["Error 404: Not Found", "Error 500: Server Crash", "OK 200: Success"]
    })
    config = {
        "columns": ["text"],
        "find_value": "Error",
        "replace_with": "Issue",
        "match_type": "contains"
    }
    res = recipe.execute({"dataframe": df}, config)
    out_df = res["dataframe"]
    assert out_df["text"].iloc[0] == "Issue 404: Not Found"
    assert out_df["text"].iloc[1] == "Issue 500: Server Crash"
    assert out_df["text"].iloc[2] == "OK 200: Success"


# =========================================================================
# 3. REGEX REPLACE / EXTRACT TESTS
# =========================================================================

def test_regex_replace_mode():
    recipe = RegexReplaceExtractRecipe()
    df = pd.DataFrame({
        "price_str": ["$1,250.00", "$45.99", "$999,999.50"]
    })
    config = {
        "source_column": "price_str",
        "mode": "replace",
        "regex_pattern": r"[\$,]",
        "replacement": "",
        "target_column": "price_clean"
    }
    res = recipe.execute({"dataframe": df}, config)
    out_df = res["dataframe"]
    assert "price_clean" in out_df.columns
    assert out_df["price_clean"].tolist() == ["1250.00", "45.99", "999999.50"]


def test_regex_extract_named_groups():
    recipe = RegexReplaceExtractRecipe()
    df = pd.DataFrame({
        "contact": ["John Doe <john@company.com>", "Jane Smith <jane@agency.org>"]
    })
    config = {
        "source_column": "contact",
        "mode": "extract",
        "regex_pattern": r"(?P<name>[^<]+)<\s*(?P<email>[^>]+)>"
    }
    res = recipe.execute({"dataframe": df}, config)
    out_df = res["dataframe"]
    assert "name" in out_df.columns
    assert "email" in out_df.columns
    assert out_df["name"].str.strip().tolist() == ["John Doe", "Jane Smith"]
    assert out_df["email"].tolist() == ["john@company.com", "jane@agency.org"]


def test_regex_extract_all():
    recipe = RegexReplaceExtractRecipe()
    df = pd.DataFrame({
        "post": ["Trending #AI and #DataScience today!", "Nothing here", "#Tech and #ML"]
    })
    config = {
        "source_column": "post",
        "mode": "extract_all",
        "regex_pattern": r"#\w+",
        "target_column": "hashtags",
        "delimiter": "; "
    }
    res = recipe.execute({"dataframe": df}, config)
    out_df = res["dataframe"]
    assert out_df["hashtags"].iloc[0] == "#AI; #DataScience"
    assert pd.isna(out_df["hashtags"].iloc[1]) or out_df["hashtags"].iloc[1] is None
    assert out_df["hashtags"].iloc[2] == "#Tech; #ML"


# =========================================================================
# 4. PARSE DATE TESTS
# =========================================================================

def test_parse_date_basic_and_parts_extraction():
    recipe = ParseDateRecipe()
    df = pd.DataFrame({
        "signup_date": ["2026-03-15", "2026-07-04", "2026-12-25"]
    })
    config = {
        "column": "signup_date",
        "date_format": "%Y-%m-%d",
        "extract_parts": ["year", "month", "day", "day_name", "is_weekend"]
    }
    res = recipe.execute({"dataframe": df}, config)
    out_df = res["dataframe"]
    assert pd.api.types.is_datetime64_any_dtype(out_df["signup_date"])
    assert out_df["signup_date_year"].tolist() == [2026, 2026, 2026]
    assert out_df["signup_date_month"].tolist() == [3, 7, 12]
    assert out_df["signup_date_day"].tolist() == [15, 4, 25]
    # March 15 2026 is a Sunday (is_weekend == True)
    assert bool(out_df["signup_date_is_weekend"].iloc[0]) is True
    # Dec 25 2026 is a Friday (is_weekend == False)
    assert bool(out_df["signup_date_is_weekend"].iloc[2]) is False


def test_parse_date_auto_and_coerce_errors():
    recipe = ParseDateRecipe()
    df = pd.DataFrame({
        "raw_date": ["10/25/2025", "invalid_date", "2026-01-01"]
    })
    config = {
        "column": "raw_date",
        "date_format": "auto",
        "errors": "coerce",
        "output_format": "date_only"
    }
    res = recipe.execute({"dataframe": df}, config)
    out_df = res["dataframe"]
    assert out_df["raw_date"].iloc[0] == "2025-10-25"
    assert pd.isna(out_df["raw_date"].iloc[1]) or out_df["raw_date"].iloc[1] is None
    assert out_df["raw_date"].iloc[2] == "2026-01-01"


# =========================================================================
# 5. GEOCODE / DISTANCE CALCULATOR TESTS
# =========================================================================

def test_geocode_distance_point_to_point():
    recipe = GeocodeDistanceCalculatorRecipe()
    # NYC: 40.7128, -74.0060  to  London: 51.5074, -0.1278 (~5570 km)
    df = pd.DataFrame({
        "orig_lat": [40.7128],
        "orig_lon": [-74.0060],
        "dest_lat": [51.5074],
        "dest_lon": [-0.1278]
    })
    config = {
        "mode": "distance",
        "distance_method": "haversine",
        "distance_unit": "km",
        "lat1_column": "orig_lat",
        "lon1_column": "orig_lon",
        "lat2_column": "dest_lat",
        "lon2_column": "dest_lon",
        "output_column": "dist_km"
    }
    res = recipe.execute({"dataframe": df}, config)
    out_df = res["dataframe"]
    assert "dist_km" in out_df.columns
    # Approx 5570 km between NYC and London
    assert 5500 <= out_df["dist_km"].iloc[0] <= 5650


def test_geocode_gazetteer_matching():
    recipe = GeocodeDistanceCalculatorRecipe()
    df = pd.DataFrame({
        "location": ["New York, USA", "Tokyo, Japan", "Unknown City XYZ"]
    })
    config = {
        "mode": "geocode",
        "address_column": "location"
    }
    res = recipe.execute({"dataframe": df}, config)
    out_df = res["dataframe"]
    assert "latitude" in out_df.columns
    assert "longitude" in out_df.columns
    assert 40.0 <= out_df["latitude"].iloc[0] <= 41.0
    assert 35.0 <= out_df["latitude"].iloc[1] <= 36.0
    assert pd.isna(out_df["latitude"].iloc[2])


# =========================================================================
# 6. TIMEZONE CONVERTER TESTS
# =========================================================================

def test_timezone_converter_utc_to_eastern():
    recipe = TimezoneConverterRecipe()
    df = pd.DataFrame({
        "utc_time": [
            pd.Timestamp("2026-06-15 12:00:00", tz="UTC"),  # Summer (EDT, UTC-4) -> 08:00
            pd.Timestamp("2026-12-15 12:00:00", tz="UTC")   # Winter (EST, UTC-5) -> 07:00
        ]
    })
    config = {
        "column": "utc_time",
        "target_timezone": "America/New_York",
        "output_format": "naive_local_wallclock"
    }
    res = recipe.execute({"dataframe": df}, config)
    out_df = res["dataframe"]
    # Verify EDT Summer conversion (12:00 UTC -> 08:00 EDT)
    assert out_df["utc_time"].iloc[0].hour == 8
    # Verify EST Winter conversion (12:00 UTC -> 07:00 EST)
    assert out_df["utc_time"].iloc[1].hour == 7


def test_timezone_converter_naive_timestamp_localization():
    recipe = TimezoneConverterRecipe()
    df = pd.DataFrame({
        "naive_time": ["2026-05-01 10:00:00", "2026-05-02 15:30:00"]
    })
    config = {
        "column": "naive_time",
        "source_timezone": "UTC",
        "target_timezone": "Asia/Tokyo",
        "output_format": "iso8601"
    }
    res = recipe.execute({"dataframe": df}, config)
    out_df = res["dataframe"]
    # Tokyo is UTC+9, so 10:00 UTC -> 19:00 JST (+0900)
    assert "+0900" in out_df["naive_time"].iloc[0] or "+09:00" in out_df["naive_time"].iloc[0]


# =========================================================================
# 7. BUSINESS-DAY CALCULATOR & HOLIDAYS TESTS
# =========================================================================

def test_business_days_between_excluding_holidays_and_weekends():
    recipe = BusinessDayHolidayCalculatorRecipe()
    # 2026-07-01 (Wednesday) to 2026-07-08 (Wednesday)
    # Days: Wed Jul 1, Thu Jul 2, Fri Jul 3 (US Holiday obs or Jul 4 Sat), Mon Jul 6, Tue Jul 7, Wed Jul 8
    df = pd.DataFrame({
        "order_date": ["2026-07-01"],
        "delivery_date": ["2026-07-08"]
    })
    config = {
        "mode": "business_days_between",
        "start_date_column": "order_date",
        "end_date_column": "delivery_date",
        "calendar": "US",
        "output_column": "work_days"
    }
    res = recipe.execute({"dataframe": df}, config)
    out_df = res["dataframe"]
    assert "work_days" in out_df.columns
    # 5 business days in between
    assert 4 <= out_df["work_days"].iloc[0] <= 5


def test_add_business_days_sla_deadline():
    recipe = BusinessDayHolidayCalculatorRecipe()
    # 2026-05-01 is a Friday. Adding 1 business day should skip Sat May 2 & Sun May 3 -> Mon May 4
    df = pd.DataFrame({
        "ticket_created": ["2026-05-01"]
    })
    config = {
        "mode": "add_business_days",
        "start_date_column": "ticket_created",
        "n_business_days": 1,
        "calendar": "US",
        "output_column": "sla_deadline"
    }
    res = recipe.execute({"dataframe": df}, config)
    out_df = res["dataframe"]
    deadline = pd.to_datetime(out_df["sla_deadline"].iloc[0])
    assert deadline.day == 4
    assert deadline.month == 5
    assert deadline.year == 2026


def test_flag_calendar_weekend_and_holiday():
    recipe = BusinessDayHolidayCalculatorRecipe()
    df = pd.DataFrame({
        "sample_date": [
            "2026-01-01",  # New Year's Day (Holiday)
            "2026-01-03",  # Saturday (Weekend)
            "2026-01-05"   # Monday (Normal Business Day)
        ]
    })
    config = {
        "mode": "flag_calendar",
        "start_date_column": "sample_date",
        "calendar": "US"
    }
    res = recipe.execute({"dataframe": df}, config)
    out_df = res["dataframe"]
    assert bool(out_df["sample_date_is_holiday"].iloc[0]) is True
    assert bool(out_df["sample_date_is_weekend"].iloc[1]) is True
    assert bool(out_df["sample_date_is_business_day"].iloc[2]) is True


# =========================================================================
# 8. END-TO-END WORKFLOW DAG PIPELINE TEST
# =========================================================================

def test_dag_pipeline_with_custom_logic_and_domain_cleaning():
    """Chains formula_calculator -> replace_value -> regex_replace_extract -> parse_date in a single DAG."""
    from backend.app.engine.dag.graph import WorkflowGraph, WorkflowNode, WorkflowEdge
    from backend.app.engine.execution.executor import DAGExecutor

    nodes = [
        WorkflowNode(
            id="node_formula",
            recipe_id="formula_calculator",
            config={
                "target_column": "subtotal",
                "formula": "[quantity] * [unit_price]"
            }
        ),
        WorkflowNode(
            id="node_replace",
            recipe_id="replace_value",
            config={
                "columns": ["status"],
                "mapping": {"P": "Pending", "C": "Completed"}
            }
        ),
        WorkflowNode(
            id="node_regex",
            recipe_id="regex_replace_extract",
            config={
                "source_column": "code",
                "mode": "extract",
                "regex_pattern": r"(?P<prefix>[A-Z]+)-(?P<num>\d+)"
            }
        ),
        WorkflowNode(
            id="node_parse_date",
            recipe_id="parse_date",
            config={
                "column": "tx_date",
                "date_format": "%Y-%m-%d",
                "extract_parts": ["year", "month"]
            }
        )
    ]

    edges = [
        WorkflowEdge(source="node_formula", target="node_replace"),
        WorkflowEdge(source="node_replace", target="node_regex"),
        WorkflowEdge(source="node_regex", target="node_parse_date")
    ]

    graph = WorkflowGraph(nodes=nodes, edges=edges)

    df = pd.DataFrame({
        "quantity": [2, 5, 10],
        "unit_price": [15.0, 20.0, 8.5],
        "status": ["P", "C", "P"],
        "code": ["TX-1001", "ORD-2002", "INV-3003"],
        "tx_date": ["2026-04-10", "2026-04-11", "2026-04-12"]
    })

    res = DAGExecutor.execute_workflow(
        execution_id="test_custom_domain_dag",
        workflow=graph,
        initial_df=df
    )

    assert res.status == "SUCCESS"
    assert len(res.node_results) == 4
    for nr in res.node_results:
        assert nr.status == "SUCCESS"
