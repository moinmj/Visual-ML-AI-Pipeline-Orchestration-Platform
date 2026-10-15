"""
Test Suite for 6 New Data Preparation & Transformation Recipes:
1. Select (Rename / Reorder Columns) (select_rename_reorder)
2. Fuzzy-Match Deduplication (fuzzy_match_deduplication)
3. Semantic Deduplication (embedding similarity) (semantic_deduplication)
4. Data Profiling Report (data_profiling_report)
5. Schema / Contract Validation (schema_contract_validation)
6. Guardrails (validation checks) (data_guardrails)
"""
import pytest
import pandas as pd
import numpy as np

from backend.app.recipes.preprocessing.select_rename_reorder import SelectRenameReorderRecipe
from backend.app.recipes.preprocessing.fuzzy_deduplication import FuzzyMatchDeduplicationRecipe
from backend.app.recipes.preprocessing.semantic_deduplication import SemanticDeduplicationRecipe
from backend.app.recipes.preprocessing.data_profiling import DataProfilingReportRecipe
from backend.app.recipes.preprocessing.schema_validation import SchemaContractValidationRecipe
from backend.app.recipes.preprocessing.data_guardrails import DataGuardrailsRecipe


# =========================================================================
# 1. SELECT (RENAME / REORDER COLUMNS)
# =========================================================================

def test_select_rename_reorder_basic():
    recipe = SelectRenameReorderRecipe()
    df = pd.DataFrame({
        "customerID": ["C1", "C2"],
        "tenure": [10, 20],
        "MonthlyCharges": [50.0, 75.0]
    })

    config = {
        "rename_mapping": {"customerID": "id", "MonthlyCharges": "monthly_fee"},
        "column_order": ["monthly_fee", "id", "tenure"],
        "mode": "keep_all"
    }

    res = recipe.execute({"dataframe": df}, config)
    out_df = res["dataframe"]

    assert list(out_df.columns) == ["monthly_fee", "id", "tenure"]
    assert out_df["monthly_fee"].tolist() == [50.0, 75.0]
    assert res["metrics"]["renamed_columns_count"] == 2
    assert "code" in recipe.to_code(config).lower() or "rename" in recipe.to_code(config).lower()


def test_select_rename_reorder_strict_mode():
    recipe = SelectRenameReorderRecipe()
    df = pd.DataFrame({
        "col_a": [1, 2],
        "col_b": [3, 4],
        "col_c": [5, 6],
        "col_d": [7, 8]
    })

    config = {
        "rename_mapping": {"col_a": "alpha"},
        "column_order": ["col_c", "alpha"],
        "mode": "strict"
    }

    res = recipe.execute({"dataframe": df}, config)
    out_df = res["dataframe"]

    assert list(out_df.columns) == ["col_c", "alpha"]
    assert len(out_df) == 2


def test_select_rename_reorder_drop_columns():
    recipe = SelectRenameReorderRecipe()
    df = pd.DataFrame({
        "col_1": [1, 2],
        "col_2": [3, 4],
        "drop_me": [9, 9]
    })

    config = {
        "drop_columns": ["drop_me"]
    }

    res = recipe.execute({"dataframe": df}, config)
    out_df = res["dataframe"]

    assert "drop_me" not in out_df.columns
    assert list(out_df.columns) == ["col_1", "col_2"]


# =========================================================================
# 2. FUZZY-MATCH DEDUPLICATION
# =========================================================================

def test_fuzzy_match_deduplication_typos():
    recipe = FuzzyMatchDeduplicationRecipe()
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "company": [
            "Microsoft Corporation",
            "Microsoft Corp",
            "Apple Inc",
            "Google LLC"
        ],
        "city": ["Seattle", "Seattle", "Cupertino", "Mountain View"]
    })

    config = {
        "key_columns": ["company"],
        "similarity_threshold": 0.75,
        "action": "merge_keep_first"
    }

    res = recipe.execute({"dataframe": df}, config)
    out_df = res["dataframe"]

    assert len(out_df) == 3
    assert "Microsoft Corporation" in out_df["company"].values
    assert res["metrics"]["rows_removed"] == 1
    assert res["metrics"]["duplicate_clusters_count"] == 1


def test_fuzzy_match_deduplication_flag_only():
    recipe = FuzzyMatchDeduplicationRecipe()
    df = pd.DataFrame({
        "name": ["Jonathan Davis", "Jonatan Davis", "Sarah Connor"]
    })

    config = {
        "key_columns": ["name"],
        "similarity_threshold": 0.85,
        "action": "flag_only"
    }

    res = recipe.execute({"dataframe": df}, config)
    out_df = res["dataframe"]

    assert len(out_df) == 3
    assert "fuzzy_cluster_id" in out_df.columns
    assert "fuzzy_similarity_score" in out_df.columns
    assert out_df["fuzzy_cluster_id"].iloc[0] == out_df["fuzzy_cluster_id"].iloc[1]
    assert out_df["fuzzy_cluster_id"].iloc[2] == -1


# =========================================================================
# 3. SEMANTIC DEDUPLICATION (EMBEDDING SIMILARITY)
# =========================================================================

def test_semantic_deduplication_paraphrases():
    recipe = SemanticDeduplicationRecipe()
    df = pd.DataFrame({
        "id": [101, 102, 103],
        "feedback": [
            "The battery life is exceptional and charges rapidly.",
            "The battery life is truly exceptional and it charges rapidly.",
            "Customer service was terrible and unhelpful."
        ]
    })

    config = {
        "text_columns": ["feedback"],
        "similarity_threshold": 0.75,
        "action": "merge_keep_first"
    }

    res = recipe.execute({"dataframe": df}, config)
    out_df = res["dataframe"]

    assert len(out_df) == 2
    assert res["metrics"]["rows_removed"] == 1
    assert "Customer service was terrible" in out_df["feedback"].iloc[1]


def test_semantic_deduplication_flag_only():
    recipe = SemanticDeduplicationRecipe()
    df = pd.DataFrame({
        "review": [
            "Great camera quality and crisp photos.",
            "Great camera quality with crisp photos.",
            "Shipping was delayed by two weeks."
        ]
    })

    config = {
        "text_columns": ["review"],
        "similarity_threshold": 0.80,
        "action": "flag_only"
    }

    res = recipe.execute({"dataframe": df}, config)
    out_df = res["dataframe"]

    assert len(out_df) == 3
    assert "semantic_cluster_id" in out_df.columns
    assert out_df["semantic_cluster_id"].iloc[0] == out_df["semantic_cluster_id"].iloc[1]
    assert out_df["semantic_cluster_id"].iloc[2] == -1


# =========================================================================
# 4. DATA PROFILING REPORT
# =========================================================================

def test_data_profiling_report():
    recipe = DataProfilingReportRecipe()
    df = pd.DataFrame({
        "age": [25, 30, 35, np.nan, 45],
        "income": [50000.0, 60000.0, 75000.0, 80000.0, 95000.0],
        "gender": ["M", "F", "F", "M", "M"],
        "constant_col": ["X", "X", "X", "X", "X"]
    })

    config = {
        "output_mode": "passthrough_with_report",
        "null_threshold_pct": 15.0
    }

    res = recipe.execute({"dataframe": df}, config)
    assert len(res["dataframe"]) == 5
    assert "profile_table" in res

    profile_df = res["profile_table"]
    assert len(profile_df) == 4
    assert "quality_alerts" in profile_df.columns

    # check age null alert
    age_row = profile_df[profile_df["column"] == "age"].iloc[0]
    assert "High missing rate" in age_row["quality_alerts"]

    # check constant col alert
    const_row = profile_df[profile_df["column"] == "constant_col"].iloc[0]
    assert "Zero variance" in const_row["quality_alerts"]

    assert res["metrics"]["total_columns"] == 4
    assert res["metrics"]["dataset_health_score"] > 0


# =========================================================================
# 5. SCHEMA / CONTRACT VALIDATION
# =========================================================================

def test_schema_contract_validation_pass():
    recipe = SchemaContractValidationRecipe()
    df = pd.DataFrame({
        "user_id": [1, 2, 3],
        "score": [85.5, 92.0, 78.4],
        "status": ["active", "pending", "active"]
    })

    config = {
        "expected_schema": [
            {"name": "user_id", "type": "integer", "nullable": False},
            {"name": "score", "type": "float", "min_value": 0.0, "max_value": 100.0},
            {"name": "status", "type": "string", "allowed_values": ["active", "pending", "closed"]}
        ],
        "validation_mode": "flag_rows"
    }

    res = recipe.execute({"dataframe": df}, config)
    out_df = res["dataframe"]

    assert res["metrics"]["contract_passed"] is True
    assert res["metrics"]["compliance_pct"] == 100.0
    assert out_df["_contract_valid"].all()


def test_schema_contract_validation_violations():
    recipe = SchemaContractValidationRecipe()
    df = pd.DataFrame({
        "age": [25, -5, 120],  # -5 violates min_value
        "role": ["admin", "invalid_role", "user"]
    })

    config = {
        "expected_schema": [
            {"name": "age", "type": "integer", "min_value": 0, "max_value": 100},
            {"name": "role", "type": "string", "allowed_values": ["admin", "user"]}
        ],
        "validation_mode": "flag_rows"
    }

    res = recipe.execute({"dataframe": df}, config)
    out_df = res["dataframe"]

    assert res["metrics"]["contract_passed"] is False
    assert bool(out_df["_contract_valid"].iloc[0]) is True
    assert bool(out_df["_contract_valid"].iloc[1]) is False
    assert bool(out_df["_contract_valid"].iloc[2]) is False


def test_schema_contract_validation_strict_raise():
    recipe = SchemaContractValidationRecipe()
    df = pd.DataFrame({"col_x": [1, 2]})

    config = {
        "expected_schema": [{"name": "missing_col", "type": "string"}],
        "validation_mode": "strict_raise"
    }

    with pytest.raises(ValueError, match="failed Schema Contract"):
        recipe.execute({"dataframe": df}, config)


# =========================================================================
# 6. GUARDRAILS (VALIDATION CHECKS)
# =========================================================================

def test_data_guardrails_not_null_and_range():
    recipe = DataGuardrailsRecipe()
    df = pd.DataFrame({
        "price": [10.0, 25.0, -2.0, 50.0],
        "item": ["A", "B", None, "D"]
    })

    config = {
        "rules": [
            {"column": "price", "check_type": "greater_than", "min_value": 0.0},
            {"column": "item", "check_type": "not_null"}
        ],
        "on_failure": "flag_rows"
    }

    res = recipe.execute({"dataframe": df}, config)
    out_df = res["dataframe"]

    assert len(out_df) == 4
    assert "guardrail_status" in out_df.columns
    assert out_df["guardrail_status"].iloc[0] == "PASS"
    assert out_df["guardrail_status"].iloc[1] == "PASS"
    assert out_df["guardrail_status"].iloc[2] == "FAIL"  # price < 0 and item is null


def test_data_guardrails_quarantine():
    recipe = DataGuardrailsRecipe()
    df = pd.DataFrame({
        "age": [20, 30, 40, -5],
        "email": ["user1@test.com", "user2@test.com", "user3@test.com", "bad_email"]
    })

    config = {
        "rules": [
            {"column": "age", "check_type": "between", "min_value": 0, "max_value": 120},
            {"column": "email", "check_type": "regex_match", "regex_pattern": r"@"}
        ],
        "on_failure": "quarantine_invalid_rows"
    }

    res = recipe.execute({"dataframe": df}, config)
    out_df = res["dataframe"]

    assert len(out_df) == 3
    assert (out_df["age"] >= 0).all()
    assert res["metrics"]["failing_rows"] == 1


def test_data_guardrails_stop_and_error():
    recipe = DataGuardrailsRecipe()
    df = pd.DataFrame({"score": [-10, 50]})

    config = {
        "rules": [
            {"column": "score", "check_type": "greater_than", "min_value": 0.0, "severity": "error"}
        ],
        "on_failure": "stop_and_error"
    }

    with pytest.raises(ValueError, match="Guardrail Check Failure"):
        recipe.execute({"dataframe": df}, config)
