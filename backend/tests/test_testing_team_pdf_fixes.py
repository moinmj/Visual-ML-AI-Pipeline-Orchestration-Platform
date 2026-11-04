import pytest
import numpy as np
import pandas as pd

from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.preprocessing.dataset_join import DatasetJoinRecipe
from backend.app.recipes.preprocessing.duplicates import CategorySanitizerRecipe, CorrelationFilterRecipe, VarianceFilterRecipe
from backend.app.recipes.preprocessing.encoding import CategoricalEncoderRecipe
from backend.app.recipes.nlp.text_preprocessor import TextPreprocessorRecipe
from backend.app.recipes.nlp.text_vectorizer import TextVectorizerRecipe
from backend.app.recipes.preprocessing.data_guardrails import DataGuardrailsRecipe
from backend.app.recipes.preprocessing.formula_calculator import FormulaCalculatorRecipe
from backend.app.recipes.preprocessing.replace_value import ReplaceValueRecipe
from backend.app.recipes.preprocessing.column_selector import ColumnSelectorRecipe
from backend.app.recipes.preprocessing.custom_python_step import CustomPythonStepRecipe
from backend.app.recipes.preprocessing.fuzzy_deduplication import FuzzyMatchDeduplicationRecipe
from backend.app.recipes.preprocessing.semantic_deduplication import SemanticDeduplicationRecipe
from backend.app.recipes.preprocessing.data_profiling import DataProfilingReportRecipe
from backend.app.recipes.preprocessing.schema_validation import SchemaContractValidationRecipe


def test_registry_aliases_from_pdf():
    """Verify registry aliases for guardrails_validator and schema_contract_validator."""
    assert recipe_registry.has("guardrails_validator") is True
    assert recipe_registry.get("guardrails_validator").recipe_id == "data_guardrails"

    assert recipe_registry.has("schema_contract_validator") is True
    assert recipe_registry.get("schema_contract_validator").recipe_id == "schema_contract_validation"


def test_dataset_join_operators_and_indicator():
    """Verify non-equi operators (!=, >, <, >=, <=) and _merge indicator toggle enforcement."""
    df_left = pd.DataFrame({"id": [1, 2, 3], "val_left": [10, 20, 30]})
    df_right = pd.DataFrame({"id": [1, 2, 4], "val_right": [15, 25, 35]})

    recipe = DatasetJoinRecipe()

    # Test 1: Non-equi operator (> / greater_than)
    res_gt = recipe.execute(
        inputs={"left_dataframe": df_left, "right_dataframe": df_right},
        config={
            "join_type": "inner",
            "conditions": [{"left": "val_left", "operator": "greater_than", "right": "val_right"}]
        }
    )
    df_gt = res_gt["dataframe"]
    # 20 > 15, 30 > 15, 30 > 25
    assert len(df_gt) == 3
    assert (df_gt["val_left"] > df_gt["val_right"]).all()

    # Test 2: Indicator column enforcement
    res_ind = recipe.execute(
        inputs={"left_dataframe": df_left, "right_dataframe": df_right},
        config={
            "join_type": "left",
            "conditions": [{"left": "id", "operator": "=", "right": "id"}],
            "indicator": True
        }
    )
    df_ind = res_ind["dataframe"]
    assert "_merge" in df_ind.columns
    # Rows 1 and 2 match ("both"), row 3 has no match in right ("left_only")
    assert "both" in df_ind["_merge"].values
    assert "left_only" in df_ind["_merge"].values


def test_category_sanitizer_decimal_protection_and_hyphens():
    """Verify CategorySanitizer safeguards decimal points and handles hyphen space separation."""
    df = pd.DataFrame({
        "price_str": ["$29.85", "100.50", "9.99"],
        "category": ["multi-state", "high-risk", "tier-1"],
        "code": ["ABC-123", "XYZ-789", "FOO-BAR"]
    })

    recipe = CategorySanitizerRecipe()
    res = recipe.execute(
        inputs={"dataframe": df},
        config={
            "remove_punctuation": True,
            "lowercase": True,
            "replace_with_space": True
        }
    )
    df_out = res["dataframe"]
    summary = res["output_summary"]
    metrics = res["metrics"]

    # Decimal points must be preserved!
    assert "29.85" in df_out["price_str"].values[0]
    assert "100.50" in df_out["price_str"].values[1]
    assert "9.99" in df_out["price_str"].values[2]

    # Hyphens between words must be separated with space instead of merged
    assert "multi state" in df_out["category"].values[0]
    assert "high risk" in df_out["category"].values[1]

    # Summary and whole integer metrics
    assert summary["title"] == "Categorical Sanitizer Summary"
    assert isinstance(metrics["sanitized_count"], int)
    assert metrics["sanitized_count"] == 3


def test_categorical_encoder_target_encoding_discrete_strings():
    """Verify target encoding does NOT collapse discrete string classification targets to 0.0."""
    df = pd.DataFrame({
        "city": ["New York", "London", "New York", "Paris", "London", "New York"],
        "churn": ["yes", "no", "yes", "no", "no", "yes"]
    })

    recipe = CategoricalEncoderRecipe()
    res = recipe.execute(
        inputs={"dataframe": df},
        config={"method": "target", "columns": ["city"], "target_column": "churn"}
    )
    df_out = res["dataframe"]
    summary = res["output_summary"]

    # Rows must NOT collapse to 0.0! New York has 3/3 "yes", London has 0/2 "yes"
    ny_vals = df_out.loc[df["city"] == "New York", "city"].values
    assert (ny_vals > 0.5).all()
    assert summary["title"] == "Categorical Encoder Summary"

    # Validation: target_column should only be required for supervised methods
    errors_unsupervised = recipe.validate_config({"method": "one_hot"})
    assert len(errors_unsupervised) == 0


def test_text_preprocessor_numeric_safeguard_and_negations():
    """Verify text preprocessor protects numeric text columns during global cleaning and preserves negations."""
    df = pd.DataFrame({
        "feedback": ["This was not good and no help", "Great service, no issues at all", "Neutral experience with no delay"],
        "billing_total": ["$129.99", "$45.50", "$1,200.00"],
        "order_id": ["10023", "10024", "10025"]
    })

    recipe = TextPreprocessorRecipe()
    res = recipe.execute(
        inputs={"dataframe": df},
        config={
            "columns": [],  # Global cleaning mode
            "remove_stopwords": True,
            "preserve_negations": True
        }
    )
    df_out = res["dataframe"]
    summary = res["output_summary"]
    metrics = res["metrics"]

    # Billing total and order_id must NOT be wiped out!
    assert df_out["billing_total"].iloc[0] == "$129.99"
    assert df_out["order_id"].iloc[0] == "10023"
    assert "billing_total" in metrics["skipped_numeric_columns"]
    assert "order_id" in metrics["skipped_numeric_columns"]

    # Negation words ("not", "no") must be preserved in feedback
    feedback_text = df_out["feedback"].iloc[0]
    assert "not" in feedback_text.split() or "no" in feedback_text.split()

    assert summary["title"] == "Text Preprocessor Summary"
    assert isinstance(metrics["processed_columns_count"], int)


def test_text_vectorizer_word2vec_crash_fix():
    """Verify Word2Vec method executes without local variable crash and returns integer metrics."""
    df = pd.DataFrame({
        "text": [
            "Natural language processing with modern transformer embeddings",
            "Machine learning automated pipeline orchestration platform",
            "Fast predictive classification and numerical regression modeling"
        ]
    })

    recipe = TextVectorizerRecipe()
    res = recipe.execute(
        inputs={"dataframe": df},
        config={"method": "word2vec", "column": "text", "max_features": 10}
    )
    assert res["dataframe"] is not None
    assert "vectorizer" in res
    assert res["vectorizer"] is not None

    summary = res["output_summary"]
    metrics = res["metrics"]
    assert summary["title"] == "Text Vectorizer Summary"
    assert isinstance(metrics["vector_features_created"], int)
    assert isinstance(metrics["total_columns"], int)


def test_guardrails_validator_approved_set_values():
    """Verify approved set values check parses comma-separated, single word, and bracketed list correctly."""
    df = pd.DataFrame({
        "status": ["active", "pending", "closed", "unknown"],
        "country": ["USA", "CAN", "USA", "MEX"]
    })

    recipe = DataGuardrailsRecipe()

    # Comma-separated string format
    res_csv = recipe.execute(
        inputs={"dataframe": df},
        config={
            "rules": [
                {"column": "status", "check_type": "is_in_set", "set_values": "active, pending, closed"}
            ],
            "on_failure": "flag_rows"
        }
    )
    df_csv = res_csv["dataframe"]
    assert df_csv["guardrail_status"].iloc[0] == "PASS"  # active
    assert df_csv["guardrail_status"].iloc[1] == "PASS"  # pending
    assert df_csv["guardrail_status"].iloc[2] == "PASS"  # closed
    assert df_csv["guardrail_status"].iloc[3] == "FAIL"  # unknown

    # Bracketed list format
    res_bracket = recipe.execute(
        inputs={"dataframe": df},
        config={
            "rules": [
                {"column": "country", "check_type": "is_in_set", "set_values": "['USA', 'CAN']"}
            ],
            "on_failure": "flag_rows"
        }
    )
    df_bracket = res_bracket["dataframe"]
    assert df_bracket["guardrail_status"].iloc[0] == "PASS"  # USA
    assert df_bracket["guardrail_status"].iloc[1] == "PASS"  # CAN
    assert df_bracket["guardrail_status"].iloc[3] == "FAIL"  # MEX

    summary = res_csv["output_summary"]
    assert summary["title"] == "Guardrails Quality Summary"
    assert isinstance(summary["metrics"]["total_rows"], int)
    assert isinstance(summary["metrics"]["rules_passed"], int)


def test_formula_calculator_new_columns_batch_and_div_zero():
    """Verify formula calculator can create new columns, handles batch formula strings, and division by zero."""
    df = pd.DataFrame({
        "revenue": [100.0, 200.0, 300.0],
        "cost": [40.0, 80.0, 0.0]
    })

    recipe = FormulaCalculatorRecipe()

    # Test 1: Create a brand new column
    res_new = recipe.execute(
        inputs={"dataframe": df},
        config={"target_column": "profit", "formula": "[revenue] - [cost]"}
    )
    assert "profit" in res_new["dataframe"].columns
    assert list(res_new["dataframe"]["profit"]) == [60.0, 120.0, 300.0]

    # Test 2: Batch formulas passed as multiline string or string list
    res_batch = recipe.execute(
        inputs={"dataframe": df},
        config={
            "formulas": [
                "margin = [revenue] - [cost]",
                "double_rev = [revenue] * 2"
            ]
        }
    )
    assert "margin" in res_batch["dataframe"].columns
    assert "double_rev" in res_batch["dataframe"].columns
    assert res_batch["output_summary"]["title"] == "Formula Calculator Summary"

    # Test 3: Division by zero halts with ZeroDivisionError when error handling is stop_on_error or raise_error
    with pytest.raises(ZeroDivisionError):
        recipe.execute(
            inputs={"dataframe": df},
            config={
                "target_column": "ratio",
                "formula": "[revenue] / [cost]",
                "on_error": "stop_on_error"
            }
        )

    # Test 4: Division by zero outputs NaN when set_null
    res_null = recipe.execute(
        inputs={"dataframe": df},
        config={
            "target_column": "ratio",
            "formula": "[revenue] / [cost]",
            "on_error": "set_null"
        }
    )
    assert pd.isna(res_null["dataframe"]["ratio"].iloc[2])


def test_replace_value_suffix_and_summary():
    """Verify replace_value accepts suffix and returns proper summary."""
    df = pd.DataFrame({"status": ["pending", "approved", "pending"]})

    recipe = ReplaceValueRecipe()
    res = recipe.execute(
        inputs={"dataframe": df},
        config={
            "columns": ["status"],
            "find_value": "pending",
            "replace_with": "in_review",
            "output_mode": "new_column",
            "new_column_suffix": "_v2"
        }
    )
    assert "status_v2" in res["dataframe"].columns
    assert res["dataframe"]["status_v2"].iloc[0] == "in_review"
    assert res["output_summary"]["title"] == "Replace Value Summary"
    assert isinstance(res["metrics"]["total_replacements"], int)


def test_unsupervised_and_custom_titles():
    """Verify all report titles and whole integers across remaining recipes from PDF."""
    df = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0],
        "b": [1.0, 2.0, 3.0, 4.0],  # Duplicate correlation
        "c": [5.0, 5.0, 5.0, 5.0]   # Constant column
    })

    # Correlation filter
    res_corr = CorrelationFilterRecipe().execute(inputs={"dataframe": df}, config={"threshold": 0.99})
    assert res_corr["output_summary"]["title"] == "Feature Correlation Summary"
    assert res_corr["output_summary"]["badge"] == "UNSUPERVISED"
    assert isinstance(res_corr["metrics"]["columns_dropped_count"], int)

    # Variance filter
    res_var = VarianceFilterRecipe().execute(inputs={"dataframe": df}, config={"max_dominant_percentage": 0.95})
    assert res_var["output_summary"]["title"] == "Variance Summary"
    assert res_var["output_summary"]["badge"] == "UNSUPERVISED"
    assert isinstance(res_var["metrics"]["columns_dropped_count"], int)

    # Column selector
    res_sel = ColumnSelectorRecipe().execute(inputs={"dataframe": df}, config={"mode": "keep", "columns": ["a", "b"]})
    assert res_sel["output_summary"]["title"] == "Column Selection & Structure Summary"
    assert res_sel["output_summary"]["badge"] == "DATA PREPARATION"
    assert isinstance(res_sel["metrics"]["total_columns"], int)

    # Custom Python
    res_py = CustomPythonStepRecipe().execute(inputs={"dataframe": df}, config={"code": "result = df[['a']] * 2"})
    assert res_py["output_summary"]["title"] == "Custom Python Execution Summary"
    assert res_py["output_summary"]["badge"] == "DATA LOGIC"

    # Schema Validation
    res_sv = SchemaContractValidationRecipe().execute(
        inputs={"dataframe": df},
        config={"expected_schema": [{"name": "a", "type": "float"}]}
    )
    assert res_sv["output_summary"]["title"] == "Schema & Contract Validation Summary"
