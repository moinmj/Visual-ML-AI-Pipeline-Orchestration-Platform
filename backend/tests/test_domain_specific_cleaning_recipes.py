"""
Unit Test Suite for 11 Domain-Specific Cleaning Recipes:
1. Currency / Unit Normalizer (currency_unit_normalizer)
2. Address Parser / Standardizer (address_parser_standardizer)
3. Phone Number Validator / Formatter (phone_number_validator)
4. Email / Domain Validator (email_domain_validator)
5. JSON / XML Flattener & Nester (json_xml_flattener_nester)
6. Hierarchical / Tree Flattener (hierarchical_tree_flattener)
7. Cross-Tab / Contingency Table Generator (cross_tab_generator)
8. Date Range Expander (date_range_expander)
9. Data Anonymization (Differential Privacy) (data_anonymization_differential_privacy)
10. Compliance Tokenization (PCI/PII masking) (compliance_tokenization)
11. Sampling Operator (random/stratified/systematic) (sampling_operator)
"""
import pytest
import pandas as pd
import numpy as np
import json

from backend.app.recipes.preprocessing.currency_unit_normalizer import CurrencyUnitNormalizerRecipe
from backend.app.recipes.preprocessing.address_parser_standardizer import AddressParserStandardizerRecipe
from backend.app.recipes.preprocessing.phone_number_validator import PhoneNumberValidatorRecipe
from backend.app.recipes.preprocessing.email_domain_validator import EmailDomainValidatorRecipe
from backend.app.recipes.preprocessing.json_xml_flattener import JsonXmlFlattenerNesterRecipe
from backend.app.recipes.preprocessing.hierarchical_tree_flattener import HierarchicalTreeFlattenerRecipe
from backend.app.recipes.preprocessing.cross_tab_generator import CrossTabGeneratorRecipe
from backend.app.recipes.preprocessing.date_range_expander import DateRangeExpanderRecipe
from backend.app.recipes.preprocessing.differential_privacy import DifferentialPrivacyAnonymizerRecipe
from backend.app.recipes.preprocessing.compliance_tokenization import ComplianceTokenizationRecipe
from backend.app.recipes.preprocessing.sampling_operator import SamplingOperatorRecipe


# =========================================================================
# 1. CURRENCY / UNIT NORMALIZER TESTS
# =========================================================================

def test_currency_normalizer_conversion():
    recipe = CurrencyUnitNormalizerRecipe()
    df = pd.DataFrame({
        "amount": [100.0, 100.0, 1000.0],
        "curr": ["USD", "EUR", "JPY"]
    })
    config = {
        "normalization_type": "currency",
        "amount_column": "amount",
        "source_unit_column": "curr",
        "target_unit": "USD",
        "output_column": "amount_usd"
    }
    res = recipe.execute({"dataframe": df}, config)
    out_df = res["dataframe"]
    assert "amount_usd" in out_df.columns
    # 100 USD -> 100 USD
    assert out_df["amount_usd"].iloc[0] == 100.0
    # 100 EUR (~1.085) -> 108.5 USD
    assert 105.0 <= out_df["amount_usd"].iloc[1] <= 112.0
    # 1000 JPY (~0.00665) -> ~6.65 USD
    assert 5.0 <= out_df["amount_usd"].iloc[2] <= 8.0


def test_unit_normalizer_physical_dimensions():
    recipe = CurrencyUnitNormalizerRecipe()
    df = pd.DataFrame({
        "weight": [10.0, 22.0462],  # 10 kg, 22.0462 lbs (~10 kg)
        "unit": ["kg", "lbs"]
    })
    config = {
        "normalization_type": "unit",
        "dimension": "weight",
        "amount_column": "weight",
        "source_unit_column": "unit",
        "target_unit": "kg",
        "output_column": "weight_kg"
    }
    res = recipe.execute({"dataframe": df}, config)
    out_df = res["dataframe"]
    assert "weight_kg" in out_df.columns
    assert round(out_df["weight_kg"].iloc[0], 1) == 10.0
    assert round(out_df["weight_kg"].iloc[1], 1) == 10.0


# =========================================================================
# 2. ADDRESS PARSER / STANDARDIZER TESTS
# =========================================================================

def test_address_parser_components_and_abbreviations():
    recipe = AddressParserStandardizerRecipe()
    df = pd.DataFrame({
        "full_address": [
            "350 5th Avenue, Suite 400, New York, NY 10118",
            "1600 Amphitheatre Parkway, Mountain View, California 94043"
        ]
    })
    config = {
        "address_column": "full_address",
        "column_prefix": "addr_",
        "generate_single_line_standardized": True
    }
    res = recipe.execute({"dataframe": df}, config)
    out_df = res["dataframe"]

    assert out_df["addr_street_number"].iloc[0] == "350"
    assert "Ave" in out_df["addr_street_name"].iloc[0]
    assert out_df["addr_state"].iloc[0] == "NY"
    assert out_df["addr_postal_code"].iloc[0] == "10118"

    assert out_df["addr_street_number"].iloc[1] == "1600"
    assert "Pkwy" in out_df["addr_street_name"].iloc[1]
    assert out_df["addr_state"].iloc[1] == "CA"
    assert out_df["addr_postal_code"].iloc[1] == "94043"


# =========================================================================
# 3. PHONE NUMBER VALIDATOR / FORMATTER TESTS
# =========================================================================

def test_phone_number_e164_formatting():
    recipe = PhoneNumberValidatorRecipe()
    df = pd.DataFrame({
        "raw_phone": [
            "(555) 234-5678",
            "+1-555-987-6543",
            "invalid_phone_text",
            "5551234567"
        ]
    })
    config = {
        "phone_column": "raw_phone",
        "default_country": "US",
        "target_format": "e164",
        "output_column": "clean_phone"
    }
    res = recipe.execute({"dataframe": df}, config)
    out_df = res["dataframe"]
    assert out_df["clean_phone"].iloc[0] == "+15552345678"
    assert out_df["clean_phone"].iloc[1] == "+15559876543"
    assert bool(out_df["raw_phone_is_valid"].iloc[2]) is False
    assert out_df["clean_phone"].iloc[3] == "+15551234567"


# =========================================================================
# 4. EMAIL / DOMAIN VALIDATOR TESTS
# =========================================================================

def test_email_domain_validator():
    recipe = EmailDomainValidatorRecipe()
    df = pd.DataFrame({
        "email": [
            "sarah.connor@acme-corp.com",
            "fake_user@mailinator.com",  # Disposable
            "admin@megacorp.org",        # Role-based
            "bad_email_at_nothing"       # Invalid
        ]
    })
    config = {
        "email_column": "email",
        "flag_disposable": True,
        "flag_role_based": True
    }
    res = recipe.execute({"dataframe": df}, config)
    out_df = res["dataframe"]
    assert bool(out_df["email_is_valid"].iloc[0]) is True
    assert bool(out_df["email_is_corporate"].iloc[0]) is True

    # Disposable flag
    assert bool(out_df["email_is_disposable"].iloc[1]) is True

    # Role based flag
    assert bool(out_df["email_is_role_based"].iloc[2]) is True

    # Invalid flag
    assert bool(out_df["email_is_valid"].iloc[3]) is False


# =========================================================================
# 5. JSON / XML FLATTENER & NESTER TESTS
# =========================================================================

def test_json_flattener_and_nester():
    flatten_recipe = JsonXmlFlattenerNesterRecipe()
    raw_json_str = json.dumps({
        "user": {"name": "Alice", "age": 30},
        "location": {"city": "Seattle", "zip": "98101"}
    })
    df = pd.DataFrame({"payload": [raw_json_str]})

    # 1. Flatten
    flat_config = {
        "mode": "flatten",
        "source_column": "payload",
        "separator": "."
    }
    flat_res = flatten_recipe.execute({"dataframe": df}, flat_config)
    flat_df = flat_res["dataframe"]
    assert "user.name" in flat_df.columns
    assert "location.city" in flat_df.columns
    assert flat_df["user.name"].iloc[0] == "Alice"
    assert flat_df["location.city"].iloc[0] == "Seattle"

    # 2. Nest back
    nest_config = {
        "mode": "nest",
        "columns_to_nest": ["user.name", "user.age", "location.city", "location.zip"],
        "target_column": "bundled_json"
    }
    nest_res = flatten_recipe.execute({"dataframe": flat_df}, nest_config)
    nest_df = nest_res["dataframe"]
    assert "bundled_json" in nest_df.columns
    rebuilt = json.loads(nest_df["bundled_json"].iloc[0])
    assert rebuilt["user"]["name"] == "Alice"
    assert rebuilt["location"]["city"] == "Seattle"


# =========================================================================
# 6. HIERARCHICAL / TREE FLATTENER TESTS
# =========================================================================

def test_hierarchical_tree_flattener():
    recipe = HierarchicalTreeFlattenerRecipe()
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "parent_id": [None, 1, 2],
        "name": ["CEO", "VP Engineering", "Staff Engineer"]
    })
    config = {
        "id_column": "id",
        "parent_id_column": "parent_id",
        "label_column": "name",
        "path_separator": " > "
    }
    res = recipe.execute({"dataframe": df}, config)
    out_df = res["dataframe"]
    assert out_df["tree_depth"].tolist() == [0, 1, 2]
    assert out_df["tree_path"].iloc[2] == "CEO > VP Engineering > Staff Engineer"
    assert out_df["level_0"].tolist() == ["CEO", "CEO", "CEO"]
    assert out_df["level_1"].iloc[1] == "VP Engineering"
    assert out_df["level_2"].iloc[2] == "Staff Engineer"
    assert bool(out_df["is_leaf"].iloc[2]) is True


# =========================================================================
# 7. CROSS-TAB / CONTINGENCY TABLE TESTS
# =========================================================================

def test_cross_tab_generator():
    recipe = CrossTabGeneratorRecipe()
    df = pd.DataFrame({
        "department": ["Sales", "Sales", "Eng", "Eng", "Eng"],
        "seniority": ["Junior", "Senior", "Junior", "Senior", "Senior"]
    })
    config = {
        "index_column": "department",
        "column_column": "seniority",
        "aggfunc": "count",
        "margins": True
    }
    res = recipe.execute({"dataframe": df}, config)
    out_df = res["dataframe"]
    assert "department" in out_df.columns
    assert "Junior" in out_df.columns
    assert "Senior" in out_df.columns
    assert "Total" in out_df.columns
    assert len(out_df) == 3  # Eng, Sales, Total


# =========================================================================
# 8. DATE RANGE EXPANDER TESTS
# =========================================================================

def test_date_range_expander_with_proration():
    recipe = DateRangeExpanderRecipe()
    df = pd.DataFrame({
        "booking_id": ["B101"],
        "checkin": ["2026-06-01"],
        "checkout": ["2026-06-03"],  # 3 days: June 1, 2, 3
        "total_revenue": [300.0]
    })
    config = {
        "start_date_column": "checkin",
        "end_date_column": "checkout",
        "frequency": "day",
        "prorate_columns": ["total_revenue"],
        "date_column_name": "stay_date"
    }
    res = recipe.execute({"dataframe": df}, config)
    out_df = res["dataframe"]
    assert len(out_df) == 3
    # 300 / 3 = 100 per day
    assert (out_df["total_revenue"] == 100.0).all()
    assert out_df["interval_index"].tolist() == [0, 1, 2]


# =========================================================================
# 9. DIFFERENTIAL PRIVACY ANONYMIZER TESTS
# =========================================================================

def test_differential_privacy_anonymizer():
    recipe = DifferentialPrivacyAnonymizerRecipe()
    df = pd.DataFrame({
        "salary": [50000.0, 75000.0, 120000.0, 95000.0],
        "credit_score": [650, 720, 800, 690]
    })
    config = {
        "columns": ["salary", "credit_score"],
        "epsilon": 1.0,
        "mechanism": "laplace",
        "clip_negative": True,
        "output_mode": "new_column",
        "new_column_suffix": "_noisy"
    }
    res = recipe.execute({"dataframe": df}, config)
    out_df = res["dataframe"]
    assert "salary_noisy" in out_df.columns
    assert "credit_score_noisy" in out_df.columns
    assert (out_df["salary_noisy"] >= 0).all()
    # Ensure noise was actually added (not identical)
    assert not np.allclose(out_df["salary"].values, out_df["salary_noisy"].values)


# =========================================================================
# 10. COMPLIANCE TOKENIZATION TESTS
# =========================================================================

def test_compliance_tokenization_credit_card_and_hmac():
    recipe = ComplianceTokenizationRecipe()
    df = pd.DataFrame({
        "card_num": ["4111-2222-3333-4444", "5500 0000 0000 9876"],
        "ssn": ["123-45-6789", "987-65-4321"],
        "customer_id": ["CUST_ALPHA", "CUST_BETA"]
    })
    config = {
        "columns": ["card_num"],
        "strategy": "pci_credit_card"
    }
    res = recipe.execute({"dataframe": df}, config)
    out_df = res["dataframe"]
    assert out_df["card_num"].iloc[0] == "****-****-****-4444"
    assert out_df["card_num"].iloc[1] == "****-****-****-9876"

    # Test HMAC SHA-256
    config_hmac = {
        "columns": ["customer_id"],
        "strategy": "hmac_sha256",
        "salt": "test_salt_key"
    }
    res_hmac = recipe.execute({"dataframe": df}, config_hmac)
    assert res_hmac["dataframe"]["customer_id"].iloc[0].startswith("tok_")


# =========================================================================
# 11. SAMPLING OPERATOR TESTS
# =========================================================================

def test_sampling_operator_stratified_and_random():
    recipe = SamplingOperatorRecipe()
    df = pd.DataFrame({
        "id": range(100),
        "category": ["A"] * 80 + ["B"] * 20
    })
    # Stratified 25% sample
    config = {
        "sampling_method": "stratified",
        "sample_mode": "fraction",
        "sample_fraction": 0.25,
        "strata_column": "category",
        "random_state": 42
    }
    res = recipe.execute({"dataframe": df}, config)
    out_df = res["dataframe"]
    assert len(out_df) == 25
    # Category A: 80 * 0.25 = 20, Category B: 20 * 0.25 = 5
    cat_counts = out_df["category"].value_counts().to_dict()
    assert cat_counts["A"] == 20
    assert cat_counts["B"] == 5


# =========================================================================
# 12. END-TO-END DAG PIPELINE TEST
# =========================================================================

def test_dag_pipeline_domain_specific_cleaning():
    """Chains currency normalizer -> email validator -> compliance tokenization -> sampling in a single DAG."""
    from backend.app.engine.dag.graph import WorkflowGraph, WorkflowNode, WorkflowEdge
    from backend.app.engine.execution.executor import DAGExecutor

    nodes = [
        WorkflowNode(
            id="node_curr",
            recipe_id="currency_unit_normalizer",
            config={
                "normalization_type": "currency",
                "amount_column": "raw_amount",
                "source_unit_column": "currency",
                "target_unit": "USD",
                "output_column": "usd_amount"
            }
        ),
        WorkflowNode(
            id="node_email",
            recipe_id="email_domain_validator",
            config={
                "email_column": "contact_email",
                "flag_disposable": True
            }
        ),
        WorkflowNode(
            id="node_token",
            recipe_id="compliance_tokenization",
            config={
                "columns": ["credit_card"],
                "strategy": "pci_credit_card"
            }
        ),
        WorkflowNode(
            id="node_sample",
            recipe_id="sampling_operator",
            config={
                "sampling_method": "random",
                "sample_mode": "count",
                "sample_count": 2,
                "random_state": 42
            }
        )
    ]

    edges = [
        WorkflowEdge(source="node_curr", target="node_email"),
        WorkflowEdge(source="node_email", target="node_token"),
        WorkflowEdge(source="node_token", target="node_sample")
    ]

    graph = WorkflowGraph(nodes=nodes, edges=edges)

    df = pd.DataFrame({
        "raw_amount": [100.0, 200.0, 500.0, 1000.0],
        "currency": ["USD", "EUR", "GBP", "USD"],
        "contact_email": ["user1@company.com", "user2@test.org", "user3@agency.io", "user4@dev.net"],
        "credit_card": ["4111-2222-3333-1111", "5500-0000-0000-2222", "3782-0000-0000-3333", "6011-0000-0000-4444"]
    })

    res = DAGExecutor.execute_workflow(
        execution_id="test_domain_cleaning_dag",
        workflow=graph,
        initial_df=df
    )

    assert res.status == "SUCCESS"
    assert len(res.node_results) == 4
    for nr in res.node_results:
        assert nr.status == "SUCCESS"
