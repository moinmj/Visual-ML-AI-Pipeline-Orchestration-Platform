import pytest
import pandas as pd
import numpy as np
from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.preprocessing.column_selector import ColumnSelectorRecipe
from backend.app.recipes.preprocessing.data_type_converter import DataTypeConverterRecipe
from backend.app.recipes.preprocessing.outlier_handler import OutlierHandlerRecipe
from backend.app.recipes.preprocessing.feature_selector import FeatureSelectorRecipe


def test_recipes_registered_in_catalog():
    assert recipe_registry.get("column_selector") is not None
    assert recipe_registry.get("data_type_converter") is not None
    assert recipe_registry.get("outlier_handler") is not None
    assert recipe_registry.get("feature_selector") is not None

    for r_id in ["column_selector", "data_type_converter", "outlier_handler", "feature_selector"]:
        r = recipe_registry.get(r_id)
        schema = r.get_schema()
        assert schema["type"] == "object"
        assert "properties" in schema
        code = r.to_code({})
        assert isinstance(code, str) and len(code) > 0


def test_column_selector_keep_and_drop():
    df = pd.DataFrame({
        "col_a": [1, 2, 3],
        "col_b": [4, 5, 6],
        "col_c": [7, 8, 9],
        "target": [0, 1, 0]
    })
    recipe = ColumnSelectorRecipe()

    # Test 'keep' mode
    res_keep = recipe.execute({"dataframe": df}, {"mode": "keep", "columns": ["col_a", "target"]})
    df_keep = res_keep["dataframe"]
    assert list(df_keep.columns) == ["col_a", "target"]
    assert len(df_keep) == 3

    # Test 'drop' mode
    res_drop = recipe.execute({"dataframe": df}, {"mode": "drop", "columns": ["col_b"]})
    df_drop = res_drop["dataframe"]
    assert "col_b" not in df_drop.columns
    assert "col_a" in df_drop.columns and "col_c" in df_drop.columns and "target" in df_drop.columns


def test_data_type_converter():
    df = pd.DataFrame({
        "str_num": ["10.5", "20.2", "invalid", "40.0"],
        "str_int": ["1", "2", "3", "4"],
        "str_date": ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"],
        "str_cat": ["apple", "banana", "apple", "banana"],
        "str_bool": ["true", "false", "1", "0"]
    })
    recipe = DataTypeConverterRecipe()

    config = {
        "conversions": {
            "str_num": "numeric",
            "str_int": "integer",
            "str_date": "datetime",
            "str_cat": "categorical",
            "str_bool": "boolean"
        },
        "errors": "coerce"
    }

    res = recipe.execute({"dataframe": df}, config)
    df_out = res["dataframe"]

    assert pd.api.types.is_numeric_dtype(df_out["str_num"])
    assert np.isnan(df_out["str_num"].iloc[2])  # "invalid" coerced to NaN
    assert pd.api.types.is_integer_dtype(df_out["str_int"])
    assert pd.api.types.is_datetime64_any_dtype(df_out["str_date"])
    assert isinstance(df_out["str_cat"].dtype, pd.CategoricalDtype)
    assert pd.api.types.is_bool_dtype(df_out["str_bool"])
    assert df_out["str_bool"].tolist() == [True, False, True, False]


def test_data_type_converter_walmart_use_case():
    # Exactly replicates user request: Date DD-MM-YYYY, Store as categorical ID, Holiday_Flag as boolean
    df = pd.DataFrame({
        "Store": [1, 2, 3, 1],
        "Date": ["05-02-2010", "12-02-2010", "19-02-2010", "26-02-2010"],
        "Weekly_Sales": [24924.50, 46039.49, 41595.55, 19403.54],
        "Holiday_Flag": [0, 1, 0, 0],
        "Temperature": [42.31, 38.51, 39.93, 46.63],
        "Fuel_Price": [2.572, 2.548, 2.514, 2.561],
        "CPI": [211.096, 211.242, 211.289, 211.319],
        "Unemployment": [8.106, 8.106, 8.106, 8.106]
    })
    recipe = DataTypeConverterRecipe()

    config = {
        "conversions": {
            "Store": "categorical",
            "Date": "datetime",
            "Holiday_Flag": "boolean"
        },
        "datetime_format": "DD-MM-YYYY"
    }

    res = recipe.execute({"dataframe": df}, config)
    df_out = res["dataframe"]

    # 1. Date is parsed with dayfirst DD-MM-YYYY
    assert pd.api.types.is_datetime64_any_dtype(df_out["Date"])
    assert df_out["Date"].iloc[0].day == 5
    assert df_out["Date"].iloc[0].month == 2
    assert df_out["Date"].iloc[0].year == 2010

    # 2. Store is categorical (string-backed category, not ordinal int)
    assert isinstance(df_out["Store"].dtype, pd.CategoricalDtype)
    assert "1" in df_out["Store"].cat.categories

    # 3. Holiday_Flag is boolean
    assert pd.api.types.is_bool_dtype(df_out["Holiday_Flag"])
    assert df_out["Holiday_Flag"].tolist() == [False, True, False, False]

    # 4. Weekly_Sales, Temperature, etc. remained numeric untouched
    assert pd.api.types.is_numeric_dtype(df_out["Weekly_Sales"])
    assert pd.api.types.is_numeric_dtype(df_out["Temperature"])
    assert pd.api.types.is_numeric_dtype(df_out["Fuel_Price"])


def test_outlier_handler():
    # Construct series with extreme outliers
    normal_data = [10.0, 11.0, 10.5, 9.8, 10.2, 10.1, 10.4, 9.9, 10.0] * 3
    data_with_outliers = normal_data + [500.0, -500.0]  # extreme upper and lower
    df = pd.DataFrame({
        "feature_x": data_with_outliers,
        "feature_y": [1.0] * len(data_with_outliers)
    })
    recipe = OutlierHandlerRecipe()

    # 1. Test IQR with clip (Winsorization)
    res_clip = recipe.execute({"dataframe": df}, {"method": "iqr", "action": "clip", "threshold": 1.5})
    df_clip = res_clip["dataframe"]
    assert len(df_clip) == len(df)  # Clip keeps all rows!
    assert df_clip["feature_x"].max() < 100.0
    assert df_clip["feature_x"].min() > 0.0

    # 2. Test Z-Score with filter (Drop outlier rows)
    res_filter = recipe.execute({"dataframe": df}, {"method": "z_score", "action": "filter", "threshold": 2.0})
    df_filter = res_filter["dataframe"]
    assert len(df_filter) < len(df)
    assert 500.0 not in df_filter["feature_x"].values
    assert -500.0 not in df_filter["feature_x"].values

    # 3. Test Quantile with flag
    res_flag = recipe.execute({"dataframe": df}, {"method": "quantile", "action": "flag", "lower_quantile": 0.05, "upper_quantile": 0.95})
    df_flag = res_flag["dataframe"]
    assert "feature_x_is_outlier" in df_flag.columns
    assert df_flag["feature_x_is_outlier"].sum() >= 1


def test_feature_selector():
    np.random.seed(42)
    n = 100
    # Create informative feature x1 and x2, noisy features x3, x4, x5
    y = np.random.choice([0, 1], size=n)
    x1 = y * 5.0 + np.random.normal(0, 1, size=n)
    x2 = y * -3.0 + np.random.normal(0, 1, size=n)
    x3 = np.random.normal(0, 5, size=n)
    x4 = np.random.normal(0, 5, size=n)
    x5 = np.random.normal(0, 5, size=n)

    df = pd.DataFrame({
        "informative_1": x1,
        "informative_2": x2,
        "noise_1": x3,
        "noise_2": x4,
        "noise_3": x5,
        "target": y
    })
    recipe = FeatureSelectorRecipe()

    # 1. Test SelectKBest with k=2
    res_k = recipe.execute(
        {"dataframe": df},
        {"method": "select_k_best", "k": 2, "target_column": "target", "task_type": "classification"}
    )
    df_k = res_k["dataframe"]

    # Target column must be preserved!
    assert "target" in df_k.columns
    # Informative features should be ranked higher than random noise
    assert "informative_1" in df_k.columns
    assert len(df_k.columns) == 3  # 2 selected features + 1 target

    # 2. Test Mutual Info
    res_mi = recipe.execute(
        {"dataframe": df},
        {"method": "mutual_info", "k": 2, "target_column": "target"}
    )
    df_mi = res_mi["dataframe"]
    assert "target" in df_mi.columns
    assert len(df_mi.columns) == 3

    # 3. Test Variance Threshold
    df_constant = df.copy()
    df_constant["zero_var"] = 42.0  # constant column
    res_vt = recipe.execute(
        {"dataframe": df_constant},
        {"method": "variance_threshold", "target_column": "target"}
    )
    df_vt = res_vt["dataframe"]
    assert "zero_var" not in df_vt.columns
    assert "target" in df_vt.columns
