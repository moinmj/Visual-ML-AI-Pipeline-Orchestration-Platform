import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from backend.app.recipes.base.recipe import BaseRecipe


class TrainTestSplitRecipe(BaseRecipe):
    recipe_id = "train_test_split"
    name = "Train / Test Splitter"
    version = "1.0.0"
    category = "splitting"
    description = "Splits a dataset into training and testing subsets based on target column and split ratio."
    input_types = ["dataframe"]
    output_types = ["train_data", "test_data"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "target_column": {
                    "type": "string",
                    "title": "Target Variable (Y)",
                    "description": "The column to predict."
                },
                "test_size": {
                    "type": "number",
                    "title": "Test Split Ratio",
                    "default": 0.2,
                    "minimum": 0.05,
                    "maximum": 0.5
                },
                "random_state": {
                    "type": "integer",
                    "title": "Random Seed",
                    "default": 42
                },
                "stratify": {
                    "type": "boolean",
                    "title": "Stratified Split",
                    "default": False,
                    "description": "Preserve target class distribution in splits."
                }
            },
            "required": ["target_column"]
        }

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors = []
        target_col = config.get("target_column")
        if not target_col or not str(target_col).strip() or str(target_col).strip() in ["-- Select Column --", "(None)"]:
            errors.append("Target variable 'target_column' (Y) is required for Train / Test Split and cannot be empty.")
        return errors

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        df: pd.DataFrame = inputs.get("dataframe")
        if df is None:
            raise ValueError("TrainTestSplit expects 'dataframe' in inputs.")

        target_col = config.get("target_column")
        if not target_col or not str(target_col).strip() or str(target_col).strip() in ["-- Select Column --", "(None)"]:
            raise ValueError(
                "Target variable 'target_column' (Y) is required for Train / Test Split, but was left empty. "
                "Please configure a valid target column to predict."
            )
        target_col = str(target_col).strip()
        if target_col not in df.columns:
            matching = [c for c in df.columns if c.lower() == target_col.lower()]
            if matching:
                target_col = matching[0]
            else:
                raise ValueError(
                    f"Specified target column '{target_col}' was not found in dataset columns: {list(df.columns)}. "
                    "Please select an existing column."
                )

        test_size = float(config.get("test_size", 0.2))
        random_state = int(config.get("random_state", 42))
        stratify_flag = config.get("stratify", False)

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # If y is categorical / string or continuous float in classification, cast or encode
        target_classes = None
        target_encoder = None
        if not pd.api.types.is_numeric_dtype(y):
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y.astype(str)), index=y.index, name=target_col)
            target_classes = [str(c) for c in le.classes_]
            target_encoder = le
        elif pd.api.types.is_float_dtype(y) and y.nunique() <= 10:
            # Discrete float classes like 0.0, 1.0 -> cast to integer
            y = y.astype(int)

        strat = y if (stratify_flag and y.nunique() > 1 and y.value_counts().min() > 1) else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=strat
        )

        res = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "feature_names": list(X.columns),
            "target_column": target_col
        }
        if target_classes:
            res["target_classes"] = target_classes
            res["target_encoder"] = target_encoder
        return res
