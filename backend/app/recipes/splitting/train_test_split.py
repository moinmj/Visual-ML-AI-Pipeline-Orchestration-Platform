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

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        df: pd.DataFrame = inputs.get("dataframe")
        if df is None:
            raise ValueError("TrainTestSplit expects 'dataframe' in inputs.")

        target_col = config.get("target_column")
        if not target_col or target_col not in df.columns:
            # Fallback to last column if user didn't specify or name changed
            target_col = df.columns[-1]

        test_size = float(config.get("test_size", 0.2))
        random_state = int(config.get("random_state", 42))
        stratify_flag = config.get("stratify", False)

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # If y is categorical / string or continuous float in classification, cast or encode
        if not pd.api.types.is_numeric_dtype(y):
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y.astype(str)), index=y.index, name=target_col)
        elif pd.api.types.is_float_dtype(y) and y.nunique() <= 10:
            # Discrete float classes like 0.0, 1.0 -> cast to integer
            y = y.astype(int)

        strat = y if (stratify_flag and y.nunique() > 1 and y.value_counts().min() > 1) else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=strat
        )

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "feature_names": list(X.columns),
            "target_column": target_col
        }
