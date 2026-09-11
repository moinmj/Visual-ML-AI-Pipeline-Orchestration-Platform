import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.neighbors import NearestNeighbors
from backend.app.recipes.base.recipe import BaseRecipe


class ClassImbalanceResamplerRecipe(BaseRecipe):
    recipe_id = "class_imbalance_resampler"
    name = "Class Imbalance Resampler (SMOTE)"
    version = "1.0.0"
    category = "preprocessing"
    description = (
        "Balances skewed class distributions in classification datasets using SMOTE "
        "(Synthetic Minority Over-sampling Technique), Random Over-Sampling, or Random Under-Sampling."
    )
    input_types = ["dataframe", "train_data"]
    output_types = ["dataframe", "train_data"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "strategy": {
                    "type": "string",
                    "title": "Resampling Strategy",
                    "enum": ["smote", "random_oversample", "random_undersample"],
                    "default": "smote",
                    "description": (
                        "SMOTE generates synthetic minority examples using k-nearest neighbors. "
                        "Random Over-Sample duplicates minority samples. "
                        "Random Under-Sample downsamples majority classes."
                    )
                },
                "target_column": {
                    "type": "string",
                    "title": "Target Variable (Y)",
                    "description": "The class label column (required if placed before Train/Test Split)."
                },
                "sampling_ratio": {
                    "type": "number",
                    "title": "Sampling Ratio (Minority / Majority)",
                    "default": 1.0,
                    "minimum": 0.1,
                    "maximum": 1.0,
                    "description": "Desired ratio of minority class to majority class after resampling (1.0 = balanced 50/50)."
                },
                "k_neighbors": {
                    "type": "integer",
                    "title": "K-Nearest Neighbors (SMOTE only)",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 20,
                    "description": "Number of nearest neighbors used to construct synthetic samples."
                },
                "random_state": {
                    "type": "integer",
                    "title": "Random Seed",
                    "default": 42
                }
            },
            "required": ["strategy"]
        }

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors = []
        strategy = config.get("strategy", "smote")
        if strategy not in ["smote", "random_oversample", "random_undersample"]:
            errors.append(f"Invalid resampling strategy '{strategy}'. Must be 'smote', 'random_oversample', or 'random_undersample'.")
        return errors

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        strategy = config.get("strategy", "smote")
        sampling_ratio = float(config.get("sampling_ratio", 1.0))
        k_neighbors = int(config.get("k_neighbors", 5))
        random_state = int(config.get("random_state", 42))

        # Check execution mode: Post-Split (X_train, y_train) vs Pre-Split (dataframe)
        has_split_data = "X_train" in inputs and "y_train" in inputs
        has_dataframe = "dataframe" in inputs and inputs["dataframe"] is not None

        if not has_split_data and not has_dataframe:
            raise ValueError(
                "ClassImbalanceResampler expects either partitioned training data ('X_train' and 'y_train') "
                "or a 'dataframe' in its inputs."
            )

        if has_split_data:
            X = inputs["X_train"]
            y = inputs["y_train"]
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)
            if not isinstance(y, pd.Series):
                y = pd.Series(y, name="target")

            X_resampled, y_resampled, orig_dist, resamp_dist = self._resample(
                X=X,
                y=y,
                strategy=strategy,
                sampling_ratio=sampling_ratio,
                k_neighbors=k_neighbors,
                random_state=random_state
            )

            result = dict(inputs)
            result["X_train"] = X_resampled
            result["y_train"] = y_resampled
            result["output_summary"] = {
                "strategy": strategy,
                "original_samples": len(y),
                "resampled_samples": len(y_resampled),
                "samples_delta": len(y_resampled) - len(y),
                "original_distribution": orig_dist,
                "resampled_distribution": resamp_dist
            }
            return result

        else:
            df: pd.DataFrame = inputs["dataframe"].copy()
            target_col = config.get("target_column") or inputs.get("target_column")
            if not target_col or target_col not in df.columns:
                # Try case-insensitive matching
                matching = [c for c in df.columns if target_col and c.lower() == str(target_col).lower()]
                if matching:
                    target_col = matching[0]
                else:
                    raise ValueError(
                        f"Target column '{target_col}' not found in dataset columns: {list(df.columns)}. "
                        "Please configure 'target_column' in the Class Imbalance Resampler settings."
                    )

            X = df.drop(columns=[target_col])
            y = df[target_col]

            X_resampled, y_resampled, orig_dist, resamp_dist = self._resample(
                X=X,
                y=y,
                strategy=strategy,
                sampling_ratio=sampling_ratio,
                k_neighbors=k_neighbors,
                random_state=random_state
            )

            df_resampled = pd.concat([X_resampled.reset_index(drop=True), y_resampled.reset_index(drop=True)], axis=1)
            result = dict(inputs)
            result["dataframe"] = df_resampled
            result["target_column"] = target_col
            result["output_summary"] = {
                "strategy": strategy,
                "original_samples": len(df),
                "resampled_samples": len(df_resampled),
                "samples_delta": len(df_resampled) - len(df),
                "original_distribution": orig_dist,
                "resampled_distribution": resamp_dist
            }
            return result

    def _resample(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        strategy: str,
        sampling_ratio: float,
        k_neighbors: int,
        random_state: int
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, int], Dict[str, int]]:
        orig_dist = {str(k): int(v) for k, v in y.value_counts().items()}

        if y.nunique() <= 1:
            # Cannot balance a single class
            return X, y, orig_dist, orig_dist

        if strategy == "smote":
            X_res, y_res = self._apply_smote(X, y, sampling_ratio, k_neighbors, random_state)
        elif strategy == "random_undersample":
            X_res, y_res = self._apply_random_undersample(X, y, sampling_ratio, random_state)
        else:
            X_res, y_res = self._apply_random_oversample(X, y, sampling_ratio, random_state)

        resamp_dist = {str(k): int(v) for k, v in y_res.value_counts().items()}
        return X_res, y_res, orig_dist, resamp_dist

    def _apply_random_oversample(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sampling_ratio: float,
        random_state: int
    ) -> Tuple[pd.DataFrame, pd.Series]:
        counts = y.value_counts()
        majority_count = counts.max()
        target_count = max(int(majority_count * sampling_ratio), counts.min())

        rng = np.random.RandomState(random_state)
        X_parts = []
        y_parts = []

        for cls, count in counts.items():
            X_cls = X[y == cls]
            y_cls = y[y == cls]

            if count < target_count:
                num_to_sample = target_count - count
                sample_indices = rng.choice(X_cls.index, size=num_to_sample, replace=True)
                X_oversampled = pd.concat([X_cls, X.loc[sample_indices]], axis=0)
                y_oversampled = pd.concat([y_cls, y.loc[sample_indices]], axis=0)
                X_parts.append(X_oversampled)
                y_parts.append(y_oversampled)
            else:
                X_parts.append(X_cls)
                y_parts.append(y_cls)

        return pd.concat(X_parts, axis=0).reset_index(drop=True), pd.concat(y_parts, axis=0).reset_index(drop=True)

    def _apply_random_undersample(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sampling_ratio: float,
        random_state: int
    ) -> Tuple[pd.DataFrame, pd.Series]:
        counts = y.value_counts()
        minority_count = counts.min()
        target_majority_count = max(int(minority_count / max(sampling_ratio, 0.01)), minority_count)

        rng = np.random.RandomState(random_state)
        X_parts = []
        y_parts = []

        for cls, count in counts.items():
            X_cls = X[y == cls]
            y_cls = y[y == cls]

            if count > target_majority_count:
                sample_indices = rng.choice(X_cls.index, size=target_majority_count, replace=False)
                X_parts.append(X.loc[sample_indices])
                y_parts.append(y.loc[sample_indices])
            else:
                X_parts.append(X_cls)
                y_parts.append(y_cls)

        return pd.concat(X_parts, axis=0).reset_index(drop=True), pd.concat(y_parts, axis=0).reset_index(drop=True)

    def _apply_smote(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sampling_ratio: float,
        k_neighbors: int,
        random_state: int
    ) -> Tuple[pd.DataFrame, pd.Series]:
        # If imbalanced-learn is available in environment, delegate to it
        try:
            from imblearn.over_sampling import SMOTE
            k = min(k_neighbors, y.value_counts().min() - 1)
            if k >= 1:
                sm = SMOTE(sampling_strategy=sampling_ratio, k_neighbors=k, random_state=random_state)
                X_res, y_res = sm.fit_resample(X, y)
                return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)
        except Exception:
            pass

        # Robust built-in pure Scikit-Learn / Numpy SMOTE fallback
        numeric_cols = list(X.select_dtypes(include=[np.number]).columns)
        non_numeric_cols = [c for c in X.columns if c not in numeric_cols]

        if not numeric_cols:
            # Fall back to random oversample if there are no numeric features
            return self._apply_random_oversample(X, y, sampling_ratio, random_state)

        counts = y.value_counts()
        majority_count = counts.max()
        target_count = max(int(majority_count * sampling_ratio), counts.min())

        rng = np.random.RandomState(random_state)
        synthetic_X_rows = []
        synthetic_y_rows = []

        for cls, count in counts.items():
            if count >= target_count:
                continue

            num_synthetic = target_count - count
            X_cls = X[y == cls]
            X_cls_num = X_cls[numeric_cols].values

            n_samples_cls = len(X_cls_num)
            k = min(k_neighbors, n_samples_cls - 1)

            if k < 1:
                # If only 1 sample, duplicate it directly
                sample_indices = rng.choice(X_cls.index, size=num_synthetic, replace=True)
                for idx in sample_indices:
                    synthetic_X_rows.append(X.loc[idx].to_dict())
                    synthetic_y_rows.append(cls)
                continue

            nn = NearestNeighbors(n_neighbors=k + 1).fit(X_cls_num)
            _, neighbor_indices = nn.kneighbors(X_cls_num)

            for _ in range(num_synthetic):
                base_idx = rng.randint(0, n_samples_cls)
                # Random neighbor (avoid index 0 which is the sample itself)
                neighbor_choice = rng.randint(1, k + 1)
                neighbor_idx = neighbor_indices[base_idx, neighbor_choice]

                base_vec = X_cls_num[base_idx]
                neighbor_vec = X_cls_num[neighbor_idx]
                gap = neighbor_vec - base_vec
                synth_num_vec = base_vec + rng.uniform(0.0, 1.0) * gap

                row_dict = dict(zip(numeric_cols, synth_num_vec))
                if non_numeric_cols:
                    base_row = X_cls.iloc[base_idx]
                    for col in non_numeric_cols:
                        row_dict[col] = base_row[col]

                synthetic_X_rows.append(row_dict)
                synthetic_y_rows.append(cls)

        if not synthetic_X_rows:
            return X.copy(), y.copy()

        df_synthetic = pd.DataFrame(synthetic_X_rows)[X.columns]
        y_synthetic = pd.Series(synthetic_y_rows, name=y.name)

        X_final = pd.concat([X, df_synthetic], axis=0).reset_index(drop=True)
        y_final = pd.concat([y, y_synthetic], axis=0).reset_index(drop=True)

        return X_final, y_final
