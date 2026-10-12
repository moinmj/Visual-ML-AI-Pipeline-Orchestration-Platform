import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class PCARecipe(BaseRecipe):
    """
    Principal Component Analysis (PCA) Recipe.
    Linear dimensionality reduction that projects data onto orthogonal axes of maximal variance.
    """

    recipe_id = "pca"
    name = "PCA (Principal Component Analysis)"
    version = "1.0.0"
    category = "unsupervised"
    description = (
        "Reduces many numeric features to a specified number of principal components. "
        "Captures maximum variance while removing multicollinearity, outputting transformed tables and variance ratios."
    )
    input_types = ["dataframe"]
    output_types = ["dataframe", "model"]

    inputs: List[RecipePort] = [
        RecipePort(
            id="input",
            label="Input Table",
            type="dataframe",
            required=True,
            max_connections=1,
            description="Dataset to reduce dimensionality"
        )
    ]
    outputs: List[RecipePort] = [
        RecipePort(
            id="output_1",
            label="PCA Output Table",
            type="dataframe",
            description="Dataset with principal component columns"
        ),
        RecipePort(
            id="output_2",
            label="PCA Model",
            type="model",
            description="Fitted PCA transformer"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "n_components": {
                    "type": "integer",
                    "title": "Number of Components",
                    "default": 2,
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Number of principal dimensions to extract (e.g. 2 or 3 for 2D/3D visualization)."
                },
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "title": "Numeric Feature Columns",
                    "format": "column",
                    "ui:widget": "column_select",
                    "description": "Columns to reduce. If empty, all numeric columns will be included."
                },
                "prefix": {
                    "type": "string",
                    "title": "Component Column Prefix",
                    "default": "PCA_",
                    "description": "Prefix used when naming the generated component columns (e.g. 'PCA_1', 'PCA_2')."
                },
                "scale_features": {
                    "type": "boolean",
                    "title": "Standardize Features (Z-Score)",
                    "default": True,
                    "description": "Recommended. Scales features to unit variance so high-magnitude columns do not artificially dominate principal axes."
                },
                "append_components": {
                    "type": "boolean",
                    "title": "Append Components to Table",
                    "default": True,
                    "description": "When true, appends PCA columns to the existing dataset. When false, outputs only the component columns."
                },
                "whiten": {
                    "type": "boolean",
                    "title": "Whiten Components",
                    "default": False,
                    "description": "When true, components are scaled to have unit variance (decorrelated white noise)."
                },
                "random_state": {
                    "type": "integer",
                    "title": "Random Seed",
                    "default": 42
                }
            }
        }

    def execute(
        self,
        inputs: Dict[str, Any],
        config: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        df = inputs.get("dataframe")
        if df is None or not isinstance(df, pd.DataFrame):
            df = inputs.get("left_dataframe") or inputs.get("input")
            if df is None and context and isinstance(context, dict):
                df = context.get("dataframe")

        if df is None or not isinstance(df, pd.DataFrame):
            return {
                "dataframe": pd.DataFrame(),
                "output_1": pd.DataFrame(),
                "output": pd.DataFrame(),
                "model": None,
                "output_summary": {"error": "No valid input DataFrame provided for PCA."}
            }

        if df.empty:
            return {
                "dataframe": df.copy(),
                "output_1": df.copy(),
                "output": df.copy(),
                "model": None,
                "metrics": {"total_rows": 0, "n_components": 0},
                "output_summary": {"message": "Input DataFrame is empty. No PCA performed."}
            }

        # Resolve columns
        cols_cfg = config.get("columns")
        feature_cols: List[str] = []
        if isinstance(cols_cfg, list) and cols_cfg:
            feature_cols = [str(c) for c in cols_cfg if str(c) in df.columns]
        elif isinstance(cols_cfg, str) and cols_cfg in df.columns:
            feature_cols = [cols_cfg]

        if not feature_cols:
            feature_cols = [
                col for col in df.columns
                if pd.api.types.is_numeric_dtype(df[col])
            ]

        if not feature_cols:
            return {
                "dataframe": df.copy(),
                "output_1": df.copy(),
                "output": df.copy(),
                "model": None,
                "metrics": {"total_rows": len(df), "n_components": 0},
                "output_summary": {
                    "title": "PCA",
                    "message": "No numeric columns found or selected to reduce.",
                    "status": "SKIPPED"
                }
            }

        matrix_df = df[feature_cols].copy()
        matrix_df = matrix_df.fillna(matrix_df.median(numeric_only=True)).fillna(0)

        n_comp_req = int(config.get("n_components", 2))
        # PCA requires n_components <= min(n_samples, n_features)
        n_components = max(1, min(n_comp_req, len(matrix_df), len(feature_cols)))

        scale_features = bool(config.get("scale_features", True))
        whiten = bool(config.get("whiten", False))
        random_state = int(config.get("random_state", 42))
        prefix = str(config.get("prefix", "PCA_") or "PCA_")
        append_components = bool(config.get("append_components", True))

        X = matrix_df.to_numpy(dtype=float)
        scaler = None
        if scale_features:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components, whiten=whiten, random_state=random_state)
        X_pca = pca.fit_transform(X)

        component_col_names = [f"{prefix}{i + 1}" for i in range(n_components)]
        pca_df = pd.DataFrame(X_pca, columns=component_col_names, index=df.index)

        if append_components:
            out_df = pd.concat([df.copy(), pca_df], axis=1)
        else:
            out_df = pca_df

        # Calculate explained variance and feature loadings
        var_ratios = [round(float(v), 4) for v in pca.explained_variance_ratio_]
        total_var = round(float(np.sum(pca.explained_variance_ratio_)), 4)
        total_var_pct = round(total_var * 100.0, 2)

        loadings_dict: Dict[str, Dict[str, float]] = {}
        for idx, comp_name in enumerate(component_col_names):
            weights = pca.components_[idx]
            top_weights = {
                feature_cols[j]: round(float(weights[j]), 4)
                for j in np.argsort(np.abs(weights))[::-1]
            }
            loadings_dict[comp_name] = top_weights

        metrics: Dict[str, Any] = {
            "n_components": n_components,
            "features_reduced": feature_cols,
            "explained_variance_ratio": var_ratios,
            "cumulative_variance_ratio": [round(float(v), 4) for v in np.cumsum(pca.explained_variance_ratio_)],
            "total_explained_variance_pct": total_var_pct,
            "singular_values": [round(float(s), 4) for s in pca.singular_values_],
            "feature_loadings": loadings_dict
        }

        # Scatter preview if >= 2 components
        scatter_sample = []
        if n_components >= 2:
            scatter_sample = pca_df[[component_col_names[0], component_col_names[1]]].head(150).to_dict(orient="records")
        metrics["scatter_preview"] = scatter_sample

        summary = {
            "title": "PCA Completed",
            "n_components": n_components,
            "total_variance_explained": f"{total_var_pct}%",
            "status": "SUCCESS"
        }

        return {
            "dataframe": out_df,
            "output_1": out_df,
            "output": out_df,
            "model": pca,
            "scaler": scaler,
            "metrics": metrics,
            "pca_summary": metrics,
            "output_summary": summary
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        n_comp = config.get("n_components", 2)
        cols = config.get("columns", [])
        col_str = repr(cols) if cols else "[c for c in df.select_dtypes(include='number').columns]"
        prefix = config.get("prefix", "PCA_")

        return (
            f"# PCA Dimensionality Reduction\n"
            f"from sklearn.decomposition import PCA\n"
            f"from sklearn.preprocessing import StandardScaler\n\n"
            f"features = {col_str}\n"
            f"X = df[features].fillna(df[features].median()).to_numpy()\n"
            f"X_scaled = StandardScaler().fit_transform(X)\n\n"
            f"pca = PCA(n_components={n_comp}, random_state=42)\n"
            f"X_pca = pca.fit_transform(X_scaled)\n"
            f"for i in range({n_comp}):\n"
            f"    df[f'{prefix}{{i+1}}'] = X_pca[:, i]\n"
        )
