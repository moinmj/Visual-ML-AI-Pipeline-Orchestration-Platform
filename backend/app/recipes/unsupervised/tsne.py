import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class TSNERecipe(BaseRecipe):
    """
    t-Distributed Stochastic Neighbor Embedding (t-SNE) Recipe.
    Non-linear dimensionality reduction technique well-suited for embedding high-dimensional data for 2D/3D visualization.
    """

    recipe_id = "tsne"
    name = "t-SNE"
    version = "1.0.0"
    category = "unsupervised"
    description = (
        "Projects high-dimensional data to 2D or 3D for visual exploration. "
        "Models pairwise similarities to preserve local clusters and manifold neighborhoods."
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
            description="Dataset to embed"
        )
    ]
    outputs: List[RecipePort] = [
        RecipePort(
            id="output_1",
            label="t-SNE Output Table",
            type="dataframe",
            description="Dataset with t-SNE coordinate columns"
        ),
        RecipePort(
            id="output_2",
            label="t-SNE Model",
            type="model",
            description="Fitted t-SNE embedding object"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "n_components": {
                    "type": "integer",
                    "title": "Target Dimensions (Components)",
                    "enum": [2, 3],
                    "default": 2,
                    "description": "Number of dimensions to project down to (2 for standard 2D scatter plots, 3 for 3D exploration)."
                },
                "perplexity": {
                    "type": "number",
                    "title": "Perplexity",
                    "default": 30.0,
                    "minimum": 2.0,
                    "maximum": 100.0,
                    "description": "Related to the number of nearest neighbors used in other manifold learning algorithms. Larger datasets usually require larger perplexity (automatically adjusted if dataset is small)."
                },
                "learning_rate": {
                    "type": "string",
                    "title": "Learning Rate",
                    "enum": ["auto", "100", "200", "500", "1000"],
                    "default": "auto",
                    "description": "The learning rate for t-SNE optimization. 'auto' adapts based on sample size."
                },
                "max_iter": {
                    "type": "integer",
                    "title": "Maximum Iterations",
                    "default": 1000,
                    "minimum": 250,
                    "maximum": 5000,
                    "description": "Maximum number of optimization iterations for gradient descent."
                },
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "title": "Feature Columns to Embed",
                    "format": "column",
                    "ui:widget": "column_select",
                    "description": "Columns to embed. If empty, all numeric columns will be used."
                },
                "prefix": {
                    "type": "string",
                    "title": "Coordinate Column Prefix",
                    "default": "tSNE_",
                    "description": "Prefix used when naming the generated coordinate columns (e.g. 'tSNE_1', 'tSNE_2')."
                },
                "scale_features": {
                    "type": "boolean",
                    "title": "Standardize Features (Z-Score)",
                    "default": True,
                    "description": "Normalizes variance across columns prior to calculating Euclidean distances."
                },
                "append_components": {
                    "type": "boolean",
                    "title": "Append Coordinates to Table",
                    "default": True,
                    "description": "When true, appends t-SNE columns to original data. When false, outputs only coordinates."
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
                "output_summary": {"error": "No valid input DataFrame provided for t-SNE."}
            }

        if df.empty:
            return {
                "dataframe": df.copy(),
                "output_1": df.copy(),
                "output": df.copy(),
                "model": None,
                "metrics": {"total_rows": 0, "n_components": 0},
                "output_summary": {"message": "Input DataFrame is empty. No t-SNE performed."}
            }

        # Resolve columns with fallback to feature_columns alias
        cols_cfg = config.get("columns") or config.get("feature_columns")
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
                    "title": "t-SNE",
                    "message": "No numeric columns found or selected for t-SNE embedding.",
                    "status": "SKIPPED"
                }
            }

        matrix_df = df[feature_cols].copy()
        matrix_df = matrix_df.fillna(matrix_df.median(numeric_only=True)).fillna(0)

        n_samples = len(matrix_df)
        n_components = int(config.get("n_components", 2))
        if n_components not in [2, 3]:
            n_components = 2

        # Defensive auto-clamping of perplexity for small datasets:
        # scikit-learn requires: perplexity < n_samples
        requested_perp = float(config.get("perplexity", 30.0))
        max_possible_perp = max(1.0, float(n_samples - 1) / 3.0)
        effective_perp = min(requested_perp, max_possible_perp)
        # Ensure at least 1.0
        effective_perp = max(1.0, effective_perp)

        lr_cfg = config.get("learning_rate", "auto")
        learning_rate = "auto" if lr_cfg == "auto" else float(lr_cfg)
        max_iter = int(config.get("max_iter", config.get("n_iter", 1000)))
        random_state = int(config.get("random_state", 42))
        scale_features = bool(config.get("scale_features", config.get("standardize", True)))
        prefix = str(config.get("prefix", "tSNE_") or "tSNE_")
        append_components = bool(config.get("append_components", config.get("append_to_input", True)))

        X = matrix_df.to_numpy(dtype=float)
        scaler = None
        if scale_features:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        from sklearn.manifold import TSNE
        tsne = TSNE(
            n_components=n_components,
            perplexity=effective_perp,
            learning_rate=learning_rate,
            max_iter=max_iter,
            random_state=random_state,
            init="pca" if n_samples >= n_components else "random"
        )
        X_tsne = tsne.fit_transform(X)

        coord_col_names = [f"{prefix}{i + 1}" for i in range(n_components)]
        tsne_df = pd.DataFrame(X_tsne, columns=coord_col_names, index=df.index)

        if append_components:
            out_df = pd.concat([df.copy(), tsne_df], axis=1)
        else:
            out_df = tsne_df

        kl_div = getattr(tsne, "kl_divergence_", None)
        n_iter_done = getattr(tsne, "n_iter_", max_iter)

        metrics: Dict[str, Any] = {
            "n_components": n_components,
            "perplexity_requested": requested_perp,
            "perplexity_used": round(effective_perp, 2),
            "kl_divergence": round(float(kl_div), 4) if kl_div is not None else None,
            "iterations_completed": int(n_iter_done),
            "features_embedded": feature_cols,
            "total_rows": n_samples
        }

        # 2D scatter preview
        scatter_sample = tsne_df[[coord_col_names[0], coord_col_names[1]]].head(150).to_dict(orient="records")
        metrics["scatter_preview"] = scatter_sample

        summary = {
            "title": "t-SNE Embedding Completed",
            "n_components": n_components,
            "perplexity": round(effective_perp, 2),
            "kl_divergence": metrics["kl_divergence"],
            "status": "SUCCESS"
        }

        return {
            "dataframe": out_df,
            "output_1": out_df,
            "output": out_df,
            "model": tsne,
            "scaler": scaler,
            "metrics": metrics,
            "tsne_summary": metrics,
            "output_summary": summary
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        n_comp = config.get("n_components", 2)
        perp = config.get("perplexity", 30.0)
        prefix = config.get("prefix", "tSNE_")

        return (
            f"# t-SNE High-Dimensional Visualization\n"
            f"from sklearn.manifold import TSNE\n"
            f"from sklearn.preprocessing import StandardScaler\n\n"
            f"features = [c for c in df.select_dtypes(include='number').columns]\n"
            f"X = df[features].fillna(df[features].median()).to_numpy()\n"
            f"X_scaled = StandardScaler().fit_transform(X)\n\n"
            f"perp = min({perp}, max(1.0, (len(X) - 1) / 3.0))\n"
            f"tsne = TSNE(n_components={n_comp}, perplexity=perp, random_state=42)\n"
            f"X_tsne = tsne.fit_transform(X_scaled)\n"
            f"for i in range({n_comp}):\n"
            f"    df[f'{prefix}{{i+1}}'] = X_tsne[:, i]\n"
        )
