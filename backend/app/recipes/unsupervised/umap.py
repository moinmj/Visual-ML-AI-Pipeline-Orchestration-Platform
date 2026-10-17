import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class UMAPRecipe(BaseRecipe):
    """
    UMAP (Uniform Manifold Approximation and Projection) Recipe.
    Non-linear dimensionality reduction preserving both local and global manifold structure.
    """

    recipe_id = "umap"
    name = "UMAP"
    version = "1.0.0"
    category = "unsupervised"
    description = (
        "Non-linear dimensionality reduction for visualization and clustering. "
        "Preserves global and local neighborhood structures, projecting high-dimensional data into 2D/3D representations."
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
            label="UMAP Output Table",
            type="dataframe",
            description="Dataset with UMAP component columns"
        ),
        RecipePort(
            id="output_2",
            label="UMAP Model",
            type="model",
            description="Fitted UMAP / Manifold transformer"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "n_components": {
                    "type": "integer",
                    "title": "Number of Dimensions (Components)",
                    "default": 2,
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Dimensionality of the target embedded space (2 or 3 for visual exploration)."
                },
                "n_neighbors": {
                    "type": "integer",
                    "title": "Number of Neighbors",
                    "default": 15,
                    "minimum": 2,
                    "maximum": 200,
                    "description": "Balance between local vs global structure. Smaller values focus on very local details; larger values capture broader trends."
                },
                "min_dist": {
                    "type": "number",
                    "title": "Minimum Distance",
                    "default": 0.1,
                    "minimum": 0.0,
                    "maximum": 0.99,
                    "description": "How tightly UMAP packs points together. Low values result in dense clumps; higher values preserve broad topology."
                },
                "metric": {
                    "type": "string",
                    "title": "Distance Metric",
                    "enum": ["euclidean", "manhattan", "cosine", "correlation"],
                    "default": "euclidean",
                    "description": "Metric to measure distance in high-dimensional feature space."
                },
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "title": "Feature Columns",
                    "format": "column",
                    "ui:widget": "column_select",
                    "description": "Columns to include in manifold learning. If empty, all numeric columns will be used."
                },
                "prefix": {
                    "type": "string",
                    "title": "Component Column Prefix",
                    "default": "UMAP_",
                    "description": "Prefix used when naming the generated component columns (e.g. 'UMAP_1', 'UMAP_2')."
                },
                "scale_features": {
                    "type": "boolean",
                    "title": "Standardize Features (Z-Score)",
                    "default": True,
                    "description": "Scales features before projection to balance column scales."
                },
                "append_components": {
                    "type": "boolean",
                    "title": "Append Components to Table",
                    "default": True,
                    "description": "When true, appends UMAP columns to the original dataset. When false, outputs only the component columns."
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
                "output_summary": {"error": "No valid input DataFrame provided for UMAP."}
            }

        if df.empty:
            return {
                "dataframe": df.copy(),
                "output_1": df.copy(),
                "output": df.copy(),
                "model": None,
                "metrics": {"total_rows": 0, "n_components": 0},
                "output_summary": {"message": "Input DataFrame is empty. No UMAP performed."}
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
                    "title": "UMAP",
                    "message": "No numeric columns found or selected to embed.",
                    "status": "SKIPPED"
                }
            }

        matrix_df = df[feature_cols].copy()
        matrix_df = matrix_df.fillna(matrix_df.median(numeric_only=True)).fillna(0)

        n_comp_req = int(config.get("n_components", 2))
        n_components = max(1, min(n_comp_req, len(matrix_df), len(feature_cols)))

        n_neighbors_req = int(config.get("n_neighbors", 15))
        n_neighbors = max(2, min(n_neighbors_req, len(matrix_df) - 1)) if len(matrix_df) > 2 else 2
        min_dist = float(config.get("min_dist", 0.1))
        metric = str(config.get("metric", "euclidean"))
        random_state = int(config.get("random_state", 42))
        scale_features = bool(config.get("scale_features", config.get("standardize", True)))
        prefix = str(config.get("prefix", "UMAP_") or "UMAP_")
        append_components = bool(config.get("append_components", config.get("append_to_input", True)))

        X = matrix_df.to_numpy(dtype=float)
        scaler = None
        if scale_features:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        fitted_model = None
        engine_used = "native_umap"
        X_umap = None

        # Check for native umap-learn package
        try:
            import umap
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric=metric,
                random_state=random_state
            )
            X_umap = reducer.fit_transform(X)
            fitted_model = reducer
        except ImportError:
            # Fallback to Scikit-Learn Spectral Embedding / Manifold for high-fidelity non-linear reduction
            engine_used = "spectral_manifold_fallback"
            from sklearn.manifold import SpectralEmbedding
            embedding = SpectralEmbedding(
                n_components=n_components,
                n_neighbors=n_neighbors,
                random_state=random_state,
                affinity="nearest_neighbors" if metric in ["euclidean", "manhattan"] else "rbf"
            )
            try:
                X_umap = embedding.fit_transform(X)
                fitted_model = embedding
            except Exception:
                # If nearest neighbors graph is disconnected, fall back to PCA projection
                from sklearn.decomposition import PCA
                pca_fallback = PCA(n_components=n_components, random_state=random_state)
                X_umap = pca_fallback.fit_transform(X)
                fitted_model = pca_fallback
                engine_used = "pca_approximation"

        # Condition coordinate scale to prevent microscopic 1e-7 exponential notation artifacts
        for i in range(n_components):
            c_std = float(np.std(X_umap[:, i]))
            if c_std > 0 and c_std < 1e-3:
                X_umap[:, i] = (X_umap[:, i] - np.mean(X_umap[:, i])) / c_std

        component_col_names = [f"{prefix}{i + 1}" for i in range(n_components)]
        umap_df = pd.DataFrame(np.round(X_umap, 5), columns=component_col_names, index=df.index)

        if append_components:
            out_df = pd.concat([df.copy(), umap_df], axis=1)
        else:
            out_df = umap_df

        # Interactive 2D/3D scatter plot data
        sample_size = min(300, len(umap_df))
        sample_indices = np.random.RandomState(random_state).choice(len(umap_df), sample_size, replace=False) if len(umap_df) > sample_size else np.arange(len(umap_df))

        color_col = None
        for col_cand in ["cluster", "CLUSTER", "label", "target", "Churn", "churn", "category"]:
            if col_cand in df.columns:
                color_col = col_cand
                break

        scatter_points: List[Dict[str, Any]] = []
        for idx in sample_indices:
            row_umap = umap_df.iloc[idx]
            x_val = round(float(row_umap[component_col_names[0]]), 4)
            y_val = round(float(row_umap[component_col_names[1]]), 4) if n_components >= 2 else 0.0
            p_dict: Dict[str, Any] = {
                "x": x_val,
                "y": y_val,
                component_col_names[0]: x_val
            }
            if n_components >= 2:
                p_dict[component_col_names[1]] = y_val
            if n_components >= 3 and len(component_col_names) >= 3:
                z_val = round(float(row_umap[component_col_names[2]]), 4)
                p_dict["z"] = z_val
                p_dict[component_col_names[2]] = z_val

            if color_col is not None and idx < len(df):
                val_color = df.iloc[idx][color_col]
                p_dict["group"] = str(val_color)
                p_dict["label"] = str(val_color)
                p_dict["cluster"] = str(val_color)

            scatter_points.append(p_dict)

        scatter_plot: Dict[str, Any] = {
            "x_axis": component_col_names[0],
            "y_axis": component_col_names[1] if n_components >= 2 else component_col_names[0],
            "z_axis": component_col_names[2] if n_components >= 3 else None,
            "dimensions": min(3, n_components),
            "points": scatter_points,
            "total_points": len(scatter_points),
            "has_color_group": color_col is not None,
            "color_feature": color_col
        }

        metrics: Dict[str, Any] = {
            "n_components": n_components,
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "metric": metric,
            "engine": engine_used,
            "features_embedded": feature_cols,
            "total_rows": len(df),
            "scatter_plot": scatter_plot,
            "scatter_preview": scatter_points[:150]
        }

        summary = {
            "title": "UMAP Embedding Completed",
            "engine": engine_used,
            "n_components": n_components,
            "n_neighbors": n_neighbors,
            "metric": metric,
            "scatter_plot": scatter_plot,
            "status": "SUCCESS"
        }

        return {
            "dataframe": out_df,
            "output_1": out_df,
            "output": out_df,
            "model": fitted_model,
            "scaler": scaler,
            "metrics": metrics,
            "umap_summary": metrics,
            "output_summary": summary
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        n_comp = config.get("n_components", 2)
        n_neigh = config.get("n_neighbors", 15)
        min_d = config.get("min_dist", 0.1)
        prefix = config.get("prefix", "UMAP_")

        return (
            f"# UMAP Non-Linear Embedding\n"
            f"import umap\n"
            f"from sklearn.preprocessing import StandardScaler\n\n"
            f"features = [c for c in df.select_dtypes(include='number').columns]\n"
            f"X = df[features].fillna(df[features].median()).to_numpy()\n"
            f"X_scaled = StandardScaler().fit_transform(X)\n\n"
            f"reducer = umap.UMAP(n_components={n_comp}, n_neighbors={n_neigh}, min_dist={min_d}, random_state=42)\n"
            f"X_umap = reducer.fit_transform(X_scaled)\n"
            f"for i in range({n_comp}):\n"
            f"    df[f'{prefix}{{i+1}}'] = X_umap[:, i]\n"
        )
