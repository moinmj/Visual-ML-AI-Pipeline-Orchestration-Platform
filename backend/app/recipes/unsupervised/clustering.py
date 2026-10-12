import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class ClusteringRecipe(BaseRecipe):
    """
    Unsupervised Clustering Recipe (KMeans, MiniBatchKMeans, DBSCAN, Agglomerative).
    Partitions rows into clusters based on multi-dimensional feature similarity.
    """

    recipe_id = "clustering"
    name = "Clustering (KMeans / DBSCAN)"
    version = "1.0.0"
    category = "unsupervised"
    description = (
        "Groups similar rows into clusters using KMeans, MiniBatchKMeans, DBSCAN, or Agglomerative Clustering. "
        "Appends cluster labels and outputs fitted cluster models and distribution metrics."
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
            description="Dataset to cluster"
        )
    ]
    outputs: List[RecipePort] = [
        RecipePort(
            id="output_1",
            label="Clustered Table",
            type="dataframe",
            description="Primary dataset with cluster labels"
        ),
        RecipePort(
            id="output_2",
            label="Cluster Model",
            type="model",
            description="Fitted clustering model"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "algorithm": {
                    "type": "string",
                    "title": "Clustering Algorithm",
                    "enum": ["KMeans", "MiniBatchKMeans", "DBSCAN", "Agglomerative"],
                    "default": "KMeans",
                    "description": "Choice of unsupervised clustering algorithm."
                },
                "n_clusters": {
                    "type": "integer",
                    "title": "Number of Clusters (k)",
                    "default": 3,
                    "minimum": 2,
                    "maximum": 100,
                    "description": "Number of clusters to form (used for KMeans, MiniBatchKMeans, Agglomerative)."
                },
                "eps": {
                    "type": "number",
                    "title": "DBSCAN Epsilon (Neighborhood Radius)",
                    "default": 0.5,
                    "minimum": 0.001,
                    "maximum": 50.0,
                    "description": "Maximum distance between two samples for one to be considered in the neighborhood of the other (DBSCAN only)."
                },
                "min_samples": {
                    "type": "integer",
                    "title": "DBSCAN Min Samples",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 200,
                    "description": "Number of samples in a neighborhood for a point to be considered a core point (DBSCAN only)."
                },
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "title": "Feature Columns to Cluster",
                    "format": "column",
                    "ui:widget": "column_select",
                    "description": "Numeric columns used to calculate distance. If empty, all detected numeric columns will be used."
                },
                "scale_features": {
                    "type": "boolean",
                    "title": "Standardize Features (Z-Score)",
                    "default": True,
                    "description": "Standardizes feature values (zero mean, unit variance) so differing column units do not skew cluster distances."
                },
                "cluster_column_name": {
                    "type": "string",
                    "title": "Cluster Label Column Name",
                    "default": "cluster",
                    "description": "Column name added to the output table storing cluster assignments."
                },
                "random_state": {
                    "type": "integer",
                    "title": "Random Seed",
                    "default": 42,
                    "description": "Random seed for reproducible centroid initialization."
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
                "output_summary": {"error": "No valid input DataFrame provided for Clustering."}
            }

        if df.empty:
            return {
                "dataframe": df.copy(),
                "output_1": df.copy(),
                "output": df.copy(),
                "model": None,
                "metrics": {"total_rows": 0, "n_clusters": 0},
                "output_summary": {"message": "Input DataFrame is empty. No clustering performed."}
            }

        out_df = df.copy()

        # Resolve columns
        cols_cfg = config.get("columns")
        feature_cols: List[str] = []
        if isinstance(cols_cfg, list) and cols_cfg:
            feature_cols = [str(c) for c in cols_cfg if str(c) in out_df.columns]
        elif isinstance(cols_cfg, str) and cols_cfg in out_df.columns:
            feature_cols = [cols_cfg]

        if not feature_cols:
            # Auto-detect numeric columns
            feature_cols = [
                col for col in out_df.columns
                if pd.api.types.is_numeric_dtype(out_df[col])
            ]

        if not feature_cols:
            return {
                "dataframe": out_df,
                "output_1": out_df,
                "output": out_df,
                "model": None,
                "metrics": {"total_rows": len(out_df), "n_clusters": 0},
                "output_summary": {
                    "title": "Clustering",
                    "message": "No numeric columns found or selected to cluster. Please supply numeric features.",
                    "status": "SKIPPED"
                }
            }

        # Prepare numeric feature matrix
        matrix_df = out_df[feature_cols].copy()
        # Impute missing values with column median or 0
        matrix_df = matrix_df.fillna(matrix_df.median(numeric_only=True)).fillna(0)

        scale_features = bool(config.get("scale_features", True))
        X = matrix_df.to_numpy(dtype=float)

        scaler = None
        if scale_features:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        algorithm = config.get("algorithm", "KMeans")
        n_clusters = int(config.get("n_clusters", 3))
        # Clamp n_clusters to row count
        n_clusters = max(1, min(n_clusters, len(out_df)))
        random_state = int(config.get("random_state", 42))
        cluster_col = str(config.get("cluster_column_name", "cluster") or "cluster")

        fitted_model = None
        labels: np.ndarray

        if algorithm == "DBSCAN":
            from sklearn.cluster import DBSCAN
            eps = float(config.get("eps", 0.5))
            min_samples = int(config.get("min_samples", 5))
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)
            fitted_model = dbscan

        elif algorithm == "MiniBatchKMeans":
            from sklearn.cluster import MiniBatchKMeans
            mb_kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                batch_size=min(1024, max(100, len(out_df)))
            )
            labels = mb_kmeans.fit_predict(X)
            fitted_model = mb_kmeans

        elif algorithm == "Agglomerative":
            from sklearn.cluster import AgglomerativeClustering
            agg = AgglomerativeClustering(n_clusters=n_clusters)
            labels = agg.fit_predict(X)
            fitted_model = agg

        else:  # Default to KMeans
            from sklearn.cluster import KMeans
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                n_init="auto"
            )
            labels = kmeans.fit_predict(X)
            fitted_model = kmeans

        out_df[cluster_col] = labels

        # Metrics computation
        unique_labels = sorted(list(set(labels)))
        cluster_dist: Dict[str, Any] = {}
        for lbl in unique_labels:
            cnt = int((labels == lbl).sum())
            pct = round((cnt / len(labels)) * 100.0, 2)
            name = f"Cluster {lbl}" if lbl != -1 else "Noise (-1)"
            cluster_dist[name] = {"count": cnt, "percentage": pct}

        metrics: Dict[str, Any] = {
            "algorithm": algorithm,
            "total_rows": len(out_df),
            "features_used": feature_cols,
            "n_clusters_found": len([l for l in unique_labels if l != -1]),
            "cluster_distribution": cluster_dist,
            "noise_count": int((labels == -1).sum()) if -1 in unique_labels else 0
        }

        # Inertia for KMeans
        if hasattr(fitted_model, "inertia_"):
            metrics["inertia"] = round(float(fitted_model.inertia_), 4)

        # Silhouette score (sampled if data > 2000 rows to ensure sub-second response)
        non_noise_mask = (labels != -1)
        valid_labels = labels[non_noise_mask]
        if len(set(valid_labels)) > 1 and len(valid_labels) >= 4:
            try:
                from sklearn.metrics import silhouette_score
                X_sub = X[non_noise_mask]
                if len(X_sub) > 2000:
                    idx = np.random.RandomState(42).choice(len(X_sub), 2000, replace=False)
                    s_score = silhouette_score(X_sub[idx], valid_labels[idx])
                else:
                    s_score = silhouette_score(X_sub, valid_labels)
                metrics["silhouette_score"] = round(float(s_score), 4)
            except Exception:
                pass

        # 2D Scatter preview (first two features + cluster)
        scatter_sample = []
        if len(feature_cols) >= 2:
            sub_df = out_df[[feature_cols[0], feature_cols[1], cluster_col]].head(150)
            scatter_sample = sub_df.to_dict(orient="records")
        metrics["scatter_preview"] = scatter_sample

        summary = {
            "title": "Clustering Completed",
            "algorithm": algorithm,
            "n_clusters": metrics["n_clusters_found"],
            "features": feature_cols,
            "status": "SUCCESS"
        }

        return {
            "dataframe": out_df,
            "output_1": out_df,
            "output": out_df,
            "model": fitted_model,
            "scaler": scaler,
            "metrics": metrics,
            "output_summary": summary,
            "cluster_summary": metrics
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        alg = config.get("algorithm", "KMeans")
        k = config.get("n_clusters", 3)
        cols = config.get("columns", [])
        col_str = repr(cols) if cols else "[c for c in df.select_dtypes(include='number').columns]"
        cluster_name = config.get("cluster_column_name", "cluster")

        return (
            f"# Clustering ({alg})\n"
            f"from sklearn.cluster import {alg}\n"
            f"from sklearn.preprocessing import StandardScaler\n\n"
            f"features = {col_str}\n"
            f"X = df[features].fillna(df[features].median()).to_numpy()\n"
            f"X_scaled = StandardScaler().fit_transform(X)\n\n"
            f"model = {alg}(n_clusters={k}, random_state=42)\n"
            f"df['{cluster_name}'] = model.fit_predict(X_scaled)\n"
        )
