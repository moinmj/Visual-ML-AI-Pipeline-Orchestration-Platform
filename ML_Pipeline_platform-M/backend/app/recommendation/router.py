from fastapi import APIRouter, Depends, HTTPException, status, Body
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from backend.app.infrastructure.database.session import get_db
from backend.app.infrastructure.storage.storage_manager import storage_manager
from backend.app.datasets.models import Dataset
from backend.app.recommendation.recommender import AIRecommender
from backend.app.recipes.base.registry import recipe_registry

router = APIRouter(prefix="/recommend", tags=["AI Recommendation & Auto-Architect"])


class RecommendationRequest(BaseModel):
    dataset_id: Optional[str] = Field(None, description="Optional ID of an uploaded dataset")
    target_column: Optional[str] = Field(None, description="Optional target label column for supervised tasks")
    time_column: Optional[str] = Field(None, description="Optional timestamp column for time-series tasks")
    task_type: Optional[str] = Field(None, description="Optional forced task: 'classification', 'regression', 'time_series_forecasting', 'anomaly_detection'")
    preset: Optional[str] = Field("balanced", description="Optimization preset: 'balanced', 'speed', 'accuracy', 'explainability'")
    dataframe_records: Optional[List[Dict[str, Any]]] = Field(None, description="Optional inline list of records to profile")


class AutoWireRequest(BaseModel):
    nodes: List[Dict[str, Any]] = Field(..., description="List of nodes on the whiteboard canvas to wire")


@router.post("/pipeline")
async def recommend_pipeline(
    payload: RecommendationRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    AI Recommendation & Pipeline Auto-Architect Endpoint.
    Analyzes dataset profile, determines optimal task type, ranks Tier-1 ML models,
    and returns a ready-to-render visual DAG (nodes, edges, layout, parameters).
    """
    df = None
    dataset_name = "Sample Dataset"

    if payload.dataset_id:
        result = await db.execute(select(Dataset).where(Dataset.id == payload.dataset_id))
        ds = result.scalar_one_or_none()
        if ds and ds.storage_path:
            dataset_name = ds.name
            try:
                df = storage_manager.read_dataframe(ds.storage_path)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Could not read dataset file '{ds.storage_path}': {str(e)}"
                )
    elif payload.dataframe_records:
        df = pd.DataFrame(payload.dataframe_records)

    # Fallback to default tabular benchmark dataset if none supplied
    if df is None:
        df = pd.DataFrame({
            "age": [25, 34, 45, 52, 23, 40, 60, 31, 29, 48],
            "income": [50000, 65000, 85000, 110000, 48000, 72000, 95000, 58000, 62000, 89000],
            "tenure_months": [12, 36, 60, 48, 6, 24, 72, 18, 15, 50],
            "contract_type": ["Monthly", "Annual", "Two-Year", "Annual", "Monthly", "Monthly", "Two-Year", "Monthly", "Annual", "Two-Year"],
            "monthly_charges": [70.5, 89.0, 105.2, 98.4, 65.0, 80.0, 115.0, 75.0, 82.5, 102.0],
            "churn": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0]
        })

    recommendation = AIRecommender.recommend_pipeline(
        df=df,
        target_column=payload.target_column,
        task_type=payload.task_type
    )

    # Update dataset_id in the synthesized ingestion node
    if payload.dataset_id and "recommended_dag" in recommendation:
        dag = recommendation["recommended_dag"]
        for n in dag.get("nodes", []):
            if n.get("recipe_id") == "csv_loader":
                n["config"]["dataset_id"] = payload.dataset_id
                n["label"] = f"📄 {dataset_name[:18]}"
                if "node_configs" in dag and n["id"] in dag["node_configs"]:
                    dag["node_configs"][n["id"]]["config"]["dataset_id"] = payload.dataset_id
                    dag["node_configs"][n["id"]]["label"] = f"📄 {dataset_name[:18]}"

    return recommendation


@router.post("/autowire")
async def autowire_nodes(payload: AutoWireRequest):
    """
    Intelligent DAG Auto-Wiring Endpoint.
    Analyzes unwired whiteboard components, sorts them topologically by recipe
    hierarchy (Ingestion -> Preprocessing -> Split -> Training -> Evaluation / Governance),
    and generates optimal directed connections (edges).
    """
    curr_nodes = payload.nodes
    if len(curr_nodes) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Auto-Wire requires at least 2 nodes on the canvas."
        )

    # Hierarchy ranking for automatic topological pipeline construction
    category_weights = {
        "ingestion": 1,
        "preprocessing": 2,
        "splitting": 3,
        "training": 4,
        "forecasting": 4,
        "anomaly": 4,
        "evaluation": 5,
        "governance": 6
    }

    def get_node_weight(node: Dict[str, Any]) -> int:
        r_id = node.get("recipe_id") or node.get("data", {}).get("recipe_id")
        recipe = recipe_registry.get(r_id) if r_id else None
        if recipe:
            return category_weights.get(recipe.category, 3)
        # Inferred from id/label
        label = str(node.get("label") or node.get("id") or "").lower()
        if "csv" in label or "loader" in label:
            return 1
        elif "impute" in label or "scale" in label or "encode" in label:
            return 2
        elif "split" in label:
            return 3
        elif "xgb" in label or "lightgbm" in label or "catboost" in label or "model" in label or "train" in label:
            return 4
        elif "eval" in label or "metric" in label:
            return 5
        elif "gov" in label or "audit" in label:
            return 6
        return 3

    # Sort nodes by pipeline category weight, then by horizontal position x
    sorted_nodes = sorted(
        curr_nodes,
        key=lambda n: (
            get_node_weight(n),
            n.get("position", {}).get("x", 0) if isinstance(n.get("position"), dict) else 0
        )
    )

    edges = []
    split_node_id = None
    model_node_id = None
    eval_node_id = None
    prev_node_id = None

    for idx, node in enumerate(sorted_nodes):
        n_id = node["id"]
        weight = get_node_weight(node)

        if weight == 3: # Splitter
            split_node_id = n_id
        elif weight == 4: # Model trainer
            model_node_id = n_id
        elif weight == 5: # Evaluator
            eval_node_id = n_id

        if idx > 0 and prev_node_id:
            edges.append({
                "id": f"e_{prev_node_id}_{n_id}",
                "source": prev_node_id,
                "target": n_id,
                "animated": True
            })
        prev_node_id = n_id

    # Secondary edge: Splitter -> Evaluator (for X_test/y_test propagation)
    if split_node_id and eval_node_id:
        split_eval_exists = any(e["source"] == split_node_id and e["target"] == eval_node_id for e in edges)
        if not split_eval_exists:
            edges.append({
                "id": f"e_{split_node_id}_{eval_node_id}",
                "source": split_node_id,
                "target": eval_node_id,
                "animated": True
            })

    return {
        "status": "AUTOWIRED",
        "nodes_count": len(curr_nodes),
        "edges_count": len(edges),
        "edges": edges
    }
