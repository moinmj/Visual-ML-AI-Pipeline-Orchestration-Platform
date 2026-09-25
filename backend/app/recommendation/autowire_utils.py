from typing import List, Dict, Any, Optional
from backend.app.recipes.base.registry import recipe_registry


def ensure_semantic_edge_handles(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Intelligently assigns correct source_handle and target_handle to DAG edges
    so that visual whiteboard ports (e.g. Tr -> Trainer, Te -> Evaluator, M -> Evaluator)
    connect to the correct semantic handles instead of collapsing to handle 0.
    """
    # 1. Build metadata lookup for all canvas nodes
    node_map: Dict[str, Dict[str, Any]] = {}
    for n in nodes:
        nid = n.get("id")
        if not nid:
            continue
        r_id = n.get("recipe_id") or n.get("data", {}).get("recipe_id") or ""
        recipe = recipe_registry.get(r_id) if r_id else None
        cat = recipe.category if recipe else ""
        label = str(n.get("label") or nid or "").lower()
        node_map[nid] = {
            "recipe_id": r_id,
            "category": cat,
            "label": label
        }

    # 2. Track incoming edge indices for multi-input nodes (e.g. joins)
    join_incoming_counts: Dict[str, int] = {}
    updated_edges: List[Dict[str, Any]] = []

    for edge in edges:
        e = dict(edge)
        src_id = e.get("source")
        tgt_id = e.get("target")

        src_meta = node_map.get(src_id, {})
        tgt_meta = node_map.get(tgt_id, {})

        src_recipe = src_meta.get("recipe_id", "")
        tgt_recipe = tgt_meta.get("recipe_id", "")
        src_cat = src_meta.get("category", "")
        tgt_cat = tgt_meta.get("category", "")

        is_src_splitter = (
            src_cat == "splitting"
            or "split" in src_recipe
            or any(k in src_recipe for k in ["train_test_split", "stratified_split", "time_series_split", "walk_forward_split"])
        )
        is_tgt_evaluator = (
            tgt_cat == "evaluation"
            or tgt_recipe == "model_evaluator"
            or "eval" in tgt_recipe
        )
        is_src_trainer = (
            src_cat in ["training", "forecasting"]
            or "trainer" in src_recipe
            or "forecaster" in src_recipe
            or any(k in src_recipe for k in ["xgb", "lightgbm", "catboost", "random_forest", "logistic_regression", "prophet", "arima"])
        )

        # Case A: Splitter -> Model Evaluator (Strictly Test Data 'Te')
        if is_src_splitter and is_tgt_evaluator:
            e["source_handle"] = "output_2"  # Te (Test Data port)
            e["target_handle"] = "input_2"   # Te (Test Data input port)

        # Case B: Splitter -> Model Trainer or Resampler (Strictly Train Data 'Tr')
        elif is_src_splitter and (is_src_trainer or "resampler" in tgt_recipe or "smote" in tgt_recipe or tgt_cat == "training"):
            e["source_handle"] = "output_1"  # Tr (Train Data port)
            if not e.get("target_handle"):
                e["target_handle"] = "input"

        # Case C: Model Trainer -> Model Evaluator (Strictly Trained Model 'M')
        elif is_src_trainer and is_tgt_evaluator:
            if not e.get("source_handle"):
                e["source_handle"] = "output"
            e["target_handle"] = "input_1"       # M (Model port)

        # Case D: Model Evaluator -> Governance / MLflow
        elif is_tgt_evaluator and tgt_cat == "governance":
            if not e.get("source_handle"):
                e["source_handle"] = "output_1"
            if not e.get("target_handle"):
                e["target_handle"] = "input"

        # Case E: Multi-Dataset Join Target
        elif tgt_recipe == "dataset_join":
            cnt = join_incoming_counts.get(tgt_id, 0)
            if not e.get("target_handle"):
                e["target_handle"] = "left" if cnt == 0 else "right"
            join_incoming_counts[tgt_id] = cnt + 1

        # Case F: IF / Condition Branching
        elif src_recipe in ["if_condition", "if_else"]:
            if not e.get("source_handle"):
                e["source_handle"] = "true"

        # Case G: Switch Node Branching
        elif src_recipe == "switch":
            if not e.get("source_handle"):
                e["source_handle"] = "case_1"

        updated_edges.append(e)

    # 3. Structural Integrity: Ensure Splitter -> Evaluator Te edge exists if both are on canvas
    split_node_id = None
    eval_node_id = None
    for nid, meta in node_map.items():
        r = meta.get("recipe_id", "")
        c = meta.get("category", "")
        if c == "splitting" or "split" in r:
            split_node_id = nid
        elif c == "evaluation" or r == "model_evaluator":
            eval_node_id = nid

    if split_node_id and eval_node_id and split_node_id != eval_node_id:
        has_split_eval_edge = any(
            e.get("source") == split_node_id and e.get("target") == eval_node_id
            for e in updated_edges
        )
        if not has_split_eval_edge:
            updated_edges.append({
                "id": f"e_{split_node_id}_{eval_node_id}",
                "source": split_node_id,
                "target": eval_node_id,
                "source_handle": "output_2",
                "target_handle": "input_2",
                "animated": True
            })

    return updated_edges
