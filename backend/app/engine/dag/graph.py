import pandas as pd
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict, deque
from pydantic import BaseModel, Field, model_validator
from backend.app.core.exceptions import ValidationException
from backend.app.recipes.base.registry import recipe_registry


class WorkflowNode(BaseModel):
    id: str = Field(..., description="Unique node instance ID on the canvas")
    recipe_id: str = Field(..., description="ID of the recipe component")
    config: Dict[str, Any] = Field(default_factory=dict, description="Configuration matching recipe JSON schema")
    label: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def normalize_node(cls, data: Any) -> Any:
        if isinstance(data, dict):
            node_data = data.get("data")
            if isinstance(node_data, dict):
                if not data.get("recipe_id"):
                    data["recipe_id"] = (
                        node_data.get("recipe_id")
                        or node_data.get("recipeId")
                        or node_data.get("type")
                        or data.get("type")
                    )
                if (not data.get("config") or not isinstance(data.get("config"), dict)) and isinstance(node_data.get("config"), dict):
                    data["config"] = dict(node_data["config"])
                if not data.get("label"):
                    data["label"] = node_data.get("label") or node_data.get("content")
        return data


class WorkflowEdge(BaseModel):
    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    source_handle: Optional[str] = None
    target_handle: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def normalize_handles(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if not data.get("source_handle"):
                data["source_handle"] = data.get("sourceHandle") or data.get("source_port")
            if not data.get("target_handle"):
                data["target_handle"] = data.get("targetHandle") or data.get("target_port")
        return data


class WorkflowGraph(BaseModel):
    id: Optional[str] = None
    workflow_id: Optional[str] = None
    pipeline_id: Optional[str] = None
    name: Optional[str] = None
    dataset_id: Optional[str] = None
    dataset_name: Optional[str] = None
    nodes: List[WorkflowNode]
    edges: List[WorkflowEdge]
    node_configs: Optional[Dict[str, Any]] = None

    @model_validator(mode="before")
    @classmethod
    def normalize_graph(cls, data: Any) -> Any:
        if isinstance(data, dict):
            node_configs = data.get("node_configs")
            if isinstance(node_configs, dict) and "nodes" in data and isinstance(data["nodes"], list):
                for n in data["nodes"]:
                    if isinstance(n, dict):
                        nid = n.get("id")
                        n_cfg = node_configs.get(nid)
                        if n_cfg and isinstance(n_cfg, dict):
                            # If config is missing, empty, or dict without recipe keys
                            cfg = n.get("config")
                            if not cfg or not isinstance(cfg, dict):
                                n["config"] = dict(n_cfg)
                            else:
                                for k, v in n_cfg.items():
                                    if k not in n["config"]:
                                        n["config"][k] = v
        return data

    def validate_graph(self) -> List[Dict[str, Any]]:
        """
        Validates graph structure, checks node IDs, detects cycles, orphan nodes,
        and semantic recipe input/output compatibility with clear explanation reasons.
        """
        errors: List[Dict[str, Any]] = []
        node_ids = {n.id for n in self.nodes}
        node_dict = {n.id: n for n in self.nodes}

        def add_error(message: str, node_id: Optional[str] = None, recipe_id: Optional[str] = None) -> None:
            """Appends a diagnostic entry along with the node/recipe it belongs to,
            so the frontend can jump straight to the offending recipe's settings panel."""
            errors.append({
                "message": message,
                "node_id": node_id,
                "recipe_id": recipe_id if recipe_id is not None else (node_dict[node_id].recipe_id if node_id in node_dict else None)
            })

        if len(node_ids) != len(self.nodes):
            add_error("Duplicate node IDs found in workflow.")

        for edge in self.edges:
            if edge.source not in node_ids:
                add_error(f"Edge references non-existent source node: '{edge.source}'", node_id=edge.source)
            if edge.target not in node_ids:
                add_error(f"Edge references non-existent target node: '{edge.target}'", node_id=edge.target)

        if errors:
            return errors

        # 1. Detect Orphan / Disconnected Nodes
        if len(self.nodes) > 1:
            connected_nodes = set()
            for e in self.edges:
                connected_nodes.add(e.source)
                connected_nodes.add(e.target)
            orphans = node_ids - connected_nodes
            for orphan in orphans:
                add_error(f"⚠️ Orphan Node '{orphan}' is completely disconnected. Please connect it to your pipeline or delete it.", node_id=orphan)

        # 2. Detect cycles using Kahn's algorithm
        in_degree = {n.id: 0 for n in self.nodes}
        adj = defaultdict(list)

        for edge in self.edges:
            adj[edge.source].append(edge.target)
            in_degree[edge.target] += 1

        queue = deque([node_id for node_id, deg in in_degree.items() if deg == 0])
        ordered_nodes = []
        visited_ids = set()
        visited_count = 0

        while queue:
            curr = queue.popleft()
            visited_count += 1
            visited_ids.add(curr)
            if curr in node_dict:
                ordered_nodes.append(node_dict[curr])
            for neighbor in adj[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Append any remaining unvisited nodes (in case of cycles or disconnected components)
        for n in self.nodes:
            if n.id not in visited_ids:
                ordered_nodes.append(n)

        if visited_count != len(self.nodes):
            add_error("❌ Cycle detected in workflow graph. Workflows must be Directed Acyclic Graphs (DAGs) without circular loops.")

        # 3. Semantic Recipe Contract Compatibility Checks
        parent_map = defaultdict(list)
        for edge in self.edges:
            parent_map[edge.target].append(edge.source)

        for node in ordered_nodes:
            try:
                recipe = recipe_registry.get(node.recipe_id)
            except Exception as e:
                # Planned ("coming soon") recipes are listed in the catalog but can't run yet.
                from backend.app.core.exceptions import ValidationException
                if isinstance(e, ValidationException):
                    add_error(f"❌ Node '{node.id}': {e.message}")
                continue

            parents = parent_map[node.id]

            # Model Trainer Check: Trainer requires Train/Test Split or train data
            if recipe.category == "training" and node.recipe_id not in ["model_calibration", "ensemble_stacker"]:
                has_split_parent = any("split" in p or "train" in p for p in parents)
                if not has_split_parent and parents:
                    parent_recipes = [node_dict[p].recipe_id for p in parents if p in node_dict]
                    valid_train_providers = {"train_test_split", "class_imbalance_resampler"}
                    has_train_data_output = False
                    for r_id in parent_recipes:
                        try:
                            prec = recipe_registry.get(r_id)
                            if "train_data" in getattr(prec, "output_types", []):
                                has_train_data_output = True
                                break
                        except Exception:
                            pass

                    if not any(r in valid_train_providers for r in parent_recipes) and not has_train_data_output:
                        add_error(
                            f"❌ Incompatible Connection for '{node.id}' [{recipe.name}]: Model trainers expect split partitions (X_train, y_train). "
                            f"Currently connected directly to '{', '.join(parents)}'. "
                            f"Fix: Insert a '✂️ Train / Test Splitter' between data preparation and this trainer.",
                            node_id=node.id, recipe_id=node.recipe_id
                        )

            # Model Calibration Check: Requires upstream trained model
            if node.recipe_id == "model_calibration" and parents:
                parent_recipes = [node_dict[p].recipe_id for p in parents if p in node_dict]
                has_model_output = False
                for r_id in parent_recipes:
                    try:
                        prec = recipe_registry.get(r_id)
                        if "model" in getattr(prec, "output_types", []):
                            has_model_output = True
                            break
                    except Exception:
                        pass
                if not has_model_output:
                    add_error(
                        f"❌ Incompatible Connection for '{node.id}' [{recipe.name}]: Model Calibration requires an upstream trained model (e.g. CatBoost, XGBoost, Random Forest). "
                        f"Currently connected to '{', '.join(parents)}'.",
                        node_id=node.id, recipe_id=node.recipe_id
                    )

            # Dataset Join Check: Requires at least 2 incoming dataset connections
            if node.recipe_id == "dataset_join":
                if len(parents) < 2:
                    add_error(
                        f"⚠️ Incomplete Connections for '{node.id}' [{recipe.name}]: Dataset Join requires 2 incoming dataset streams (Left and Right), "
                        f"but currently has {len(parents)} incoming connection(s). Connect two dataset nodes to this Join processor.",
                        node_id=node.id, recipe_id=node.recipe_id
                    )

        # 3b. Node Configuration & Required Schema Validation in sequential pipeline order (upstream -> downstream)
        connected_node_ids = {e.source for e in self.edges} | {e.target for e in self.edges}
        has_pipeline = len(connected_node_ids) > 0

        # Precompute which nodes on the canvas are valid vs invalid
        valid_nodes = set()
        for node in ordered_nodes:
            try:
                rec = recipe_registry.get(node.recipe_id)
                cfg = node.config or {}
                if not rec.validate_config(cfg):
                    valid_nodes.add(node.id)
            except Exception:
                pass

        for node in ordered_nodes:
            try:
                recipe = recipe_registry.get(node.recipe_id)
            except Exception:
                continue

            cfg = node.config or {}
            cfg_errors = recipe.validate_config(cfg)
            if not cfg_errors:
                continue

            is_connected = node.id in connected_node_ids
            # A node is treated as an incomplete draft node if:
            # 1. There is an active pipeline on canvas (has_pipeline) and this node is disconnected from it; OR
            # 2. There are no edges (edges == 0), multiple nodes (> 1), and at least one other node is valid/configured.
            is_draft = (has_pipeline and not is_connected) or (not has_pipeline and len(self.nodes) > 1 and len(valid_nodes) > 0)

            for ce in cfg_errors:
                if str(ce).startswith("⚠️") or str(ce).startswith("💡"):
                    add_error(f"{ce}", node_id=node.id, recipe_id=node.recipe_id)
                elif is_draft:
                    add_error(
                        f"⚠️ Incomplete Draft Node '{node.id}' [{recipe.name}]: {ce} (Disconnected from active pipeline)",
                        node_id=node.id, recipe_id=node.recipe_id
                    )
                else:
                    add_error(f"❌ Configuration Error for '{node.id}' [{recipe.name}]: {ce}", node_id=node.id, recipe_id=node.recipe_id)

        # 4. ML Best Practice Checks & Recommendations
        all_recipe_ids = {n.recipe_id for n in self.nodes}
        for node in self.nodes:
            try:
                recipe = recipe_registry.get(node.recipe_id)
            except Exception:
                continue

            # Check if ML Trainer is present without any Categorical Encoder in the pipeline
            if recipe.category == "training":
                if "categorical_encoder" not in all_recipe_ids:
                    add_error(
                        f"💡 Pro-Tip for '{node.id}' [{recipe.name}]: No 'Categorical Feature Encoder' processor found in pipeline. "
                        f"If your dataset contains text/string categories (e.g. Region, Category, Status), "
                        f"insert a 'Categorical Feature Encoder' before Train/Test Split to boost model accuracy.",
                        node_id=node.id, recipe_id=node.recipe_id
                    )

                # Check for discrete 0/1 target column in Regression mode
                cfg = node.config or {}
                if str(cfg.get("task_type", "")).lower() == "regression":
                    target_col = cfg.get("target_column")
                    if not target_col:
                        for p_id in parent_map.get(node.id, []):
                            if p_id in node_dict:
                                p_cfg = node_dict[p_id].config or {}
                                if p_cfg.get("target_column"):
                                    target_col = p_cfg.get("target_column")
                                    break
                    if target_col:
                        is_binary_discrete = False
                        for n in self.nodes:
                            if n.config and isinstance(n.config.get("dataframe"), pd.DataFrame):
                                df_c = n.config["dataframe"]
                                if target_col in df_c.columns:
                                    u_vals = set(df_c[target_col].dropna().unique())
                                    if len(u_vals) <= 2 and (u_vals.issubset({0, 1, 0.0, 1.0, True, False, "0", "1", "yes", "no"}) or len(u_vals) <= 2):
                                        is_binary_discrete = True
                                        break
                        if not is_binary_discrete:
                            t_lower = str(target_col).lower()
                            binary_keywords = ["churn", "target_churn", "fraud", "default", "label", "binary", "class", "target_class", "survived", "promoted", "converted"]
                            if any(k in t_lower for k in binary_keywords) or t_lower.startswith("is_"):
                                is_binary_discrete = True

                        if is_binary_discrete:
                            add_error(
                                f"⚠️ Inline Validation Warning for '{node.id}' [{recipe.name}]: "
                                f"Target column '{target_col}' appears to be a discrete 0/1 category column, but Regression mode is selected. "
                                f"Regression is intended for continuous numeric quantities (such as sales_amount, price, or temperature). "
                                f"Remedy: Switch ML Task Type to 'classification' for binary 0/1 predictions, or select a continuous numeric target like 'sales_amount'.",
                                node_id=node.id, recipe_id=node.recipe_id
                            )

        return errors

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Returns structured diagnostic breakdown categorized into blocking errors,
        warnings, and actionable best-practice recommendations for frontend canvas display.
        """
        raw_items = self.validate_graph()
        errors = [item for item in raw_items if item["message"].startswith("❌") or (not item["message"].startswith("⚠️") and not item["message"].startswith("💡"))]
        warnings = [item for item in raw_items if item["message"].startswith("⚠️")]
        recommendations = [item for item in raw_items if item["message"].startswith("💡")]

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "recommendations": recommendations
        }

    def get_topological_order(self) -> List[WorkflowNode]:
        """
        Returns nodes in valid execution order using Kahn's topological sort.
        """
        diag = self.get_diagnostics()
        if not diag["is_valid"]:
            error_details = " | ".join(e["message"] for e in diag["errors"])
            raise ValidationException(f"Invalid workflow DAG structure: {error_details}", errors=diag["errors"])

        in_degree = {n.id: 0 for n in self.nodes}
        adj = defaultdict(list)
        node_map = {n.id: n for n in self.nodes}

        for edge in self.edges:
            adj[edge.source].append(edge.target)
            in_degree[edge.target] += 1

        queue = deque([node_id for node_id, deg in in_degree.items() if deg == 0])
        ordered_nodes = []

        while queue:
            curr = queue.popleft()
            ordered_nodes.append(node_map[curr])
            for neighbor in adj[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return ordered_nodes