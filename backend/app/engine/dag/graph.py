from typing import List, Dict, Any, Optional, Set
from collections import defaultdict, deque
from pydantic import BaseModel, Field
from backend.app.core.exceptions import ValidationException
from backend.app.recipes.base.registry import recipe_registry


class WorkflowNode(BaseModel):
    id: str = Field(..., description="Unique node instance ID on the canvas")
    recipe_id: str = Field(..., description="ID of the recipe component")
    config: Dict[str, Any] = Field(default_factory=dict, description="Configuration matching recipe JSON schema")
    label: Optional[str] = None


class WorkflowEdge(BaseModel):
    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    source_handle: Optional[str] = None
    target_handle: Optional[str] = None


class WorkflowGraph(BaseModel):
    nodes: List[WorkflowNode]
    edges: List[WorkflowEdge]

    def validate_graph(self) -> List[str]:
        """
        Validates graph structure, checks node IDs, detects cycles, orphan nodes,
        and semantic recipe input/output compatibility with clear explanation reasons.
        """
        errors = []
        node_ids = {n.id for n in self.nodes}

        if len(node_ids) != len(self.nodes):
            errors.append("Duplicate node IDs found in workflow.")

        for edge in self.edges:
            if edge.source not in node_ids:
                errors.append(f"Edge references non-existent source node: '{edge.source}'")
            if edge.target not in node_ids:
                errors.append(f"Edge references non-existent target node: '{edge.target}'")

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
                errors.append(f"⚠️ Orphan Node '{orphan}' is completely disconnected. Please connect it to your pipeline or delete it.")

        # 2. Detect cycles using Kahn's algorithm
        in_degree = {n.id: 0 for n in self.nodes}
        adj = defaultdict(list)

        for edge in self.edges:
            adj[edge.source].append(edge.target)
            in_degree[edge.target] += 1

        queue = deque([node_id for node_id, deg in in_degree.items() if deg == 0])
        visited_count = 0

        while queue:
            curr = queue.popleft()
            visited_count += 1
            for neighbor in adj[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if visited_count != len(self.nodes):
            errors.append("❌ Cycle detected in workflow graph. Workflows must be Directed Acyclic Graphs (DAGs) without circular loops.")

        # 3. Semantic Recipe Contract Compatibility Checks
        parent_map = defaultdict(list)
        for edge in self.edges:
            parent_map[edge.target].append(edge.source)

        node_dict = {n.id: n for n in self.nodes}
        for node in self.nodes:
            try:
                recipe = recipe_registry.get(node.recipe_id)
            except Exception:
                continue

            parents = parent_map[node.id]

            # Model Trainer Check: Trainer requires Train/Test Split or train data
            if recipe.category == "training":
                has_split_parent = any("split" in p or "train" in p for p in parents)
                if not has_split_parent and parents:
                    parent_recipes = [node_dict[p].recipe_id for p in parents if p in node_dict]
                    if not any(r in ["train_test_split"] for r in parent_recipes):
                        errors.append(
                            f"❌ Incompatible Connection for '{node.id}' [{recipe.name}]: Model trainers expect split partitions (X_train, y_train). "
                            f"Currently connected directly to '{', '.join(parents)}'. "
                            f"Fix: Insert a '✂️ Train / Test Splitter' between data preparation and this trainer."
                        )

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
                    errors.append(
                        f"💡 Pro-Tip for '{node.id}' [{recipe.name}]: No 'Categorical Feature Encoder' processor found in pipeline. "
                        f"If your dataset contains text/string categories (e.g. Region, Category, Status), "
                        f"insert a 'Categorical Feature Encoder' before Train/Test Split to boost model accuracy."
                    )

        return errors

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Returns structured diagnostic breakdown categorized into blocking errors,
        warnings, and actionable best-practice recommendations for frontend canvas display.
        """
        raw_items = self.validate_graph()
        errors = [item for item in raw_items if item.startswith("❌") or (not item.startswith("⚠️") and not item.startswith("💡"))]
        warnings = [item for item in raw_items if item.startswith("⚠️")]
        recommendations = [item for item in raw_items if item.startswith("💡")]

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
        errors = self.validate_graph()
        # Filter out warnings from hard blocking execution errors
        blocking_errors = [e for e in errors if not e.startswith("⚠️")]
        if blocking_errors:
            raise ValidationException("Invalid workflow DAG structure", errors=blocking_errors)

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
