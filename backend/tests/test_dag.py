import pytest
from backend.app.engine.dag.graph import WorkflowGraph, WorkflowNode, WorkflowEdge
from backend.app.engine.execution.executor import DAGExecutor


def test_dag_cycle_detection():
    # Construct a cycle: node1 -> node2 -> node3 -> node1
    nodes = [
        WorkflowNode(id="node1", recipe_id="missing_value_imputer"),
        WorkflowNode(id="node2", recipe_id="feature_scaler"),
        WorkflowNode(id="node3", recipe_id="categorical_encoder")
    ]
    edges = [
        WorkflowEdge(source="node1", target="node2"),
        WorkflowEdge(source="node2", target="node3"),
        WorkflowEdge(source="node3", target="node1")
    ]

    graph = WorkflowGraph(nodes=nodes, edges=edges)
    errors = graph.validate_graph()
    assert len(errors) > 0
    assert any("Cycle detected" in e for e in errors)


def test_dag_valid_topological_sort():
    # Valid linear DAG: node1 -> node2 -> node3
    nodes = [
        WorkflowNode(id="node3", recipe_id="categorical_encoder"),
        WorkflowNode(id="node1", recipe_id="missing_value_imputer"),
        WorkflowNode(id="node2", recipe_id="feature_scaler")
    ]
    edges = [
        WorkflowEdge(source="node1", target="node2"),
        WorkflowEdge(source="node2", target="node3")
    ]

    graph = WorkflowGraph(nodes=nodes, edges=edges)
    assert len(graph.validate_graph()) == 0

    order = graph.get_topological_order()
    order_ids = [n.id for n in order]
    assert order_ids == ["node1", "node2", "node3"]
