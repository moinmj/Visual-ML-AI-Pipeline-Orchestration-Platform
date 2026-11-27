"""Tests for guarded cyclic edge validation and LLM hub-and-spoke cyclic execution."""

import json
from unittest.mock import patch

from backend.app.engine.dag.graph import WorkflowGraph, WorkflowNode, WorkflowEdge
from backend.app.engine.execution.executor import DAGExecutor


def test_unguarded_cycle_is_still_rejected():
    wg = WorkflowGraph(
        nodes=[
            WorkflowNode(id="n1", recipe_id="manual_data_entry", config={"data": [{"a": 1}]}),
            WorkflowNode(id="n2", recipe_id="row_filter", config={}),
        ],
        edges=[
            WorkflowEdge(source="n1", target="n2"),
            WorkflowEdge(source="n2", target="n1"),
        ],
    )
    diag = wg.get_diagnostics()
    assert diag["is_valid"] is False
    assert any("unguarded cycle" in e["message"].lower() for e in diag["errors"])


def test_guarded_llm_hub_and_spoke_cycle_is_valid():
    wg = WorkflowGraph(
        nodes=[
            WorkflowNode(id="ingest", recipe_id="manual_data_entry", config={"data": [{"task": "Research Acme"}]}),
            WorkflowNode(id="hub", recipe_id="llm_agent", config={"max_iterations": 5}),
            WorkflowNode(id="agent_web", recipe_id="web_browsing_agent", config={}),
            WorkflowNode(id="agent_doc", recipe_id="document_qa_agent", config={}),
        ],
        edges=[
            WorkflowEdge(source="ingest", target="hub"),
            WorkflowEdge(source="hub", target="agent_web"),
            WorkflowEdge(source="agent_web", target="hub"),
            WorkflowEdge(source="hub", target="agent_doc"),
            WorkflowEdge(source="agent_doc", target="hub"),
        ],
    )
    diag = wg.get_diagnostics()
    assert diag["is_valid"] is True, f"Expected valid cyclic graph, got errors: {diag['errors']}"
    assert wg.has_cycles() is True
    entries = [n.id for n in wg.get_entry_nodes()]
    assert entries == ["ingest"]


def test_cyclic_hub_and_spoke_llm_routing_and_return():
    wg = WorkflowGraph(
        nodes=[
            WorkflowNode(id="ingest", recipe_id="manual_data_entry", config={"data": [{"task": "Analyze Acme SLA"}]}),
            WorkflowNode(id="hub", recipe_id="llm_agent", config={"max_iterations": 5}),
            WorkflowNode(id="agent_web", recipe_id="web_browsing_agent", config={}),
            WorkflowNode(id="agent_doc", recipe_id="document_qa_agent", config={}),
        ],
        edges=[
            WorkflowEdge(source="ingest", target="hub"),
            WorkflowEdge(source="hub", target="agent_web"),
            WorkflowEdge(source="agent_web", target="hub"),
            WorkflowEdge(source="hub", target="agent_doc"),
            WorkflowEdge(source="agent_doc", target="hub"),
        ],
    )

    hub_calls = {"count": 0}

    def fake_live_llm(system_prompt, user_prompt, **kwargs):
        # Hub routing calls happen when Connected Downstream Agents is in user_prompt
        if "Connected Downstream Agents:" in user_prompt:
            hub_calls["count"] += 1
            if hub_calls["count"] == 1:
                payload = {"next_node_id": "agent_web", "answer": "Checking web first"}
            elif hub_calls["count"] == 2:
                assert "agent_web" in user_prompt
                payload = {"next_node_id": "agent_doc", "answer": "Now checking docs"}
            else:
                assert "agent_doc" in user_prompt
                payload = {"next_node_id": "FINISH", "answer": "Final synthesized answer from web + docs."}
            return {
                "content": json.dumps(payload),
                "execution_mode": "ONLINE_LIVE",
                "provider_used": "groq",
                "model_used": "test-model",
                "fallback_reason": "None (Live Online LLM)",
            }
        return {
            "content": "Agent sub-task result",
            "execution_mode": "ONLINE_LIVE",
            "provider_used": "groq",
            "model_used": "test-model",
            "fallback_reason": "None (Live Online LLM)",
        }

    with patch("backend.app.recipes.agents.llm_client.call_live_llm", side_effect=fake_live_llm):
        events = []
        res = DAGExecutor.execute_workflow(
            execution_id="test_cyclic_hub_001",
            workflow=wg,
            include_node_outputs=True,
            on_event=events.append,
        )

    assert res.status == "SUCCESS"
    executed_ids = [r.node_id for r in res.node_results if r.status == "SUCCESS"]
    assert executed_ids == ["ingest", "hub", "agent_web", "hub", "agent_doc", "hub"]
    assert res.node_outputs["hub"]["final_answer"] == "Final synthesized answer from web + docs."

    # Verify flow visibility events & execution_trace
    assert [t["node_id"] for t in res.execution_trace] == ["ingest", "hub", "agent_web", "hub", "agent_doc", "hub"]
    assert [t["iteration"] for t in res.execution_trace] == [1, 1, 1, 2, 1, 3]
    edge_events = [e for e in events if e.get("type") == "edge_traversed"]
    assert [(e["source"], e["target"], e["is_back_edge"]) for e in edge_events] == [
        ("ingest", "hub", False),
        ("hub", "agent_web", False),
        ("agent_web", "hub", True),
        ("hub", "agent_doc", False),
        ("agent_doc", "hub", True),
    ]


def test_cyclic_max_iterations_safety_cap_prevents_infinite_loop():
    wg = WorkflowGraph(
        nodes=[
            WorkflowNode(id="ingest", recipe_id="manual_data_entry", config={"data": [{"task": "Loop forever"}]}),
            WorkflowNode(id="hub", recipe_id="llm_agent", config={"max_iterations": 2}),
            WorkflowNode(id="agent_web", recipe_id="web_browsing_agent", config={"max_iterations": 5}),
        ],
        edges=[
            WorkflowEdge(source="ingest", target="hub"),
            WorkflowEdge(source="hub", target="agent_web"),
            WorkflowEdge(source="agent_web", target="hub"),
        ],
    )

    def always_loop_llm(system_prompt, user_prompt, **kwargs):
        if "Connected Downstream Agents:" in user_prompt:
            return {
                "content": json.dumps({"next_node_id": "agent_web", "answer": "Loop again"}),
                "execution_mode": "ONLINE_LIVE",
                "provider_used": "groq",
                "model_used": "test-model",
                "fallback_reason": "None (Live Online LLM)",
            }
        return {
            "content": "Web result",
            "execution_mode": "ONLINE_LIVE",
            "provider_used": "groq",
            "model_used": "test-model",
            "fallback_reason": "None (Live Online LLM)",
        }

    with patch("backend.app.recipes.agents.llm_client.call_live_llm", side_effect=always_loop_llm):
        res = DAGExecutor.execute_workflow(
            execution_id="test_cyclic_cap_001",
            workflow=wg,
        )

    assert res.status == "SUCCESS"
    hub_runs = [r for r in res.node_results if r.node_id == "hub"]
    assert len(hub_runs) == 2
    assert any("Loop cap reached" in log for log in res.logs)


def test_hub_receives_agent_description_and_worker_receives_hub_instruction():
    wg = WorkflowGraph(
        nodes=[
            WorkflowNode(id="ingest", recipe_id="manual_data_entry", config={"data": [{"task": "Generic row task"}]}),
            WorkflowNode(
                id="hub",
                recipe_id="multi_agent_orchestrator",
                config={"query": "Find Acme pricing online", "max_iterations": 4, "worker_roles": ["Researcher"]},
            ),
            WorkflowNode(
                id="agent_web",
                recipe_id="web_browsing_agent",
                config={"agent_description": "Specialist for live web pricing lookup"},
            ),
        ],
        edges=[
            WorkflowEdge(source="ingest", target="hub"),
            WorkflowEdge(source="hub", target="agent_web"),
            WorkflowEdge(source="agent_web", target="hub"),
        ],
    )

    seen_prompts = []

    def inspect_prompts_llm(system_prompt, user_prompt, **kwargs):
        seen_prompts.append((system_prompt, user_prompt))
        if "Connected Downstream Agents:" in system_prompt:
            assert "Specialist for live web pricing lookup" in system_prompt
            if len([p for p in seen_prompts if "Connected Downstream Agents:" in p[0]]) == 1:
                assert "Find Acme pricing online" in user_prompt
                return {
                    "content": json.dumps({"next_node_id": "agent_web", "answer": "Search exact Acme Corp enterprise pricing"}),
                    "execution_mode": "ONLINE_LIVE",
                    "provider_used": "groq",
                    "model_used": "test-model",
                    "fallback_reason": "None (Live Online LLM)",
                }
            return {
                "content": json.dumps({"next_node_id": "FINISH", "answer": "Done with pricing"}),
                "execution_mode": "ONLINE_LIVE",
                "provider_used": "groq",
                "model_used": "test-model",
                "fallback_reason": "None (Live Online LLM)",
            }
        # Worker call must receive hub_instruction instead of generic row task
        assert "Search exact Acme Corp enterprise pricing" in user_prompt
        return {
            "content": "Acme pricing is $99/seat",
            "execution_mode": "ONLINE_LIVE",
            "provider_used": "groq",
            "model_used": "test-model",
            "fallback_reason": "None (Live Online LLM)",
        }

    with patch("backend.app.recipes.agents.llm_client.call_live_llm", side_effect=inspect_prompts_llm):
        res = DAGExecutor.execute_workflow(execution_id="test_hub_instr_001", workflow=wg, include_node_outputs=True)

    assert res.status == "SUCCESS"
    assert res.node_outputs["hub"]["final_synthesis"] == "Done with pricing"
