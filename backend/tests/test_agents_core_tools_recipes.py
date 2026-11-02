"""
Unit tests for Phase 4: Core Agents & Tools Recipes.
Tests all 8 recipes across standard workflows and edge cases:
- LlmAgentRecipe
- MultiAgentOrchestratorRecipe
- AgentHandoffRouterRecipe
- WebBrowsingAgentRecipe
- DocumentQaAgentRecipe
- MultimodalAgentRecipe
- CodeExecutionSandboxRecipe
- ChatbotEndpointRecipe
"""

import json
import pytest
import pandas as pd
import numpy as np

from backend.app.recipes.agents.llm_agent import LlmAgentRecipe
from backend.app.recipes.agents.multi_agent_orchestrator import MultiAgentOrchestratorRecipe
from backend.app.recipes.agents.agent_handoff_router import AgentHandoffRouterRecipe
from backend.app.recipes.agents.web_browsing_agent import WebBrowsingAgentRecipe
from backend.app.recipes.agents.document_qa_agent import DocumentQaAgentRecipe
from backend.app.recipes.agents.multimodal_agent import MultimodalAgentRecipe
from backend.app.recipes.agents.code_execution_sandbox import CodeExecutionSandboxRecipe
from backend.app.recipes.agents.chatbot_endpoint import ChatbotEndpointRecipe
from backend.app.recipes.base.registry import recipe_registry


# ─────────────────────────────────────────────────────────────────────────────
# 1. LLM Agent (ReAct) Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestLlmAgentRecipe:
    def test_react_calculator_tool(self):
        recipe = LlmAgentRecipe()
        inputs = {"task": "Calculate 45 * 12 for the revenue projection"}
        res = recipe.execute(inputs, {"tools": ["calculator"]})

        assert res["status"] == "success"
        assert "540" in res["final_answer"]
        assert res["tools_called"] >= 1
        assert "dataframe" in res
        assert not res["dataframe"].empty

    def test_react_data_lookup_tool(self):
        recipe = LlmAgentRecipe()
        inputs = {
            "task": "Look up region for the target customer",
            "context": {"region": "North America", "tier": "Enterprise"}
        }
        res = recipe.execute(inputs, {"tools": ["data_lookup"]})

        assert res["status"] == "success"
        assert "North America" in res["final_answer"]

    def test_react_dataframe_batch(self):
        recipe = LlmAgentRecipe()
        df = pd.DataFrame({
            "task": ["Calculate 100 / 4", "Look up tier"],
            "tier": ["Platinum", "Gold"]
        })
        res = recipe.execute({"dataframe": df}, {"tools": ["calculator", "data_lookup"]})

        assert "dataframe" in res
        out_df = res["dataframe"]
        assert "agent_final_answer" in out_df.columns
        assert "25" in out_df.iloc[0]["agent_final_answer"]

    def test_react_empty_task(self):
        recipe = LlmAgentRecipe()
        res = recipe.execute({"task": ""}, {})
        assert res["status"] == "success"
        assert res["tools_called"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# 2. Multi-Agent Orchestrator Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestMultiAgentOrchestratorRecipe:
    def test_supervisor_worker_synthesis(self):
        recipe = MultiAgentOrchestratorRecipe()
        inputs = {"goal": "Architect a fault-tolerant RAG retrieval service"}
        config = {
            "worker_roles": ["Research Specialist", "Systems Engineer", "QA Reviewer"]
        }
        res = recipe.execute(inputs, config)

        assert res["status"] == "success"
        assert "supervisor_plan" in res
        assert len(res["worker_contributions"]) == 3
        assert "QA Reviewer" in res["worker_contributions"]
        assert "Executive Multi-Agent Synthesis" in res["final_synthesis"]

    def test_orchestrator_dataframe_input(self):
        recipe = MultiAgentOrchestratorRecipe()
        df = pd.DataFrame({
            "goal": ["Optimize database query latency", "Deploy security guardrail"]
        })
        res = recipe.execute({"dataframe": df}, {})

        assert "dataframe" in res
        out_df = res["dataframe"]
        assert "supervisor_plan" in out_df.columns
        assert "final_synthesis" in out_df.columns
        assert len(out_df) == 2

    def test_orchestrator_empty_goal(self):
        recipe = MultiAgentOrchestratorRecipe()
        res = recipe.execute({"goal": ""}, {})
        assert res["status"] == "success"
        assert "Unable to orchestrate" in res["final_synthesis"]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Agent Handoff Router Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestAgentHandoffRouterRecipe:
    def test_billing_routing(self):
        recipe = AgentHandoffRouterRecipe()
        inputs = {"message": "I was charged twice on my credit card invoice and need a refund"}
        res = recipe.execute(inputs, {})

        assert res["routed_agent_id"] == "billing_agent"
        assert res["confidence_score"] > 0.5

    def test_technical_support_routing(self):
        recipe = AgentHandoffRouterRecipe()
        inputs = {"message": "The pipeline execution failed with timeout error and 500 status"}
        res = recipe.execute(inputs, {})

        assert res["routed_agent_id"] == "technical_support_agent"
        assert res["confidence_score"] > 0.5

    def test_human_escalation_routing(self):
        recipe = AgentHandoffRouterRecipe()
        inputs = {"message": "This service is completely unacceptable, let me speak with a manager or supervisor immediately"}
        res = recipe.execute(inputs, {})

        assert res["routed_agent_id"] == "human_escalation"

    def test_handoff_dataframe_batch(self):
        recipe = AgentHandoffRouterRecipe()
        df = pd.DataFrame({
            "query": [
                "Can I get an enterprise sales quote for 50 seats?",
                "My payment failed on the invoice",
                "How do I fix the API connection error?"
            ]
        })
        res = recipe.execute({"dataframe": df}, {"message_column": "query"})

        assert "dataframe" in res
        out_df = res["dataframe"]
        assert out_df.iloc[0]["routed_agent_id"] == "sales_onboarding_agent"
        assert out_df.iloc[1]["routed_agent_id"] == "billing_agent"
        assert out_df.iloc[2]["routed_agent_id"] == "technical_support_agent"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Web Browsing Agent Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestWebBrowsingAgentRecipe:
    def test_web_crawl_simulation(self):
        recipe = WebBrowsingAgentRecipe()
        inputs = {"url": "https://docs.datalect.app/orchestration/recipes"}
        res = recipe.execute(inputs, {"extract_key_findings": True})

        assert res["status"] == "success"
        assert "extracted_content" in res
        assert len(res["key_findings"]) >= 1

    def test_web_search_query(self):
        recipe = WebBrowsingAgentRecipe()
        inputs = {"query": "latest advancements in multimodal LLM agents"}
        res = recipe.execute(inputs, {})

        assert res["status"] == "success"
        assert "multimodal LLM agents" in res["extracted_content"]

    def test_web_empty_input(self):
        recipe = WebBrowsingAgentRecipe()
        res = recipe.execute({}, {})
        assert res["status"] == "success"
        assert res["status_code"] == 200


# ─────────────────────────────────────────────────────────────────────────────
# 5. Document QA Agent Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestDocumentQaAgentRecipe:
    def test_document_qa_answering(self):
        recipe = DocumentQaAgentRecipe()
        context = (
            "Datalect Orchestration platform provides visual node workflows. "
            "The execution latency for local pipelines averages under 50 milliseconds. "
            "All pipelines support automated retry policies and enterprise authentication."
        )
        inputs = {
            "question": "What is the average execution latency for local pipelines?",
            "context": context
        }
        res = recipe.execute(inputs, {})

        assert res["status"] == "success"
        assert "50 milliseconds" in res["answer"]
        assert res["confidence"] > 0.7

    def test_document_qa_dataframe_batch(self):
        recipe = DocumentQaAgentRecipe()
        df = pd.DataFrame({
            "doc": [
                "Refunds are processed within 5 business days of request submission.",
                "System maintenance occurs every Sunday at 02:00 UTC."
            ],
            "q": [
                "How many days does a refund take?",
                "When is system maintenance scheduled?"
            ]
        })
        res = recipe.execute({"dataframe": df}, {"context_column": "doc", "question_column": "q"})

        assert "dataframe" in res
        out_df = res["dataframe"]
        assert "5 business days" in out_df.iloc[0]["qa_answer"]
        assert "Sunday at 02:00 UTC" in out_df.iloc[1]["qa_answer"]


# ─────────────────────────────────────────────────────────────────────────────
# 6. Multimodal Agent Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestMultimodalAgentRecipe:
    def test_image_analysis(self):
        recipe = MultimodalAgentRecipe()
        inputs = {"image_url": "https://assets.example.com/invoice_scan_sample.png"}
        res = recipe.execute(inputs, {"modality": "image", "enable_ocr": True})

        assert res["status"] == "success"
        assert "visual_description" in res
        assert "ocr_text" in res
        assert res["metrics"]["modality"] == "image"

    def test_audio_analysis(self):
        recipe = MultimodalAgentRecipe()
        inputs = {"audio_url": "https://assets.example.com/customer_call_001.wav"}
        res = recipe.execute(inputs, {"modality": "audio"})

        assert res["status"] == "success"
        assert "transcript" in res
        assert res["metrics"]["modality"] == "audio"

    def test_multimodal_dataframe_batch(self):
        recipe = MultimodalAgentRecipe()
        df = pd.DataFrame({
            "media_uri": [
                "https://images.site/product_catalog_photo.jpg",
                "https://audio.site/voice_memo_sample.mp3"
            ]
        })
        res = recipe.execute({"dataframe": df}, {"media_column": "media_uri"})

        assert "dataframe" in res
        out_df = res["dataframe"]
        assert "multimodal_summary" in out_df.columns
        assert len(out_df) == 2


# ─────────────────────────────────────────────────────────────────────────────
# 7. Code Execution Sandbox Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestCodeExecutionSandboxRecipe:
    def test_successful_code_execution(self):
        recipe = CodeExecutionSandboxRecipe()
        code = (
            "import math\n"
            "values = [10, 20, 30, 40]\n"
            "print(f'Sum of values: {sum(values)}')\n"
            "result = math.sqrt(sum(values))\n"
        )
        res = recipe.execute({"code": code}, {})

        assert res["status"] == "success"
        assert "Sum of values: 100" in res["stdout"]
        assert res["result"] == 10.0
        assert res["error"] is None

    def test_context_variable_injection(self):
        recipe = CodeExecutionSandboxRecipe()
        code = "result = user_count * price_per_user"
        context = {"user_count": 25, "price_per_user": 40}
        res = recipe.execute({"code": code, "context": context}, {})

        assert res["status"] == "success"
        assert res["result"] == 1000

    def test_syntax_error_resilience(self):
        recipe = CodeExecutionSandboxRecipe()
        code = "def broken_fn(:\n    return True"
        res = recipe.execute({"code": code}, {})

        assert res["status"] == "error"
        assert "SyntaxError" in res["error"]

    def test_restricted_module_import_security(self):
        recipe = CodeExecutionSandboxRecipe()
        code = "import os\nresult = os.name"
        res = recipe.execute({"code": code}, {"allowed_modules": ["math", "json"]})

        assert res["status"] == "error"
        assert "ImportError" in res["error"]

    def test_code_sandbox_dataframe_batch(self):
        recipe = CodeExecutionSandboxRecipe()
        df = pd.DataFrame({
            "script": [
                "result = 2 ** 10",
                "result = len('hello world')"
            ]
        })
        res = recipe.execute({"dataframe": df}, {"code_column": "script"})

        assert "dataframe" in res
        out_df = res["dataframe"]
        assert out_df.iloc[0]["execution_result_result"] == "1024"
        assert out_df.iloc[1]["execution_result_result"] == "11"


# ─────────────────────────────────────────────────────────────────────────────
# 8. Chatbot Endpoint Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestChatbotEndpointRecipe:
    def test_greeting_interaction(self):
        recipe = ChatbotEndpointRecipe()
        inputs = {"message": "Hello, how can you help me today?"}
        res = recipe.execute(inputs, {"bot_name": "Atlas Bot"})

        assert "Atlas Bot" in res["reply"]
        assert len(res["quick_replies"]) >= 2
        assert "session_id" in res
        assert res["metrics"]["intent"] == "greeting"

    def test_pricing_intent_quick_replies(self):
        recipe = ChatbotEndpointRecipe()
        inputs = {"message": "What is the cost and pricing for enterprise?"}
        res = recipe.execute(inputs, {})

        assert "enterprise" in res["reply"].lower()
        assert "Request demo" in res["quick_replies"]
        assert res["metrics"]["intent"] == "pricing"

    def test_chatbot_dataframe_batch(self):
        recipe = ChatbotEndpointRecipe()
        df = pd.DataFrame({
            "message": [
                "Hi there",
                "How much does the platform cost?",
                "API returned error code 500"
            ]
        })
        res = recipe.execute({"dataframe": df}, {})

        assert "dataframe" in res
        out_df = res["dataframe"]
        assert "assistant_reply" in out_df.columns
        assert "intent" in out_df.columns
        assert list(out_df["intent"]) == ["greeting", "pricing", "technical"]

    def test_empty_message_handling(self):
        recipe = ChatbotEndpointRecipe()
        res = recipe.execute({"message": ""}, {"bot_name": "SupportBot"})

        assert "SupportBot" in res["reply"]
        assert len(res["quick_replies"]) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# 9. Registry Resolution & Aliases Verification
# ─────────────────────────────────────────────────────────────────────────────
class TestPhase4RegistryResolution:
    def test_registry_contains_all_phase4_recipes(self):
        phase4_ids = [
            "llm_agent",
            "multi_agent_orchestrator",
            "agent_handoff_router",
            "web_browsing_agent",
            "document_qa_agent",
            "multimodal_agent",
            "code_execution_sandbox",
            "chatbot_endpoint",
        ]
        for rid in phase4_ids:
            recipe = recipe_registry.get(rid)
            assert recipe is not None
            assert recipe.category == "Agents & LLM Orchestration"

    def test_registry_aliases_resolve_correctly(self):
        aliases_map = {
            "agent": "llm_agent",
            "orchestrator": "multi_agent_orchestrator",
            "handoff_router": "agent_handoff_router",
            "web_agent": "web_browsing_agent",
            "doc_qa": "document_qa_agent",
            "multimodal": "multimodal_agent",
            "code_sandbox": "code_execution_sandbox",
            "chatbot": "chatbot_endpoint",
        }
        for alias, expected_id in aliases_map.items():
            resolved = recipe_registry.get(alias)
            assert resolved.recipe_id == expected_id
