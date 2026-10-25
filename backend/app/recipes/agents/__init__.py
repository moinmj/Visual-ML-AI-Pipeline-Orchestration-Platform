"""
Agents & LLM Orchestration Recipes Package.
Provides all 18 enterprise recipes across:
- Prompt & Memory Management (Phase 1)
- RAG & Retrieval (Phase 2)
- Governance & Ops (Phase 3)
- Core Agents & Tools (Phase 4)
"""
# Phase 1: Prompt & Memory Management
from backend.app.recipes.agents.prompt_template_manager import PromptTemplateManagerRecipe
from backend.app.recipes.agents.conversation_memory import ConversationMemoryRecipe
from backend.app.recipes.agents.structured_output_enforcer import StructuredOutputEnforcerRecipe

# Phase 2: RAG & Retrieval
from backend.app.recipes.agents.embedding_generator import EmbeddingGeneratorRecipe
from backend.app.recipes.agents.vector_db_indexer import VectorDbIndexerRecipe
from backend.app.recipes.agents.rag_pipeline import RagPipelineRecipe

# Phase 3: Governance & Ops
from backend.app.recipes.agents.agent_guardrail import AgentGuardrailRecipe
from backend.app.recipes.agents.cost_aware_model_router import CostAwareModelRouterRecipe
from backend.app.recipes.agents.llm_evaluation import LlmEvaluationRecipe
from backend.app.recipes.agents.fine_tuning_trigger import FineTuningTriggerRecipe

# Phase 4: Core Agents & Tools
from backend.app.recipes.agents.llm_agent import LlmAgentRecipe
from backend.app.recipes.agents.multi_agent_orchestrator import MultiAgentOrchestratorRecipe
from backend.app.recipes.agents.agent_handoff_router import AgentHandoffRouterRecipe
from backend.app.recipes.agents.web_browsing_agent import WebBrowsingAgentRecipe
from backend.app.recipes.agents.document_qa_agent import DocumentQaAgentRecipe
from backend.app.recipes.agents.multimodal_agent import MultimodalAgentRecipe
from backend.app.recipes.agents.code_execution_sandbox import CodeExecutionSandboxRecipe
from backend.app.recipes.agents.chatbot_endpoint import ChatbotEndpointRecipe

__all__ = [
    # Phase 1
    "PromptTemplateManagerRecipe",
    "ConversationMemoryRecipe",
    "StructuredOutputEnforcerRecipe",
    # Phase 2
    "EmbeddingGeneratorRecipe",
    "VectorDbIndexerRecipe",
    "RagPipelineRecipe",
    # Phase 3
    "AgentGuardrailRecipe",
    "CostAwareModelRouterRecipe",
    "LlmEvaluationRecipe",
    "FineTuningTriggerRecipe",
    # Phase 4
    "LlmAgentRecipe",
    "MultiAgentOrchestratorRecipe",
    "AgentHandoffRouterRecipe",
    "WebBrowsingAgentRecipe",
    "DocumentQaAgentRecipe",
    "MultimodalAgentRecipe",
    "CodeExecutionSandboxRecipe",
    "ChatbotEndpointRecipe",
]
