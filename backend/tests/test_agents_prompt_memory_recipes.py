"""
Comprehensive Unit Tests for Agents & LLM Orchestration (Phase 1):
- PromptTemplateManagerRecipe
- ConversationMemoryRecipe
- StructuredOutputEnforcerRecipe
"""
import json
import pytest
import pandas as pd
import numpy as np

from backend.app.recipes.agents.prompt_template_manager import PromptTemplateManagerRecipe
from backend.app.recipes.agents.conversation_memory import ConversationMemoryRecipe
from backend.app.recipes.agents.structured_output_enforcer import StructuredOutputEnforcerRecipe
from backend.app.core.exceptions import ValidationException


# =============================================================================
# 1. PROMPT TEMPLATE MANAGER TESTS
# =============================================================================

def test_prompt_template_manager_basic_interpolation():
    recipe = PromptTemplateManagerRecipe()
    df = pd.DataFrame([
        {"customer_name": "Alice", "product": "Laptop", "issue": "Battery drains fast"},
        {"customer_name": "Bob", "product": "Smartphone", "issue": "Screen flickering"}
    ])
    config = {
        "template": "Customer {{customer_name}} bought {product} and reported: {{issue}}.",
        "output_column": "prompt_result"
    }
    res = recipe.execute({"input": df}, config)
    out_df = res["dataframe"]
    assert "prompt_result" in out_df.columns
    assert out_df["prompt_result"].iloc[0] == "Customer Alice bought Laptop and reported: Battery drains fast."
    assert out_df["prompt_result"].iloc[1] == "Customer Bob bought Smartphone and reported: Screen flickering."
    assert res["metrics"]["total_prompts_rendered"] == 2


def test_prompt_template_manager_few_shot_and_chat_messages():
    recipe = PromptTemplateManagerRecipe()
    df = pd.DataFrame([{"inquiry": "How do I reset my password?"}])
    config = {
        "template": "User query: {{inquiry}}",
        "system_prompt": "You are a tier-1 IT support bot.",
        "output_mode": "chat_messages_json",
        "output_column": "rendered_text",
        "few_shot_examples": [
            {"input": "Where is my invoice?", "output": "Navigate to Account > Billing."}
        ]
    }
    res = recipe.execute({"input": df}, config)
    out_df = res["dataframe"]
    assert "rendered_text" in out_df.columns
    assert "messages_payload" in out_df.columns

    messages = json.loads(out_df["messages_payload"].iloc[0])
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a tier-1 IT support bot."
    assert messages[1]["role"] == "user"
    assert "Where is my invoice?" in messages[1]["content"]
    assert "User query: How do I reset my password?" in messages[1]["content"]


def test_prompt_template_manager_missing_variables_and_history():
    recipe = PromptTemplateManagerRecipe()
    df = pd.DataFrame([
        {"user": "Carol", "known_attr": "Gold", "missing_attr": np.nan, "history": "Prior ticket: Login issue."}
    ])
    # Mode fill_na
    config_fill = {
        "template": "Hello {{user}}, status: {{missing_attr}}, details: {{unknown_col}}",
        "missing_variable_handling": "fill_na",
        "na_placeholder": "[UNKNOWN]",
        "conversation_history_column": "history"
    }
    res_fill = recipe.execute({"input": df}, config_fill)
    out_text = res_fill["dataframe"]["rendered_prompt"].iloc[0]
    assert "[UNKNOWN]" in out_text
    assert "Prior ticket: Login issue." in out_text

    # Mode raise_error
    config_raise = {
        "template": "Hello {{user}}, details: {{unknown_col}}",
        "missing_variable_handling": "raise_error"
    }
    with pytest.raises(ValidationException):
        recipe.execute({"input": df}, config_raise)


def test_prompt_template_manager_empty_dataframe():
    recipe = PromptTemplateManagerRecipe()
    df = pd.DataFrame(columns=["user", "text"])
    config = {"template": "Hello {{user}}", "output_mode": "chat_messages_json"}
    res = recipe.execute({"input": df}, config)
    out_df = res["dataframe"]
    assert len(out_df) == 0
    assert "rendered_prompt" in out_df.columns
    assert "messages_payload" in out_df.columns


# =============================================================================
# 2. CONVERSATION MEMORY TESTS
# =============================================================================

def test_conversation_memory_sliding_window_aggregate():
    recipe = ConversationMemoryRecipe()
    # 6 turns in one session
    df = pd.DataFrame([
        {"session_id": "S1", "turn_id": 1, "role": "user", "message": "Hi, I have a question."},
        {"session_id": "S1", "turn_id": 2, "role": "assistant", "message": "Hello! How can I help?"},
        {"session_id": "S1", "turn_id": 3, "role": "user", "message": "My order #123 is delayed."},
        {"session_id": "S1", "turn_id": 4, "role": "assistant", "message": "Let me check order #123."},
        {"session_id": "S1", "turn_id": 5, "role": "user", "message": "Any updates?"},
        {"session_id": "S1", "turn_id": 6, "role": "assistant", "message": "It will arrive tomorrow."}
    ])
    # Keep last 3 turns
    config = {
        "session_id_column": "session_id",
        "role_column": "role",
        "message_column": "message",
        "order_column": "turn_id",
        "memory_type": "sliding_window",
        "window_size": 3,
        "output_mode": "aggregate_per_session",
        "output_format": "both"
    }
    res = recipe.execute({"input": df}, config)
    out_df = res["dataframe"]
    assert len(out_df) == 1
    assert out_df["retained_turns"].iloc[0] == 3
    assert out_df["total_turns"].iloc[0] == 6

    transcript = out_df["conversation_transcript"].iloc[0]
    assert "Let me check order #123." in transcript
    assert "It will arrive tomorrow." in transcript
    assert "Hi, I have a question." not in transcript  # Slid out of window


def test_conversation_memory_token_buffer():
    recipe = ConversationMemoryRecipe()
    df = pd.DataFrame([
        {"session_id": "S_LONG", "role": "user", "message": "A" * 400},   # ~100 tokens
        {"session_id": "S_LONG", "role": "assistant", "message": "B" * 400},  # ~100 tokens
        {"session_id": "S_LONG", "role": "user", "message": "C" * 400},   # ~100 tokens
        {"session_id": "S_LONG", "role": "assistant", "message": "D" * 400}   # ~100 tokens
    ])
    # Set tight token budget allowing roughly 2 messages (~220 tokens)
    config = {
        "session_id_column": "session_id",
        "memory_type": "token_buffer",
        "max_token_limit": 220,
        "output_mode": "aggregate_per_session"
    }
    res = recipe.execute({"input": df}, config)
    out_df = res["dataframe"]
    assert len(out_df) == 1
    assert out_df["retained_turns"].iloc[0] <= 2


def test_conversation_memory_enrich_each_turn_and_role_normalization():
    recipe = ConversationMemoryRecipe()
    df = pd.DataFrame([
        {"session_id": "S2", "speaker": "human", "text": "Turn 1"},
        {"session_id": "S2", "speaker": "bot", "text": "Turn 2"},
        {"session_id": "S2", "speaker": "client", "text": "Turn 3"}
    ])
    config = {
        "session_id_column": "session_id",
        "role_column": "speaker",
        "message_column": "text",
        "output_mode": "enrich_each_turn",
        "memory_type": "full_history"
    }
    res = recipe.execute({"input": df}, config)
    out_df = res["dataframe"]
    assert len(out_df) == 3
    assert "conversation_transcript" in out_df.columns
    # Check that roles were mapped to User/Assistant
    assert "User: Turn 1" in out_df["conversation_transcript"].iloc[0]
    assert "Assistant: Turn 2" in out_df["conversation_transcript"].iloc[1]


# =============================================================================
# 3. STRUCTURED OUTPUT ENFORCER TESTS
# =============================================================================

def test_structured_output_enforcer_clean_json_unpacking():
    recipe = StructuredOutputEnforcerRecipe()
    df = pd.DataFrame([
        {"raw_llm": '{"intent": "flight_booking", "destination": "Paris", "confidence": 0.95}'},
        {"raw_llm": '{"intent": "hotel_cancel", "destination": "London", "confidence": 0.88}'}
    ])
    config = {
        "source_column": "raw_llm",
        "expected_schema": {
            "properties": {
                "intent": {"type": "string"},
                "destination": {"type": "string"},
                "confidence": {"type": "number"}
            },
            "required": ["intent", "destination"]
        },
        "extraction_mode": "keep_both"
    }
    res = recipe.execute({"input": df}, config)
    out_df = res["dataframe"]
    assert out_df["intent"].tolist() == ["flight_booking", "hotel_cancel"]
    assert out_df["destination"].tolist() == ["Paris", "London"]
    assert out_df["confidence"].tolist() == [0.95, 0.88]
    assert out_df["is_valid_json"].all()
    assert out_df["schema_valid"].all()


def test_structured_output_enforcer_markdown_and_syntax_auto_repair():
    recipe = StructuredOutputEnforcerRecipe()
    # LLM responses with markdown fences, single quotes, trailing commas, missing closing brace
    df = pd.DataFrame([
        # Markdown fence
        {"llm_output": '```json\n{"intent": "refund", "amount": 50}\n```'},
        # Single quotes + trailing comma + Python True
        {"llm_output": "{'intent': 'dispute', 'is_urgent': True, 'amount': 120,}"},
        # Truncated missing brace
        {"llm_output": '{"intent": "support", "amount": 0'}
    ])
    config = {
        "source_column": "llm_output",
        "expected_schema": {
            "properties": {
                "intent": {"type": "string", "default": "unknown"},
                "amount": {"type": "number", "default": 0}
            },
            "required": ["intent"]
        },
        "auto_repair": True,
        "extraction_mode": "unpack_columns"
    }
    res = recipe.execute({"input": df}, config)
    out_df = res["dataframe"]
    assert out_df["intent"].iloc[0] == "refund"
    assert out_df["intent"].iloc[1] == "dispute"
    assert out_df["intent"].iloc[2] == "support"
    assert out_df["amount"].iloc[0] == 50
    assert out_df["amount"].iloc[1] == 120


def test_structured_output_enforcer_regex_fallback_and_diagnostics():
    recipe = StructuredOutputEnforcerRecipe()
    # Totally malformed text response
    df = pd.DataFrame([
        {"raw": "Sure! I found that intent: account_lockout and confidence: 0.99 for this user."},
        {"raw": "I am unable to parse this request."}
    ])
    config = {
        "source_column": "raw",
        "expected_schema": {
            "properties": {
                "intent": {"type": "string", "default": "general"},
                "confidence": {"type": "number", "default": 0.0}
            },
            "required": ["intent"]
        },
        "handle_malformed": "regex_extraction",
        "audit_columns": True
    }
    res = recipe.execute({"input": df}, config)
    out_df = res["dataframe"]
    assert out_df["intent"].iloc[0] == "account_lockout"
    assert out_df["confidence"].iloc[0] == 0.99
    assert out_df["intent"].iloc[1] == "general"  # Default fallback
    assert out_df["is_valid_json"].iloc[0] == False
    assert out_df["is_valid_json"].iloc[1] == False


def test_structured_output_enforcer_empty_dataframe():
    recipe = StructuredOutputEnforcerRecipe()
    df = pd.DataFrame(columns=["llm_output"])
    config = {
        "source_column": "llm_output",
        "expected_schema": {
            "properties": {"action": {"type": "string"}}
        }
    }
    res = recipe.execute({"input": df}, config)
    out_df = res["dataframe"]
    assert len(out_df) == 0
    assert "action" in out_df.columns
