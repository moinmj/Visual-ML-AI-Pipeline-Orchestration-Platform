"""Unit tests for shared live LLM client and transparent fallback diagnostics."""

from unittest.mock import MagicMock, patch

from backend.app.recipes.agents.llm_client import call_live_llm
from backend.app.recipes.agents.rag_pipeline import RagPipelineRecipe


def test_call_live_llm_explicit_offline():
    res = call_live_llm(
        system_prompt="You are helpful.",
        user_prompt="Hello",
        provider="offline_grounded",
    )
    assert res["content"] is None
    assert res["execution_mode"] == "OFFLINE_FALLBACK"
    assert res["provider_used"] == "offline_fallback"
    assert "explicitly selected" in res["fallback_reason"].lower()


def test_call_live_llm_reports_token_exhausted_429():
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.text = '{"error": {"message": "Rate limit / tokens exhausted"}}'

    with patch("backend.app.recipes.agents.llm_client.httpx.post", return_value=mock_resp), \
         patch("backend.app.recipes.agents.llm_client._collect_groq_keys", return_value=["gsk_test_1"]):
        res = call_live_llm(
            system_prompt="sys",
            user_prompt="hi",
            provider="groq",
        )

    assert res["content"] is None
    assert res["execution_mode"] == "OFFLINE_FALLBACK"
    assert "429" in res["fallback_reason"]
    assert "exhausted" in res["fallback_reason"].lower()


def test_call_live_llm_online_success():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"choices": [{"message": {"content": "Live answer"}}]}

    with patch("backend.app.recipes.agents.llm_client.httpx.post", return_value=mock_resp), \
         patch("backend.app.recipes.agents.llm_client._collect_groq_keys", return_value=["gsk_test_1"]):
        res = call_live_llm(
            system_prompt="sys",
            user_prompt="hi",
            provider="groq",
        )

    assert res["content"] == "Live answer"
    assert res["execution_mode"] == "ONLINE_LIVE"
    assert res["provider_used"] == "groq"
    assert res["fallback_reason"] == "None (Live Online LLM)"


def test_rag_pipeline_exposes_fallback_diagnostics():
    recipe = RagPipelineRecipe()
    res = recipe.execute(
        inputs={"dataframe": [{"id": "d1", "text": "Refunds are processed within 5 business days."}]},
        config={"query": "How long do refunds take?", "llm_provider": "offline_grounded", "top_k": 1},
    )
    df = res["dataframe"]
    assert df.iloc[0]["execution_mode"] == "OFFLINE_FALLBACK"
    assert df.iloc[0]["llm_provider_used"] == "offline_grounded"
    assert "offline" in df.iloc[0]["fallback_reason"].lower()
    assert res["summary"]["execution_mode"] == "OFFLINE_FALLBACK"


def test_voice_ai_agent_live_and_fallback():
    from backend.app.recipes.agents.voice_ai_agent import VoiceAiAgentRecipe

    recipe = VoiceAiAgentRecipe()

    # 1. Missing API key falls back transparently
    res_fb = recipe.execute(
        inputs={"dataframe": [{"phone": "+14155550199", "name": "Alice"}]},
        config={"provider": "vapi", "api_key": ""},
        context={"hub_instruction": "Qualify Alice for 50-seat plan"},
    )
    assert res_fb["dataframe"].iloc[0]["execution_mode"] == "OFFLINE_FALLBACK"
    assert "VAPI_API_KEY" in res_fb["dataframe"].iloc[0]["fallback_reason"]
    assert "Qualify Alice for 50-seat plan" in res_fb["dataframe"].iloc[0]["call_summary"]

    # 2. Mocked live Vapi API call succeeds
    mock_resp = MagicMock()
    mock_resp.status_code = 201
    mock_resp.json.return_value = {"id": "call_vapi_123", "status": "queued"}
    with patch("backend.app.recipes.agents.voice_ai_agent.httpx.post", return_value=mock_resp):
        res_live = recipe.execute(
            inputs={"dataframe": [{"phone": "+14155550199", "name": "Alice"}]},
            config={"provider": "vapi", "api_key": "vapi_test_key"},
        )
    assert res_live["dataframe"].iloc[0]["execution_mode"] == "ONLINE_LIVE"
    assert res_live["dataframe"].iloc[0]["call_id"] == "call_vapi_123"
    assert res_live["dataframe"].iloc[0]["provider_used"] == "vapi"
