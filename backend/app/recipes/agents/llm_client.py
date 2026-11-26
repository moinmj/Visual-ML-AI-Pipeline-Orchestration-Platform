"""
Shared Live LLM Client with Automatic Key Rotation and Transparent Offline Fallback Diagnostics.

Primary mode: Online live LLM (Groq -> Gemini -> OpenAI, depending on configured keys and provider).
Fallback mode: Returns structured failure reasons (e.g. token/quota exhausted, missing key, timeout)
so recipes can fall back to deterministic local execution while surfacing clear diagnostics.
"""

import os
import base64
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx


def _get_env_or_setting(key: str, default: str = "") -> str:
    val = os.getenv(key)
    if val is not None and str(val).strip():
        return str(val).strip()
    try:
        from backend.app.core.config import settings
        s_val = getattr(settings, key, None)
        if s_val is not None and str(s_val).strip():
            return str(s_val).strip()
    except Exception:
        pass
    return default


def _collect_groq_keys() -> List[str]:
    keys: List[str] = []
    for k in [
        "GROQ_API_KEY",
        "GROQ_API_KEY_1",
        "GROQ_API_KEY_2",
        "GROQ_API_KEY_3",
        "GROQ_API_KEY_4",
        "GROQ_API_KEY_5",
        "GROQ_API_KEY_6",
    ]:
        v = _get_env_or_setting(k)
        if v and v not in keys:
            keys.append(v)
    return keys


def _classify_http_error(status_code: int, body_text: str = "") -> str:
    low = (body_text or "").lower()
    if status_code == 429 or any(t in low for t in ("rate limit", "rate_limit", "quota", "tokens", "resource_exhausted", "exceeded")):
        return f"Token/rate limit exhausted (HTTP {status_code})"
    if status_code in (401, 403) or any(t in low for t in ("invalid api key", "invalid_api_key", "unauthorized", "permission")):
        return f"Invalid or unauthorized API key (HTTP {status_code})"
    if status_code >= 500:
        return f"Provider server error (HTTP {status_code})"
    return f"API error (HTTP {status_code})"


def _classify_exception(exc: Exception) -> str:
    name = type(exc).__name__
    msg = str(exc).lower()
    if "timeout" in name.lower() or "timed out" in msg:
        return f"Request timed out ({name})"
    if "connect" in name.lower() or "dns" in msg or "network" in msg:
        return f"Network unreachable ({name})"
    return f"Request failed ({name}: {str(exc)[:120]})"


def _call_groq(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str],
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> Dict[str, Any]:
    keys = _collect_groq_keys()
    if not keys:
        return {"ok": False, "reason": "Missing GROQ_API_KEY in .env"}

    target_model = model or _get_env_or_setting("ML_PIPELINE_GROQ_MODEL") or _get_env_or_setting("GROQ_MODEL") or "llama-3.3-70b-versatile"
    base_url = _get_env_or_setting("GROQ_BASE_URL", "https://api.groq.com").rstrip("/")
    url = f"{base_url}/openai/v1/chat/completions"

    last_reason = "Groq request failed"
    for idx, api_key in enumerate(keys):
        try:
            resp = httpx.post(
                url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": target_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=timeout,
            )
            if resp.status_code == 200:
                content = (resp.json()["choices"][0]["message"].get("content") or "").strip()
                if content:
                    return {
                        "ok": True,
                        "content": content,
                        "provider_used": "groq",
                        "model_used": target_model,
                    }
                last_reason = "Empty response content (model used all tokens in reasoning)"
                continue
            last_reason = _classify_http_error(resp.status_code, resp.text)
            # Rotate to next key on 429 or 401
            if resp.status_code in (429, 401, 403) and idx < len(keys) - 1:
                continue
            return {"ok": False, "reason": f"Groq: {last_reason}"}
        except Exception as exc:
            last_reason = _classify_exception(exc)
            return {"ok": False, "reason": f"Groq: {last_reason}"}

    return {"ok": False, "reason": f"Groq (all {len(keys)} keys failed): {last_reason}"}


def _call_gemini(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str],
    temperature: float,
    max_tokens: int,
    timeout: float,
    image_path: Optional[str] = None,
) -> Dict[str, Any]:
    api_key = _get_env_or_setting("GEMINI_API_KEY")
    if not api_key:
        return {"ok": False, "reason": "Missing GEMINI_API_KEY in .env"}

    target_model = model or _get_env_or_setting("GEMINI_MODEL") or "gemini-2.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={api_key}"

    parts: List[Dict[str, Any]] = [{"text": f"{system_prompt}\n\n{user_prompt}".strip()}]
    if image_path and not str(image_path).startswith(("http://", "https://")):
        p = Path(str(image_path))
        if p.exists() and p.is_file():
            try:
                mime, _ = mimetypes.guess_type(str(p))
                b64 = base64.b64encode(p.read_bytes()).decode("ascii")
                parts.append({"inlineData": {"mimeType": mime or "image/png", "data": b64}})
            except Exception:
                pass

    try:
        resp = httpx.post(
            url,
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"parts": parts}],
                "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
            },
            timeout=timeout,
        )
        if resp.status_code == 200:
            candidates = resp.json().get("candidates") or []
            if candidates:
                c_parts = candidates[0].get("content", {}).get("parts") or []
                text = "".join(p.get("text", "") for p in c_parts).strip()
                if text:
                    return {
                        "ok": True,
                        "content": text,
                        "provider_used": "gemini",
                        "model_used": target_model,
                    }
            return {"ok": False, "reason": "Gemini: Empty response candidates"}
        return {"ok": False, "reason": f"Gemini: {_classify_http_error(resp.status_code, resp.text)}"}
    except Exception as exc:
        return {"ok": False, "reason": f"Gemini: {_classify_exception(exc)}"}


def _call_openai(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str],
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> Dict[str, Any]:
    api_key = _get_env_or_setting("OPENAI_API_KEY")
    if not api_key:
        return {"ok": False, "reason": "Missing OPENAI_API_KEY in .env"}

    target_model = model or _get_env_or_setting("OPENAI_MODEL") or "gpt-4o-mini"
    try:
        resp = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": target_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=timeout,
        )
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"]
            return {
                "ok": True,
                "content": content,
                "provider_used": "openai",
                "model_used": target_model,
            }
        return {"ok": False, "reason": f"OpenAI: {_classify_http_error(resp.status_code, resp.text)}"}
    except Exception as exc:
        return {"ok": False, "reason": f"OpenAI: {_classify_exception(exc)}"}


def call_live_llm(
    system_prompt: str,
    user_prompt: str,
    provider: str = "auto",
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 512,
    timeout: float = 15.0,
    image_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Calls an online live LLM provider with automatic fallback diagnostics.

    Returns:
        {
            "content": Optional[str],
            "execution_mode": "ONLINE_LIVE" | "OFFLINE_FALLBACK",
            "provider_used": str,
            "model_used": str,
            "fallback_reason": str,
        }
    """
    prov = (provider or "auto").strip().lower()
    if prov in ("offline", "offline_grounded", "local", "local_fast"):
        return {
            "content": None,
            "execution_mode": "OFFLINE_FALLBACK",
            "provider_used": "offline_fallback",
            "model_used": "deterministic_local",
            "fallback_reason": "Offline mode explicitly selected",
        }

    # Determine ordered providers to try
    if prov == "auto":
        order = ["groq", "gemini", "openai"] if not image_path else ["gemini", "groq", "openai"]
    elif prov in ("groq", "gemini", "openai"):
        order = [prov]
        # In auto-resilient mode, only stick to requested provider if explicitly chosen,
        # so error reason accurately reflects that provider.
    else:
        order = ["groq", "gemini", "openai"]

    errors: List[str] = []
    for p in order:
        if p == "groq":
            res = _call_groq(system_prompt, user_prompt, model if prov == "groq" else None, temperature, max_tokens, timeout)
        elif p == "gemini":
            res = _call_gemini(system_prompt, user_prompt, model if prov == "gemini" else None, temperature, max_tokens, timeout, image_path=image_path)
        else:
            res = _call_openai(system_prompt, user_prompt, model if prov == "openai" else None, temperature, max_tokens, timeout)

        if res.get("ok"):
            return {
                "content": res["content"],
                "execution_mode": "ONLINE_LIVE",
                "provider_used": res["provider_used"],
                "model_used": res["model_used"],
                "fallback_reason": "None (Live Online LLM)",
            }
        errors.append(res.get("reason", f"{p} failed"))

    return {
        "content": None,
        "execution_mode": "OFFLINE_FALLBACK",
        "provider_used": "offline_fallback",
        "model_used": "deterministic_local",
        "fallback_reason": " | ".join(errors) if errors else "No online LLM provider succeeded",
    }


def llm_provider_schema_props() -> Dict[str, Any]:
    """Shared schema properties for selecting live LLM provider and model."""
    return {
        "llm_provider": {
            "type": "string",
            "title": "LLM Provider",
            "input_type": "dropdown",
            "enum": ["auto", "groq", "gemini", "openai", "offline"],
            "default": "auto",
            "description": "Online LLM provider ('auto' tries Groq -> Gemini -> OpenAI). Falls back to offline mode with reason if unavailable."
        },
        "llm_model": {
            "type": "string",
            "title": "LLM Model (Optional)",
            "default": "",
            "description": "Optional model ID override (e.g. llama-3.3-70b-versatile, gemini-2.5-flash, gpt-4o-mini). Leave blank for default."
        },
    }


def worker_description_prop(default_desc: str) -> Dict[str, Any]:
    """Shared schema property describing a worker agent's role to an upstream Hub LLM."""
    return {
        "agent_description": {
            "type": "string",
            "title": "Agent Role Description (For Hub Routing)",
            "default": default_desc,
            "description": "Describes what this agent does so a connected Central LLM/Orchestrator knows when to route tasks to it."
        }
    }
