"""
Voice AI Agent (Vapi / Bland / Retell) Recipe.
(Agents & LLM Orchestration -> Core Agents)

Initiates live two-way conversational AI phone calls via Vapi, Bland AI, or Retell AI,
with automatic offline simulation fallback and transparent execution diagnostics.
"""

import os
import json
import time
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd

from backend.app.recipes.base.recipe import BaseRecipe
from backend.app.recipes.agents.llm_client import (
    _get_env_or_setting,
    _classify_http_error,
    _classify_exception,
    worker_description_prop,
)


class VoiceAiAgentRecipe(BaseRecipe):
    recipe_id = "voice_ai_agent"
    name = "Voice AI Agent (Vapi / Bland / Retell)"
    version = "1.0.0"
    category = "Agents & LLM Orchestration"
    group = "Agents & LLM Orchestration"
    subgroup = "Core Agents"
    media_type = "agents"
    description = "Places two-way conversational AI phone calls via Vapi, Bland AI, or Retell AI with call status and transcript output."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "provider": {
                    "type": "string",
                    "title": "Voice AI Provider",
                    "input_type": "dropdown",
                    "enum": ["vapi", "bland", "retell", "simulation"],
                    "default": "vapi",
                    "description": "External Voice AI provider ('vapi', 'bland', 'retell') or 'simulation' for offline demo."
                },
                "api_key": {
                    "type": "string",
                    "title": "API Key (Optional if set in .env)",
                    "input_type": "password",
                    "default": "",
                    "description": "Provider API key. Falls back to VAPI_API_KEY, BLAND_API_KEY, or RETELL_API_KEY in .env."
                },
                "assistant_id": {
                    "type": "string",
                    "title": "Assistant / Agent ID (Vapi / Retell)",
                    "default": "",
                    "description": "Assistant ID (Vapi), Agent ID (Retell), or Phone Number ID."
                },
                "phone_number_id": {
                    "type": "string",
                    "title": "Caller Phone / Phone Number ID",
                    "default": "",
                    "description": "Outbound caller number (Bland/Retell) or Vapi phoneNumberId."
                },
                "phone_column": {
                    "type": "string",
                    "title": "Lead Phone Column",
                    "default": "phone",
                    "description": "Column containing lead phone numbers in E.164 format (e.g. +14155551234)."
                },
                "default_phone": {
                    "type": "string",
                    "title": "Fallback Phone Number",
                    "default": "",
                    "description": "Used if phone_column is absent from input."
                },
                "task_column": {
                    "type": "string",
                    "title": "Call Script / Instruction Column",
                    "default": "",
                    "description": "Optional column holding per-lead call instructions."
                },
                "system_prompt": {
                    "type": "string",
                    "title": "Voice Agent Prompt / Goal",
                    "input_type": "textarea",
                    "default": "You are a friendly sales qualification agent. Confirm budget, timeline, and offer to book a viewing.",
                    "description": "Conversational instructions sent to the Voice AI call (overridden by Hub instruction if invoked in a cyclic workflow)."
                },
                "first_message": {
                    "type": "string",
                    "title": "Opening Greeting",
                    "default": "Hi! Calling to follow up on your inquiry. Do you have a quick minute?",
                    "description": "First sentence spoken when the lead answers."
                },
                "language": {
                    "type": "string",
                    "title": "Language",
                    "default": "en",
                    "description": "Spoken language code (e.g. en, ar, es)."
                },
                **worker_description_prop("Calls leads via Voice AI (Vapi/Bland/Retell) to qualify them and return call outcome & transcript."),
            },
            "required": ["provider"]
        }

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        prov = str(config.get("provider", "vapi")).strip().lower()
        if prov not in ("vapi", "bland", "retell", "simulation"):
            return [f"Unsupported Voice AI provider '{prov}'. Must be 'vapi', 'bland', 'retell', or 'simulation'."]
        return []

    def _resolve_api_key(self, provider: str, config: Dict[str, Any]) -> str:
        explicit = str(config.get("api_key") or "").strip()
        if explicit:
            return explicit
        env_map = {
            "vapi": "VAPI_API_KEY",
            "bland": "BLAND_API_KEY",
            "retell": "RETELL_API_KEY",
        }
        return _get_env_or_setting(env_map.get(provider, ""))

    def _dispatch_call(
        self,
        provider: str,
        api_key: str,
        to_number: str,
        prompt: str,
        first_message: str,
        assistant_id: str,
        phone_number_id: str,
        language: str,
    ) -> Dict[str, Any]:
        if provider == "simulation":
            return {"ok": False, "reason": "Offline simulation mode explicitly selected"}
        if not api_key:
            return {"ok": False, "reason": f"Missing {provider.upper()}_API_KEY in .env or node config"}
        if not to_number:
            return {"ok": False, "reason": "Missing destination phone number"}

        try:
            if provider == "vapi":
                payload: Dict[str, Any] = {
                    "customer": {"number": to_number},
                }
                if phone_number_id:
                    payload["phoneNumberId"] = phone_number_id
                if assistant_id:
                    payload["assistantId"] = assistant_id
                else:
                    payload["assistant"] = {
                        "firstMessage": first_message,
                        "model": {"provider": "openai", "model": "gpt-4o-mini", "messages": [{"role": "system", "content": prompt}]},
                    }
                resp = httpx.post(
                    "https://api.vapi.ai/call",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=15.0,
                )
                if resp.status_code in (200, 201):
                    data = resp.json()
                    return {"ok": True, "call_id": data.get("id", ""), "call_status": data.get("status", "queued"), "raw": data}
                return {"ok": False, "reason": f"Vapi: {_classify_http_error(resp.status_code, resp.text)}"}

            if provider == "bland":
                payload = {
                    "phone_number": to_number,
                    "task": prompt,
                    "first_sentence": first_message,
                    "language": language,
                }
                if phone_number_id:
                    payload["from"] = phone_number_id
                resp = httpx.post(
                    "https://api.bland.ai/v1/calls",
                    headers={"authorization": api_key, "Content-Type": "application/json"},
                    json=payload,
                    timeout=15.0,
                )
                if resp.status_code in (200, 201):
                    data = resp.json()
                    return {"ok": True, "call_id": data.get("call_id", ""), "call_status": data.get("status", "queued"), "raw": data}
                return {"ok": False, "reason": f"Bland: {_classify_http_error(resp.status_code, resp.text)}"}

            if provider == "retell":
                payload = {
                    "to_number": to_number,
                    "from_number": phone_number_id,
                    "override_agent_id": assistant_id or None,
                    "retell_llm_dynamic_variables": {"task": prompt, "first_message": first_message},
                }
                payload = {k: v for k, v in payload.items() if v is not None and v != ""}
                resp = httpx.post(
                    "https://api.retellai.com/v2/create-phone-call",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=15.0,
                )
                if resp.status_code in (200, 201):
                    data = resp.json()
                    return {"ok": True, "call_id": data.get("call_id", ""), "call_status": data.get("call_status", "registered"), "raw": data}
                return {"ok": False, "reason": f"Retell: {_classify_http_error(resp.status_code, resp.text)}"}

        except Exception as exc:
            return {"ok": False, "reason": f"{provider.capitalize()}: {_classify_exception(exc)}"}

        return {"ok": False, "reason": f"Unsupported provider: {provider}"}

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        start_time = time.time()
        raw_input = next((inputs[k] for k in ("dataframe", "input", "df", "data") if k in inputs and inputs[k] is not None), None)

        if isinstance(raw_input, pd.DataFrame):
            df = raw_input.copy()
        elif isinstance(raw_input, list):
            df = pd.DataFrame(raw_input)
        elif isinstance(raw_input, dict) and "records" in raw_input:
            df = pd.DataFrame(raw_input["records"])
        elif isinstance(raw_input, dict):
            df = pd.DataFrame([raw_input])
        else:
            df = pd.DataFrame([{"phone": config.get("default_phone") or "+14155550100", "name": "Demo Lead"}])

        if df.empty:
            out_df = pd.DataFrame(columns=["call_id", "call_status", "call_answered", "call_summary", "execution_mode", "provider_used", "fallback_reason"])
            return {"dataframe": out_df, "status": "success", "summary": {"total_calls": 0}}

        provider = str(config.get("provider") or "vapi").strip().lower()
        api_key = self._resolve_api_key(provider, config)
        assistant_id = str(config.get("assistant_id") or "").strip()
        phone_number_id = str(config.get("phone_number_id") or "").strip()
        phone_col = str(config.get("phone_column") or "phone").strip()
        default_phone = str(config.get("default_phone") or "").strip()
        task_col = str(config.get("task_column") or "").strip()
        base_prompt = str(config.get("system_prompt") or "Qualify lead budget and timeline.").strip()
        first_message = str(config.get("first_message") or "Hi! Calling to follow up on your inquiry.").strip()
        language = str(config.get("language") or "en").strip()
        hub_instr = str(context.get("hub_instruction") or "").strip() if isinstance(context, dict) else ""

        call_ids, statuses, answered_flags, summaries = [], [], [], []
        exec_modes, providers_used, fallback_reasons = [], [], []

        for idx, row in df.iterrows():
            to_num = str(row.get(phone_col) or default_phone).strip() if phone_col in df.columns else default_phone
            if to_num in ("nan", "None"):
                to_num = default_phone
            row_prompt = str(row.get(task_col) or "").strip() if (task_col and task_col in df.columns) else ""
            active_prompt = hub_instr or row_prompt or base_prompt

            res = self._dispatch_call(
                provider=provider,
                api_key=api_key,
                to_number=to_num,
                prompt=active_prompt,
                first_message=first_message,
                assistant_id=assistant_id,
                phone_number_id=phone_number_id,
                language=language,
            )
            if res.get("ok"):
                call_ids.append(res["call_id"])
                statuses.append(res["call_status"])
                answered_flags.append(True)
                summaries.append(f"Dispatched live {provider} call ({res['call_id']}) to {to_num}.")
                exec_modes.append("ONLINE_LIVE")
                providers_used.append(provider)
                fallback_reasons.append("None (Live Voice AI)")
            else:
                sim_id = f"sim_{provider}_{idx+1}"
                call_ids.append(sim_id)
                statuses.append("simulated_completed")
                answered_flags.append(idx % 2 == 0)
                summaries.append(
                    f"[Simulated {provider} call to {to_num}] Answered={idx % 2 == 0}. Prompt: {active_prompt[:120]}"
                )
                exec_modes.append("OFFLINE_FALLBACK")
                providers_used.append("offline_simulation")
                fallback_reasons.append(res.get("reason", "Voice AI fallback"))

        out_df = df.copy().reset_index(drop=True)
        out_df["call_id"] = call_ids
        out_df["call_status"] = statuses
        out_df["call_answered"] = answered_flags
        out_df["call_summary"] = summaries
        out_df["execution_mode"] = exec_modes
        out_df["provider_used"] = providers_used
        out_df["fallback_reason"] = fallback_reasons

        duration_ms = round((time.time() - start_time) * 1000, 2)
        return {
            "dataframe": out_df,
            "status": "success",
            "answer": summaries[0] if summaries else "",
            "summary": {
                "total_calls": len(out_df),
                "execution_mode": exec_modes[0] if exec_modes else "none",
                "provider_used": providers_used[0] if providers_used else "none",
                "fallback_reason": fallback_reasons[0] if fallback_reasons else "none",
                "duration_ms": duration_ms,
            },
        }
