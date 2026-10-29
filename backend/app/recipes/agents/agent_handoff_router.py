"""
Agent-to-Agent Handoff Router Recipe.
(Agents & LLM Orchestration -> Core Agents)

Analyzes conversational intent and dynamically routes dialogues between
specialized enterprise agents (Billing, Technical Support, Sales, Human Escalation).
"""

import re
import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from backend.app.recipes.base.recipe import BaseRecipe

_DEFAULT_AGENT_DOMAINS = [
    {
        "id": "human_escalation",
        "name": "Human Supervisor & Escalations",
        "keywords": ["human", "agent", "representative", "manager", "supervisor", "lawyer", "complaint", "sue", "unacceptable"],
        "weight": 1.2
    },
    {
        "id": "billing_agent",
        "name": "Billing & Refund Specialist",
        "keywords": ["refund", "billing", "invoice", "charge", "receipt", "payment", "credit card", "subscription", "price", "cost"],
        "weight": 1.0
    },
    {
        "id": "technical_support_agent",
        "name": "Technical & API Support",
        "keywords": ["error", "bug", "crash", "api", "token", "timeout", "exception", "broken", "traceback", "status 500", "connect"],
        "weight": 1.0
    },
    {
        "id": "sales_onboarding_agent",
        "name": "Enterprise Sales & Onboarding",
        "keywords": ["demo", "trial", "purchase", "enterprise plan", "quote", "sales", "contract", "seats", "upgrade"],
        "weight": 1.0
    }
]


def _route_message(
    message: str,
    agent_domains: List[Dict[str, Any]],
    fallback_id: str = "general_assistant_agent"
) -> Tuple[str, str, str, float]:
    """
    Evaluates message keywords and semantic intent against agent domains.
    Returns (agent_id, agent_name, rationale, confidence_score).
    """
    clean_msg = str(message).lower()
    if not clean_msg.strip():
        return fallback_id, "General Assistant", "Empty inquiry: assigned to fallback", 0.30

    best_agent = None
    best_score = 0.0
    matching_keywords = []

    for domain in agent_domains:
        d_id = domain.get("id", "agent")
        d_name = domain.get("name", d_id)
        keywords = domain.get("keywords", [])
        weight = float(domain.get("weight", 1.0))

        matches = [kw for kw in keywords if re.search(r"\b" + re.escape(kw) + r"\b", clean_msg)]
        if matches:
            score = min(1.0, (len(matches) * 0.35 + 0.30) * weight)
            if score > best_score:
                best_score = score
                best_agent = domain
                matching_keywords = matches

    if best_agent and best_score >= 0.40:
        a_id = best_agent.get("id")
        a_name = best_agent.get("name")
        reason = f"Intent matches {len(matching_keywords)} trigger keyword(s): {', '.join(matching_keywords)}"
        return a_id, a_name, reason, round(best_score, 3)

    return fallback_id, "General Assistant", "No specialized domain threshold exceeded: assigned to fallback", 0.50


class AgentHandoffRouterRecipe(BaseRecipe):
    recipe_id = "agent_handoff_router"
    name = "Agent-to-Agent Handoff Router"
    version = "1.0.0"
    category = "Agents & LLM Orchestration"
    group = "Agents & LLM Orchestration"
    subgroup = "Core Agents"
    media_type = "agents"
    description = "Routes a conversation or task dynamically between specialised agents based on intent and domain signals."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "message_column": {
                    "type": "string",
                    "title": "Message / Dialogue Column",
                    "description": "Column holding the message to evaluate. Auto-detected if empty."
                },
                "fallback_agent_id": {
                    "type": "string",
                    "title": "Fallback Agent ID",
                    "default": "general_assistant_agent",
                    "description": "Destination agent when no specialized domain criteria is met."
                },
                "output_agent_column": {
                    "type": "string",
                    "title": "Output Agent Column",
                    "default": "assigned_agent_id",
                    "description": "Target column name storing the routed agent ID."
                }
            },
            "required": ["fallback_agent_id"]
        }

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        return []

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        start_time = time.time()

        raw_input = None
        for k in ["dataframe", "input", "df", "data"]:
            if k in inputs and inputs[k] is not None:
                raw_input = inputs[k]
                break

        if raw_input is None and inputs:
            direct_msg = None
            for mk in ["message", "query", "prompt", "text", "inquiry"]:
                if mk in inputs and inputs[mk] is not None:
                    direct_msg = inputs[mk]
                    break
            if direct_msg is not None:
                df = pd.DataFrame([{"message": str(direct_msg)}])
            else:
                df = pd.DataFrame()
        elif isinstance(raw_input, pd.DataFrame):
            df = raw_input.copy()
        elif isinstance(raw_input, list):
            df = pd.DataFrame(raw_input)
        elif isinstance(raw_input, dict) and "records" in raw_input:
            df = pd.DataFrame(raw_input["records"])
        elif isinstance(raw_input, dict):
            df = pd.DataFrame([raw_input])
        else:
            df = pd.DataFrame([{"message": str(raw_input or "")}])

        msg_col = config.get("message_column")
        fallback_id = str(config.get("fallback_agent_id") or "general_assistant_agent").strip()
        output_col = str(config.get("output_agent_column") or "assigned_agent_id").strip()

        # Edge case: Empty DataFrame
        if df.empty:
            out_df = pd.DataFrame(columns=[output_col, "routed_agent_id", "routing_confidence", "assigned_agent_name", "handoff_reason", "confidence_score", "handoff_payload"])
            return {
                "dataframe": out_df,
                "status": "success",
                "routed_agent_id": fallback_id,
                "confidence_score": 0.0,
                "summary": {"total_routed": 0, "status": "empty"}
            }

        # Resolve column
        if not msg_col or str(msg_col).strip() in ["", "-- Select Column --", "(None)"]:
            candidates = ["message", "query", "prompt", "text", "inquiry", "input"]
            found = [c for c in candidates if c in df.columns]
            msg_col = found[0] if found else df.columns[0]
        else:
            msg_col = str(msg_col).strip()

        if msg_col not in df.columns:
            df[msg_col] = ""

        agent_ids: List[str] = []
        agent_names: List[str] = []
        reasons: List[str] = []
        confidences: List[float] = []
        payloads: List[str] = []

        agent_counts: Dict[str, int] = {}

        for i in range(len(df)):
            row = df.iloc[i]
            msg_val = str(row[msg_col]) if pd.notna(row[msg_col]) else ""
            a_id, a_name, reason, conf = _route_message(msg_val, _DEFAULT_AGENT_DOMAINS, fallback_id)

            agent_ids.append(a_id)
            agent_names.append(a_name)
            reasons.append(reason)
            confidences.append(conf)

            payload = {
                "destination_agent": a_id,
                "origin_message": msg_val,
                "routing_confidence": conf,
                "session_attributes": row.to_dict()
            }
            payloads.append(json.dumps(payload))
            agent_counts[a_id] = agent_counts.get(a_id, 0) + 1

        out_df = df.copy().reset_index(drop=True)
        out_df[output_col] = agent_ids
        out_df["routed_agent_id"] = agent_ids
        out_df["routing_confidence"] = confidences
        out_df["assigned_agent_name"] = agent_names
        out_df["handoff_reason"] = reasons
        out_df["confidence_score"] = confidences
        out_df["handoff_payload"] = payloads

        duration_ms = round((time.time() - start_time) * 1000, 2)
        summary = {
            "total_routed": len(out_df),
            "distribution": agent_counts,
            "duration_ms": duration_ms
        }

        first_id = agent_ids[0] if agent_ids else fallback_id
        first_conf = confidences[0] if confidences else 0.0

        return {
            "dataframe": out_df,
            "status": "success",
            "routed_agent_id": first_id,
            "confidence_score": first_conf,
            "summary": summary
        }
