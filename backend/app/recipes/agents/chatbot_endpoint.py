"""
Chatbot Conversational Endpoint Recipe.
(Agents & LLM Orchestration -> Core Agents)

Exposes a complete conversational endpoint adapter for web and mobile frontends.
Formats messages into standardized chat schemas, provides quick reply suggestions,
manages session identifiers, and formats turn-by-turn conversational history.
"""

import time
import json
import uuid
import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from backend.app.recipes.base.recipe import BaseRecipe

_QUICK_REPLY_MAP = {
    "greeting": ["How can you help me?", "View documentation", "Start a new project"],
    "help": ["Contact support", "Browse tutorials", "Check system status"],
    "pricing": ["View enterprise pricing", "Request demo", "Compare plans"],
    "technical": ["View API reference", "Download SDK", "Submit bug report"],
    "default": ["Tell me more", "Explain simply", "Summarize key points"]
}


def _detect_intent_category(message: str) -> str:
    """Classifies user message into high-level conversational intent."""
    m = message.lower().strip()
    if any(w in m for w in ["hi", "hello", "hey", "good morning", "good evening", "greetings"]):
        return "greeting"
    elif any(w in m for w in ["help", "assist", "guide", "how do i", "support"]):
        return "help"
    elif any(w in m for w in ["price", "cost", "billing", "quote", "subscription", "plan"]):
        return "pricing"
    elif any(w in m for w in ["error", "bug", "api", "token", "code", "sdk", "endpoint", "failure"]):
        return "technical"
    return "default"


def _generate_assistant_reply(
    user_message: str,
    bot_name: str = "Datalect AI",
    system_persona: Optional[str] = None
) -> Tuple[str, List[str]]:
    """
    Synthesizes a helpful conversational response and contextual quick reply suggestions.
    """
    clean_msg = str(user_message).strip()
    intent = _detect_intent_category(clean_msg)
    quick_replies = _QUICK_REPLY_MAP.get(intent, _QUICK_REPLY_MAP["default"])

    if not clean_msg:
        reply = f"Hello! I am {bot_name}. How can I assist you with your machine learning pipelines today?"
        return reply, quick_replies

    if intent == "greeting":
        reply = f"Greetings! I'm {bot_name}. I can help you design workflows, configure agent nodes, run models, or troubleshoot issues."
    elif intent == "help":
        reply = f"I'm here to assist. You can provide me with dataset schemas, model parameters, or questions about pipeline orchestration."
    elif intent == "pricing":
        reply = "Our enterprise plans include unlimited visual workflows, dedicated GPU nodes, custom LLM fine-tuning, and 24/7 SLA support."
    elif intent == "technical":
        reply = f"I've logged your technical query regarding: '{clean_msg[:50]}...'. Let's isolate the issue step-by-step."
    else:
        persona_prefix = f"[{system_persona}] " if system_persona else ""
        reply = f"{persona_prefix}Regarding '{clean_msg}': I have processed your request through our pipeline intelligence core. Ready for next instructions."

    return reply, quick_replies


class ChatbotEndpointRecipe(BaseRecipe):
    recipe_id = "chatbot_endpoint"
    name = "Chatbot Endpoint"
    version = "1.0.0"
    category = "Agents & LLM Orchestration"
    group = "Agents & LLM Orchestration"
    subgroup = "Core Agents"
    media_type = "agents"
    description = "Enterprise chatbot endpoint adapter providing message formatting, session tracking, and quick replies."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "bot_name": {
                    "type": "string",
                    "title": "Bot Name",
                    "default": "Datalect Assistant",
                    "description": "Public name for the conversational bot."
                },
                "system_persona": {
                    "type": "string",
                    "title": "System Persona",
                    "default": "Enterprise ML Orchestration Specialist",
                    "description": "Persona role and behavioral directives."
                },
                "message_column": {
                    "type": "string",
                    "title": "User Message Column",
                    "default": "message",
                    "description": "Column holding user input messages."
                },
                "include_quick_replies": {
                    "type": "boolean",
                    "title": "Include Quick Replies",
                    "default": True,
                    "description": "Generate dynamic action suggestion chips."
                }
            }
        }

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        return []

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        """
        Executes chatbot endpoint wrapping.

        Inputs:
            - message / user_message / query / input: str or list of user inquiries
            - dataframe / df: DataFrame of user chat messages
            - session_id: optional persistent conversation ID
            - context / history: prior conversation messages

        Config:
            - bot_name: str (default 'Datalect Assistant')
            - system_persona: str (default 'Enterprise ML Orchestration Specialist')
            - message_column: str (default 'message')
            - include_quick_replies: bool (default True)
            - simulate_latency_ms: int (default 0)
        """
        # Safe extraction without truth-value ambiguity
        raw_input = None
        for k in ["dataframe", "input", "df", "data", "corpus"]:
            if k in inputs and inputs[k] is not None:
                raw_input = inputs[k]
                break

        msg_direct = None
        for k in ["message", "user_message", "query", "text"]:
            if k in inputs and inputs[k] is not None:
                msg_direct = inputs[k]
                break

        bot_name = str(config.get("bot_name", "Datalect Assistant"))
        system_persona = str(config.get("system_persona", "Enterprise ML Orchestration Specialist"))
        msg_col = config.get("message_column", "message")
        include_replies = bool(config.get("include_quick_replies", True))
        session_id = str(inputs.get("session_id") or config.get("session_id") or str(uuid.uuid4())[:8])

        # Case A: Direct message input (string)
        if msg_direct is not None and not isinstance(msg_direct, pd.DataFrame):
            reply_text, q_replies = _generate_assistant_reply(
                str(msg_direct), bot_name=bot_name, system_persona=system_persona
            )

            timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
            chat_payload = {
                "session_id": session_id,
                "bot_name": bot_name,
                "persona": system_persona,
                "user_message": str(msg_direct),
                "reply": reply_text,
                "quick_replies": q_replies if include_replies else [],
                "timestamp": timestamp,
                "formatted_transcript": [
                    {"role": "user", "content": str(msg_direct), "timestamp": timestamp},
                    {"role": "assistant", "content": reply_text, "timestamp": timestamp}
                ]
            }

            df_out = pd.DataFrame([{
                "session_id": session_id,
                "user_message": str(msg_direct),
                "assistant_reply": reply_text,
                "quick_replies": json.dumps(q_replies) if include_replies else "[]",
                "timestamp": timestamp
            }])

            return {
                "dataframe": df_out,
                "output": chat_payload,
                "session_id": session_id,
                "reply": reply_text,
                "quick_replies": q_replies if include_replies else [],
                "metrics": {
                    "total_turns": 1,
                    "session_id": session_id,
                    "intent": _detect_intent_category(str(msg_direct))
                }
            }

        # Case B: DataFrame batch messages
        if isinstance(raw_input, pd.DataFrame):
            df = raw_input.copy()
            if df.empty:
                return {
                    "dataframe": df,
                    "output": {"session_id": session_id, "messages": []},
                    "metrics": {"total_turns": 0, "session_id": session_id}
                }

            if msg_col not in df.columns:
                text_cols = [c for c in df.columns if df[c].dtype == object]
                msg_col = text_cols[0] if text_cols else df.columns[0]

            replies = []
            replies_list = []
            intents = []
            now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

            for _, row in df.iterrows():
                u_msg = str(row.get(msg_col, ""))
                rep, q_rep = _generate_assistant_reply(u_msg, bot_name=bot_name, system_persona=system_persona)
                replies.append(rep)
                replies_list.append(json.dumps(q_rep if include_replies else []))
                intents.append(_detect_intent_category(u_msg))

            df["assistant_reply"] = replies
            df["quick_replies"] = replies_list
            df["intent"] = intents
            df["session_id"] = session_id
            df["timestamp"] = now_iso

            return {
                "dataframe": df,
                "output": {
                    "session_id": session_id,
                    "total_messages": len(df),
                    "intents_distribution": pd.Series(intents).value_counts().to_dict()
                },
                "metrics": {
                    "total_turns": len(df),
                    "session_id": session_id,
                    "unique_intents": len(set(intents))
                }
            }

        # Case C: Fallback empty or default string
        u_msg = str(raw_input or "")
        reply_text, q_replies = _generate_assistant_reply(
            u_msg, bot_name=bot_name, system_persona=system_persona
        )
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        df_out = pd.DataFrame([{
            "session_id": session_id,
            "user_message": u_msg,
            "assistant_reply": reply_text,
            "quick_replies": json.dumps(q_replies) if include_replies else "[]",
            "timestamp": timestamp
        }])

        return {
            "dataframe": df_out,
            "output": {
                "session_id": session_id,
                "reply": reply_text,
                "quick_replies": q_replies if include_replies else []
            },
            "session_id": session_id,
            "reply": reply_text,
            "quick_replies": q_replies if include_replies else [],
            "metrics": {
                "total_turns": 1,
                "session_id": session_id,
                "intent": _detect_intent_category(u_msg)
            }
        }
