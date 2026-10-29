"""
Conversational Memory / State Store Recipe.
Group: Agents & LLM Orchestration
Sub-group: Prompt & Memory Management
Category: agents

Manages, stores, and compiles multi-turn dialogue history, session buffers,
and state contexts across conversational agents and LLM pipelines.
"""
import json
import logging
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort
from backend.app.core.exceptions import ValidationException

logger = logging.getLogger(__name__)


class ConversationMemoryRecipe(BaseRecipe):
    recipe_id = "conversation_memory"
    name = "Conversational Memory / State Store"
    version = "1.0.0"
    category = "agents"
    group = "Agents & LLM Orchestration"
    subgroup = "Prompt & Memory Management"
    description = "Manages, stores, and compiles multi-turn conversational history with sliding window, token buffer, and session state tracking."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="input",
            label="Input Table",
            type="dataframe",
            required=True,
            max_connections=1,
            description="Dataset containing conversation turns (session_id, role, message)"
        )
    ]

    outputs = [
        RecipePort(
            id="dataframe",
            label="Memory Table",
            type="dataframe",
            description="Dataset with compiled multi-turn conversational history and structured message buffers"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "session_id_column": {
                    "type": "string",
                    "title": "Session / User ID Column",
                    "input_type": "column_select",
                    "default": "session_id",
                    "description": "Column identifying unique conversation threads or users. If left blank or not found, all rows are treated as a single conversation."
                },
                "role_column": {
                    "type": "string",
                    "title": "Role Column",
                    "input_type": "column_select",
                    "default": "role",
                    "description": "Column designating speaker role (e.g. 'user', 'assistant', 'system', 'agent')."
                },
                "message_column": {
                    "type": "string",
                    "title": "Message / Text Column",
                    "input_type": "column_select",
                    "default": "message",
                    "description": "Column containing the dialogue utterance or message content."
                },
                "order_column": {
                    "type": "string",
                    "title": "Order / Timestamp Column (Optional)",
                    "input_type": "column_select",
                    "default": "",
                    "description": "Optional column to sort conversation turns chronologically (e.g. 'turn_id', 'timestamp', 'created_at')."
                },
                "memory_type": {
                    "type": "string",
                    "title": "Memory Strategy",
                    "input_type": "dropdown",
                    "enum": ["sliding_window", "token_buffer", "full_history"],
                    "default": "sliding_window",
                    "description": "'sliding_window' retains the last K turns; 'token_buffer' caps by estimated token count pruning oldest messages; 'full_history' includes all turns."
                },
                "window_size": {
                    "type": "integer",
                    "title": "Sliding Window Size (Turns)",
                    "input_type": "number",
                    "default": 6,
                    "description": "Maximum number of most recent conversation turns to keep in memory."
                },
                "max_token_limit": {
                    "type": "integer",
                    "title": "Max Token Limit (for token_buffer)",
                    "input_type": "number",
                    "default": 1500,
                    "description": "Maximum estimated tokens allowed in the conversational memory context."
                },
                "output_mode": {
                    "type": "string",
                    "title": "Output Granularity Mode",
                    "input_type": "dropdown",
                    "enum": ["aggregate_per_session", "enrich_each_turn"],
                    "default": "aggregate_per_session",
                    "description": "'aggregate_per_session' produces 1 summarized row per session; 'enrich_each_turn' preserves all rows and appends historical context up to that turn."
                },
                "output_format": {
                    "type": "string",
                    "title": "History Representation Format",
                    "input_type": "dropdown",
                    "enum": ["transcript", "messages_json", "both"],
                    "default": "both",
                    "description": "'transcript' generates formatted text (User: ... \n Assistant: ...); 'messages_json' produces standard array of chat dicts; 'both' outputs both columns."
                },
                "system_prefix": {
                    "type": "string",
                    "title": "Default System Instruction",
                    "input_type": "textarea",
                    "default": "You are a helpful conversational AI assistant.",
                    "description": "System message to prepend to the memory buffer when formatted as chat messages."
                }
            },
            "required": ["message_column"]
        }

    def _estimate_tokens(self, text: str) -> int:
        """Heuristic token estimator (~4 characters per token)."""
        if not text:
            return 0
        return max(1, int(len(text) / 4.0))

    def _normalize_role(self, raw_role: Any) -> str:
        if pd.isna(raw_role) or not raw_role:
            return "user"
        r = str(raw_role).lower().strip()
        if r in ["user", "human", "client", "customer"]:
            return "user"
        elif r in ["assistant", "ai", "bot", "agent", "system_response"]:
            return "assistant"
        elif r in ["system", "instruction"]:
            return "system"
        return "user"

    def _compile_session_memory(
        self,
        turns: List[Dict[str, Any]],
        memory_type: str,
        window_size: int,
        max_token_limit: int,
        system_prefix: str
    ) -> Dict[str, Any]:
        """Applies memory retention strategy to a list of ordered turns."""
        if not turns:
            messages = [{"role": "system", "content": system_prefix}] if system_prefix else []
            return {
                "transcript": "",
                "messages_json": json.dumps(messages),
                "retained_turns_count": 0,
                "total_turns_count": 0,
                "estimated_tokens": self._estimate_tokens(system_prefix)
            }

        total_turns = len(turns)
        selected_turns = turns

        if memory_type == "sliding_window":
            if window_size > 0 and len(turns) > window_size:
                selected_turns = turns[-window_size:]
        elif memory_type == "token_buffer":
            # Work backwards from newest turns until token budget reached
            budget = max_token_limit - self._estimate_tokens(system_prefix)
            accumulated = []
            curr_tokens = 0
            for turn in reversed(turns):
                t_tokens = self._estimate_tokens(turn["message"])
                if curr_tokens + t_tokens <= budget or not accumulated:
                    accumulated.append(turn)
                    curr_tokens += t_tokens
                else:
                    break
            selected_turns = list(reversed(accumulated))

        # Build transcript and message payload
        transcript_lines = []
        messages_list = []
        if system_prefix and system_prefix.strip():
            messages_list.append({"role": "system", "content": system_prefix.strip()})

        for t in selected_turns:
            role = t["role"]
            msg = t["message"]
            label = "User" if role == "user" else ("Assistant" if role == "assistant" else "System")
            transcript_lines.append(f"{label}: {msg}")
            messages_list.append({"role": role, "content": msg})

        transcript_str = "\n".join(transcript_lines)
        total_tokens = sum(self._estimate_tokens(m["content"]) for m in messages_list)

        return {
            "transcript": transcript_str,
            "messages_json": json.dumps(messages_list),
            "retained_turns_count": len(selected_turns),
            "total_turns_count": total_turns,
            "estimated_tokens": total_tokens
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        df = inputs.get("dataframe") if "dataframe" in inputs else inputs.get("input")
        if df is None:
            df = inputs.get("df")
        if df is None:
            raise ValidationException("ConversationMemory requires a valid tabular dataset on 'input'.")

        session_col = config.get("session_id_column", "session_id")
        role_col = config.get("role_column", "role")
        msg_col = config.get("message_column", "message")
        order_col = config.get("order_column", "").strip()
        memory_type = config.get("memory_type", "sliding_window")
        window_size = int(config.get("window_size", 6))
        max_tokens = int(config.get("max_token_limit", 1500))
        output_mode = config.get("output_mode", "aggregate_per_session")
        output_format = config.get("output_format", "both")
        system_prefix = config.get("system_prefix", "You are a helpful conversational AI assistant.")

        # Validate message column
        if msg_col not in df.columns:
            # Fallback to any string/text column
            text_candidates = [c for c in df.columns if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == "object"]
            if text_candidates:
                msg_col = text_candidates[0]
                logger.info(f"Target message column not found. Defaulting to candidate '{msg_col}'.")
            else:
                raise ValidationException(f"Message column '{msg_col}' not found in input table.")

        src_df = df.copy()

        # Handle empty dataframe edge case
        if len(src_df) == 0:
            if output_mode == "aggregate_per_session":
                res = pd.DataFrame(columns=[session_col if session_col in src_df.columns else "session_id", "conversation_transcript", "messages_payload", "retained_turns", "estimated_tokens"])
            else:
                res = src_df.copy()
                res["conversation_transcript"] = pd.Series(dtype=object)
                res["messages_payload"] = pd.Series(dtype=object)
            return {"dataframe": res}

        # Handle missing session_id
        if not session_col or session_col not in src_df.columns:
            src_df["_session_id_internal"] = "session_default"
            active_session_col = "_session_id_internal"
            remove_session_col = True
        else:
            active_session_col = session_col
            remove_session_col = False

        # Optional sort
        if order_col and order_col in src_df.columns:
            src_df = src_df.sort_values(by=[active_session_col, order_col], ascending=[True, True])

        # Group turns by session
        session_groups = src_df.groupby(active_session_col, sort=False)

        if output_mode == "aggregate_per_session":
            records = []
            for sess_id, grp in session_groups:
                turns = []
                for _, row in grp.iterrows():
                    m_val = str(row[msg_col]).strip() if pd.notna(row[msg_col]) else ""
                    if not m_val:
                        continue
                    r_val = self._normalize_role(row[role_col] if role_col in grp.columns else "user")
                    turns.append({"role": r_val, "message": m_val})

                mem = self._compile_session_memory(
                    turns=turns,
                    memory_type=memory_type,
                    window_size=window_size,
                    max_token_limit=max_tokens,
                    system_prefix=system_prefix
                )

                rec = {
                    active_session_col: sess_id,
                    "retained_turns": mem["retained_turns_count"],
                    "total_turns": mem["total_turns_count"],
                    "estimated_tokens": mem["estimated_tokens"]
                }
                if output_format in ["transcript", "both"]:
                    rec["conversation_transcript"] = mem["transcript"]
                if output_format in ["messages_json", "both"]:
                    rec["messages_payload"] = mem["messages_json"]

                # Extract last user message if present
                last_turn = turns[-1]["message"] if turns else ""
                rec["latest_message"] = last_turn
                records.append(rec)

            out_df = pd.DataFrame(records)
            if remove_session_col and "_session_id_internal" in out_df.columns:
                out_df = out_df.drop(columns=["_session_id_internal"])

        else:  # enrich_each_turn
            # For each turn, compute running history up to and including that turn
            transcripts = []
            payloads = []
            tokens_list = []
            retained_list = []

            for sess_id, grp in session_groups:
                running_turns = []
                for _, row in grp.iterrows():
                    m_val = str(row[msg_col]).strip() if pd.notna(row[msg_col]) else ""
                    r_val = self._normalize_role(row[role_col] if role_col in grp.columns else "user")
                    if m_val:
                        running_turns.append({"role": r_val, "message": m_val})

                    mem = self._compile_session_memory(
                        turns=list(running_turns),
                        memory_type=memory_type,
                        window_size=window_size,
                        max_token_limit=max_tokens,
                        system_prefix=system_prefix
                    )
                    transcripts.append(mem["transcript"])
                    payloads.append(mem["messages_json"])
                    tokens_list.append(mem["estimated_tokens"])
                    retained_list.append(mem["retained_turns_count"])

            out_df = src_df.copy()
            if output_format in ["transcript", "both"]:
                out_df["conversation_transcript"] = transcripts
            if output_format in ["messages_json", "both"]:
                out_df["messages_payload"] = payloads
            out_df["history_tokens"] = tokens_list
            out_df["retained_turns"] = retained_list

            if remove_session_col and "_session_id_internal" in out_df.columns:
                out_df = out_df.drop(columns=["_session_id_internal"])

        return {
            "dataframe": out_df,
            "metrics": {
                "total_sessions_processed": len(session_groups),
                "memory_strategy": memory_type,
                "window_size": window_size,
                "output_mode": output_mode
            }
        }

    def infer_output_schema(self, input_schemas: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        output_mode = config.get("output_mode", "aggregate_per_session")
        output_format = config.get("output_format", "both")
        session_col = config.get("session_id_column", "session_id")

        if output_mode == "aggregate_per_session":
            schema = {
                session_col: "string",
                "latest_message": "string",
                "retained_turns": "integer",
                "total_turns": "integer",
                "estimated_tokens": "integer"
            }
            if output_format in ["transcript", "both"]:
                schema["conversation_transcript"] = "string"
            if output_format in ["messages_json", "both"]:
                schema["messages_payload"] = "string"
            return schema
        else:
            schema = dict(input_schemas.get("input", {}))
            if output_format in ["transcript", "both"]:
                schema["conversation_transcript"] = "string"
            if output_format in ["messages_json", "both"]:
                schema["messages_payload"] = "string"
            schema["history_tokens"] = "integer"
            schema["retained_turns"] = "integer"
            return schema
