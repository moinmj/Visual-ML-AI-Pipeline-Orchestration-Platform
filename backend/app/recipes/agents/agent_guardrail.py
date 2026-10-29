"""
Agent Guardrail & Moderation Layer (Agents & LLM Orchestration -> Governance & Ops).

Scans, sanitizes, and filters prompts and LLM completions for:
- PII (Personally Identifiable Information) leaks: Credit cards, SSN, emails, phone numbers, IP addresses, API keys
- Prompt Injections: System prompt leak attempts, jailbreaks, instruction overrides
- Toxicity & Policy Violations: Profanity, harassment, hate speech, dangerous instructions
- Enterprise Custom Banned Keywords
"""

import re
import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from backend.app.recipes.base.recipe import BaseRecipe

# ─────────────────────────────────────────────────────────────────────────────
# Built-in Regex Patterns for PII and Prompt Injections
# ─────────────────────────────────────────────────────────────────────────────
_PII_PATTERNS = {
    "credit_card": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b"),
    "phone": re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "api_key": re.compile(r"\b(?:sk-[a-zA-Z0-9]{20,48}|ghp_[a-zA-Z0-9]{36}|AIza[0-9A-Za-z-_]{35})\b"),
    "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
}

_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|system)\s+prompts?", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(in\s+)?(developer\s+mode|dan\s+mode|unrestricted)", re.IGNORECASE),
    re.compile(r"repeat\s+(the\s+)?(system\s+prompt|initial\s+instructions|secret\s+key)", re.IGNORECASE),
    re.compile(r"reveal\s+your\s+(instructions|system\s+prompt|internal\s+guidelines)", re.IGNORECASE),
    re.compile(r"bypass\s+(all\s+)?(content\s+filters|safety\s+guidelines)", re.IGNORECASE),
]

_TOXICITY_KEYWORDS = {
    "exploit", "ddos", "ransomware", "malware", "keylogger",
    "kill", "attack", "bomb", "weapon", "terrorist", "suicide",
    "hack", "crack", "steal", "fraud", "phishing"
}


def _scan_text(
    text: str,
    check_pii: bool = True,
    check_injection: bool = True,
    check_toxicity: bool = True,
    banned_keywords: Optional[List[str]] = None
) -> Tuple[bool, List[str], str, float]:
    """
    Scans a single text string against enabled guardrail policies.
    Returns (is_safe, violations_list, redacted_text, risk_score).
    """
    if not text or not str(text).strip():
        return True, [], "", 0.0

    original = str(text)
    redacted = original
    violations = []
    risk_points = 0.0

    # 1. Check PII
    if check_pii:
        for pii_type, pattern in _PII_PATTERNS.items():
            matches = pattern.findall(redacted)
            if matches:
                violations.append(f"pii_{pii_type}")
                risk_points += 0.3 * len(matches)
                redacted = pattern.sub(f"[REDACTED_{pii_type.upper()}]", redacted)

    # 2. Check Prompt Injection
    if check_injection:
        for pattern in _INJECTION_PATTERNS:
            if pattern.search(original):
                violations.append("prompt_injection_attempt")
                risk_points += 0.6
                break

    # 3. Check Toxicity & Dangerous Operations
    if check_toxicity:
        words = re.findall(r"\w+", original.lower())
        detected_toxic = set(words).intersection(_TOXICITY_KEYWORDS)
        if detected_toxic:
            violations.append(f"toxicity_flags({','.join(sorted(detected_toxic))})")
            risk_points += 0.25 * len(detected_toxic)

    # 4. Check Custom Banned Keywords
    if banned_keywords:
        lower_orig = original.lower()
        for kw in banned_keywords:
            kw_clean = str(kw).strip().lower()
            if kw_clean and kw_clean in lower_orig:
                violations.append(f"banned_keyword({kw_clean})")
                risk_points += 0.35
                # Case-insensitive redaction
                kw_re = re.compile(re.escape(kw_clean), re.IGNORECASE)
                redacted = kw_re.sub("[REDACTED_KEYWORD]", redacted)

    is_safe = len(violations) == 0
    risk_score = min(round(risk_points, 3), 1.0)

    return is_safe, violations, redacted, risk_score


class AgentGuardrailRecipe(BaseRecipe):
    recipe_id = "agent_guardrail"
    name = "Agent Guardrail & Moderation Layer"
    version = "1.0.0"
    category = "agents"
    media_type = "agents"
    description = "Scans, sanitizes, and filters prompts and completions for safety violations, PII leaks, prompt injections, and policy breaches."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "target_column": {
                    "type": "string",
                    "title": "Target Content Column",
                    "description": "Column containing prompt or LLM response to scan. Auto-detected if empty."
                },
                "action_on_violation": {
                    "type": "string",
                    "title": "Action on Violation",
                    "enum": ["redact_in_place", "block_and_flag", "log_only"],
                    "default": "redact_in_place",
                    "description": "'redact_in_place' replaces sensitive substrings; 'block_and_flag' replaces entire message with a blocked message; 'log_only' annotates violations without altering content."
                },
                "blocked_fallback_message": {
                    "type": "string",
                    "title": "Blocked Fallback Message",
                    "default": "[BLOCKED: Content violates enterprise AI safety policy.]",
                    "description": "Text displayed when content is blocked."
                },
                "check_pii": {
                    "type": "boolean",
                    "title": "Check PII (SSN, Cards, Emails, Keys)",
                    "default": True,
                    "description": "Scan and redact personally identifiable information."
                },
                "check_prompt_injection": {
                    "type": "boolean",
                    "title": "Check Prompt Injection",
                    "default": True,
                    "description": "Scan for system prompt leaks, jailbreaks, and instruction overrides."
                },
                "check_toxicity": {
                    "type": "boolean",
                    "title": "Check Toxicity & Dangerous Directives",
                    "default": True,
                    "description": "Scan for harmful keywords and attack vectors."
                },
                "banned_keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "title": "Custom Banned Keywords",
                    "default": [],
                    "description": "List of company-specific blocked terms."
                },
                "max_input_length": {
                    "type": "integer",
                    "title": "Max Character Length",
                    "default": 10000,
                    "description": "Flags or truncates inputs exceeding this character limit to prevent resource exhaustion."
                }
            },
            "required": ["action_on_violation"]
        }

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors = []
        action = config.get("action_on_violation", "redact_in_place")
        if action not in ["redact_in_place", "block_and_flag", "log_only"]:
            errors.append(f"Invalid action_on_violation '{action}'.")
        return errors

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        start_time = time.time()

        # 1. Resolve Input DataFrame safely without evaluating bool(df)
        raw_input = None
        for k in ["dataframe", "input", "df"]:
            if k in inputs and inputs[k] is not None:
                raw_input = inputs[k]
                break

        if isinstance(raw_input, pd.DataFrame):
            df = raw_input.copy()
        elif isinstance(raw_input, list):
            df = pd.DataFrame(raw_input)
        elif isinstance(raw_input, dict) and "records" in raw_input:
            df = pd.DataFrame(raw_input["records"])
        elif isinstance(raw_input, dict):
            df = pd.DataFrame([raw_input])
        else:
            df = pd.DataFrame()

        target_col = config.get("target_column")
        action = str(config.get("action_on_violation") or "redact_in_place").strip().lower()
        blocked_fallback = str(config.get("blocked_fallback_message") or "[BLOCKED: Content violates enterprise AI safety policy.]")
        check_pii = bool(config.get("check_pii", True))
        check_injection = bool(config.get("check_prompt_injection", True))
        check_toxicity = bool(config.get("check_toxicity", True))
        banned_kws = config.get("banned_keywords") or []
        max_length = int(config.get("max_input_length") or 10000)

        # 2. Edge Case: Empty DataFrame
        if df.empty:
            out_df = pd.DataFrame(columns=["is_safe", "risk_score", "violations_found", "sanitized_content"])
            return {
                "dataframe": out_df,
                "summary": {"total_scanned": 0, "violations_detected": 0, "status": "empty"}
            }

        # 3. Resolve Target Column
        if not target_col or str(target_col).strip() in ["", "-- Select Column --", "(None)"]:
            # Priority order for agent pipelines
            candidate_cols = ["rendered_prompt", "prompt", "message", "generated_answer", "text", "query"]
            found = [c for c in candidate_cols if c in df.columns]
            if found:
                target_col = found[0]
            else:
                str_cols = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype).startswith("str")]
                target_col = str_cols[0] if str_cols else df.columns[0]
        else:
            target_col = str(target_col).strip()

        if target_col not in df.columns:
            df[target_col] = ""

        # 4. Process Rows
        is_safe_list: List[bool] = []
        violations_list: List[str] = []
        sanitized_content_list: List[str] = []
        risk_score_list: List[float] = []

        total_violations = 0

        for raw_val in df[target_col]:
            if pd.isna(raw_val) or raw_val is None:
                is_safe_list.append(True)
                violations_list.append("[]")
                sanitized_content_list.append("")
                risk_score_list.append(0.0)
                continue

            text_val = str(raw_val)
            # Truncate if exceeds max_length
            if len(text_val) > max_length:
                text_val = text_val[:max_length]

            safe, viols, redacted_text, score = _scan_text(
                text=text_val,
                check_pii=check_pii,
                check_injection=check_injection,
                check_toxicity=check_toxicity,
                banned_keywords=banned_kws
            )

            if not safe:
                total_violations += 1

            is_safe_list.append(safe)
            violations_list.append(json.dumps(viols))
            risk_score_list.append(score)

            if safe or action == "log_only":
                sanitized_content_list.append(text_val)
            elif action == "block_and_flag":
                sanitized_content_list.append(blocked_fallback)
            else:  # redact_in_place
                sanitized_content_list.append(redacted_text)

        # 5. Build Output DataFrame
        out_df = df.copy().reset_index(drop=True)
        out_df["is_safe"] = is_safe_list
        out_df["risk_score"] = risk_score_list
        out_df["violations_found"] = violations_list
        out_df["sanitized_content"] = sanitized_content_list

        duration_ms = round((time.time() - start_time) * 1000, 2)
        summary = {
            "total_scanned": len(out_df),
            "violations_detected": total_violations,
            "violation_rate": round(total_violations / len(out_df), 3) if len(out_df) > 0 else 0.0,
            "target_column": target_col,
            "action_applied": action,
            "duration_ms": duration_ms
        }

        return {
            "dataframe": out_df,
            "summary": summary
        }
