"""
Fine-Tuning Dataset Formatter & Trigger (Agents & LLM Orchestration -> Governance & Ops).

Packages high-quality evaluated interaction datasets into standard fine-tuning formats:
- OpenAI Chat JSONL format ({"messages": [...]})
- Alpaca Instruct format ({"instruction": ..., "output": ...})
- ShareGPT format ({"conversations": [...]})
- Prompt-Completion JSONL format ({"prompt": ..., "completion": ...})

Applies quality filters (evaluation verdict PASS, safety check), calculates token metrics,
and partitions data into train / validation splits for training jobs.
"""

import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from backend.app.recipes.base.recipe import BaseRecipe


class FineTuningTriggerRecipe(BaseRecipe):
    recipe_id = "fine_tuning_trigger"
    name = "Fine-Tuning Dataset Formatter & Trigger"
    version = "1.0.0"
    category = "agents"
    media_type = "agents"
    description = "Formats, validates, and packages evaluated prompt-response datasets into standard JSONL fine-tuning formats with train/val splits."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "format_type": {
                    "type": "string",
                    "title": "Dataset Format Type",
                    "enum": ["openai_chat_jsonl", "alpaca_instruct", "sharegpt", "completion_jsonl"],
                    "default": "openai_chat_jsonl",
                    "description": "Standard fine-tuning target format."
                },
                "instruction_column": {
                    "type": "string",
                    "title": "Instruction / Query Column",
                    "description": "Column holding prompt, question, or instruction. Auto-detected if empty."
                },
                "response_column": {
                    "type": "string",
                    "title": "Response / Target Column",
                    "description": "Column holding ideal completion or response. Auto-detected if empty."
                },
                "system_prompt": {
                    "type": "string",
                    "title": "System Prompt",
                    "default": "You are a professional enterprise AI assistant.",
                    "description": "System role instruction injected into formatted chat messages."
                },
                "filter_rejected": {
                    "type": "boolean",
                    "title": "Filter Out Low Quality / Rejected Examples",
                    "default": True,
                    "description": "Excludes rows where evaluation_verdict == 'FAIL' or is_safe == False."
                },
                "validation_split_ratio": {
                    "type": "number",
                    "title": "Validation Split Ratio (e.g. 0.1 for 10%)",
                    "default": 0.1,
                    "minimum": 0.0,
                    "maximum": 0.5,
                    "description": "Fraction of examples reserved for validation."
                },
                "output_column": {
                    "type": "string",
                    "title": "Output JSONL Column Name",
                    "default": "fine_tuning_jsonl",
                    "description": "Target column name storing the serialized JSON line string."
                }
            },
            "required": ["format_type"]
        }

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors = []
        fmt = config.get("format_type", "openai_chat_jsonl")
        if fmt not in ["openai_chat_jsonl", "alpaca_instruct", "sharegpt", "completion_jsonl"]:
            errors.append(f"Invalid format_type '{fmt}'.")
        ratio = config.get("validation_split_ratio", 0.1)
        if not isinstance(ratio, (int, float)) or ratio < 0.0 or ratio > 0.5:
            errors.append("validation_split_ratio must be between 0.0 and 0.5.")
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

        format_type = str(config.get("format_type") or "openai_chat_jsonl").strip().lower()
        inst_col = config.get("instruction_column")
        resp_col = config.get("response_column")
        system_prompt = str(config.get("system_prompt") or "You are a professional enterprise AI assistant.")
        filter_rejected = bool(config.get("filter_rejected", True))
        val_ratio_raw = config.get("validation_split_ratio")
        val_ratio = float(val_ratio_raw) if val_ratio_raw is not None else 0.1
        output_col = str(config.get("output_column") or "fine_tuning_jsonl").strip()

        # 2. Edge Case: Empty DataFrame
        if df.empty:
            out_df = pd.DataFrame(columns=[output_col, "dataset_split", "estimated_tokens"])
            return {
                "dataframe": out_df,
                "summary": {"total_examples": 0, "status": "empty"}
            }

        # 3. Resolve Columns
        if not inst_col or str(inst_col).strip() in ["", "-- Select Column --", "(None)"]:
            candidates = ["prompt", "query", "instruction", "input", "question"]
            found = [c for c in candidates if c in df.columns]
            inst_col = found[0] if found else df.columns[0]
        else:
            inst_col = str(inst_col).strip()

        if not resp_col or str(resp_col).strip() in ["", "-- Select Column --", "(None)"]:
            candidates = ["generated_answer", "response", "output", "sanitized_content", "answer"]
            found = [c for c in candidates if c in df.columns and c != inst_col]
            resp_col = found[0] if found else (df.columns[1] if len(df.columns) > 1 else df.columns[0])
        else:
            resp_col = str(resp_col).strip()

        # 4. Filter Low Quality / Safety Violations if enabled
        filtered_df = df
        if filter_rejected:
            if "evaluation_verdict" in df.columns:
                filtered_df = filtered_df[filtered_df["evaluation_verdict"] == "PASS"]
            if "is_safe" in df.columns:
                filtered_df = filtered_df[filtered_df["is_safe"] == True]

        filtered_df = filtered_df.reset_index(drop=True)

        if filtered_df.empty:
            out_df = pd.DataFrame(columns=[output_col, "dataset_split", "estimated_tokens"])
            return {
                "dataframe": out_df,
                "summary": {"total_examples": 0, "filtered_out": len(df), "status": "all_filtered"}
            }

        # 5. Format Records
        formatted_jsonls: List[str] = []
        splits: List[str] = []
        token_estimates: List[int] = []

        total_rows = len(filtered_df)
        if val_ratio == 0.0 or total_rows <= 1:
            val_cutoff = total_rows
        else:
            val_cutoff = max(1, int(total_rows * (1.0 - val_ratio)))

        for i in range(total_rows):
            split_tag = "train" if i < val_cutoff else "validation"

            prompt_txt = str(filtered_df.iloc[i].get(inst_col, "") or "").strip()
            resp_txt = str(filtered_df.iloc[i].get(resp_col, "") or "").strip()

            # Word count approximation: ~1.3 tokens per word
            approx_tokens = int(len((prompt_txt + " " + resp_txt).split()) * 1.33) + 10

            if format_type == "alpaca_instruct":
                record = {
                    "instruction": prompt_txt,
                    "input": "",
                    "output": resp_txt
                }
            elif format_type == "sharegpt":
                record = {
                    "conversations": [
                        {"from": "system", "value": system_prompt},
                        {"from": "human", "value": prompt_txt},
                        {"from": "gpt", "value": resp_txt}
                    ]
                }
            elif format_type == "completion_jsonl":
                record = {
                    "prompt": prompt_txt,
                    "completion": resp_txt
                }
            else:  # openai_chat_jsonl (standard)
                record = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt_txt},
                        {"role": "assistant", "content": resp_txt}
                    ]
                }

            formatted_jsonls.append(json.dumps(record, ensure_ascii=False))
            splits.append(split_tag)
            token_estimates.append(approx_tokens)

        # 6. Build Output DataFrame
        out_df = filtered_df.copy().reset_index(drop=True)
        out_df[output_col] = formatted_jsonls
        out_df["dataset_split"] = splits
        out_df["estimated_tokens"] = token_estimates

        duration_ms = round((time.time() - start_time) * 1000, 2)
        summary = {
            "format_type": format_type,
            "total_examples": len(out_df),
            "train_examples": sum(1 for s in splits if s == "train"),
            "val_examples": sum(1 for s in splits if s == "validation"),
            "total_estimated_tokens": sum(token_estimates),
            "filtered_out_count": len(df) - len(out_df),
            "duration_ms": duration_ms
        }

        return {
            "dataframe": out_df,
            "summary": summary
        }
