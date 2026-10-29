"""
Prompt Template Manager Recipe.
Group: Agents & LLM Orchestration
Sub-group: Prompt & Memory Management
Category: agents

Manages, renders, and structures reusable system and user prompt templates
with dynamic variable interpolation, few-shot examples, and chat message assembling.
"""
import re
import json
import logging
from typing import Dict, Any, List, Optional, Union
import pandas as pd

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort
from backend.app.core.exceptions import ValidationException

logger = logging.getLogger(__name__)


class PromptTemplateManagerRecipe(BaseRecipe):
    recipe_id = "prompt_template_manager"
    name = "Prompt Template Manager"
    version = "1.0.0"
    category = "agents"
    group = "Agents & LLM Orchestration"
    subgroup = "Prompt & Memory Management"
    description = "Manages reusable, versioned prompt templates with dynamic variable interpolation, few-shot examples, and chat message format assembly."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="input",
            label="Input Table",
            type="dataframe",
            required=True,
            max_connections=1,
            description="Dataset containing columns to interpolate into prompt templates"
        )
    ]

    outputs = [
        RecipePort(
            id="dataframe",
            label="Prompt Table",
            type="dataframe",
            description="Dataset augmented with rendered prompts and structured chat message payloads"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "template": {
                    "type": "string",
                    "title": "User Prompt Template",
                    "input_type": "textarea",
                    "description": "User prompt template text. Use {{column_name}} or {column_name} placeholders for dynamic variables. E.g. 'Classify the sentiment of customer query: {{inquiry}} from {{company}}'.",
                    "default": "Analyze the following text: {{text}}"
                },
                "system_prompt": {
                    "type": "string",
                    "title": "System Persona / Instruction",
                    "input_type": "textarea",
                    "description": "Optional system-level instructions or persona (e.g. 'You are a senior data analyst and financial auditor.').",
                    "default": "You are a helpful and precise enterprise AI assistant."
                },
                "output_mode": {
                    "type": "string",
                    "title": "Output Format Mode",
                    "input_type": "dropdown",
                    "enum": ["rendered_string", "chat_messages_json", "separate_columns"],
                    "default": "rendered_string",
                    "description": "'rendered_string' generates a single interpolated text column; 'chat_messages_json' produces standard OpenAI/Groq messages array JSON; 'separate_columns' outputs system_prompt and user_prompt columns."
                },
                "output_column": {
                    "type": "string",
                    "title": "Output Prompt Column",
                    "input_type": "text",
                    "description": "Name of the resulting column containing the rendered prompt.",
                    "default": "rendered_prompt"
                },
                "few_shot_examples": {
                    "type": "array",
                    "title": "Few-Shot Learning Examples",
                    "input_type": "json_editor",
                    "items": {
                        "type": "object",
                        "properties": {
                            "input": {"type": "string"},
                            "output": {"type": "string"}
                        }
                    },
                    "default": [],
                    "description": "Optional list of few-shot examples [{'input': '...', 'output': '...'}] prepended to the prompt to steer LLM accuracy."
                },
                "few_shot_header": {
                    "type": "string",
                    "title": "Few-Shot Header Title",
                    "input_type": "text",
                    "default": "Here are some reference examples:\n",
                    "description": "Introductory text placed before the few-shot examples."
                },
                "variable_mappings": {
                    "type": "object",
                    "title": "Variable to Column Mappings",
                    "input_type": "json_editor",
                    "default": {},
                    "description": "Optional custom dictionary mapping template placeholder names to specific dataframe columns or static constants. E.g. {'user_query': 'raw_text'}."
                },
                "missing_variable_handling": {
                    "type": "string",
                    "title": "Missing Variable Handling",
                    "input_type": "dropdown",
                    "enum": ["fill_na", "keep_placeholder", "raise_error"],
                    "default": "fill_na",
                    "description": "What to do when a placeholder in the template is not found in the row: 'fill_na' replaces with na_placeholder, 'keep_placeholder' preserves {{variable}}, 'raise_error' halts execution."
                },
                "na_placeholder": {
                    "type": "string",
                    "title": "Null / Missing Value Placeholder",
                    "input_type": "text",
                    "default": "N/A",
                    "description": "Text to substitute when a column value is NaN, null, or missing."
                },
                "conversation_history_column": {
                    "type": "string",
                    "title": "Conversation History Column (Optional)",
                    "input_type": "column_select",
                    "default": "",
                    "description": "Optional column containing pre-formatted multi-turn conversation history to append to the prompt."
                }
            },
            "required": ["template"]
        }

    def _format_few_shot_text(self, examples: List[Dict[str, Any]], header: str) -> str:
        if not examples:
            return ""
        lines = [header.rstrip()]
        for i, ex in enumerate(examples, 1):
            inp = ex.get("input", "")
            out = ex.get("output", "")
            lines.append(f"Example {i}:\nInput: {inp}\nOutput: {out}\n")
        return "\n".join(lines).strip() + "\n\n"

    def _extract_variables(self, template_str: str) -> List[str]:
        """Extracts variable names from both {{var}} and {var} patterns."""
        # Find double curly {{var}} first
        double_vars = re.findall(r"\{\{([a-zA-Z0-9_\.\-]+)\}\}", template_str)
        # Find single curly {var}
        single_vars = re.findall(r"(?<!\{)\{([a-zA-Z0-9_\.\-]+)\}(?!\})", template_str)
        return list(dict.fromkeys(double_vars + single_vars))

    def _render_template_for_row(
        self,
        template_str: str,
        row_dict: Dict[str, Any],
        mappings: Dict[str, Any],
        missing_handling: str,
        na_placeholder: str
    ) -> str:
        rendered = template_str

        # Find all placeholders
        pattern = re.compile(r"\{\{([a-zA-Z0-9_\.\-]+)\}\}|\{([a-zA-Z0-9_\.\-]+)\}")
        
        def replacer(match):
            var_name = match.group(1) or match.group(2)
            # Check mappings first, then row_dict
            source_key = mappings.get(var_name, var_name)
            
            if source_key in row_dict:
                val = row_dict[source_key]
                if pd.isna(val) or val is None:
                    return str(na_placeholder)
                return str(val)
            elif var_name in mappings:
                # Static mapping value
                val = mappings[var_name]
                return str(val)
            else:
                if missing_handling == "raise_error":
                    raise ValidationException(f"Template placeholder '{var_name}' not found in dataset columns.")
                elif missing_handling == "keep_placeholder":
                    return match.group(0)
                else:  # fill_na
                    return str(na_placeholder)

        return pattern.sub(replacer, rendered)

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        df = inputs.get("dataframe") if "dataframe" in inputs else inputs.get("input")
        if df is None:
            df = inputs.get("df")
        if df is None:
            raise ValidationException("PromptTemplateManager requires a valid tabular dataset on 'input'.")

        template = config.get("template", "Analyze the following text: {{text}}")
        system_prompt = config.get("system_prompt", "You are a helpful and precise enterprise AI assistant.")
        output_mode = config.get("output_mode", "rendered_string")
        output_col = config.get("output_column", "rendered_prompt")
        few_shot_examples = config.get("few_shot_examples") or []
        few_shot_header = config.get("few_shot_header", "Here are some reference examples:\n")
        mappings = config.get("variable_mappings") or {}
        missing_handling = config.get("missing_variable_handling", "fill_na")
        na_placeholder = config.get("na_placeholder", "N/A")
        history_col = config.get("conversation_history_column", "").strip()

        result_df = df.copy()

        # Handle empty dataframe edge case
        if len(result_df) == 0:
            if output_mode == "rendered_string":
                result_df[output_col] = pd.Series(dtype=object)
            elif output_mode == "chat_messages_json":
                result_df[output_col] = pd.Series(dtype=object)
                result_df["messages_payload"] = pd.Series(dtype=object)
            elif output_mode == "separate_columns":
                result_df["system_prompt"] = pd.Series(dtype=object)
                result_df[output_col] = pd.Series(dtype=object)
            return {"dataframe": result_df}

        few_shot_text = self._format_few_shot_text(few_shot_examples, few_shot_header)

        rendered_prompts = []
        messages_payloads = []
        system_prompts = []

        for idx, row in result_df.iterrows():
            row_dict = row.to_dict()
            rendered_user_text = self._render_template_for_row(
                template_str=template,
                row_dict=row_dict,
                mappings=mappings,
                missing_handling=missing_handling,
                na_placeholder=na_placeholder
            )

            # Prepend few-shot if present
            if few_shot_text:
                full_user_content = few_shot_text + rendered_user_text
            else:
                full_user_content = rendered_user_text

            # Append conversation history if column specified and present
            if history_col and history_col in row_dict:
                hist_val = row_dict[history_col]
                if pd.notna(hist_val) and str(hist_val).strip():
                    full_user_content += f"\n\nConversation Context:\n{str(hist_val)}"

            rendered_prompts.append(full_user_content)
            system_prompts.append(system_prompt)

            # Build standard chat messages structure
            chat_messages = []
            if system_prompt and system_prompt.strip():
                chat_messages.append({"role": "system", "content": system_prompt.strip()})
            chat_messages.append({"role": "user", "content": full_user_content})
            messages_payloads.append(json.dumps(chat_messages))

        if output_mode == "rendered_string":
            result_df[output_col] = rendered_prompts
        elif output_mode == "chat_messages_json":
            result_df[output_col] = rendered_prompts
            result_df["messages_payload"] = messages_payloads
        elif output_mode == "separate_columns":
            result_df["system_prompt"] = system_prompts
            result_df[output_col] = rendered_prompts

        # Metadata metrics
        variables_found = self._extract_variables(template)
        return {
            "dataframe": result_df,
            "metrics": {
                "total_prompts_rendered": len(result_df),
                "template_variables": variables_found,
                "few_shot_examples_count": len(few_shot_examples),
                "output_mode": output_mode
            }
        }

    def infer_output_schema(self, input_schemas: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        schema = dict(input_schemas.get("input", {}))
        output_mode = config.get("output_mode", "rendered_string")
        output_col = config.get("output_column", "rendered_prompt")

        if output_mode == "rendered_string":
            schema[output_col] = "string"
        elif output_mode == "chat_messages_json":
            schema[output_col] = "string"
            schema["messages_payload"] = "string"
        elif output_mode == "separate_columns":
            schema["system_prompt"] = "string"
            schema[output_col] = "string"

        return schema
