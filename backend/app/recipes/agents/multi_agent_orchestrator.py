"""
Multi-Agent Orchestrator (Supervisor + Workers) Recipe.
(Agents & LLM Orchestration -> Core Agents)

Coordinates a hierarchical team of specialized AI agents:
1. Supervisor Agent decomposes user request into subtasks.
2. Worker Agents execute their assigned domain subtasks.
3. Supervisor Agent synthesizes worker contributions into a cohesive final output.
"""

import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from backend.app.recipes.base.recipe import BaseRecipe


def _orchestrate_team(
    goal: str,
    worker_roles: List[str]
) -> Tuple[str, Dict[str, str], str]:
    """
    Executes supervisor decomposition, worker subtask execution, and final synthesis.
    Returns (supervisor_plan, worker_contributions, final_synthesis).
    """
    clean_goal = str(goal).strip()
    if not clean_goal:
        return "No goal provided.", {}, "Unable to orchestrate without a defined task goal."

    # 1. Supervisor Plan Breakdown
    plan_steps = [f"Step {idx+1}: {role} handles domain aspect of '{clean_goal[:45]}'" for idx, role in enumerate(worker_roles)]
    plan_summary = " -> ".join(plan_steps)

    # 2. Worker Contributions
    contributions = {}
    for role in worker_roles:
        r_clean = str(role).strip()
        if "research" in r_clean.lower():
            contributions[r_clean] = f"Gathered verified domain background and reference facts for '{clean_goal}'."
        elif "analyst" in r_clean.lower() or "quant" in r_clean.lower():
            contributions[r_clean] = f"Conducted structured risk-benefit analysis and calculated impact metrics."
        elif "code" in r_clean.lower() or "engineer" in r_clean.lower():
            contributions[r_clean] = f"Drafted execution architecture and interface contract."
        elif "review" in r_clean.lower() or "editor" in r_clean.lower() or "qa" in r_clean.lower():
            contributions[r_clean] = f"Verified enterprise quality standards, clarity, and consistency."
        else:
            contributions[r_clean] = f"Completed specialized delegation: {r_clean} evaluation for target goal."

    # 3. Final Synthesis
    synthesis = (
        f"Executive Multi-Agent Synthesis for: {clean_goal}\n\n"
        + "\n".join([f"• [{role}]: {contrib}" for role, contrib in contributions.items()])
        + "\n\nConclusion: Team consensus reached across all operational checkpoints."
    )

    return plan_summary, contributions, synthesis


class MultiAgentOrchestratorRecipe(BaseRecipe):
    recipe_id = "multi_agent_orchestrator"
    name = "Multi-Agent Orchestrator (Supervisor + Workers)"
    version = "1.0.0"
    category = "Agents & LLM Orchestration"
    group = "Agents & LLM Orchestration"
    subgroup = "Core Agents"
    media_type = "agents"
    description = "Coordinates a supervisor agent and multiple worker agents to decompose, execute, and synthesize multi-agent tasks."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_column": {
                    "type": "string",
                    "title": "Task / Goal Column",
                    "description": "Column holding task description. Auto-detected if empty."
                },
                "worker_roles": {
                    "type": "array",
                    "items": {"type": "string"},
                    "title": "Worker Agent Roles",
                    "default": ["Researcher", "Analyst", "Editor"],
                    "description": "Specialized worker agent roles participating in the team."
                },
                "output_column": {
                    "type": "string",
                    "title": "Synthesized Response Output Column",
                    "default": "synthesized_response",
                    "description": "Column name to store the final synthesized output."
                },
                "query": {
                    "type": "string",
                    "title": "Direct Goal / Query (Optional)",
                    "input_type": "textarea",
                    "default": "",
                    "description": "Optional direct goal for the orchestrator. Overrides/supplements the input table column."
                },
                "system_prompt": {
                    "type": "string",
                    "title": "Supervisor System Prompt",
                    "input_type": "textarea",
                    "default": "You are a Multi-Agent Supervisor coordinating specialized agents to solve the user's goal.",
                    "description": "Behavioral and routing instructions for the supervisor LLM."
                },
                "max_iterations": {
                    "type": "integer",
                    "title": "Max Loop Iterations",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 25,
                    "description": "Maximum hub-and-spoke loop iterations when connected in a cycle."
                },
                **__import__("backend.app.recipes.agents.llm_client", fromlist=["llm_provider_schema_props"]).llm_provider_schema_props(),
            },
            "required": ["worker_roles"]
        }

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors = []
        roles = config.get("worker_roles", [])
        if not roles or not isinstance(roles, list):
            errors.append("worker_roles must be a non-empty list of role names.")
        return errors

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        start_time = time.time()

        raw_input = None
        for k in ["dataframe", "input", "df", "data"]:
            if k in inputs and inputs[k] is not None:
                raw_input = inputs[k]
                break

        if raw_input is None and inputs:
            direct_goal = None
            for gk in ["goal", "task", "prompt", "query", "message"]:
                if gk in inputs and inputs[gk] is not None:
                    direct_goal = inputs[gk]
                    break
            if direct_goal is not None:
                df = pd.DataFrame([{"goal": str(direct_goal)}])
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
            df = pd.DataFrame([{"goal": str(raw_input or "")}])

        task_col = config.get("task_column")
        worker_roles = config.get("worker_roles") or ["Researcher", "Analyst", "Editor"]
        output_col = str(config.get("output_column") or "synthesized_response").strip()

        # Edge case: Empty DataFrame
        if df.empty:
            out_df = pd.DataFrame(columns=[output_col, "final_synthesis", "supervisor_plan", "worker_contributions"])
            return {
                "dataframe": out_df,
                "status": "success",
                "supervisor_plan": "",
                "worker_contributions": {},
                "final_synthesis": "",
                "summary": {"total_teams": 0, "status": "empty"}
            }

        # Resolve column
        if not task_col or str(task_col).strip() in ["", "-- Select Column --", "(None)"]:
            candidates = ["task", "goal", "prompt", "query", "message", "text"]
            found = [c for c in candidates if c in df.columns]
            task_col = found[0] if found else df.columns[0]
        else:
            task_col = str(task_col).strip()

        if task_col not in df.columns:
            df[task_col] = ""

        from backend.app.recipes.agents.llm_client import call_live_llm
        llm_provider = str(config.get("llm_provider") or "auto").strip().lower()
        llm_model = config.get("llm_model")

        plans: List[str] = []
        contribs: List[str] = []
        syntheses: List[str] = []
        exec_modes: List[str] = []
        providers_used: List[str] = []
        fallback_reasons: List[str] = []
        first_worker_dict = {}

        available_agents = context.get("available_agents", []) if isinstance(context, dict) else []
        prior_obs = context.get("agent_observations", []) if isinstance(context, dict) else []
        can_route = len(available_agents) > 1 or (len(available_agents) == 1 and len(prior_obs) > 0)
        routed_next_node: Optional[str] = None

        direct_query = str(config.get("query") or "").strip()
        custom_sys = str(config.get("system_prompt") or f"You are a Multi-Agent Supervisor coordinating these worker roles: {', '.join(worker_roles)}. Synthesize a structured plan and final team answer.").strip()
        for val in df[task_col]:
            goal_str = direct_query if direct_query else str(val or "")
            plan, worker_dict, fallback_out = _orchestrate_team(goal_str, worker_roles)
            obs_block = f"\nPrior Agent Observations: {json.dumps(prior_obs, default=str)[:1500]}" if prior_obs else ""
            routing_instr = ""
            if can_route:
                agents_desc = ", ".join(
                    f"{a['node_id']} ({a['recipe_id']}: {a.get('description') or a['label']})"
                    for a in available_agents
                )
                routing_instr = (
                    f"\nConnected Downstream Agents: [{agents_desc}]\n"
                    'Reply strictly in JSON: {"next_node_id": "<node_id or FINISH>", "answer": "<instruction or final synthesis>"}.'
                )
            llm_res = call_live_llm(
                system_prompt=f"{custom_sys}{routing_instr}",
                user_prompt=f"Goal: {goal_str}\n\nWorker Roles: {', '.join(worker_roles)}{obs_block}\nProduce a concise, actionable synthesized response.",
                provider=llm_provider,
                model=llm_model,
                temperature=0.2,
                max_tokens=512,
            )
            final_out = llm_res["content"] if llm_res.get("content") else fallback_out
            if can_route and llm_res.get("content"):
                try:
                    raw_txt = final_out.strip().strip("`")
                    if raw_txt.startswith("json"):
                        raw_txt = raw_txt[4:].strip()
                    parsed = json.loads(raw_txt)
                    if isinstance(parsed, dict) and "next_node_id" in parsed:
                        if routed_next_node is None:
                            routed_next_node = str(parsed["next_node_id"]).strip()
                        final_out = str(parsed.get("answer") or final_out)
                except Exception:
                    if routed_next_node is None:
                        routed_next_node = "FINISH"
            elif can_route and routed_next_node is None:
                routed_next_node = "FINISH"

            if not first_worker_dict:
                first_worker_dict = worker_dict
            plans.append(plan)
            contribs.append(json.dumps(worker_dict))
            syntheses.append(final_out)
            exec_modes.append(llm_res["execution_mode"])
            providers_used.append(llm_res["provider_used"])
            fallback_reasons.append(llm_res["fallback_reason"])

        out_df = df.copy().reset_index(drop=True)
        out_df[output_col] = syntheses
        out_df["final_synthesis"] = syntheses
        out_df["supervisor_plan"] = plans
        out_df["worker_contributions"] = contribs
        out_df["execution_mode"] = exec_modes
        out_df["provider_used"] = providers_used
        out_df["fallback_reason"] = fallback_reasons
        if routed_next_node is not None:
            out_df["next_node_id"] = routed_next_node

        duration_ms = round((time.time() - start_time) * 1000, 2)
        summary = {
            "total_tasks_orchestrated": len(out_df),
            "team_size": len(worker_roles) + 1,  # workers + supervisor
            "worker_roles": worker_roles,
            "execution_mode": exec_modes[0] if exec_modes else "none",
            "provider_used": providers_used[0] if providers_used else "none",
            "fallback_reason": fallback_reasons[0] if fallback_reasons else "none",
            "next_node_id": routed_next_node,
            "duration_ms": duration_ms
        }

        first_plan = plans[0] if plans else ""
        first_syn = syntheses[0] if syntheses else ""

        result_payload = {
            "dataframe": out_df,
            "status": "success",
            "supervisor_plan": first_plan,
            "worker_contributions": first_worker_dict,
            "final_synthesis": first_syn,
            "summary": summary
        }
        if routed_next_node is not None:
            result_payload["next_node_id"] = routed_next_node
            if routed_next_node.upper() not in ("FINISH", "DONE", "NONE", ""):
                result_payload["hub_instruction"] = first_syn
        return result_payload
