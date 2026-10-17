from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class CostAttributionReportRecipe(BaseRecipe):
    recipe_id = "cost_attribution_report"
    name = "Cost Attribution Report"
    version = "1.0.0"
    category = "monitoring"
    description = "Breaks down platform and compute costs by pipeline, team or project."
    input_types = ["dataframe"]
    output_types = ["report", "dataframe"]

    def get_inputs(self) -> List[RecipePort]:
        return [
            RecipePort(id="input", label="Input Table (Optional)", type="dataframe", required=False, description="Optional execution telemetry or cost dataset")
        ]

    def get_outputs(self) -> List[RecipePort]:
        return [
            RecipePort(id="output", label="Output (report)", type="report", description="Executive FinOps Cost Attribution Report"),
            RecipePort(id="output_table", label="Cost Attribution Table", type="dataframe", description="Granular spend attribution by department and pipeline")
        ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "attribution_dimension": {
                    "type": "string",
                    "title": "Primary Breakdown Dimension",
                    "enum": [
                        "Team / Department",
                        "Pipeline & Workflow",
                        "Project",
                        "Resource Type (Compute / Storage / API)"
                    ],
                    "default": "Team / Department"
                },
                "time_period": {
                    "type": "string",
                    "title": "Reporting Time Period",
                    "enum": [
                        "Last 7 Days",
                        "Current Month (MTD)",
                        "Last 30 Days",
                        "Quarter to Date (QTD)",
                        "Annual Projection"
                    ],
                    "default": "Current Month (MTD)"
                },
                "monthly_budget_usd": {
                    "type": "number",
                    "title": "Monthly Budget Ceiling ($)",
                    "default": 5000.0,
                    "minimum": 100.0,
                    "description": "Total monthly allocated budget ceiling across all workloads."
                },
                "include_idle_capacity_cost": {
                    "type": "boolean",
                    "title": "Include Shared Idle / Standby Overhead",
                    "default": True,
                    "description": "Distribute idle standby cluster capacity proportionately across teams."
                },
                "currency": {
                    "type": "string",
                    "title": "Display Currency",
                    "enum": ["USD ($)", "EUR (€)", "GBP (£)", "INR (₹)"],
                    "default": "USD ($)"
                }
            }
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        df = inputs.get("dataframe")
        if df is None and isinstance(inputs.get("input"), pd.DataFrame):
            df = inputs["input"]

        dim = config.get("attribution_dimension", "Team / Department")
        period = config.get("time_period", "Current Month (MTD)")
        total_budget = float(config.get("monthly_budget_usd", 5000.0))
        include_idle = bool(config.get("include_idle_capacity_cost", True))
        currency_str = str(config.get("currency", "USD ($)"))
        if "EUR" in currency_str or "€" in currency_str:
            curr_code = "EUR"
            curr_symbol = "€"
        elif "GBP" in currency_str or "£" in currency_str:
            curr_code = "GBP"
            curr_symbol = "£"
        elif "INR" in currency_str or "₹" in currency_str:
            curr_code = "INR"
            curr_symbol = "₹"
        else:
            curr_code = "USD"
            curr_symbol = "$"

        # Check if incoming dataframe has user-provided cost attribution ledger
        has_custom_cost_cols = False
        if isinstance(df, pd.DataFrame) and not df.empty:
            lower_cols = [c.lower() for c in df.columns]
            if any(k in lower_cols for k in ["cost", "spend", "amount", "charge"]) and any(k in lower_cols for k in ["team", "department", "pipeline", "project"]):
                has_custom_cost_cols = True

        if has_custom_cost_cols:
            # Aggregate custom dataset
            cost_col = next(c for c in df.columns if c.lower() in ["cost", "spend", "amount", "charge"])
            group_col = next(c for c in df.columns if c.lower() in ["team", "department", "pipeline", "project"])
            
            agg = df.groupby(group_col)[cost_col].sum().reset_index()
            total_spend = float(agg[cost_col].sum())
            
            records = []
            idle_pool = (total_spend * 0.12) if include_idle else 0.0
            top_entity = str(agg.iloc[0][group_col]) if not agg.empty else "Platform"
            top_spend = float(agg[cost_col].max()) if not agg.empty else 0.0

            for _, r in agg.iterrows():
                entity_spend = float(r[cost_col])
                share_pct = (entity_spend / max(total_spend, 1e-6)) * 100.0
                budget_alloc = (total_budget * (share_pct / 100.0))
                status = "OVER_BUDGET" if entity_spend > budget_alloc * 1.1 else ("WARNING" if entity_spend > budget_alloc * 0.9 else "ON_BUDGET")
                idle_val = round(entity_spend * 0.12 if include_idle else 0.0, 2)
                total_val = round(entity_spend * (1.12 if include_idle else 1.0), 2)
                records.append({
                    "Entity": str(r[group_col]),
                    "Dimension": dim,
                    "Direct_Spend": f"{curr_symbol}{entity_spend:,.2f}",
                    "Idle_Overhead": f"{curr_symbol}{idle_val:,.2f}",
                    "Total_Cost": f"{curr_symbol}{total_val:,.2f}",
                    "Share_of_Total_Pct": f"{round(share_pct, 1)}%",
                    "Status": status
                })
            attribution_df = pd.DataFrame(records)
            total_actual_spend = round(total_spend + idle_pool, 2)
        else:
            # Synthesize Enterprise FinOps Multi-Tenant Ledger
            raw_attribution = [
                {"Department": "Customer Intelligence AI", "Project": "Retention & Churn", "Pipeline": "Real-time Churn Prediction", "Resource_Type": "Serverless API Compute", "Executions": 1420000, "Compute_Hours": 182.4, "Direct_Cost": 948.50},
                {"Department": "Data Science & Research", "Project": "Customer Segmentation", "Pipeline": "Customer Cluster Engine (KMeans/UMAP)", "Resource_Type": "Batch Worker Nodes", "Executions": 28400, "Compute_Hours": 146.0, "Direct_Cost": 765.20},
                {"Department": "Risk & Fraud Prevention", "Project": "Gateway Security", "Pipeline": "Transaction Anomaly Detector", "Resource_Type": "GPU Triton Cluster", "Executions": 2850000, "Compute_Hours": 320.5, "Direct_Cost": 1280.00},
                {"Department": "Marketing Personalization", "Project": "Next Best Offer", "Pipeline": "Market Basket (Apriori/FP-Growth)", "Resource_Type": "CPU Worker Pool", "Executions": 42000, "Compute_Hours": 88.2, "Direct_Cost": 395.40},
                {"Department": "Core Platform ML", "Project": "Feature Store & ETL", "Pipeline": "Feature Preprocessing & Scaling", "Resource_Type": "Distributed Spark Engine", "Executions": 185000, "Compute_Hours": 112.0, "Direct_Cost": 512.80},
                {"Department": "Data Science & Research", "Project": "Explainability & Governance", "Pipeline": "Model Drift & Health Operator", "Resource_Type": "Monitoring Cron Workers", "Executions": 12400, "Compute_Hours": 44.5, "Direct_Cost": 185.60}
            ]

            total_direct = sum(r["Direct_Cost"] for r in raw_attribution)
            # Idle pool is ~14% of direct compute pool
            idle_pool = (total_direct * 0.14) if include_idle else 0.0
            total_actual_spend = round(total_direct + idle_pool, 2)

            dept_totals: Dict[str, float] = {}
            for item in raw_attribution:
                direct = item["Direct_Cost"]
                allocated_idle = (direct / total_direct) * idle_pool if total_direct > 0 else 0.0
                total_item_cost = round(direct + allocated_idle, 2)
                dept = item["Department"]
                dept_totals[dept] = dept_totals.get(dept, 0.0) + total_item_cost

            top_entity = max(dept_totals, key=lambda k: dept_totals[k])
            top_spend = dept_totals[top_entity]

            rows = []
            for item in raw_attribution:
                direct = item["Direct_Cost"]
                allocated_idle = (direct / total_direct) * idle_pool if total_direct > 0 else 0.0
                total_item_cost = round(direct + allocated_idle, 2)
                budget_share = round((total_item_cost / total_budget) * 100.0, 1)
                
                # Status based on proportion
                status = "WARNING" if budget_share > 25.0 else "ON_BUDGET"

                rows.append({
                    "Department": item["Department"],
                    "Project": item["Project"],
                    "Pipeline": item["Pipeline"],
                    "Resource_Type": item["Resource_Type"],
                    "Executions": f"{item['Executions']:,}",
                    "Compute_Hours": item["Compute_Hours"],
                    "Direct_Spend": f"{curr_symbol}{direct:,.2f}",
                    "Idle_Overhead": f"{curr_symbol}{allocated_idle:,.2f}",
                    "Total_Cost": f"{curr_symbol}{total_item_cost:,.2f}",
                    "Budget_Utilization_Pct": f"{budget_share}%",
                    "Budget_Status": status
                })

            attribution_df = pd.DataFrame(rows)

        # 2. Financial KPIs & Executive Summary
        budget_util_pct = round((total_actual_spend / total_budget) * 100.0, 1)
        if budget_util_pct > 100.0:
            budget_health = "OVER_BUDGET"
            recommendation = "Review high-frequency GPU pipelines and consider downscaling non-critical clusters."
        elif budget_util_pct > 85.0:
            budget_health = "WARNING"
            recommendation = "Workload approaching monthly budget cap. Optimize batch sizes and pruning policies."
        else:
            budget_health = "HEALTHY"
            recommendation = "Platform spend is well within approved financial envelope."

        report = {
            "title": f"FinOps Cloud Compute Cost Attribution ({period}) - {curr_code}",
            "reporting_period": period,
            "currency": currency_str,
            "currency_code": curr_code,
            "currency_symbol": curr_symbol,
            f"total_spend_{curr_code.lower()}": f"{curr_symbol}{total_actual_spend:,.2f}",
            f"monthly_budget_{curr_code.lower()}": f"{curr_symbol}{total_budget:,.2f}",
            "total_platform_spend": f"{curr_symbol}{total_actual_spend:,.2f}",
            "monthly_budget": f"{curr_symbol}{total_budget:,.2f}",
            "budget_utilization": f"{budget_util_pct}%",
            "financial_health": budget_health,
            "top_cost_driver": f"{top_entity} ({curr_symbol}{top_spend:,.2f})",
            "idle_overhead_allocated": f"{curr_symbol}{idle_pool:,.2f}" if include_idle else "Excluded",
            "finops_recommendation": recommendation
        }

        metrics = {
            f"total_spend_{curr_code.lower()}": total_actual_spend,
            f"monthly_budget_{curr_code.lower()}": total_budget,
            f"idle_overhead_{curr_code.lower()}": round(idle_pool, 2) if include_idle else 0.0,
            "total_spend_usd": total_actual_spend,
            "monthly_budget_usd": total_budget,
            "idle_overhead_usd": round(idle_pool, 2) if include_idle else 0.0,
            "budget_utilization_pct": budget_util_pct,
            "status": budget_health,
            "top_spending_department": top_entity,
            "currency": curr_code,
            "currency_symbol": curr_symbol
        }

        output_summary = {
            f"Total Spend ({curr_code})": f"{curr_symbol}{total_actual_spend:,.2f}",
            f"Monthly Budget ({curr_code})": f"{curr_symbol}{total_budget:,.2f}",
            "Budget Utilization": f"{budget_util_pct}%",
            "Status": budget_health,
            "Top Spending Department": top_entity,
            f"Idle Overhead ({curr_code})": f"{curr_symbol}{idle_pool:,.2f}" if include_idle else f"{curr_symbol}0.00",
            "currency": curr_code,
            "currency_symbol": curr_symbol,
            "formatted_spend": f"{curr_symbol}{total_actual_spend:,.2f}",
            "formatted_budget": f"{curr_symbol}{total_budget:,.2f}",
            "budget_utilization_pct": budget_util_pct
        }

        ret = {
            "report": report,
            "metrics": metrics,
            "output_summary": output_summary,
            "dataframe": attribution_df,
            "output_table": attribution_df
        }
        if isinstance(df, pd.DataFrame):
            ret["input_dataframe"] = df
        return ret

    def to_code(self, config: Dict[str, Any]) -> str:
        curr = config.get("currency", "USD ($)")
        budget = config.get("monthly_budget_usd", 5000.0)
        dim = config.get("attribution_dimension", "Team / Department")
        return (
            f"# FinOps Cost Attribution Report\n"
            f"# Dimension: {dim} | Monthly Budget: {budget} ({curr})\n"
            f"# Enterprise Cloud Infrastructure & Spend Governance\n"
        )
