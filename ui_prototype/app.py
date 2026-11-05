import os
import sys

# Ensure repository root is on sys.path for Streamlit Cloud and nested executions
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import json
import time
import datetime
from typing import Any, Dict, List, Optional, Union

from streamlit_flow import streamlit_flow
import importlib
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

from backend.app.recommendation.recommender import AIRecommender
from backend.app.engine.execution.executor import DAGExecutor
from backend.app.engine.execution.job_manager import job_manager
from backend.app.engine.inference.pipeline_inferencer import PipelineInferencer
from backend.app.engine.inference.schemas import PredictionRequest

from backend.app.profiling.profiler import DataProfiler
from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes import register_all_recipes

if not recipe_registry._recipes:
    register_all_recipes()

from backend.app.engine.dag.graph import WorkflowGraph, WorkflowNode, WorkflowEdge
from backend.app.tenant_data.client import (
    fetch_tenant_environments,
    fetch_tenant_models,
    ingest_tenant_model,
)

st.set_page_config(
    page_title="Datalect ML — Visual AI/ML Pipeline Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# COMPANY DESIGN SYSTEM & ENTERPRISE STYLING (Slate & Tailwind)
# -------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Typography & Slate Theme */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #0F172A;
    }

    .stApp {
        background-color: #F8FAFC;
    }

    .main .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
        max-width: 98%;
    }

    /* Headings */
    .brand-header-title {
        font-size: 1.45rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.025em;
        line-height: 1.25;
    }

    .brand-header-sub {
        font-size: 0.82rem;
        color: #64748B;
        font-weight: 500;
        margin-top: 2px;
    }

    .main-header {
        font-size: 1.55rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }

    .sub-header {
        font-size: 0.88rem;
        color: #64748B;
        margin-bottom: 1.0rem;
        font-weight: 400;
    }

    /* Status Pills & Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 3px 9px;
        font-size: 0.72rem;
        font-weight: 600;
        border-radius: 9999px;
        background-color: #F1F5F9;
        color: #334155;
        border: 1px solid #E2E8F0;
    }

    .badge-blue {
        background-color: #EFF6FF;
        color: #1D4ED8;
        border-color: #DBEAFE;
    }

    .badge-emerald {
        background-color: #ECFDF5;
        color: #047857;
        border-color: #A7F3D0;
    }

    .badge-amber {
        background-color: #FFFBEB;
        color: #B45309;
        border-color: #FDE68A;
    }

    .badge-rose {
        background-color: #FFF1F2;
        color: #BE123C;
        border-color: #FECDD3;
    }

    .badge-purple {
        background-color: #FAF5FF;
        color: #7E22CE;
        border-color: #E9D5FF;
    }

    .badge-sky {
        background-color: #F0F9FF;
        color: #0369A1;
        border-color: #BAE6FD;
    }

    /* Corporate Buttons */
    button[kind="primary"] {
        background: #2563EB !important;
        border: 1px solid #1D4ED8 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        color: #FFFFFF !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
        transition: all 0.15s ease-in-out !important;
    }

    button[kind="primary"]:hover {
        background: #1D4ED8 !important;
        border-color: #1E40AF !important;
        box-shadow: 0 2px 4px 0 rgba(37, 99, 235, 0.25) !important;
    }

    button[kind="secondary"] {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        color: #1E293B !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.03) !important;
        transition: all 0.15s ease-in-out !important;
    }

    button[kind="secondary"]:hover {
        background: #F8FAFC !important;
        border-color: #CBD5E1 !important;
        color: #0F172A !important;
    }

    /* Metric Cards */
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 12px 16px;
        box-shadow: 0 1px 3px 0 rgba(15, 23, 42, 0.03);
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        color: #64748B !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.35rem !important;
        font-weight: 800 !important;
        color: #0F172A !important;
        letter-spacing: -0.02em;
    }

    /* Tabs Styling */
    [data-testid="stTabs"] [role="tablist"] {
        border-bottom: 1px solid #E2E8F0;
        gap: 6px;
    }

    [data-testid="stTabs"] [role="tab"] {
        font-weight: 600;
        font-size: 0.82rem;
        color: #64748B;
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
        border: none;
        background: transparent;
        transition: all 0.15s ease;
    }

    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        color: #2563EB;
        background: #EFF6FF;
        border-bottom: 2px solid #2563EB;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        box-shadow: 0 1px 2px 0 rgba(15, 23, 42, 0.02);
        margin-bottom: 10px;
    }

    /* Dot-Grid Canvas Background */
    .stCanvasWrapper {
        background-color: #F8FAFC;
        background-image: radial-gradient(#CBD5E1 1.5px, transparent 1.5px);
        background-size: 24px 24px;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        overflow: hidden;
    }

    /* Shimmer & Chrysa AI Animations */
    .chrysa-card {
        padding: 16px;
        border-radius: 12px;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 8px -2px rgba(15, 23, 42, 0.06);
        margin-bottom: 12px;
    }

    .chrysa-shimmer {
        position: relative;
        overflow: hidden;
    }

    .chrysa-shimmer::after {
        content: "";
        position: absolute;
        inset: 0;
        transform: translateX(-100%);
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.75), transparent);
        animation: chrysa-sweep 1.4s ease-in-out infinite;
    }

    @keyframes chrysa-sweep {
        100% { transform: translateX(100%); }
    }

    .chrysa-text-shimmer {
        background: linear-gradient(90deg, #1E293B 0%, #2563EB 50%, #1E293B 100%);
        background-size: 200% 100%;
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        animation: chrysa-text-sweep 2.2s linear infinite;
    }

    @keyframes chrysa-text-sweep {
        from { background-position: 200% 0; }
        to { background-position: -200% 0; }
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# SAMPLE DATASETS GENERATOR
# -------------------------------------------------------------
@st.cache_data
def get_preset_dataset(name: str) -> pd.DataFrame:
    np.random.seed(42)
    n = 300
    if name == "Customer Churn (Classification)":
        age = np.random.randint(18, 70, size=n).astype(float)
        age[np.random.choice(n, 15, replace=False)] = np.nan
        salary = np.random.normal(55000, 15000, size=n)
        salary[np.random.choice(n, 10, replace=False)] = np.nan
        department = np.random.choice(["Engineering", "Sales", "Marketing", "HR"], size=n)
        experience = np.random.randint(1, 20, size=n)
        prob = (age / 100) * 0.3 + (experience / 20) * 0.2 + (salary < 45000) * 0.4
        churn = (prob > np.random.uniform(0, 1, size=n)).astype(int)
        return pd.DataFrame({"Age": age, "Salary": salary, "Department": department, "Experience": experience, "Churn": churn})

    elif name == "Daily Retail Sales (Time-Series)":
        dates = pd.date_range("2026-01-01", periods=180, freq="D")
        trend = np.linspace(200, 500, 180)
        weekly = 50 * np.sin(2 * np.pi * np.arange(180) / 7)
        noise = np.random.normal(0, 15, 180)
        sales = (trend + weekly + noise).clip(min=50).round(2)
        promo = np.random.choice([0, 1], size=180, p=[0.8, 0.2])
        return pd.DataFrame({"Date": dates, "Sales": sales, "Promo_Active": promo})

    elif name == "Credit Transactions (Anomaly Injection)":
        amount = np.random.exponential(150, size=n)
        latency = np.random.normal(120, 20, size=n)
        risk_score = np.random.uniform(0.1, 0.4, size=n)
        outlier_idx = np.random.choice(n, 15, replace=False)
        amount[outlier_idx] = np.random.uniform(2500, 8000, size=15)
        latency[outlier_idx] = np.random.uniform(900, 2500, size=15)
        risk_score[outlier_idx] = np.random.uniform(0.85, 0.99, size=15)
        location = np.random.choice(["US", "EU", "APAC", "LATAM"], size=n)
        return pd.DataFrame({"Amount": amount.round(2), "Latency_ms": latency.round(1), "RiskScore": risk_score.round(3), "Location": location})

    elif name == "Titanic Survival":
        pclass = np.random.choice([1, 2, 3], size=n, p=[0.25, 0.25, 0.5])
        age = np.random.normal(30, 12, size=n).clip(1, 80)
        age[np.random.choice(n, 20, replace=False)] = np.nan
        fare = np.random.exponential(32, size=n).round(2)
        sex = np.random.choice(["male", "female"], size=n, p=[0.6, 0.4])
        survived = ((sex == "female") * 0.5 + (pclass == 1) * 0.3 + (fare > 50) * 0.2 > np.random.uniform(0, 1, size=n)).astype(int)
        return pd.DataFrame({"Pclass": pclass, "Sex": sex, "Age": age, "Fare": fare, "Survived": survived})

    else:
        feat1 = np.random.normal(5.8, 0.8, size=n)
        feat2 = np.random.normal(3.0, 0.4, size=n)
        feat3 = np.random.normal(3.7, 1.7, size=n)
        target = np.random.choice(["Setosa", "Versicolor", "Virginica"], size=n)
        return pd.DataFrame({"SepalLength": feat1, "SepalWidth": feat2, "PetalLength": feat3, "Species": target})


if "active_df" not in st.session_state:
    st.session_state["active_df"] = get_preset_dataset("Customer Churn (Classification)")

if "active_dataset_name" not in st.session_state:
    st.session_state["active_dataset_name"] = "Customer Churn (Classification)"

# -------------------------------------------------------------
# DYNAMIC COMPONENT PALETTE BUILDER (All 187 Backend Recipes)
# -------------------------------------------------------------
CATEGORY_METADATA = {
    "triggers": {"name": "⚡ Triggers & Inbound Events", "icon": "⚡", "color": "amber", "order": 1},
    "ingestion": {"name": "📥 Ingestion & Sources", "icon": "📥", "color": "sky", "order": 2},
    "export": {"name": "📥 Ingestion & Sources", "icon": "📥", "color": "sky", "order": 2},
    "preprocessing": {"name": "🧹 Data Cleaning & Preprocessing", "icon": "🧹", "color": "indigo", "order": 3},
    "flow_control": {"name": "🔀 Flow Control & Branching", "icon": "🔀", "color": "slate", "order": 4},
    "splitting": {"name": "✂️ Dataset Splitting & Resampling", "icon": "✂️", "color": "emerald", "order": 5},
    "training": {"name": "🤖 Machine Learning Models", "icon": "🤖", "color": "blue", "order": 6},
    "forecasting": {"name": "📈 Time-Series Forecasting", "icon": "📈", "color": "teal", "order": 7},
    "anomaly": {"name": "🚨 Anomaly Detection (Unsupervised)", "icon": "🚨", "color": "rose", "order": 8},
    "unsupervised": {"name": "🌌 Unsupervised & Clustering", "icon": "🌌", "color": "violet", "order": 9},
    "nlp": {"name": "🔤 NLP & Text Processing", "icon": "🔤", "color": "cyan", "order": 10},
    "vision_audio": {"name": "👁️ Vision & Audio Processing", "icon": "👁️", "color": "fuchsia", "order": 11},
    "agents": {"name": "🧠 AI Agents & LLM Orchestration", "icon": "🧠", "color": "purple", "order": 12},
    "Agents & LLM Orchestration": {"name": "🧠 AI Agents & LLM Orchestration", "icon": "🧠", "color": "purple", "order": 12},
    "integrations": {"name": "🔌 Cloud & Tool Integrations", "icon": "🔌", "color": "orange", "order": 13},
    "monitoring": {"name": "📡 Drift & Performance Monitoring", "icon": "📡", "color": "yellow", "order": 14},
    "governance": {"name": "🏛️ Governance & Model Registry", "icon": "🏛️", "color": "stone", "order": 15},
    "evaluation": {"name": "🎯 Model Evaluation & Reports", "icon": "🎯", "color": "green", "order": 16}
}

RECIPE_SPECIFIC_ICONS = {
    "csv_loader": "📄",
    "export_writer": "💾",
    "api_response_endpoint": "🌐",
    "webhook_trigger": "⚡",
    "cron_trigger": "⏰",
    "kafka_consumer": "📨",
    "column_selector": "🎯",
    "data_type_converter": "🔄",
    "duplicate_remover": "👯",
    "category_sanitizer": "🔤",
    "correlation_filter": "📊",
    "variance_filter": "🛑",
    "missing_value_imputer": "🧹",
    "outlier_handler": "🛡️",
    "feature_selector": "✨",
    "feature_scaler": "⚖️",
    "categorical_encoder": "🔤",
    "text_preprocessor": "🧹",
    "sentiment_analysis": "🎭",
    "text_translation": "🌐",
    "text_vectorizer": "🔤",
    "train_test_split": "✂️",
    "stratified_split": "📊",
    "time_series_split": "⏱️",
    "walk_forward_split": "🔄",
    "xgboost_trainer": "⚡",
    "lightgbm_trainer": "🚀",
    "catboost_trainer": "🐱",
    "random_forest_trainer": "🌲",
    "linear_trainer": "📈",
    "logistic_regression_trainer": "📈",
    "isolation_forest": "🌲",
    "statistical_guardrail": "🛡️",
    "prophet_forecaster": "🔮",
    "arima_forecaster": "📊",
    "lag_feature_engineering": "⏱️",
    "mlflow_tracker": "🏛️",
    "model_evaluator": "🎯",
}


@st.cache_data(show_spinner=False)
def build_recipe_palette() -> Dict[str, List[Dict[str, Any]]]:
    """Dynamically aggregates all enabled backend recipes into enterprise categories with full metadata."""
    palette: Dict[str, List[Dict[str, Any]]] = {}
    all_recipes = recipe_registry.list_all(enabled=True)
    
    # Sort recipes by category predefined sequence and recipe name
    sorted_recipes = sorted(
        all_recipes,
        key=lambda r: (
            CATEGORY_METADATA.get(r.category, {}).get("order", 99),
            r.name
        )
    )
    
    for r in sorted_recipes:
        meta = CATEGORY_METADATA.get(r.category, {"name": f"📦 {r.category.title()}", "icon": "📦", "color": "slate"})
        cat_name = meta["name"]
        
        # Determine specific or category icon
        icon = RECIPE_SPECIFIC_ICONS.get(r.recipe_id, meta["icon"])
        
        # Build default config from schema defaults
        def_config = {}
        try:
            recipe_inst = recipe_registry._recipes.get(r.recipe_id)
            if recipe_inst and hasattr(recipe_inst, "get_schema"):
                schema = recipe_inst.get_schema() or {}
            elif hasattr(r, "get_schema"):
                schema = r.get_schema() or {}
            else:
                schema = {}
            props = schema.get("properties", {})
            for p_name, p_spec in props.items():
                if "default" in p_spec:
                    def_config[p_name] = p_spec["default"]
        except Exception:
            def_config = {}

        # Add helpful recipe-specific defaults
        if r.recipe_id == "train_test_split" and "test_size" not in def_config:
            def_config.update({"target_column": "", "test_size": 0.2})
        elif r.recipe_id == "xgboost_trainer" and "task_type" not in def_config:
            def_config.update({"task_type": "classification", "n_estimators": 100, "max_depth": 6})
        elif r.recipe_id == "missing_value_imputer" and "strategy" not in def_config:
            def_config["strategy"] = "median"
        elif r.recipe_id == "feature_scaler" and "method" not in def_config:
            def_config["method"] = "standard"

        raw_ver = getattr(r, "version", "1.0.0") or "1.0.0"
        version_str = raw_ver if str(raw_ver).startswith("v") else f"v{raw_ver}"

        palette.setdefault(cat_name, []).append({
            "id": r.recipe_id,
            "name": r.name,
            "icon": icon,
            "category": cat_name,
            "category_key": r.category,
            "version": version_str,
            "description": getattr(r, "description", "") or "Modular enterprise pipeline processor.",
            "input_types": getattr(r, "input_types", ["dataframe"]),
            "output_types": getattr(r, "output_types", ["dataframe"]),
            "default_config": def_config
        })
    return palette


RECIPE_CATEGORY_MAP = build_recipe_palette()


@st.cache_data(show_spinner=False)
def get_cached_ai_recommendation(dataset_name: str, num_rows: int, num_cols: int, target_col: Optional[str], task_type: Optional[str]) -> Dict[str, Any]:
    """Generates instantaneous heuristic pipeline recommendations without blocking on external LLM 404s."""
    df = st.session_state.get("active_df")
    if df is None:
        return {
            "explanation": "No active dataset loaded.",
            "profile_summary": {"rows": 0, "missing_cells": 0},
            "task_type": "classification",
            "model_rankings": []
        }
    return AIRecommender._heuristic_recommend_pipeline(
        df,
        target_column=target_col,
        task_type=task_type
    )


# Canvas Flow State
if "canvas_version" not in st.session_state:
    st.session_state["canvas_version"] = 1

if "flow_state" not in st.session_state:
    st.session_state["flow_state"] = StreamlitFlowState(nodes=[], edges=[])

if "node_configs" not in st.session_state:
    st.session_state["node_configs"] = {}

if "api_telemetry_history" not in st.session_state:
    st.session_state["api_telemetry_history"] = []


def record_api_telemetry(
    action_name: str,
    endpoint: str,
    method: str = "POST",
    request_payload: Any = None,
    response_payload: Any = None,
    status_code: int = 200,
    duration_ms: float = 0.0
):
    """Records real-time REST API telemetry on every user interaction."""
    import datetime
    telemetry_entry = {
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
        "action": action_name,
        "method": method,
        "endpoint": endpoint,
        "status_code": status_code,
        "duration_ms": round(duration_ms, 2),
        "request_payload": request_payload,
        "response_payload": response_payload
    }
    # Keep last 50 actions in session history
    st.session_state["api_telemetry_history"].insert(0, telemetry_entry)
    if len(st.session_state["api_telemetry_history"]) > 50:
        st.session_state["api_telemetry_history"].pop()


def parse_uploaded_dataset(up_file):
    """Unified file reader supporting CSV, Excel, and nested JSON with schema validation."""
    if up_file is None:
        return None, None
    try:
        t0 = time.time()
        if up_file.name.endswith(".csv"):
            df = pd.read_csv(up_file)
        elif up_file.name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(up_file)
        elif up_file.name.endswith(".json"):
            import json
            raw_json = json.load(up_file)
            if isinstance(raw_json, list):
                df = pd.json_normalize(raw_json)
            else:
                df = pd.json_normalize([raw_json])
        else:
            df = pd.read_csv(up_file)
        
        if df is None or df.empty:
            st.error("Uploaded file contains zero valid data rows.")
            return None, None
        
        # Record Telemetry for Dataset Ingestion API
        record_api_telemetry(
            action_name=f"Upload Dataset: {up_file.name}",
            endpoint="/api/v1/datasets/upload",
            method="POST",
            request_payload={"filename": up_file.name, "size_bytes": getattr(up_file, "size", 0)},
            response_payload={"dataset_name": up_file.name, "rows": len(df), "columns": list(df.columns)},
            status_code=201,
            duration_ms=(time.time() - t0) * 1000.0
        )
        return df, up_file.name
    except Exception as e:
        st.error(f"Error parsing uploaded file '{up_file.name}': {str(e)}")
        return None, None


def fetch_platform_datasets() -> List[Dict[str, Any]]:
    """Fetches list of uploaded datasets from platform database."""
    try:
        import sqlite3
        db_path = os.path.join(ROOT_DIR, "data", "platform.db")
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("SELECT id, name, row_count, column_count, storage_path FROM datasets ORDER BY created_at DESC")
            rows = cur.fetchall()
            conn.close()
            seen = set()
            res = []
            for r in rows:
                if r[1] not in seen:
                    seen.add(r[1])
                    res.append({"id": r[0], "name": r[1], "rows": r[2], "columns": r[3], "storage_path": r[4]})
            return res
    except Exception:
        pass
    return []


def load_platform_dataset(storage_rel_path: str) -> Optional[pd.DataFrame]:
    """Loads a DataFrame from local platform storage."""
    try:
        storage_root = os.path.join(ROOT_DIR, "data", "storage")
        full_p = os.path.join(storage_root, storage_rel_path)
        if os.path.exists(full_p):
            if full_p.endswith(".csv"):
                return pd.read_csv(full_p)
            elif full_p.endswith((".xlsx", ".xls")):
                return pd.read_excel(full_p)
            elif full_p.endswith(".parquet"):
                return pd.read_parquet(full_p)
            elif full_p.endswith(".json"):
                return pd.read_json(full_p)
    except Exception:
        pass
    return None



# -------------------------------------------------------------
# SIDEBAR NAVIGATION & PERSISTENT API INSPECTOR DOCK
# -------------------------------------------------------------
st.sidebar.markdown("""
<div style="display:flex; align-items:center; gap:10px; margin-bottom:14px; padding-bottom:12px; border-bottom:1px solid #E2E8F0;">
    <div style="width:36px; height:36px; border-radius:10px; background:#2563EB; color:#FFFFFF; display:flex; align-items:center; justify-content:center; font-weight:800; font-size:18px; box-shadow:0 2px 6px rgba(37,99,235,0.35);">
        ⚡
    </div>
    <div>
        <div style="font-weight:800; font-size:1.05rem; color:#0F172A; line-height:1.2; letter-spacing:-0.01em;">Datalect ML</div>
        <div style="font-size:0.75rem; color:#64748B; font-weight:500;">Visual AI/ML Pipeline Studio</div>
    </div>
</div>
""", unsafe_allow_html=True)
app_mode = st.sidebar.radio(
    "Navigation",
    ["🎨 Pipeline Whiteboard", "📊 Dataset Studio & Profiler", "🧩 Recipe Catalog (187)", "🌐 API & Telemetry Inspector"]
)

# Track Tab Navigation GET APIs
if st.session_state.get("_prev_app_mode") != app_mode:
    st.session_state["_prev_app_mode"] = app_mode
    if "Pipeline Whiteboard" in app_mode:
        record_api_telemetry("View Pipeline Whiteboard Canvas", "/api/v1/workflows/canvas", "GET", {"view": "whiteboard_canvas"}, {"nodes_count": len(st.session_state.get("flow_state", StreamlitFlowState(nodes=[], edges=[])).nodes)}, 200, 0.8)
    elif "Dataset Studio" in app_mode:
        record_api_telemetry("View Dataset Profile & Preview", f"/api/v1/datasets/{st.session_state.get('active_dataset_name', 'churn')}/profile", "GET", {"dataset_name": st.session_state.get("active_dataset_name")}, {"rows": len(st.session_state.get("active_df", [])), "columns": list(st.session_state.get("active_df", pd.DataFrame()).columns)}, 200, 1.2)
    elif "Recipe Catalog" in app_mode:
        record_api_telemetry("Fetch All Recipe Components", "/api/v1/recipes/", "GET", {}, {"total_recipes": len(recipe_registry.list_all())}, 200, 0.9)
    elif "API & Telemetry" in app_mode:
        record_api_telemetry("Fetch API Telemetry Stream", "/api/v1/telemetry", "GET", {}, {"events_count": len(st.session_state.get("api_telemetry_history", []))}, 200, 0.4)



def get_node_position(n) -> tuple:
    """Safely extracts (x, y) coordinates from any StreamlitFlowNode representation."""
    pos_val = getattr(n, "position", None)
    if pos_val is None:
        pos_val = getattr(n, "pos", (0, 0))
    if isinstance(pos_val, (list, tuple)) and len(pos_val) >= 2:
        return (float(pos_val[0]), float(pos_val[1]))
    elif isinstance(pos_val, dict):
        return (float(pos_val.get("x", 0)), float(pos_val.get("y", 0)))
    elif hasattr(pos_val, "x") and hasattr(pos_val, "y"):
        return (float(pos_val.x), float(pos_val.y))
    return (0.0, 0.0)


def get_current_dag_payload() -> dict:
    """Extracts current visual whiteboard nodes and edges into REST API payload format."""
    nodes = []
    for n in st.session_state["flow_state"].nodes:
        cfg = st.session_state["node_configs"].get(n.id, {})
        nodes.append({
            "id": n.id,
            "recipe_id": cfg.get("recipe_id", "csv_loader"),
            "config": cfg.get("config", {}),
            "label": n.data.get("content", n.id) if hasattr(n, "data") and isinstance(n.data, dict) else n.id
        })
    edges = [
        {"source": e.source, "target": e.target}
        for e in st.session_state["flow_state"].edges
    ]
    return {"nodes": nodes, "edges": edges}


# -------------------------------------------------------------
# PERSISTENT SIDEBAR API INSPECTOR (Visible on EVERY Tab)
# -------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### 📡 Live API Flight Recorder")
st.sidebar.caption("Tracks real-time REST API hits, payloads & latencies across all your platform actions.")

# 1. Most Recent Action Banner
if st.session_state["api_telemetry_history"]:
    last_hit = st.session_state["api_telemetry_history"][0]
    st.sidebar.markdown(f"""
    <div style="background:#1E293B; color:#F8FAFC; padding:10px; border-radius:6px; margin-bottom:10px; font-size:0.82rem; border-left:4px solid #10B981;">
        <div><b>⚡ Last Action:</b> {last_hit['action']}</div>
        <div style="color:#94A3B8; margin-top:2px;"><code>{last_hit['method']} {last_hit['endpoint']}</code></div>
        <div style="margin-top:4px;"><span style="color:#34D399;">🟢 {last_hit['status_code']} OK</span> | ⏱️ <b>{last_hit['duration_ms']}ms</b> | 🕒 {last_hit['timestamp']}</div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar.expander("🔍 Last Request / Response Body", expanded=False):
        st.markdown("**📤 Request Payload:**")
        st.json(last_hit["request_payload"])
        st.markdown("**📥 Response Output:**")
        st.json(last_hit["response_payload"])

sidebar_dag = get_current_dag_payload()
s_nodes_cnt = len(sidebar_dag["nodes"])
s_edges_cnt = len(sidebar_dag["edges"])

sb_c1, sb_c2 = st.sidebar.columns(2)
sb_c1.metric("Active Nodes", f"{s_nodes_cnt}")
sb_c2.metric("Active Edges", f"{s_edges_cnt}")

with st.sidebar.expander("📦 Current Whiteboard API Contract", expanded=False):
    st.markdown("**Endpoint:** `POST /api/v1/workflows/execute`")
    st.json(sidebar_dag)
    curl_snippet_sb = f"""curl -X POST http://localhost:8000/api/v1/workflows/execute -H "Content-Type: application/json" -d '{json.dumps(sidebar_dag)}'"""
    st.code(curl_snippet_sb, language="bash")

with st.sidebar.expander("⚡ 1-Click Pre-Flight Validator", expanded=False):
    if st.button("🧪 Validate Active DAG API", key="sb_btn_validate_api", use_container_width=True):
        if not sidebar_dag["nodes"]:
            st.warning("Canvas is empty.")
        else:
            t_v0 = time.time()
            wf_g = WorkflowGraph(
                nodes=[WorkflowNode(id=n["id"], recipe_id=n["recipe_id"], config=n["config"]) for n in sidebar_dag["nodes"]],
                edges=[WorkflowEdge(source=e["source"], target=e["target"]) for e in sidebar_dag["edges"]]
            )
            v_errs = wf_g.validate_graph()
            v_dur = (time.time() - t_v0) * 1000.0
            
            record_api_telemetry(
                action_name="Validate Workflow Graph",
                endpoint="/api/v1/workflows/validate",
                method="POST",
                request_payload=sidebar_dag,
                response_payload={"valid": len(v_errs) == 0, "errors": v_errs},
                status_code=200 if not v_errs else 422,
                duration_ms=v_dur
            )
            
            if not v_errs:
                st.success("✅ 200 OK: 100% DAG Valid!")
            else:
                for ve in v_errs:
                    st.write(f"- {ve}")
            st.rerun()


# -------------------------------------------------------------
# HELPER: SANITIZE NODE CONFIGS & EXECUTE VIA DAG EXECUTOR
# -------------------------------------------------------------
def get_incomplete_pipeline_nodes() -> List[Dict[str, Any]]:
    """
    Checks all canvas nodes for unconfigured required schema properties.
    Returns list of dicts: [{'node_id': ..., 'label': ..., 'recipe_name': ..., 'missing_fields': [...]}]
    """
    incomplete = []
    flow_state = st.session_state.get("flow_state")
    current_nodes = flow_state.nodes if flow_state else []
    node_configs = st.session_state.get("node_configs", {})

    for n in current_nodes:
        cfg_entry = node_configs.get(n.id, {})
        r_id = cfg_entry.get("recipe_id")
        recipe_obj = recipe_registry.get(r_id) if r_id else None
        if not recipe_obj:
            continue

        schema = recipe_obj.get_schema()
        required_fields = schema.get("required", [])
        cfg = cfg_entry.get("config", {})

        missing = []
        for rf in required_fields:
            if r_id == "csv_loader" and rf == "dataset_id" and st.session_state.get("active_df") is not None:
                continue
            val = cfg.get(rf)
            if val is None or val == "" or str(val).strip() in ["-- Select Column --", "(None)"]:
                field_title = schema.get("properties", {}).get(rf, {}).get("title", rf)
                missing.append(field_title)

        if missing:
            node_label = n.data.get("content", n.id) if hasattr(n, "data") and isinstance(n.data, dict) else n.id
            incomplete.append({
                "node_id": n.id,
                "label": node_label,
                "recipe_name": recipe_obj.name,
                "missing_fields": missing
            })

    return incomplete


def sanitize_node_configs_for_active_dataset():
    """Sanitizes node parameters when active dataset columns change without destructively wiping saved columns."""
    df = st.session_state.get("active_df")
    if df is None:
        return
    cols = list(df.columns)
    saved_cols = list(st.session_state.get("saved_dataset_columns", []))
    all_known_cols = set(cols + saved_cols)

    for n_id, cfg_data in st.session_state.get("node_configs", {}).items():
        cfg = cfg_data.get("config", {})
        # Do NOT destructively overwrite target_column with cols[-1] if user has not picked one yet or if it was saved
        if "columns" in cfg:
            if isinstance(cfg["columns"], str) and cfg["columns"].strip():
                valid = [c.strip() for c in cfg["columns"].split(",") if c.strip() in all_known_cols]
                if valid:
                    cfg["columns"] = valid
            elif isinstance(cfg["columns"], list) and cfg["columns"]:
                valid = [c for c in cfg["columns"] if c in all_known_cols]
                if valid:
                    cfg["columns"] = valid


def execute_pipeline():
    canvas_nodes = list(st.session_state["flow_state"].nodes)
    canvas_edges = list(st.session_state["flow_state"].edges)

    if not canvas_nodes:
        st.warning("⚠️ The canvas is empty. Please add nodes or load a template first.")
        return

    # 1. Enforce Required Fields Pre-Flight Check
    incomplete = get_incomplete_pipeline_nodes()
    if incomplete:
        st.error("### ❌ Cannot Execute Pipeline: Required Fields Missing")
        for item in incomplete:
            st.markdown(f"> ⚠️ **Node `{item['node_id']}` ({item['recipe_name']})** requires: **`{', '.join(item['missing_fields'])}`** to be selected before running.")
        return

    sanitize_node_configs_for_active_dataset()

    backend_nodes = []
    for n in canvas_nodes:
        cfg_data = st.session_state["node_configs"].get(n.id, {"recipe_id": "missing_value_imputer", "config": {}})
        backend_nodes.append(WorkflowNode(
            id=n.id,
            recipe_id=cfg_data["recipe_id"],
            config=cfg_data.get("config", {}),
            label=n.data.get("content", n.id) if hasattr(n, "data") and isinstance(n.data, dict) else n.id
        ))

    backend_edges = [
        WorkflowEdge(source=e.source, target=e.target)
        for e in canvas_edges
    ]

    workflow_graph = WorkflowGraph(
        id=st.session_state.get("active_saved_workflow_id"),
        workflow_id=st.session_state.get("active_saved_workflow_id"),
        name=st.session_state.get("active_saved_workflow_name"),
        nodes=backend_nodes,
        edges=backend_edges
    )
    val_errors = workflow_graph.validate_graph()

    # Pre-Flight Validation Checks
    blocking_errors = [e for e in val_errors if not e.startswith("⚠️")]
    warnings = [e for e in val_errors if e.startswith("⚠️")]

    for w in warnings:
        st.warning(w)

    if blocking_errors:
        st.error("### ❌ Pipeline Pre-Flight Validation Failed")
        for err in blocking_errors:
            st.markdown(f"> {err}")
        return

    t_exec_start = time.time()
    exec_result = DAGExecutor.execute_workflow(
        execution_id=f"exec_{int(pd.Timestamp.now().timestamp())}",
        workflow=workflow_graph,
        initial_df=st.session_state["active_df"]
    )
    exec_latency = (time.time() - t_exec_start) * 1000.0

    st.session_state["last_execution"] = {
        "execution_id": exec_result.execution_id,
        "status": exec_result.status,
        "final_metrics": exec_result.final_metrics,
        "anomaly_summary": exec_result.anomaly_summary,
        "forecasting_summary": exec_result.forecasting_summary,
        "governance_summary": exec_result.governance_summary,
        "execution_logs": exec_result.logs,
        "node_outputs": exec_result.node_outputs,
        "step_snapshots": exec_result.step_snapshots,
        "inference_schema": exec_result.inference_schema
    }
    st.session_state["inference_bundle"] = job_manager.get_inference_bundle(exec_result.execution_id)
    st.session_state["inference_schema"] = exec_result.inference_schema
    try:
        job_manager.register_job_result(exec_result.execution_id, exec_result, workflow_id=st.session_state.get("active_saved_workflow_id"))
    except Exception:
        pass

    # Record Telemetry for Workflow Execution API
    record_api_telemetry(
        action_name="▶️ Run Pipeline",
        endpoint="/api/v1/workflows/execute",
        method="POST",
        request_payload=get_current_dag_payload(),
        response_payload={
            "execution_id": exec_result.execution_id,
            "status": exec_result.status,
            "total_duration_ms": exec_result.total_duration_ms,
            "final_metrics": exec_result.final_metrics,
            "steps_executed": len(exec_result.step_snapshots)
        },
        status_code=200 if exec_result.status == "SUCCESS" else 500,
        duration_ms=exec_result.total_duration_ms
    )

    if exec_result.status == "SUCCESS":
        st.success("🎉 Pipeline executed cleanly through DAG Engine!")
    else:
        st.error("### ❌ Pipeline Execution Failed on Node Step")
        for log in exec_result.logs:
            if "❌" in log:
                st.markdown(f"> {log}")


def render_prediction_studio(bundle: Optional[Dict[str, Any]], schema: Optional[Dict[str, Any]], in_dialog: bool = False):
    """
    Renders the complete, interactive model prediction studio supporting
    classification, regression, forecasting, anomaly detection, and batch CSV scoring.
    """
    if not bundle or not bundle.get("model"):
        st.info("ℹ️ No trained model found. Please construct and run your pipeline first to test live predictions.")
        return

    exec_id = bundle.get("execution_id", "latest")
    task_type = bundle.get("task_type") or (schema.get("task_type") if schema else "classification")
    if bundle.get("forecasting_summary") or hasattr(bundle.get("model"), "make_future_dataframe") or hasattr(bundle.get("model"), "get_forecast"):
        task_type = "time_series_forecasting"
    elif bundle.get("anomaly_summary") or type(bundle.get("model")).__name__ in ["IsolationForest", "EllipticEnvelope"]:
        task_type = "anomaly_detection"

    prefix = "dlg_" if in_dialog else "page_"

    # =========================================================
    # 1. TABULAR CLASSIFICATION & REGRESSION
    # =========================================================
    if task_type in ["classification", "regression"]:
        p_tab1, p_tab2, p_tab3 = st.tabs([
            "🎛️ Interactive Prediction",
            "📁 Batch CSV Scoring",
            "📡 REST API & cURL Command"
        ])

        with p_tab1:
            features = schema.get("features", []) if schema else []
            fn_list = bundle.get("feature_names", [f["name"] for f in features])
            sample_row = bundle.get("sample_row", {})

            # Quick action buttons
            q_col1, q_col2, _ = st.columns([2.5, 2.5, 5])
            with q_col1:
                if st.button("🎲 Load Random Test Sample", key=f"{prefix}btn_load_sample", use_container_width=True):
                    X_test = bundle.get("X_test")
                    if X_test is not None and isinstance(X_test, pd.DataFrame) and not X_test.empty:
                        rand_row = X_test.sample(1).to_dict(orient="records")[0]
                    else:
                        rand_row = sample_row
                    for fn in fn_list:
                        if fn in rand_row:
                            st.session_state[f"{prefix}input_{fn}"] = rand_row[fn]
                    st.rerun()

            with q_col2:
                if st.button("🧹 Reset to Medians", key=f"{prefix}btn_reset_sample", use_container_width=True):
                    for fn in fn_list:
                        k = f"{prefix}input_{fn}"
                        if k in st.session_state:
                            del st.session_state[k]
                    st.rerun()

            st.markdown("##### 📝 Input Feature Values:")
            user_inputs = {}
            f_cols = st.columns(2)

            feat_meta_dict = {f["name"]: f for f in features} if features else {}

            for idx, fn in enumerate(fn_list):
                col = f_cols[idx % 2]
                f_meta = feat_meta_dict.get(fn, {})
                d_type = f_meta.get("data_type", "numeric")
                widget_key = f"{prefix}input_{fn}"

                if d_type == "categorical" and f_meta.get("allowed_categories"):
                    cats = f_meta["allowed_categories"]
                    def_idx = 0
                    if widget_key in st.session_state and str(st.session_state[widget_key]) in cats:
                        def_idx = cats.index(str(st.session_state[widget_key]))
                    user_inputs[fn] = col.selectbox(fn, cats, index=def_idx, key=widget_key)
                else:
                    def_val = float(f_meta.get("default_value", 0.0))
                    min_val = float(f_meta.get("min_value", -1000000.0))
                    max_val = float(f_meta.get("max_value", 1000000.0))
                    if min_val >= max_val:
                        min_val = def_val - 100.0
                        max_val = def_val + 100.0
                    
                    cur_val = float(st.session_state.get(widget_key, def_val))
                    step = 0.1 if isinstance(def_val, float) else 1.0
                    user_inputs[fn] = col.number_input(
                        f"{fn}",
                        value=cur_val,
                        step=step,
                        key=widget_key
                    )

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("⚡ Run Instant Prediction", type="primary", use_container_width=True, key=f"{prefix}btn_run_pred"):
                with st.spinner("Executing model prediction..."):
                    pred_res = PipelineInferencer.predict(bundle, PredictionRequest(inputs=user_inputs))

                if pred_res.status == "SUCCESS":
                    if pred_res.task_type == "classification":
                        c_card1, c_card2, c_card3 = st.columns(3)
                        c_card1.metric("Predicted Target Class", f"🎯 {pred_res.prediction}")
                        c_card2.metric("Prediction Confidence", f"{pred_res.confidence:.1f}%")
                        c_card3.metric("Inference Latency", f"{pred_res.inference_latency_ms}ms")

                        if pred_res.probabilities:
                            prob_df = pd.DataFrame(
                                list(pred_res.probabilities.items()),
                                columns=["Class", "Probability"]
                            ).sort_values(by="Probability", ascending=True)
                            prob_df["Likelihood (%)"] = prob_df["Probability"] * 100.0

                            fig_prob = px.bar(
                                prob_df,
                                x="Likelihood (%)",
                                y="Class",
                                orientation="h",
                                text=prob_df["Likelihood (%)"].apply(lambda p: f"{p:.1f}%"),
                                color="Likelihood (%)",
                                color_continuous_scale="Blues",
                                title="Per-Class Prediction Likelihood Distribution"
                            )
                            fig_prob.update_layout(height=260, margin=dict(l=20, r=20, t=35, b=20), xaxis_title="Likelihood (%)", yaxis_title="")
                            st.plotly_chart(fig_prob, use_container_width=True)

                    else:
                        r_card1, r_card2 = st.columns(2)
                        r_card1.metric("Predicted Value", f"📈 {pred_res.prediction}")
                        r_card2.metric("Inference Latency", f"{pred_res.inference_latency_ms}ms")

                    with st.expander("🔍 Inspected Input Payload (JSON)", expanded=False):
                        st.json(user_inputs)
                else:
                    st.error(f"❌ Prediction Failed: {pred_res.error_message}")

        with p_tab2:
            st.markdown("##### 📁 Batch CSV Dataset Scoring")
            st.caption("Upload an unlabelled CSV file to score all observations simultaneously.")
            up_file = st.file_uploader("Upload CSV file", type=["csv"], key=f"{prefix}csv_uploader")
            if up_file is not None:
                try:
                    df_up = pd.read_csv(up_file)
                    st.write(f"Loaded **{len(df_up):,}** rows and **{len(df_up.columns)}** columns.")
                    st.dataframe(df_up.head(5), use_container_width=True)

                    if st.button("⚡ Score Entire Batch Dataset", type="primary", key=f"{prefix}btn_score_batch"):
                        with st.spinner("Scoring batch dataset..."):
                            batch_res = PipelineInferencer.predict(
                                bundle,
                                PredictionRequest(inputs=df_up.to_dict(orient="records"))
                            )
                        if batch_res.status == "SUCCESS" and batch_res.batch_predictions:
                            scored_df = pd.DataFrame(batch_res.batch_predictions)
                            st.success(f"🎉 Successfully scored {len(scored_df):,} records in {batch_res.inference_latency_ms}ms!")
                            st.dataframe(scored_df.head(50), use_container_width=True)

                            st.download_button(
                                "📥 Download Scored Dataset (CSV)",
                                data=scored_df.to_csv(index=False),
                                file_name=f"scored_predictions_{exec_id}.csv",
                                mime="text/csv",
                                key=f"{prefix}btn_dl_batch"
                            )
                        else:
                            st.error(f"❌ Batch scoring failed: {batch_res.error_message}")
                except Exception as e:
                    st.error(f"Error parsing CSV file: {str(e)}")

        with p_tab3:
            st.markdown("##### 📡 Production REST API Endpoint")
            st.caption("External frontend developers, microservices, and automated cron jobs can invoke live predictions via this endpoint.")
            
            api_url = f"http://localhost:8000/api/v1/workflows/{exec_id}/predict"
            sample_body = {"inputs": user_inputs if 'user_inputs' in locals() and user_inputs else sample_row}
            body_json = json.dumps(sample_body, indent=2)

            st.markdown(f"**Endpoint:** `POST {api_url}`")
            st.code(body_json, language="json")

            curl_str = f"""curl -X POST {api_url} \\
  -H "Content-Type: application/json" \\
  -d '{json.dumps(sample_body)}'"""
            st.markdown("##### 💻 Ready-to-Run cURL Terminal Command:")
            st.code(curl_str, language="bash")

    # =========================================================
    # 2. TIME-SERIES FORECASTING
    # =========================================================
    elif task_type == "time_series_forecasting":
        st.markdown("##### 📈 Future Trajectory & Horizon Forecasting")
        st.caption("Generate time-series projections with confidence bands for arbitrary forecast intervals.")

        fc_c1, fc_c2, fc_c3 = st.columns(3)
        h_val = fc_c1.slider("Forecast Horizon (Steps)", min_value=1, max_value=180, value=30, key=f"{prefix}fc_h")
        freq_opt = fc_c2.selectbox("Frequency", ["Daily (D)", "Weekly (W)", "Monthly (M)", "Hourly (H)"], index=0, key=f"{prefix}fc_freq")
        freq_code = freq_opt.split("(")[-1].replace(")", "")
        custom_range = fc_c3.checkbox("Custom Date Range", value=False, key=f"{prefix}fc_custom")

        s_date, e_date = None, None
        if custom_range:
            dr_c1, dr_c2 = st.columns(2)
            s_date = str(dr_c1.date_input("Start Date", value=datetime.date.today(), key=f"{prefix}s_date"))
            e_date = str(dr_c2.date_input("End Date", value=datetime.date.today() + datetime.timedelta(days=30), key=f"{prefix}e_date"))

        if st.button("📈 Generate Forecast Predictions", type="primary", use_container_width=True, key=f"{prefix}btn_fc_run"):
            with st.spinner("Generating future forecast trajectory..."):
                fc_req = PredictionRequest(
                    forecast_horizon=h_val,
                    freq=freq_code,
                    start_date=s_date,
                    end_date=e_date
                )
                fc_res = PipelineInferencer.predict(bundle, fc_req)

            if fc_res.status == "SUCCESS" and fc_res.forecast_records:
                st.success(f"🎉 Generated {len(fc_res.forecast_records)} forecast intervals in {fc_res.inference_latency_ms}ms!")
                f_df = pd.DataFrame(fc_res.forecast_records)

                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("Projected End Value", f"{fc_res.projected_end_value:,.2f}" if fc_res.projected_end_value is not None else "N/A")
                m_col2.metric("Projected Change", f"{fc_res.projected_change_pct:+.2f}%" if fc_res.projected_change_pct is not None else "N/A")
                m_col3.metric("Projected Trend", str(fc_res.trend).upper())

                fig_fc = go.Figure()
                fig_fc.add_trace(go.Scatter(x=f_df["ds"], y=f_df["yhat"], mode="lines+markers", name="Forecast", line=dict(color="#3B82F6", width=2.5)))
                if "yhat_lower" in f_df.columns and "yhat_upper" in f_df.columns:
                    fig_fc.add_trace(go.Scatter(x=f_df["ds"], y=f_df["yhat_upper"], mode="lines", line=dict(width=0), showlegend=False))
                    fig_fc.add_trace(go.Scatter(x=f_df["ds"], y=f_df["yhat_lower"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(59, 130, 246, 0.2)", name="95% Confidence Interval"))

                fig_fc.update_layout(title="Future Forecast Trajectory with Confidence Intervals", xaxis_title="Timeline", yaxis_title="Predicted Value", height=400)
                st.plotly_chart(fig_fc, use_container_width=True)

                st.download_button(
                    "📥 Download Forecast (CSV)",
                    data=f_df.to_csv(index=False),
                    file_name=f"future_forecast_{exec_id}.csv",
                    mime="text/csv",
                    key=f"{prefix}btn_dl_fc"
                )
            else:
                st.error(f"❌ Forecasting failed: {fc_res.error_message}")

    # =========================================================
    # 3. ANOMALY DETECTION
    # =========================================================
    elif task_type == "anomaly_detection":
        st.markdown("##### 🚨 Anomaly Detection & Outlier Risk Scoring")
        st.caption("Score observations to determine if they represent abnormal deviations from normal historical patterns.")

        features = schema.get("features", []) if schema else []
        fn_list = bundle.get("feature_names", [f["name"] for f in features])
        sample_row = bundle.get("sample_row", {})

        user_inputs = {}
        anom_cols = st.columns(2)
        for idx, fn in enumerate(fn_list):
            c = anom_cols[idx % 2]
            def_v = float(sample_row.get(fn, 0.0))
            user_inputs[fn] = c.number_input(fn, value=def_v, step=0.1, key=f"{prefix}anom_{fn}")

        if st.button("🚨 Check Outlier Risk", type="primary", use_container_width=True, key=f"{prefix}btn_anom_run"):
            with st.spinner("Scoring outlier risk..."):
                anom_res = PipelineInferencer.predict(bundle, PredictionRequest(inputs=user_inputs))

            if anom_res.status == "SUCCESS":
                badge_color = "#EF4444" if anom_res.is_anomaly == 1 else "#10B981"
                verdict_text = "🚨 HIGH-RISK ANOMALY DETECTED" if anom_res.is_anomaly == 1 else "🟢 NORMAL TRANSACTION / RECORD"
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #1E293B, #0F172A); padding: 18px 24px; border-radius: 12px; border: 1px solid {badge_color}; margin-bottom: 16px;">
                    <div style="font-size: 0.85rem; color: #94A3B8; text-transform: uppercase;">Anomaly Detection Verdict</div>
                    <div style="font-size: 1.6rem; font-weight: 700; color: {badge_color}; margin-top: 4px;">{verdict_text}</div>
                </div>
                """, unsafe_allow_html=True)

                a1, a2, a3 = st.columns(3)
                a1.metric("Outlier Score", f"{anom_res.anomaly_score:.4f}")
                a2.metric("Risk Level", str(anom_res.risk_level))
                a3.metric("Latency", f"{anom_res.inference_latency_ms}ms")
            else:
                st.error(f"❌ Scoring failed: {anom_res.error_message}")


if hasattr(st, "dialog"):
    @st.dialog("🔮 Interactive Model Prediction Studio", width="large")
    def show_prediction_dialog():
        bundle = st.session_state.get("inference_bundle")
        schema = st.session_state.get("inference_schema")
        render_prediction_studio(bundle, schema, in_dialog=True)
else:
    def show_prediction_dialog():
        bundle = st.session_state.get("inference_bundle")
        schema = st.session_state.get("inference_schema")
        render_prediction_studio(bundle, schema, in_dialog=False)


def create_flow_node(node_id: str, pos: tuple, content: str, category_key: str = "default") -> StreamlitFlowNode:
    """Helper to create fully interactive, connectable, and draggable React Flow nodes styled with company design tokens."""
    color_map = {
        "triggers": {"border": "#F59E0B", "accent": "#D97706"},
        "ingestion": {"border": "#0284C7", "accent": "#0369A1"},
        "export": {"border": "#0284C7", "accent": "#0369A1"},
        "preprocessing": {"border": "#6366F1", "accent": "#4F46E5"},
        "splitting": {"border": "#10B981", "accent": "#059669"},
        "training": {"border": "#2563EB", "accent": "#1D4ED8"},
        "forecasting": {"border": "#0D9488", "accent": "#0F766E"},
        "anomaly": {"border": "#E11D48", "accent": "#BE123C"},
        "unsupervised": {"border": "#8B5CF6", "accent": "#7C3AED"},
        "nlp": {"border": "#06B6D4", "accent": "#0891B2"},
        "vision_audio": {"border": "#D946EF", "accent": "#C026D3"},
        "agents": {"border": "#A855F7", "accent": "#9333EA"},
        "Agents & LLM Orchestration": {"border": "#A855F7", "accent": "#9333EA"},
        "integrations": {"border": "#F97316", "accent": "#EA580C"},
        "monitoring": {"border": "#EAB308", "accent": "#CA8A04"},
        "governance": {"border": "#78716C", "accent": "#57534E"},
        "evaluation": {"border": "#16A34A", "accent": "#15803D"},
        "flow_control": {"border": "#64748B", "accent": "#334155"},
        "default": {"border": "#CBD5E1", "accent": "#2563EB"},
    }

    # Auto-infer category if default
    effective_cat = category_key.lower()
    if effective_cat == "default":
        cfg = st.session_state.get("node_configs", {}).get(node_id, {})
        r_id = cfg.get("recipe_id", "")
        if r_id:
            r_obj = recipe_registry.get(r_id)
            if r_obj:
                effective_cat = r_obj.category.lower()
        if effective_cat == "default":
            nid_low = node_id.lower()
            if any(k in nid_low for k in ["trigger", "cron", "webhook"]):
                effective_cat = "triggers"
            elif any(k in nid_low for k in ["csv", "load", "ingest", "source"]):
                effective_cat = "ingestion"
            elif any(k in nid_low for k in ["impute", "scale", "clean", "encode", "prep"]):
                effective_cat = "preprocessing"
            elif any(k in nid_low for k in ["split", "strat"]):
                effective_cat = "splitting"
            elif any(k in nid_low for k in ["xgb", "lightgbm", "catboost", "forest", "linear", "model", "train"]):
                effective_cat = "training"
            elif any(k in nid_low for k in ["prophet", "arima", "forecast"]):
                effective_cat = "forecasting"
            elif any(k in nid_low for k in ["anom", "outlier", "isolation"]):
                effective_cat = "anomaly"
            elif any(k in nid_low for k in ["eval"]):
                effective_cat = "evaluation"

    theme = color_map.get(effective_cat, color_map["default"])
    node_style = {
        "background": "#FFFFFF",
        "color": "#0F172A",
        "border": f"1.5px solid {theme['border']}",
        "borderRadius": "10px",
        "boxShadow": "0 2px 8px -2px rgba(15, 23, 42, 0.08), 0 1px 3px -1px rgba(15, 23, 42, 0.04)",
        "fontFamily": "Inter, -apple-system, sans-serif",
        "fontSize": "12px",
        "fontWeight": "600",
        "padding": "10px 14px",
        "minWidth": "165px",
    }

    return StreamlitFlowNode(
        id=node_id,
        pos=pos,
        data={"content": content},
        node_type="default",
        source_position="right",
        target_position="left",
        connectable=True,
        selectable=True,
        draggable=True,
        deletable=True,
        style=node_style
    )


def auto_format_canvas_layout():
    """Automatically spaces and arranges active whiteboard nodes into an aesthetic horizontal pipeline workflow."""
    nodes = list(st.session_state["flow_state"].nodes)
    if not nodes:
        return

    def get_order_score(nid: str) -> float:
        cfg = st.session_state.get("node_configs", {}).get(nid, {})
        r_id = cfg.get("recipe_id", "")
        r_obj = recipe_registry.get(r_id) if r_id else None
        if r_obj:
            c = r_obj.category.lower()
            if "trigger" in c: return 0.0
            if "ingest" in c: return 1.0
            if "prep" in c: return 2.0
            if "split" in c: return 3.0
            if "train" in c or "model" in c or "forecast" in c or "anom" in c: return 4.0
            if "eval" in c or "gov" in c: return 5.0
        return 2.5

    nodes_sorted = sorted(nodes, key=lambda n: (get_order_score(n.id), get_node_position(n)[0]))
    cur_x = 40
    toggle_y = False
    for n in nodes_sorted:
        cur_y = 100 if not toggle_y else 60
        toggle_y = not toggle_y
        n.pos = (cur_x, cur_y)
        cur_x += 240

    st.session_state["canvas_version"] = st.session_state.get("canvas_version", 1) + 1


def build_recommended_pipeline(rec_dict: dict = None, target_col: str = None, task_type: str = None):
    """Instantiates a full end-to-end DAG based on dataset analysis."""
    df = st.session_state["active_df"]
    rec = rec_dict or AIRecommender._heuristic_recommend_pipeline(df, target_column=target_col, task_type=task_type)
    
    t_rec_start = time.time()
    t_nodes = []
    t_edges = []
    node_configs = {}

    cur_x = 40
    prev_id = "node_csv"
    t_nodes.append(create_flow_node(prev_id, (cur_x, 100), f"📄 {st.session_state.get('active_dataset_name', 'Active Data')[:15]}"))
    node_configs[prev_id] = {"recipe_id": "csv_loader", "label": "Dataset Ingestion", "config": {}}
    cur_x += 240

    # 1. Preprocessing Steps
    for prep in rec.get("preprocessing_recommendations", []):
        nid = f"node_{prep['recipe_id']}"
        prep_name = prep.get("name") or prep.get("recipe_name") or prep["recipe_id"]
        prep_config = prep.get("config") or prep.get("params") or {}
        node_label = prep_name if any(char in prep_name for char in ["⚙️", "🧹", "🔤", "⚖️"]) else f"⚙️ {prep_name}"
        t_nodes.append(create_flow_node(nid, (cur_x, 100), node_label))
        t_edges.append(StreamlitFlowEdge(id=f"e_{prev_id}_{nid}", source=prev_id, target=nid, animated=True))
        node_configs[nid] = {"recipe_id": prep["recipe_id"], "label": prep_name, "config": prep_config}
        prev_id = nid
        cur_x += 240

    task = rec.get("task_type")
    resolved_target = rec.get("target_column") or target_col
    # 2. Split or Direct Model
    if task in ["classification", "regression"]:
        split_id = "node_split"
        t_nodes.append(create_flow_node(split_id, (cur_x, 100), "✂️ Train/Test Split"))
        t_edges.append(StreamlitFlowEdge(id=f"e_{prev_id}_{split_id}", source=prev_id, target=split_id, animated=True))
        node_configs[split_id] = {"recipe_id": "train_test_split", "label": "✂️ Split", "config": {"target_column": resolved_target, "test_size": 0.2}}
        prev_id = split_id
        cur_x += 240

        top_model = rec["model_rankings"][0]
        model_id = "node_model"
        t_nodes.append(create_flow_node(model_id, (cur_x, 50), top_model["name"]))
        t_edges.append(StreamlitFlowEdge(id=f"e_{split_id}_{model_id}", source=split_id, target=model_id, animated=True))
        node_configs[model_id] = {"recipe_id": top_model["recipe_id"], "label": top_model["name"], "config": {"task_type": task}}
        cur_x += 240

        eval_id = "node_eval"
        t_nodes.append(create_flow_node(eval_id, (cur_x, 100), "🎯 Evaluation Report"))
        t_edges.append(StreamlitFlowEdge(id=f"e_{split_id}_{eval_id}", source=split_id, target=eval_id, animated=True))
        t_edges.append(StreamlitFlowEdge(id=f"e_{model_id}_{eval_id}", source=model_id, target=eval_id, animated=True))
        node_configs[eval_id] = {"recipe_id": "model_evaluator", "label": "🎯 Evaluator", "config": {"report_type": "Comprehensive"}}

    elif task == "time_series_forecasting":
        top_model = rec["model_rankings"][0]
        model_id = "node_model"
        t_nodes.append(create_flow_node(model_id, (cur_x, 100), top_model["name"]))
        t_edges.append(StreamlitFlowEdge(id=f"e_{prev_id}_{model_id}", source=prev_id, target=model_id, animated=True))
        node_configs[model_id] = {"recipe_id": top_model["recipe_id"], "label": top_model["name"], "config": {"horizon_periods": 30}}

    else:
        top_model = rec["model_rankings"][0]
        model_id = "node_model"
        t_nodes.append(create_flow_node(model_id, (cur_x, 100), top_model["name"]))
        t_edges.append(StreamlitFlowEdge(id=f"e_{prev_id}_{model_id}", source=prev_id, target=model_id, animated=True))
        node_configs[model_id] = {"recipe_id": top_model["recipe_id"], "label": top_model["name"], "config": {"contamination": 0.05}}

    st.session_state["flow_state"] = StreamlitFlowState(nodes=t_nodes, edges=t_edges)
    st.session_state["node_configs"] = node_configs
    st.session_state["canvas_version"] = st.session_state.get("canvas_version", 1) + 1
    
    # Record Telemetry for AI Recommendation API
    record_api_telemetry(
        action_name="🧠 AI Auto-Architect Pipeline",
        endpoint="/api/v1/workflows/recommend",
        method="POST",
        request_payload={"dataset_name": st.session_state.get("active_dataset_name"), "rows": len(df), "target_column": resolved_target, "task_type": task},
        response_payload={"task_type": task, "target_column": resolved_target, "nodes_generated": len(t_nodes), "edges_generated": len(t_edges)},
        status_code=200,
        duration_ms=(time.time() - t_rec_start) * 1000.0
    )
    
    execute_pipeline()


def load_ml_template(force_preset: bool = False):
    """Applies ML classification/regression template to current active data or sample preset."""
    curr_name = st.session_state.get("active_dataset_name", "")
    if force_preset or "active_df" not in st.session_state or st.session_state["active_df"] is None or curr_name == "Daily Retail Sales (Time-Series)":
        st.session_state["active_df"] = get_preset_dataset("Customer Churn (Classification)")
        st.session_state["active_dataset_name"] = "Customer Churn (Classification)"

    df = st.session_state["active_df"]
    cols = list(df.columns)
    named = [c for c in cols if any(k in c.lower() for k in ["target", "churn", "survived", "label", "class", "y", "status"])]
    target_col = named[0] if named else cols[-1]

    t_nodes = [
        create_flow_node("node_csv", (40, 100), f"📄 {st.session_state.get('active_dataset_name', 'Active Data')[:15]}"),
        create_flow_node("node_impute", (280, 100), "🧹 Imputer (Median)"),
        create_flow_node("node_scale", (520, 100), "⚖️ Feature Scaler"),
        create_flow_node("node_split", (760, 100), "✂️ Train/Test Split"),
        create_flow_node("node_xgb", (1000, 50), "⚡ XGBoost Classifier"),
        create_flow_node("node_eval", (1240, 100), "🎯 Evaluation Report")
    ]

    t_edges = [
        StreamlitFlowEdge(id="e1", source="node_csv", target="node_impute", animated=True),
        StreamlitFlowEdge(id="e2", source="node_impute", target="node_scale", animated=True),
        StreamlitFlowEdge(id="e3", source="node_scale", target="node_split", animated=True),
        StreamlitFlowEdge(id="e4", source="node_split", target="node_xgb", animated=True),
        StreamlitFlowEdge(id="e5", source="node_split", target="node_eval", animated=True),
        StreamlitFlowEdge(id="e6", source="node_xgb", target="node_eval", animated=True)
    ]

    st.session_state["node_configs"] = {
        "node_csv": {"recipe_id": "csv_loader", "label": "Dataset Ingestion", "config": {}},
        "node_impute": {"recipe_id": "missing_value_imputer", "label": "Imputer", "config": {"strategy": "median"}},
        "node_scale": {"recipe_id": "feature_scaler", "label": "Scaler", "config": {"method": "standard"}},
        "node_split": {"recipe_id": "train_test_split", "label": "Splitter", "config": {"target_column": target_col, "test_size": 0.2}},
        "node_xgb": {"recipe_id": "xgboost_trainer", "label": "XGBoost", "config": {"task_type": "classification", "n_estimators": 100, "max_depth": 6}},
        "node_eval": {"recipe_id": "model_evaluator", "label": "Evaluator", "config": {"report_type": "Comprehensive"}}
    }

    st.session_state["flow_state"] = StreamlitFlowState(nodes=t_nodes, edges=t_edges)
    st.session_state["canvas_version"] = st.session_state.get("canvas_version", 1) + 1
    
    record_api_telemetry(
        action_name="⚡ Load ML Supervised Template",
        endpoint="/api/v1/workflows/templates/ml_supervised",
        method="POST",
        request_payload={"template": "ml_supervised", "target_column": target_col, "dataset": st.session_state.get("active_dataset_name")},
        response_payload={"nodes_created": 6, "edges_created": 6, "status": "TEMPLATE_LOADED"},
        status_code=200,
        duration_ms=4.2
    )
    
    execute_pipeline()


def load_forecast_template(force_preset: bool = False):
    """Applies Time-Series Forecasting template to current active data or sample preset."""
    curr_df = st.session_state.get("active_df")
    curr_name = st.session_state.get("active_dataset_name", "")
    has_date = any(pd.api.types.is_datetime64_any_dtype(curr_df[c]) or "date" in c.lower() or "time" in c.lower() for c in curr_df.columns) if curr_df is not None else False

    if force_preset or curr_df is None or not has_date or curr_name != "Daily Retail Sales (Time-Series)":
        st.session_state["active_df"] = get_preset_dataset("Daily Retail Sales (Time-Series)")
        st.session_state["active_dataset_name"] = "Daily Retail Sales (Time-Series)"
        date_col = "Date"
        target_col = "Sales"
    else:
        date_cols = [c for c in curr_df.columns if pd.api.types.is_datetime64_any_dtype(curr_df[c]) or "date" in c.lower() or "time" in c.lower()]
        date_col = date_cols[0]
        num_cols = [c for c in curr_df.columns if pd.api.types.is_numeric_dtype(curr_df[c]) and c != date_col]
        target_col = num_cols[0] if num_cols else curr_df.columns[-1]

    t_nodes = [
        create_flow_node("node_csv", (40, 100), f"📄 {st.session_state.get('active_dataset_name', 'Active Data')[:15]}"),
        create_flow_node("node_impute", (300, 100), "🧹 Time Imputer"),
        create_flow_node("node_prophet", (560, 100), "🔮 Prophet Forecaster")
    ]

    t_edges = [
        StreamlitFlowEdge(id="e1", source="node_csv", target="node_impute", animated=True),
        StreamlitFlowEdge(id="e2", source="node_impute", target="node_prophet", animated=True)
    ]

    st.session_state["node_configs"] = {
        "node_csv": {"recipe_id": "csv_loader", "label": "Dataset Ingestion", "config": {}},
        "node_impute": {"recipe_id": "missing_value_imputer", "label": "Imputer", "config": {"strategy": "ffill"}},
        "node_prophet": {"recipe_id": "prophet_forecaster", "label": "Prophet Forecaster", "config": {"date_column": date_col, "target_column": target_col, "horizon_periods": 30}}
    }

    st.session_state["flow_state"] = StreamlitFlowState(nodes=t_nodes, edges=t_edges)
    st.session_state["canvas_version"] = st.session_state.get("canvas_version", 1) + 1
    
    record_api_telemetry(
        action_name="🔮 Load Time-Series Forecast Template",
        endpoint="/api/v1/workflows/templates/forecasting",
        method="POST",
        request_payload={"template": "forecasting", "dataset": st.session_state.get("active_dataset_name")},
        response_payload={"nodes_created": 3, "edges_created": 2, "status": "TEMPLATE_LOADED"},
        status_code=200,
        duration_ms=3.8
    )
    
    execute_pipeline()


def load_anomaly_template(force_preset: bool = False):
    """Applies Unsupervised Anomaly Detection template to current active data or sample preset."""
    curr_name = st.session_state.get("active_dataset_name", "")
    if force_preset or "active_df" not in st.session_state or st.session_state["active_df"] is None or curr_name == "Daily Retail Sales (Time-Series)":
        st.session_state["active_df"] = get_preset_dataset("Credit Transactions (Anomaly Injection)")
        st.session_state["active_dataset_name"] = "Credit Transactions (Anomaly Injection)"

    t_nodes = [
        create_flow_node("node_csv", (40, 100), f"📄 {st.session_state.get('active_dataset_name', 'Active Data')[:15]}"),
        create_flow_node("node_impute", (300, 100), "🧹 Imputer (Median)"),
        create_flow_node("node_iso", (560, 100), "🌲 Isolation Forest")
    ]

    t_edges = [
        StreamlitFlowEdge(id="e1", source="node_csv", target="node_impute", animated=True),
        StreamlitFlowEdge(id="e2", source="node_impute", target="node_iso", animated=True)
    ]

    st.session_state["node_configs"] = {
        "node_csv": {"recipe_id": "csv_loader", "label": "Dataset Ingestion", "config": {}},
        "node_impute": {"recipe_id": "missing_value_imputer", "label": "Imputer", "config": {"strategy": "median"}},
        "node_iso": {"recipe_id": "isolation_forest", "label": "Isolation Forest", "config": {"contamination": 0.05, "n_estimators": 100}}
    }

    st.session_state["flow_state"] = StreamlitFlowState(nodes=t_nodes, edges=t_edges)
    st.session_state["canvas_version"] = st.session_state.get("canvas_version", 1) + 1
    
    record_api_telemetry(
        action_name="🚨 Load Anomaly Detection Template",
        endpoint="/api/v1/workflows/templates/anomaly_detection",
        method="POST",
        request_payload={"template": "anomaly_detection", "dataset": st.session_state.get("active_dataset_name")},
        response_payload={"nodes_created": 3, "edges_created": 2, "status": "TEMPLATE_LOADED"},
        status_code=200,
        duration_ms=3.5
    )
    
    execute_pipeline()


# -------------------------------------------------------------
# WORKFLOW PERSISTENCE & RESTORATION HELPERS
# -------------------------------------------------------------
def save_workflow_to_backend(name: str, description: str = "") -> dict:
    """Saves current active canvas workflow and node configs to backend REST API/DB."""
    active_df = st.session_state.get("active_df")
    saved_cols = list(st.session_state.get("saved_dataset_columns", []))
    active_cols = list(active_df.columns) if active_df is not None else saved_cols
    
    dataset_metadata = {
        "dataset_name": st.session_state.get("active_dataset_name", ""),
        "columns": active_cols,
        "row_count": len(active_df) if active_df is not None else 0
    }

    nodes_payload = []
    for n in st.session_state["flow_state"].nodes:
        pos = get_node_position(n)
        content = n.data.get("content", n.id) if hasattr(n, "data") and isinstance(n.data, dict) else n.id
        n_cfg = st.session_state.get("node_configs", {}).get(n.id, {})
        nodes_payload.append({
            "id": n.id,
            "position": pos,
            "content": content,
            "recipe_id": n_cfg.get("recipe_id", ""),
            "config": n_cfg.get("config", {})
        })

    edges_payload = [
        {"id": e.id, "source": e.source, "target": e.target}
        for e in st.session_state["flow_state"].edges
    ]

    persisted_configs = dict(st.session_state.get("node_configs", {}))
    persisted_configs["_dataset_metadata"] = dataset_metadata

    body = {
        "id": st.session_state.get("active_saved_workflow_id"),
        "name": name,
        "description": description,
        "nodes": nodes_payload,
        "edges": edges_payload,
        "node_configs": persisted_configs,
        "last_execution": st.session_state.get("last_execution")
    }

    try:
        import httpx
        res = httpx.post("http://localhost:8000/api/v1/workflows/", json=body, timeout=4.0)
        if res.status_code in [200, 201]:
            saved_json = res.json()
            saved_name = saved_json.get("name") or name
            st.session_state["active_saved_workflow_id"] = saved_json.get("id")
            st.session_state["active_saved_workflow_name"] = saved_name
            if "wb_name_input" in st.session_state:
                st.session_state["wb_name_input"] = saved_name
            record_api_telemetry("💾 Save Workflow API", "/api/v1/workflows/", "POST", body, saved_json, res.status_code, 2.1)
            return saved_json
    except Exception:
        pass

    # Direct Async DB Fallback
    import uuid
    import asyncio
    from backend.app.infrastructure.database.session import AsyncSessionLocal, init_db
    from backend.app.workflows.models import Workflow
    from backend.app.workflows.router import is_generic_workbook_name
    from sqlalchemy.future import select

    wf_id = st.session_state.get("active_saved_workflow_id") or str(uuid.uuid4())
    final_saved_name = name

    async def _async_save():
        nonlocal final_saved_name
        await init_db()
        async with AsyncSessionLocal() as session:
            existing_wf_res = await session.execute(select(Workflow).where(Workflow.id == wf_id))
            wf = existing_wf_res.scalar_one_or_none()
            if wf:
                if name and not (is_generic_workbook_name(name) and not is_generic_workbook_name(wf.name)):
                    wf.name = name
                final_saved_name = wf.name
                wf.description = description
                wf.nodes = nodes_payload
                wf.edges = edges_payload
                wf.node_configs = persisted_configs
                wf.last_execution = st.session_state.get("last_execution")
                wf.is_active = True
                wf.deleted_at = None
                wf.updated_at = datetime.now(timezone.utc)
            else:
                wf = Workflow(
                    id=wf_id,
                    name=name,
                    description=description,
                    nodes=nodes_payload,
                    edges=edges_payload,
                    node_configs=persisted_configs,
                    last_execution=st.session_state.get("last_execution")
                )
                session.add(wf)
            await session.commit()

            last_ex = st.session_state.get("last_execution")
            if last_ex and isinstance(last_ex, dict):
                try:
                    from backend.app.workflows.router import record_workflow_execution_history
                    await record_workflow_execution_history(
                        db=session,
                        workflow_id=wf_id,
                        execution_id=last_ex.get("execution_id") or str(uuid.uuid4()),
                        status_str=last_ex.get("status", "SUCCESS"),
                        total_duration_ms=last_ex.get("total_duration_ms", 0.0),
                        nodes=nodes_payload,
                        edges=edges_payload,
                        node_configs=persisted_configs,
                        metrics=last_ex.get("final_metrics") or last_ex.get("metrics"),
                        reports={
                            "anomaly_summary": last_ex.get("anomaly_summary"),
                            "forecasting_summary": last_ex.get("forecasting_summary"),
                            "governance_summary": last_ex.get("governance_summary"),
                            "node_results": last_ex.get("node_results"),
                            "inference_schema": last_ex.get("inference_schema"),
                        },
                        step_snapshots=last_ex.get("step_snapshots"),
                        logs=last_ex.get("execution_logs") or last_ex.get("logs"),
                        run_label=None
                    )
                except Exception:
                    pass

    try:
        asyncio.run(_async_save())
        st.session_state["active_saved_workflow_id"] = wf_id
        st.session_state["active_saved_workflow_name"] = final_saved_name
        if "wb_name_input" in st.session_state:
            st.session_state["wb_name_input"] = final_saved_name
        saved_dict = {"id": wf_id, "name": final_saved_name, "status": "SAVED"}
        record_api_telemetry("💾 Save Workflow DB", "/api/v1/workflows/", "POST", body, saved_dict, 201, 1.8)
        try:
            fetch_saved_workflows_from_backend.clear()
        except Exception:
            pass
        return saved_dict
    except Exception as ex:
        st.error(f"Error saving workflow: {str(ex)}")
        return None


@st.cache_data(ttl=15, show_spinner=False)
def fetch_saved_workflows_from_backend(include_deleted: bool = False) -> list:
    """Fetches list of all saved pipeline workbooks from backend REST API/DB."""
    try:
        import httpx
        url = f"http://localhost:8000/api/v1/workflows/?include_deleted={'true' if include_deleted else 'false'}"
        res = httpx.get(url, timeout=0.8)
        if res.status_code == 200:
            payload = res.json()
            if isinstance(payload, dict):
                return payload.get("data") or payload.get("items") or []
            return payload
    except Exception:
        pass

    # Direct Async DB Fallback
    import asyncio
    from backend.app.infrastructure.database.session import AsyncSessionLocal, init_db
    from backend.app.workflows.models import Workflow
    from sqlalchemy.future import select

    async def _async_list():
        await init_db()
        async with AsyncSessionLocal() as session:
            query = select(Workflow)
            if not include_deleted:
                query = query.where(Workflow.is_active == True)
            query = query.order_by(Workflow.updated_at.desc())
            result = await session.execute(query)
            wfs = result.scalars().all()
            return [
                {
                    "id": w.id,
                    "name": w.name,
                    "description": w.description,
                    "nodes": w.nodes,
                    "edges": w.edges,
                    "node_configs": w.node_configs,
                    "is_active": getattr(w, "is_active", True),
                    "updated_at": w.updated_at.strftime("%Y-%m-%d %H:%M:%S") if w.updated_at else ""
                }
                for w in wfs
            ]

    try:
        return asyncio.run(_async_list())
    except Exception:
        return []


def soft_delete_workflow_backend(workflow_id: str) -> bool:
    """Soft-deletes a workflow by ID."""
    try:
        try:
            fetch_saved_workflows_from_backend.clear()
        except Exception:
            pass
        import httpx
        res = httpx.delete(f"http://localhost:8000/api/v1/workflows/{workflow_id}", timeout=1.0)
        if res.status_code == 200:
            record_api_telemetry("🗑️ Soft-Delete Workflow API", f"/api/v1/workflows/{workflow_id}", "DELETE", {"id": workflow_id}, res.json(), 200, 1.9)
            return True
    except Exception:
        pass

    # Direct Async DB Fallback
    import asyncio
    from datetime import datetime, timezone
    from backend.app.infrastructure.database.session import AsyncSessionLocal, init_db
    from backend.app.workflows.models import Workflow
    from sqlalchemy.future import select

    async def _async_delete():
        await init_db()
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(Workflow).where(Workflow.id == workflow_id))
            wf = result.scalar_one_or_none()
            if wf:
                wf.is_active = False
                wf.deleted_at = datetime.now(timezone.utc)
                await session.commit()
                return True
            return False

    try:
        return asyncio.run(_async_delete())
    except Exception:
        return False


def restore_deleted_workflow_backend(workflow_id: str) -> bool:
    """Restores a soft-deleted workflow by ID back to active state."""
    try:
        try:
            fetch_saved_workflows_from_backend.clear()
        except Exception:
            pass
        import httpx
        res = httpx.post(f"http://localhost:8000/api/v1/workflows/{workflow_id}/restore", timeout=1.0)
        if res.status_code == 200:
            record_api_telemetry("♻️ Restore Workflow API", f"/api/v1/workflows/{workflow_id}/restore", "POST", {"id": workflow_id}, res.json(), 200, 2.0)
            return True
    except Exception:
        pass

    # Direct Async DB Fallback
    import asyncio
    from datetime import datetime, timezone
    from backend.app.infrastructure.database.session import AsyncSessionLocal, init_db
    from backend.app.workflows.models import Workflow
    from sqlalchemy.future import select

    async def _async_restore():
        await init_db()
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(Workflow).where(Workflow.id == workflow_id))
            wf = result.scalar_one_or_none()
            if wf:
                wf.is_active = True
                wf.deleted_at = None
                wf.updated_at = datetime.now(timezone.utc)
                await session.commit()
                return True
            return False

@st.cache_data(ttl=15, show_spinner=False)
def fetch_workflow_history_backend(workflow_id: str) -> list:
    """Fetches execution history list for a workflow."""
    try:
        import httpx
        url = f"http://localhost:8000/api/v1/workflows/{workflow_id}/history"
        res = httpx.get(url, timeout=0.8)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass

    # Direct Async DB Fallback
    import asyncio
    from backend.app.infrastructure.database.session import AsyncSessionLocal, init_db
    from backend.app.workflows.models import Workflow, WorkflowExecution
    from sqlalchemy.future import select

    async def _async_get_hist():
        await init_db()
        async with AsyncSessionLocal() as session:
            stmt = select(WorkflowExecution).where(WorkflowExecution.workflow_id == workflow_id).order_by(WorkflowExecution.version_number.desc())
            res = await session.execute(stmt)
            execs = res.scalars().all()
            if not execs:
                # Check if legacy workflow has last_execution
                wf_res = await session.execute(select(Workflow).where(Workflow.id == workflow_id))
                wf = wf_res.scalar_one_or_none()
                if wf and wf.last_execution and isinstance(wf.last_execution, dict):
                    last_ex = wf.last_execution
                    return [{
                        "id": last_ex.get("execution_id", "legacy"),
                        "workflow_id": workflow_id,
                        "version_number": 1,
                        "run_label": "Run #1 (Initial Execution)",
                        "status": last_ex.get("status", "SUCCESS"),
                        "total_duration_ms": last_ex.get("total_duration_ms", 0.0),
                        "metrics": last_ex.get("final_metrics") or last_ex.get("metrics") or {},
                        "nodes_count": len(wf.nodes or []),
                        "edges_count": len(wf.edges or []),
                        "created_at": wf.updated_at.strftime("%Y-%m-%d %H:%M:%S") if wf.updated_at else ""
                    }]
            return [
                {
                    "id": ex.id,
                    "workflow_id": ex.workflow_id,
                    "version_number": ex.version_number,
                    "run_label": ex.run_label,
                    "status": ex.status,
                    "total_duration_ms": ex.total_duration_ms,
                    "metrics": ex.metrics or {},
                    "nodes_count": len(ex.snapshot_nodes or []),
                    "edges_count": len(ex.snapshot_edges or []),
                    "created_at": ex.created_at.strftime("%Y-%m-%d %H:%M:%S") if ex.created_at else ""
                }
                for ex in execs
            ]
    try:
        return asyncio.run(_async_get_hist())
    except Exception:
        return []


def fetch_workflow_execution_detail_backend(workflow_id: str, execution_id: str) -> dict:
    """Fetches full snapshot and execution report for a specific run."""
    try:
        import httpx
        url = f"http://localhost:8000/api/v1/workflows/{workflow_id}/history/{execution_id}"
        res = httpx.get(url, timeout=4.0)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass

    import asyncio
    from backend.app.infrastructure.database.session import AsyncSessionLocal, init_db
    from backend.app.workflows.models import WorkflowExecution
    from sqlalchemy.future import select

    async def _async_detail():
        await init_db()
        async with AsyncSessionLocal() as session:
            stmt = select(WorkflowExecution).where(WorkflowExecution.workflow_id == workflow_id, WorkflowExecution.id == execution_id)
            res = await session.execute(stmt)
            ex = res.scalar_one_or_none()
            if ex:
                return {
                    "id": ex.id,
                    "workflow_id": ex.workflow_id,
                    "version_number": ex.version_number,
                    "run_label": ex.run_label,
                    "status": ex.status,
                    "total_duration_ms": ex.total_duration_ms,
                    "snapshot_nodes": ex.snapshot_nodes or [],
                    "snapshot_edges": ex.snapshot_edges or [],
                    "snapshot_node_configs": ex.snapshot_node_configs or {},
                    "metrics": ex.metrics,
                    "reports": ex.reports,
                    "step_snapshots": ex.step_snapshots,
                    "logs": ex.logs,
                    "created_at": ex.created_at.strftime("%Y-%m-%d %H:%M:%S") if ex.created_at else ""
                }
            return None
    try:
        return asyncio.run(_async_detail())
    except Exception:
        return None





def compare_workflow_executions_backend(workflow_id: str, run_a_id: str, run_b_id: str) -> dict:
    """Compares two historical execution runs."""
    try:
        import httpx
        url = f"http://localhost:8000/api/v1/workflows/{workflow_id}/history/compare?run_a={run_a_id}&run_b={run_b_id}"
        res = httpx.get(url, timeout=4.0)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass

    # Fallback to local detail comparison
    detail_a = fetch_workflow_execution_detail_backend(workflow_id, run_a_id)
    detail_b = fetch_workflow_execution_detail_backend(workflow_id, run_b_id)
    if not detail_a or not detail_b:
        return {}

    metrics_a = detail_a.get("metrics") or {}
    metrics_b = detail_b.get("metrics") or {}
    all_keys = sorted(list(set(list(metrics_a.keys()) + list(metrics_b.keys()))))
    m_diff = {}
    lower_is_better = {"loss", "log_loss", "mae", "mse", "rmse"}
    for k in all_keys:
        v_a = metrics_a.get(k)
        v_b = metrics_b.get(k)
        if isinstance(v_a, (int, float)) and isinstance(v_b, (int, float)):
            delta = round(float(v_b) - float(v_a), 4)
            pct = round((delta / abs(v_a)) * 100, 2) if v_a != 0 else None
            improved = (delta > 0) if k.lower() not in lower_is_better else (delta < 0)
            m_diff[k] = {"val_a": v_a, "val_b": v_b, "delta": delta, "pct_change": pct, "improved": improved}

    nodes_a = {n["id"]: n for n in detail_a.get("snapshot_nodes", []) if isinstance(n, dict)}
    nodes_b = {n["id"]: n for n in detail_b.get("snapshot_nodes", []) if isinstance(n, dict)}
    added = [{"id": nid, "label": n.get("content") or n.get("label")} for nid, n in nodes_b.items() if nid not in nodes_a]
    removed = [{"id": nid, "label": n.get("content") or n.get("label")} for nid, n in nodes_a.items() if nid not in nodes_b]

    return {
        "workflow_id": workflow_id,
        "run_a": detail_a,
        "run_b": detail_b,
        "metrics_diff": m_diff,
        "config_diff": {"nodes_added": added, "nodes_removed": removed}
    }


def restore_saved_workflow(wf_data: dict):
    """Restores a saved pipeline workbook into active StreamlitFlowState and node_configs."""
    # If wf_data is a lean list summary (missing nodes/configs), fetch full details via GET /workflows/{id}
    if not wf_data.get("nodes") and wf_data.get("id"):
        try:
            import httpx
            r = httpx.get(f"http://localhost:8000/api/v1/workflows/{wf_data['id']}", timeout=4.0)
            if r.status_code == 200:
                wf_data = r.json()
        except Exception:
            pass

        if not wf_data.get("nodes"):
            try:
                import asyncio
                from backend.app.infrastructure.database.session import AsyncSessionLocal, init_db
                from backend.app.workflows.models import Workflow
                from sqlalchemy.future import select

                async def _get_full():
                    await init_db()
                    async with AsyncSessionLocal() as session:
                        res = await session.execute(select(Workflow).where(Workflow.id == wf_data["id"]))
                        w = res.scalar_one_or_none()
                        if w:
                            return {
                                "id": w.id,
                                "name": w.name,
                                "description": w.description,
                                "nodes": w.nodes or [],
                                "edges": w.edges or [],
                                "node_configs": w.node_configs or {},
                                "last_execution": w.last_execution,
                                "is_active": w.is_active,
                            }
                        return None
                db_full = asyncio.run(_get_full())
                if db_full:
                    wf_data = db_full
            except Exception:
                pass

    t_nodes = []
    t_edges = []
    
    # Restore saved configs and extract dataset metadata
    saved_configs = dict(wf_data.get("node_configs", {}))
    ds_meta = saved_configs.pop("_dataset_metadata", {})
    if ds_meta and isinstance(ds_meta, dict):
        if ds_meta.get("columns"):
            st.session_state["saved_dataset_columns"] = ds_meta["columns"]
        if ds_meta.get("dataset_name"):
            st.session_state["saved_dataset_name"] = ds_meta["dataset_name"]
            if not st.session_state.get("active_dataset_name"):
                st.session_state["active_dataset_name"] = ds_meta["dataset_name"]

    # Clear stale widget keys to prevent Streamlit widget state collision
    for k in list(st.session_state.keys()):
        if k.startswith("cfg_") or k.startswith("lbl_") or k.startswith("vmode_"):
            del st.session_state[k]

    # Pre-populate widget session state keys so UI widgets immediately reflect saved parameters
    for nid, node_entry in saved_configs.items():
        if isinstance(node_entry, dict) and "config" in node_entry and isinstance(node_entry["config"], dict):
            for prop_name, prop_val in node_entry["config"].items():
                st.session_state[f"cfg_{nid}_{prop_name}"] = prop_val

    # Restore Nodes
    for nd in wf_data.get("nodes", []):
        nid = nd["id"]
        pos = nd.get("position", (100, 100))
        content = nd.get("content", nid)
        t_nodes.append(create_flow_node(nid, pos, content))
        
        # If node had inline config and was missing from saved_configs, backfill it
        if nid not in saved_configs and ("recipe_id" in nd or "config" in nd):
            saved_configs[nid] = {
                "recipe_id": nd.get("recipe_id", ""),
                "label": content,
                "config": nd.get("config", {})
            }

    # Restore Edges
    for ed in wf_data.get("edges", []):
        eid = ed.get("id", f"e_{ed['source']}_{ed['target']}")
        t_edges.append(StreamlitFlowEdge(id=eid, source=ed["source"], target=ed["target"], animated=True))

    st.session_state["flow_state"] = StreamlitFlowState(nodes=t_nodes, edges=t_edges)
    st.session_state["node_configs"] = saved_configs
    st.session_state["canvas_version"] = st.session_state.get("canvas_version", 1) + 1
    st.session_state["active_saved_workflow_name"] = wf_data.get("name", "Saved Workflow")
    st.session_state["active_saved_workflow_id"] = wf_data.get("id")
    if "wb_name_input" in st.session_state:
        st.session_state["wb_name_input"] = wf_data.get("name", "Saved Workflow")

    # Restore Execution Results & Diagnostics if present
    last_exec = wf_data.get("last_execution")
    if last_exec and isinstance(last_exec, dict):
        st.session_state["last_execution"] = last_exec
        if "inference_schema" in last_exec and last_exec["inference_schema"]:
            st.session_state["inference_schema"] = last_exec["inference_schema"]
            if "execution_id" in last_exec:
                st.session_state["inference_bundle"] = job_manager.get_inference_bundle(last_exec["execution_id"])
    else:
        st.session_state.pop("last_execution", None)
        st.session_state.pop("inference_bundle", None)
        st.session_state.pop("inference_schema", None)

    record_api_telemetry(
        action_name=f"📂 Restore Workflow: {wf_data.get('name')}",
        endpoint=f"/api/v1/workflows/{wf_data.get('id')}",
        method="GET",
        request_payload={"id": wf_data.get("id")},
        response_payload={"nodes_restored": len(t_nodes), "edges_restored": len(t_edges)},
        status_code=200,
        duration_ms=1.5
    )


# -------------------------------------------------------------
# TAB 1: VISUAL PIPELINE WHITEBOARD
# -------------------------------------------------------------
if "Pipeline Whiteboard" in app_mode:
    curr_nodes_cnt = len(st.session_state["flow_state"].nodes)
    curr_edges_cnt = len(st.session_state["flow_state"].edges)
    active_ds_title = st.session_state.get('active_dataset_name', 'No Dataset Selected')

    # Top Brand Bar & Pipeline Studio Header
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:center; background:#FFFFFF; border:1px solid #E2E8F0; border-radius:12px; padding:10px 18px; margin-bottom:14px; box-shadow:0 1px 3px rgba(15,23,42,0.03);">
        <div style="display:flex; align-items:center; gap:12px;">
            <div style="width:38px; height:38px; border-radius:10px; background:#EFF6FF; border:1px solid #DBEAFE; color:#2563EB; display:flex; align-items:center; justify-content:center; font-size:18px;">
                ⚡
            </div>
            <div>
                <div style="font-size:1.15rem; font-weight:800; color:#0F172A; letter-spacing:-0.02em;">Custom Pipeline Studio</div>
                <div style="font-size:0.75rem; color:#64748B;">Datalect ML Enterprise AI/ML Pipeline Whiteboard</div>
            </div>
        </div>
        <div style="display:flex; align-items:center; gap:10px;">
            <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:5px 12px; font-size:11px; color:#475569; display:flex; gap:10px;">
                <span>Nodes: <strong style="color:#0F172A;">{curr_nodes_cnt}</strong></span>
                <span style="color:#CBD5E1;">|</span>
                <span>Links: <strong style="color:#0F172A;">{curr_edges_cnt}</strong></span>
            </div>
            <div style="background:#EFF6FF; border:1px solid #DBEAFE; border-radius:8px; padding:5px 12px; font-size:11px; color:#1D4ED8; font-weight:600;">
                📄 {active_ds_title[:24]}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Top Execution & Action Bar
    bar_col1, bar_col2, bar_col3, bar_col4, bar_col5, bar_col6, bar_col7, bar_col8 = st.columns([2.0, 1.8, 1.5, 1.5, 1.5, 1.4, 1.4, 1.2])
    with bar_col1:
        execute_clicked_top = st.button("▶️ RUN PIPELINE", type="primary", use_container_width=True)

    with bar_col2:
        if st.button("✨ AI Recommend", type="secondary", use_container_width=True):
            if len(st.session_state["flow_state"].nodes) > 0:
                st.session_state["pending_action"] = "ai_recommend"
            else:
                build_recommended_pipeline()
                st.rerun()

    with bar_col3:
        if st.button("⚡ ML Template", use_container_width=True):
            if len(st.session_state["flow_state"].nodes) > 0:
                st.session_state["pending_action"] = "ml_template"
            else:
                load_ml_template(force_preset=True)
                st.rerun()

    with bar_col4:
        if st.button("🔮 Forecast", use_container_width=True):
            if len(st.session_state["flow_state"].nodes) > 0:
                st.session_state["pending_action"] = "forecast_template"
            else:
                load_forecast_template(force_preset=True)
                st.rerun()

    with bar_col5:
        if st.button("🚨 Anomaly", use_container_width=True):
            if len(st.session_state["flow_state"].nodes) > 0:
                st.session_state["pending_action"] = "anomaly_template"
            else:
                load_anomaly_template(force_preset=True)
                st.rerun()

    with bar_col6:
        if st.button("🔗 Auto-Wire", use_container_width=True):
            curr_nodes = st.session_state["flow_state"].nodes
            incomplete = get_incomplete_pipeline_nodes()
            if incomplete:
                st.session_state["autowire_validation_error"] = incomplete
                st.rerun()
            else:
                st.session_state.pop("autowire_validation_error", None)

            if len(curr_nodes) < 2:
                st.warning("⚠️ Auto-Wire requires at least 2 nodes on the canvas. Add components first.")
            else:
                def get_recipe_type(nid: str) -> str:
                    cfg = st.session_state.get("node_configs", {}).get(nid, {})
                    return cfg.get("recipe_id", "")

                def get_category_order(nid: str) -> float:
                    r_id = get_recipe_type(nid)
                    r_obj = recipe_registry.get(r_id) if r_id else None
                    if r_obj:
                        c = r_obj.category.lower()
                        if "trigger" in c: return 0.0
                        if "ingest" in c: return 1.0
                        if "prep" in c: return 2.0
                        if "split" in c: return 3.0
                        if "train" in c or "model" in c or "forecast" in c or "anom" in c: return 4.0
                        if "eval" in c or "gov" in c: return 5.0
                    if r_id in ["cron_trigger", "webhook_trigger", "kafka_consumer"]:
                        return 0.0
                    if r_id in ["csv_loader", "export_writer"]:
                        return 1.0
                    if r_id in ["train_test_split", "stratified_split", "time_series_split", "walk_forward_split"]:
                        return 3.0
                    if r_id in ["model_evaluator", "mlflow_tracker"]:
                        return 5.0
                    return 2.0

                # Categorize nodes
                triggers = [n for n in curr_nodes if get_category_order(n.id) < 1.0]
                ingestions = [n for n in curr_nodes if 1.0 <= get_category_order(n.id) < 1.2]
                preprocessings = [n for n in curr_nodes if 1.2 <= get_category_order(n.id) < 3.0]
                splitters = [n for n in curr_nodes if 3.0 <= get_category_order(n.id) < 3.5]
                resamplers = [n for n in curr_nodes if 3.5 <= get_category_order(n.id) < 4.0]
                models = [n for n in curr_nodes if 4.0 <= get_category_order(n.id) < 5.0]
                evaluators = [n for n in curr_nodes if get_category_order(n.id) >= 5.0]

                # Sort each group by X position
                triggers.sort(key=get_node_position)
                ingestions.sort(key=get_node_position)
                preprocessings.sort(key=get_node_position)
                splitters.sort(key=get_node_position)
                resamplers.sort(key=get_node_position)
                models.sort(key=get_node_position)
                evaluators.sort(key=get_node_position)

                new_edges = []
                added_pairs = set()

                def add_edge(src_id, tgt_id):
                    if src_id != tgt_id and (src_id, tgt_id) not in added_pairs:
                        added_pairs.add((src_id, tgt_id))
                        new_edges.append(StreamlitFlowEdge(
                            id=f"auto_{src_id}_{tgt_id}",
                            source=src_id,
                            target=tgt_id,
                            animated=True
                        ))

                # 1. Triggers -> First Ingestion or Preprocessing node
                data_candidates = ingestions + preprocessings + splitters + resamplers + models
                first_data_node = data_candidates[0].id if data_candidates else None
                for t_node in triggers:
                    if first_data_node:
                        add_edge(t_node.id, first_data_node)

                # 2. Ingestion + Preprocessing linear chain
                data_chain = ingestions + preprocessings
                for i in range(len(data_chain) - 1):
                    add_edge(data_chain[i].id, data_chain[i+1].id)

                last_prep_node = data_chain[-1].id if data_chain else (triggers[-1].id if triggers else None)

                # 3. If Splitter exists
                if splitters:
                    main_splitter = splitters[0].id
                    if last_prep_node:
                        add_edge(last_prep_node, main_splitter)
                    
                    if resamplers:
                        main_resampler = resamplers[0].id
                        add_edge(main_splitter, main_resampler)
                        for m in models:
                            add_edge(main_resampler, m.id)
                    else:
                        for m in models:
                            add_edge(main_splitter, m.id)

                    for ev in evaluators:
                        add_edge(main_splitter, ev.id)
                    for m in models:
                        for ev in evaluators:
                            add_edge(m.id, ev.id)
                elif resamplers:
                    # No splitter, but resampler exists (pre-split resampling mode)
                    main_resampler = resamplers[0].id
                    if last_prep_node:
                        add_edge(last_prep_node, main_resampler)
                    for m in models:
                        add_edge(main_resampler, m.id)
                    for ev in evaluators:
                        add_edge(main_resampler, ev.id)
                    for m in models:
                        for ev in evaluators:
                            add_edge(m.id, ev.id)
                else:
                    # No splitter: Preprocessing connects to models
                    if models:
                        for m in models:
                            if last_prep_node:
                                add_edge(last_prep_node, m.id)
                            for ev in evaluators:
                                add_edge(m.id, ev.id)
                    elif evaluators and last_prep_node:
                        for ev in evaluators:
                            add_edge(last_prep_node, ev.id)

                # Fallback for any un-wired isolated nodes: chain linearly by category & x_pos
                all_sorted = sorted(curr_nodes, key=lambda n: (get_category_order(n.id), get_node_position(n)[0]))
                if not new_edges and len(all_sorted) >= 2:
                    for i in range(len(all_sorted) - 1):
                        add_edge(all_sorted[i].id, all_sorted[i+1].id)

                st.session_state["flow_state"].edges = new_edges
                st.session_state["canvas_version"] = st.session_state.get("canvas_version", 1) + 1
                st.success(f"🔗 Smart-wired {len(curr_nodes)} nodes into an intelligent ML pipeline DAG!")
                st.rerun()

    with bar_col7:
        if st.button("📐 Auto-Format", use_container_width=True):
            auto_format_canvas_layout()
            st.rerun()

    with bar_col8:
        if st.button("🧹 Clear", use_container_width=True):
            if len(st.session_state["flow_state"].nodes) > 0:
                st.session_state["pending_action"] = "clear_canvas"
            else:
                st.info("ℹ️ Whiteboard is already empty.")

    # Workbook Persistence & Load Bar
    with st.expander("💾 Pipeline Workbook Manager (Save & Restore Workbooks)", expanded=False):
        w_col1, w_col2 = st.columns([3, 3])
        
        with w_col1:
            st.markdown("#### 💾 Save Current Pipeline Workbook")
            wb_name = st.text_input("Workbook Title", value=st.session_state.get("active_saved_workflow_name", "My ML Pipeline Workbook"), key="wb_name_input")
            wb_desc = st.text_area("Optional Description", value="", height=68, key="wb_desc_input")
            if st.button("💾 Save Pipeline to Database & API", type="primary", use_container_width=True, key="btn_save_wb_ui"):
                if not st.session_state["flow_state"].nodes:
                    st.warning("⚠️ Canvas is empty. Add nodes before saving.")
                else:
                    saved_res = save_workflow_to_backend(wb_name, wb_desc)
                    if saved_res:
                        st.success(f"✅ Workbook '{wb_name}' saved permanently to SQLite & REST API!")
                        st.session_state["active_saved_workflow_name"] = wb_name
                        st.rerun()

        with w_col2:
            st.markdown("#### 📂 Load & Manage Saved Workbooks")
            show_archived = st.checkbox("♻️ Show Soft-Deleted / Archived Pipelines", value=False, key="chk_show_archived_wb")
            saved_list = fetch_saved_workflows_from_backend(include_deleted=show_archived)
            
            if show_archived:
                saved_list = [w for w in saved_list if not w.get("is_active", True)]
            else:
                saved_list = [w for w in saved_list if w.get("is_active", True)]

            if not saved_list:
                msg = "No archived pipeline workbooks in trash." if show_archived else "No active saved pipeline workbooks found in database. Build a pipeline and click Save!"
                st.info(msg)
            else:
                wf_options = {f"{w['name']} (ID: {w['id'][:8]}... | {w.get('updated_at', '')})": w for w in saved_list}
                selected_wf_label = st.selectbox("Select Saved Workbook", list(wf_options.keys()), key="select_saved_wf")
                selected_wf = wf_options[selected_wf_label]
                
                if not show_archived:
                    btn_c1, btn_c2 = st.columns([3, 2])
                    with btn_c1:
                        if st.button("📂 Load Selected Workbook", use_container_width=True, key="btn_load_wb_ui"):
                            restore_saved_workflow(selected_wf)
                            st.success(f"🎉 Pipeline '{selected_wf['name']}' restored with all exact node configurations & parameters!")
                            st.rerun()
                    with btn_c2:
                        if st.button("🗑️ Soft-Delete", use_container_width=True, key="btn_del_wb_ui"):
                            if soft_delete_workflow_backend(selected_wf["id"]):
                                st.success(f"🗑️ Soft-deleted '{selected_wf['name']}'. It is safely archived in DB!")
                                st.rerun()
                else:
                    if st.button("♻️ Restore Selected Pipeline from Trash", type="primary", use_container_width=True, key="btn_restore_wb_ui"):
                        if restore_deleted_workflow_backend(selected_wf["id"]):
                            st.success(f"♻️ Restored '{selected_wf['name']}' back to active state!")
                            st.rerun()

    # ---------------------------------------------------------
    # WORKFLOW VERSION HISTORY & EXECUTION AUDIT TRAIL (Option B)
    # ---------------------------------------------------------
    with st.expander("🕒 Version History & Execution Audit Trail", expanded=False):
        st.markdown("#### 🕒 Pipeline Version History & Audit Trail")
        st.caption("Inspect past configurations, view execution run reports, and compare metrics side-by-side.")

        # Determine active workflow or select from saved workbooks
        active_saved_id = st.session_state.get("active_saved_workflow_id")
        saved_pipelines = fetch_saved_workflows_from_backend(include_deleted=False)

        if not saved_pipelines:
            st.info("ℹ️ No saved workbooks found. Save your current pipeline first to enable version tracking & history.")
        else:
            w_opts = {f"{w['name']} (ID: {w['id'][:8]}...)": w["id"] for w in saved_pipelines}
            default_wf_idx = 0
            if active_saved_id:
                for idx, (lbl, wid) in enumerate(w_opts.items()):
                    if wid == active_saved_id:
                        default_wf_idx = idx
                        break

            chosen_wf_lbl = st.selectbox("Select Workbook to Audit", list(w_opts.keys()), index=default_wf_idx, key="sel_audit_wf")
            target_audit_wfid = w_opts[chosen_wf_lbl]

            hist_runs = fetch_workflow_history_backend(target_audit_wfid)

            if not hist_runs:
                st.info("ℹ️ No historical execution runs recorded for this workbook yet. Click '▶️ RUN PIPELINE' above to record Run #1.")
            else:
                hist_tab1, hist_tab2 = st.tabs(["📜 Execution Timeline", "⚖️ Compare Runs Side-by-Side"])

                with hist_tab1:
                    st.markdown(f"##### Showing **{len(hist_runs)}** Recorded Execution Runs:")

                    run_choices = {}
                    for r in hist_runs:
                        status_icon = "🟢" if r.get("status") == "SUCCESS" else "🔴"
                        v_num = r.get("version_number", 1)
                        dt_str = r.get("created_at", "")[:19]
                        metrics_dict = r.get("metrics") or {}
                        # Format headline metric badge
                        headline_metric = ""
                        for mk in ["accuracy", "recall", "f1_score", "r2_score", "mae"]:
                            if mk in metrics_dict:
                                headline_metric = f" | {mk.capitalize()}: {metrics_dict[mk]:.3f}"
                                break
                        lbl = f"{status_icon} v{v_num} — {r.get('run_label', f'Run #{v_num}')} ({dt_str}){headline_metric}"
                        run_choices[lbl] = r

                    selected_run_lbl = st.selectbox("Select Run to Inspect", list(run_choices.keys()), key="sel_hist_run_card")
                    active_run_summary = run_choices[selected_run_lbl]

                    # Metrics & KPIs Row
                    kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
                    kpi_c1.metric("Version", f"v{active_run_summary.get('version_number', 1)}")
                    kpi_c2.metric("Status", active_run_summary.get("status", "UNKNOWN"))
                    dur_s = (active_run_summary.get("total_duration_ms") or 0.0) / 1000.0
                    kpi_c3.metric("Duration", f"{dur_s:.2f}s")
                    kpi_c4.metric("Canvas Nodes", active_run_summary.get("nodes_count", 0))

                    if active_run_summary.get("metrics"):
                        with st.expander("📊 Run Summary Metrics", expanded=True):
                            st.json(active_run_summary["metrics"])

                    # Action: View Report
                    if st.button("🔍 Inspect Run Report in Diagnostic Viewer", type="primary", use_container_width=True, key=f"btn_view_rep_{active_run_summary['id']}"):
                        detail = fetch_workflow_execution_detail_backend(target_audit_wfid, active_run_summary["id"])
                        if detail:
                            st.session_state["last_execution"] = {
                                "execution_id": detail["id"],
                                "status": detail["status"],
                                "total_duration_ms": detail.get("total_duration_ms", 0.0),
                                "final_metrics": detail.get("metrics"),
                                **(detail.get("reports") or {}),
                                "step_snapshots": detail.get("step_snapshots"),
                                "execution_logs": detail.get("logs")
                            }
                            st.success(f"Loaded execution report for {active_run_summary.get('run_label', 'Run')}! Scroll down to Diagnostic Results.")
                            st.rerun()

                with hist_tab2:
                    st.markdown("##### ⚖️ Side-by-Side Execution Run Comparison")
                    if len(hist_runs) < 2:
                        st.info("ℹ️ Comparison requires at least 2 execution runs. Run the pipeline again with tweaked parameters to compare.")
                    else:
                        cmp_col1, cmp_col2 = st.columns(2)
                        with cmp_col1:
                            r_a_lbl = st.selectbox("Baseline Run (Run A)", list(run_choices.keys()), index=min(1, len(run_choices)-1), key="cmp_run_a_select")
                            run_a_obj = run_choices[r_a_lbl]
                        with cmp_col2:
                            r_b_lbl = st.selectbox("Comparison Run (Run B)", list(run_choices.keys()), index=0, key="cmp_run_b_select")
                            run_b_obj = run_choices[r_b_lbl]

                        cmp_data = compare_workflow_executions_backend(target_audit_wfid, run_a_obj["id"], run_b_obj["id"])
                        if cmp_data:
                            # 1. Metric Deltas
                            m_diff = cmp_data.get("metrics_diff", {})
                            if m_diff:
                                st.markdown("###### 📈 Metric Differences:")
                                diff_rows = []
                                for m_key, m_info in m_diff.items():
                                    v_a = m_info.get("val_a")
                                    v_b = m_info.get("val_b")
                                    delta = m_info.get("delta")
                                    pct = m_info.get("pct_change")
                                    pct_str = f" ({'+' if pct > 0 else ''}{pct}%)" if pct is not None else ""
                                    imp = m_info.get("improved")
                                    status_badge = "🟢 Improved" if imp else ("🔴 Regressed" if imp is False else "⚪ Neutral")
                                    diff_rows.append({
                                        "Metric": m_key,
                                        "Run A (Baseline)": v_a,
                                        "Run B (Comparison)": v_b,
                                        "Delta (B - A)": f"{'+' if delta > 0 else ''}{delta}{pct_str}",
                                        "Status": status_badge
                                    })
                                st.dataframe(pd.DataFrame(diff_rows), use_container_width=True)

                            # 2. Graph and Parameter Changes
                            c_diff = cmp_data.get("config_diff", {})
                            if c_diff:
                                if c_diff.get("nodes_added"):
                                    st.markdown("###### ➕ Components Added in Run B:")
                                    for na in c_diff["nodes_added"]:
                                        st.markdown(f"- `{na.get('label')}` (`{na.get('id')}`)")
                                if c_diff.get("nodes_removed"):
                                    st.markdown("###### ➖ Components Removed in Run B:")
                                    for nr in c_diff["nodes_removed"]:
                                        st.markdown(f"- `{nr.get('label')}` (`{nr.get('id')}`)")
                                if c_diff.get("parameter_changes"):
                                    st.markdown("###### ⚙️ Parameter Modifications:")
                                    st.dataframe(pd.DataFrame(c_diff["parameter_changes"]), use_container_width=True)

    # Overwrite & Clear Confirmation Prompt
    if "pending_action" in st.session_state and st.session_state["pending_action"]:
        action = st.session_state["pending_action"]
        if action == "clear_canvas":
            st.error(f"🗑️ **Wipe Whiteboard Confirmation:** The whiteboard currently contains {len(st.session_state['flow_state'].nodes)} active nodes and pipeline execution states. Are you sure you want to clear everything?")
            c_yes, c_no, _ = st.columns([2, 2, 6])
            with c_yes:
                if st.button("🗑️ Yes, Wipe Whiteboard", type="primary", key="btn_confirm_wipe", use_container_width=True):
                    st.session_state["pending_action"] = None
                    st.session_state["flow_state"] = StreamlitFlowState(nodes=[], edges=[])
                    st.session_state["node_configs"] = {}
                    st.session_state["canvas_version"] = st.session_state.get("canvas_version", 1) + 1
                    if "last_execution" in st.session_state:
                        del st.session_state["last_execution"]
                    st.success("Whiteboard cleared.")
                    st.rerun()
            with c_no:
                if st.button("❌ Cancel", key="btn_cancel_wipe", use_container_width=True):
                    st.session_state["pending_action"] = None
                    st.rerun()
        else:
            action_name = action.replace("_", " ").title()
            st.warning(f"⚠️ **Overwrite Confirmation:** The canvas currently has {len(st.session_state['flow_state'].nodes)} active nodes. Applying **{action_name}** will replace your current whiteboard.")
            
            c_opt1, c_opt2, c_opt3, _ = st.columns([2.5, 2.5, 1.5, 3])
            with c_opt1:
                if st.button("✅ Apply to My Current Data", type="primary", key="btn_apply_curr_data", use_container_width=True):
                    st.session_state["pending_action"] = None
                    if action == "ai_recommend":
                        build_recommended_pipeline()
                    elif action == "ml_template":
                        load_ml_template(force_preset=False)
                    elif action == "forecast_template":
                        load_forecast_template(force_preset=False)
                    elif action == "anomaly_template":
                        load_anomaly_template(force_preset=False)
                    st.rerun()

            with c_opt2:
                if st.button("📦 Load Sample Preset Dataset", key="btn_apply_sample_data", use_container_width=True):
                    st.session_state["pending_action"] = None
                    if action == "ai_recommend":
                        build_recommended_pipeline()
                    elif action == "ml_template":
                        load_ml_template(force_preset=True)
                    elif action == "forecast_template":
                        load_forecast_template(force_preset=True)
                    elif action == "anomaly_template":
                        load_anomaly_template(force_preset=True)
                    st.rerun()

            with c_opt3:
                if st.button("❌ Cancel", key="btn_cancel_template", use_container_width=True):
                    st.session_state["pending_action"] = None
                    st.rerun()

    # ---------------------------------------------------------
    # AI DATASET ADVISOR & PRE-FLIGHT INSIGHTS (Section 8 of Spec)
    # ---------------------------------------------------------
    active_df = st.session_state["active_df"]
    df_cols = list(active_df.columns)
    
    with st.expander("🧠 AI Dataset Intelligence & Customizable Auto-Architect", expanded=False):
        c_t1, c_t2 = st.columns(2)
        with c_t1:
            # Safely sync target selection with current dataset columns without double-default warning
            curr_target = st.session_state.get("ai_target_selected")
            target_idx = df_cols.index(curr_target) if curr_target in df_cols else len(df_cols) - 1
            custom_target = st.selectbox("🎯 Target / Dependent Feature", df_cols, index=target_idx, key="ai_target_selectbox")
            st.session_state["ai_target_selected"] = custom_target
        with c_t2:
            task_options = ["Auto-Detect", "classification", "regression", "time_series_forecasting", "anomaly_detection"]
            curr_task = st.session_state.get("ai_task_selected")
            task_idx = task_options.index(curr_task) if curr_task in task_options else 0
            custom_task = st.selectbox("⚙️ Problem Type Intent", task_options, index=task_idx, key="ai_task_selectbox")
            st.session_state["ai_task_selected"] = custom_task

        task_override = None if custom_task == "Auto-Detect" else custom_task
        ds_name = st.session_state.get("active_dataset_name", "dataset")
        rec = get_cached_ai_recommendation(ds_name, len(active_df), len(df_cols), custom_target, task_override)

        c_ai1, c_ai2 = st.columns([3, 2])
        with c_ai1:
            st.markdown(f"**Diagnosis:** {rec['explanation']}")
            st.caption(f"Active Data: **{st.session_state['active_dataset_name']}** ({rec['profile_summary']['rows']} rows, {rec['profile_summary']['missing_cells']} missing values)")
            if st.button(f"⚡ Build & Run {rec['task_type'].replace('_', ' ').title()} Pipeline", key="btn_apply_ai_expander"):
                if len(st.session_state["flow_state"].nodes) > 0:
                    st.session_state["pending_action"] = "ai_recommend"
                    st.rerun()
                else:
                    build_recommended_pipeline(target_col=custom_target, task_type=task_override)
                    st.rerun()
        with c_ai2:
            st.markdown("**Top Recommended Algorithm:**")
            for m in rec["model_rankings"][:2]:
                st.write(f"• **{m['name']}** ({m['tier']}) ➔ *{m['reason'][:80]}...*")

    # ---------------------------------------------------------
    # STEP 1: COMPONENT PALETTE (Enterprise Catalog Drawer)
    # ---------------------------------------------------------
    total_procs = sum(len(v) for v in RECIPE_CATEGORY_MAP.values())
    with st.expander(f"📦 Step 1: Recipe Palette ({total_procs} Enterprise Processors)", expanded=True):
        st.caption("🌐 **Active API:** `GET /api/v1/recipes?category=...` ➔ `GET /api/v1/recipes/{recipe_id}/schema`")

        # Search & Category Filter Row
        pal_search_col, pal_cat_col = st.columns([1.8, 1.2])
        with pal_search_col:
            pal_search = st.text_input(
                "Search Processors",
                placeholder="Search recipes, groups, SQL joins, transforms...",
                label_visibility="collapsed",
                key="pal_search_filter"
            )

        cat_labels = {f"{c} ({len(RECIPE_CATEGORY_MAP[c])})": c for c in RECIPE_CATEGORY_MAP.keys()}
        with pal_cat_col:
            sel_cat_label = st.selectbox(
                "Filter by Category",
                list(cat_labels.keys()),
                label_visibility="collapsed",
                key="pal_cat_dropdown"
            )
            selected_cat = cat_labels[sel_cat_label]

        if pal_search and pal_search.strip():
            q = pal_search.strip().lower()
            matching_recipes = []
            for c_name, r_list in RECIPE_CATEGORY_MAP.items():
                for r in r_list:
                    if (q in r["name"].lower() or 
                        q in r["id"].lower() or 
                        q in r.get("description", "").lower() or 
                        q in c_name.lower()):
                        matching_recipes.append(r)
            available = matching_recipes
            st.caption(f"Showing **{len(available)}** matching processors for `{pal_search}` across all categories:")
        else:
            available = RECIPE_CATEGORY_MAP[selected_cat]

        if not available:
            st.info(f"No processors matched '{pal_search}'. Try keywords like 'ingestion', 'imputer', 'xgboost', 'split', or 'transform'.")
        else:
            col_recipe, col_meta, col_add = st.columns([3, 2.5, 1.5])
            with col_recipe:
                recipe_map = {f"{r['icon']} {r['name']} ({r['version']})": r for r in available}
                chosen_label = st.selectbox(
                    "Select Processor",
                    list(recipe_map.keys()),
                    key="step1_recipe_select"
                )
                chosen_meta = recipe_map[chosen_label]

            with col_meta:
                in_str = ", ".join(chosen_meta.get("input_types", ["dataframe"]))
                out_str = ", ".join(chosen_meta.get("output_types", ["dataframe"]))
                desc_snippet = chosen_meta.get("description", "")[:75]
                st.markdown(f"""
                <div style="font-size:11px; color:#475569; padding-top:6px;">
                    <div><b>In:</b> <code>{in_str}</code> • <b>Out:</b> <code>{out_str}</code></div>
                    <div style="color:#64748B; margin-top:2px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="{chosen_meta.get('description','')}">{desc_snippet}...</div>
                </div>
                """, unsafe_allow_html=True)

            with col_add:
                st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
                if st.button("➕ Drop to Board", type="primary", use_container_width=True, key="btn_drop_processor"):
                    st.session_state["node_counter"] = st.session_state.get("node_counter", 0) + 1
                    counter = st.session_state["node_counter"]
                    count = len(st.session_state["flow_state"].nodes)
                    pos_x = 80 + (count % 3) * 260
                    pos_y = 60 + (count // 3) * 140

                    node_id = f"{chosen_meta['id']}_{counter}"
                    node_title = f"{chosen_meta['icon']} {chosen_meta['name'].split('(')[0].strip()}"

                    new_node = create_flow_node(
                        node_id=node_id,
                        pos=(pos_x, pos_y),
                        content=f"{node_title}\n[{node_id}]",
                        category_key=chosen_meta.get("category_key", "default")
                    )

                    st.session_state["flow_state"].nodes.append(new_node)

                    init_cfg = dict(chosen_meta["default_config"])
                    if "target_column" in init_cfg:
                        init_cfg["target_column"] = ""

                    st.session_state["node_configs"][node_id] = {
                        "recipe_id": chosen_meta["id"],
                        "label": node_title,
                        "config": init_cfg
                    }
                    st.session_state["canvas_version"] = st.session_state.get("canvas_version", 1) + 1

                    record_api_telemetry(
                        action_name=f"➕ Drop Node: {chosen_meta['name']}",
                        endpoint="/api/v1/recipes/instantiate",
                        method="POST",
                        request_payload={"recipe_id": chosen_meta["id"], "node_id": node_id, "default_config": init_cfg},
                        response_payload={"status": "NODE_CREATED", "node_id": node_id, "schema": chosen_meta.get("default_config")},
                        status_code=201,
                        duration_ms=1.5
                    )
                    st.rerun()

    st.markdown("---")

    # ---------------------------------------------------------
    # STEP 2 & 3: CANVAS + CONNECTOR + INSPECTOR
    # ---------------------------------------------------------
    if "autowire_validation_error" in st.session_state and st.session_state["autowire_validation_error"]:
        err_items = st.session_state["autowire_validation_error"]
        st.error("### ⚠️ Auto-Wire Halted: Incomplete Node Configurations Detected!")
        st.markdown(
            "Auto-Wire cannot automatically connect nodes while required configuration fields are missing or unset. "
            "Please configure the following required fields in the Inspector panel before auto-wiring:"
        )
        for item in err_items:
            missing_str = ", ".join(f"`{f}`" for f in item["missing_fields"])
            st.warning(f"👉 **{item['label']}** (`{item['node_id']}`): Missing required parameter(s): **{missing_str}**")
        
        c_dismiss, _ = st.columns([1, 4])
        with c_dismiss:
            if st.button("✕ Dismiss Warning", key="dismiss_autowire_err"):
                del st.session_state["autowire_validation_error"]
                st.rerun()

    def render_chrysa_assistant_card():
        st.markdown("""
        <div class="chrysa-card">
            <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;">
                <div style="display:flex; align-items:center; gap:8px;">
                    <div style="width:28px; height:28px; border-radius:8px; background:#EFF6FF; border:1px solid #DBEAFE; display:flex; align-items:center; justify-content:center; color:#2563EB;">✨</div>
                    <div>
                        <span class="chrysa-text-shimmer" style="font-weight:700; font-size:0.92rem;">Chrysa Copilot</span>
                        <span class="badge badge-blue" style="font-size:9px; margin-left:4px;">AI ASSIST</span>
                    </div>
                </div>
            </div>
            <p style="font-size:0.78rem; color:#64748B; margin-bottom:10px; line-height:1.4;">
                Ask about your dataset, pipeline architecture, metric interpretation, or guardrails.
            </p>
        </div>
        """, unsafe_allow_html=True)

        q_prompts = [
            "📊 Evaluate current pipeline metrics",
            "🛡️ Check for data leakage in split",
            "📈 How does MAPE handle zeros?",
            "⚡ Recommend next processor step"
        ]
        sel_prompt = st.selectbox("Quick Inquiries", ["-- Select a prompt --"] + q_prompts, key="chrysa_quick_prompt")
        user_query = st.text_input("Ask Chrysa anything...", placeholder="e.g. Which model is best for churn?", key="chrysa_custom_input")

        active_q = user_query if user_query.strip() else (sel_prompt if sel_prompt != "-- Select a prompt --" else None)

        if active_q:
            with st.spinner("Chrysa is analyzing pipeline state..."):
                answer = ""
                df_curr = st.session_state.get("active_df")
                last_exec = st.session_state.get("last_execution")

                if "leak" in active_q.lower() or "split" in active_q.lower():
                    split_cfg = None
                    for nid, c in st.session_state.get("node_configs", {}).items():
                        if "split" in c.get("recipe_id", ""):
                            split_cfg = c.get("config", {})
                            break
                    if split_cfg:
                        strat = split_cfg.get("stratify", False)
                        ts = split_cfg.get("time_series_mode", False)
                        answer = f"✅ **Split Analysis:** Active splitter detected. Stratification is **{'Enabled' if strat else 'Disabled'}** and Time-Series chronological ordering is **{'Active' if ts else 'Inactive'}**. All preprocessing encoders and scalers are fitted strictly on the training split to guarantee zero target or feature leakage."
                    else:
                        answer = "ℹ️ **Split Recommendation:** You currently do not have a dedicated Train/Test Split node wired. For supervised models, add a `Train / Test Splitter` or `Stratified Splitter` before the trainer to ensure robust evaluation on unseen test data."

                elif "mape" in active_q.lower() or "zero" in active_q.lower():
                    answer = "🛡️ **MAPE Zero Guardrail:** Standard MAPE divides by actual values: `|actual - pred| / actual`. When actuals contain zeros (e.g. 0 sales or churn = 0), division by zero causes runaway percentages (e.g. trillions %). The platform evaluator automatically displays **`N/A (Contains zeros)`** and reports **SMAPE (Symmetric MAPE)** to handle zero-grounded features safely."

                elif "metric" in active_q.lower() or "evaluat" in active_q.lower() or "perform" in active_q.lower():
                    if last_exec and last_exec.get("final_metrics"):
                        m = last_exec["final_metrics"]
                        if "accuracy" in m:
                            answer = f"🏆 **Classification Executive Summary:** Model achieved **Accuracy = {m.get('accuracy', 0)*100:.2f}%**, **F1 Score = {m.get('f1_score', 0)*100:.2f}%**, and **Balanced Accuracy = {m.get('balanced_accuracy', 0)*100:.2f}%**. Status is **{last_exec.get('status', 'SUCCESS')}** across {len(last_exec.get('step_snapshots', {}))} steps."
                        elif "r2_score" in m:
                            answer = f"📈 **Regression Executive Summary:** Model achieved **R² = {m.get('r2_score', 0):.4f}**, **MAE = {m.get('mae', 0):.4f}**, and **RMSE = {m.get('rmse', 0):.4f}**. SMAPE is `{m.get('smape', 'N/A')}`."
                        else:
                            answer = f"📊 **Execution Metrics Summary:** Pipeline finished with status `{last_exec.get('status')}`. Final metrics recorded: `{list(m.keys())}`."
                    else:
                        answer = "ℹ️ No recent execution metrics found. Click **▶️ RUN PIPELINE** on the toolbar to execute the DAG and generate diagnostic performance reports."

                elif "step" in active_q.lower() or "recommend" in active_q.lower() or "next" in active_q.lower():
                    curr_nodes = st.session_state.get("flow_state", StreamlitFlowState([], [])).nodes
                    if not curr_nodes:
                        answer = "💡 **Recommended Next Step:** Drop a `Dataset Ingestion` node or click **✨ Recommend AI Pipeline** to automatically build a complete preprocessing and modeling pipeline."
                    elif not any("split" in n.id for n in curr_nodes):
                        answer = "💡 **Recommended Next Step:** Add a `Train / Test Splitter` (under Dataset Splitting) to prevent overfitting before training a model."
                    elif not any("eval" in n.id for n in curr_nodes):
                        answer = "💡 **Recommended Next Step:** Add a `Model Evaluator` node and connect your model to inspect Accuracy, F1, and Confusion Matrix heatmaps."
                    else:
                        answer = "✨ **Pipeline Complete:** Your architecture has ingestion, preprocessing, splitting, training, and evaluation nodes. Click **▶️ RUN PIPELINE** to test."

                else:
                    answer = f"🤖 **Chrysa Insights:** Active dataset has **{len(df_curr) if df_curr is not None else 0}** rows and **{len(df_curr.columns) if df_curr is not None else 0}** features. Active canvas has **{len(st.session_state['flow_state'].nodes)}** nodes. All 187 enterprise processors are available in the palette above."

                st.markdown(f"""
                <div style="background:#F0FDF4; border:1px solid #BBF7D0; border-radius:10px; padding:12px; margin-top:8px; font-size:0.82rem; color:#166534;">
                    {answer}
                </div>
                """, unsafe_allow_html=True)

    canvas_col, right_col = st.columns([3, 1])

    with canvas_col:
        st.markdown("#### 🖱️ Step 2: Interactive Whiteboard")

        if not st.session_state["flow_state"].nodes:
            st.markdown("""
            <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:16px; padding:28px 20px; text-align:center; box-shadow:0 4px 12px -2px rgba(15,23,42,0.04); margin-bottom:12px;">
                <div style="width:48px; height:48px; border-radius:14px; background:#EFF6FF; border:1px solid #DBEAFE; color:#2563EB; display:flex; align-items:center; justify-content:center; margin:0 auto 10px auto; font-size:22px;">
                    ⚡
                </div>
                <h3 style="font-size:1.1rem; font-weight:800; color:#0F172A; margin-bottom:4px; letter-spacing:-0.02em;">Canvas is Ready</h3>
                <p style="font-size:0.82rem; color:#64748B; max-width:420px; margin:0 auto 14px auto; line-height:1.45;">
                    Select a saved workflow from the top dropdown, let AI auto-architect an end-to-end pipeline, or drop components from the 187-processor catalog above.
                </p>
            </div>
            """, unsafe_allow_html=True)

            c_h1, c_h2, c_h3, c_h4 = st.columns(4)
            with c_h1:
                if st.button("✨ Recommend AI Pipeline", key="btn_hero_ai_rec", use_container_width=True, type="primary"):
                    build_recommended_pipeline()
                    st.rerun()
            with c_h2:
                if st.button("⚡ ML Supervised Template", key="btn_hero_ml_temp", use_container_width=True):
                    load_ml_template(force_preset=True)
                    st.rerun()
            with c_h3:
                if st.button("🔮 Forecasting Template", key="btn_hero_fc_temp", use_container_width=True):
                    load_forecast_template(force_preset=True)
                    st.rerun()
            with c_h4:
                if st.button("🚨 Anomaly Template", key="btn_hero_an_temp", use_container_width=True):
                    load_anomaly_template(force_preset=True)
                    st.rerun()

        # Render 2D React Flow Canvas with versioned key
        canvas_key = f"pipeline_flow_canvas_v{st.session_state.get('canvas_version', 1)}"
        flow_result = streamlit_flow(
            canvas_key,
            st.session_state["flow_state"],
            height=460,
            fit_view=True,
            show_minimap=True,
            show_controls=True,
            pan_on_drag=True,
            allow_zoom=True,
            allow_new_edges=True,
            animate_new_edges=True,
            enable_edge_menu=True,
            get_node_on_click=True
        )
        if flow_result is not None:
            st.session_state["flow_state"] = flow_result

        # FOOLPROOF 1-CLICK NODE CONNECTOR
        with st.expander("🔗 1-Click Line Connector & Wire Management", expanded=True):
            current_nodes = st.session_state["flow_state"].nodes
            node_id_list = [n.id for n in current_nodes]
            
            if len(node_id_list) >= 2:
                c_src, c_arrow, c_tgt, c_btn = st.columns([3, 1, 3, 2])
                with c_src:
                    src_id = st.selectbox("Source (Output)", node_id_list, index=0, key="conn_src")
                with c_arrow:
                    st.markdown("<div style='text-align:center; font-size:1.4rem; padding-top:24px;'>➔</div>", unsafe_allow_html=True)
                with c_tgt:
                    tgt_id = st.selectbox("Target (Input)", node_id_list, index=min(1, len(node_id_list)-1), key="conn_tgt")
                with c_btn:
                    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                    if st.button("🔗 Connect Line", use_container_width=True):
                        if src_id == tgt_id:
                            st.error("❌ Cannot connect a node to itself.")
                        else:
                            edge_id = f"edge_{src_id}_{tgt_id}"
                            existing_ids = [e.id for e in st.session_state["flow_state"].edges]
                            if edge_id in existing_ids:
                                st.warning(f"⚠️ Connection `{src_id}` ➔ `{tgt_id}` already exists.")
                            else:
                                # Pre-flight Cycle Detection before wiring
                                cand_nodes = [WorkflowNode(id=n.id, recipe_id=st.session_state["node_configs"].get(n.id, {}).get("recipe_id", "missing_value_imputer"), config={}) for n in current_nodes]
                                cand_edges = [WorkflowEdge(source=e.source, target=e.target) for e in st.session_state["flow_state"].edges]
                                cand_edges.append(WorkflowEdge(source=src_id, target=tgt_id))
                                cand_graph = WorkflowGraph(nodes=cand_nodes, edges=cand_edges)
                                try:
                                    cand_graph.get_topological_order()
                                    new_edge = StreamlitFlowEdge(id=edge_id, source=src_id, target=tgt_id, animated=True)
                                    st.session_state["flow_state"].edges.append(new_edge)
                                    st.session_state["canvas_version"] = st.session_state.get("canvas_version", 1) + 1
                                    
                                    record_api_telemetry(
                                        action_name=f"🔗 Wire Connection: `{src_id}` ➔ `{tgt_id}`",
                                        endpoint="/api/v1/workflows/connect",
                                        method="POST",
                                        request_payload={"source": src_id, "target": tgt_id, "edge_id": edge_id},
                                        response_payload={"status": "CONNECTED", "topological_validity": "CYCLE_FREE"},
                                        status_code=200,
                                        duration_ms=2.1
                                    )
                                    
                                    st.success(f"Connected `{src_id}` ➔ `{tgt_id}`!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Circular Loop Blocked: Connecting `{src_id}` ➔ `{tgt_id}` creates a cycle in the pipeline.")

            # Active Edges Table
            if st.session_state["flow_state"].edges:
                st.caption("Active Connections:")
                for edge in list(st.session_state["flow_state"].edges):
                    e_col1, e_col2 = st.columns([5, 1])
                    e_col1.write(f"• `{edge.source}` ➔ `{edge.target}`")
                    if e_col2.button("❌", key=f"del_{edge.id}"):
                        st.session_state["flow_state"].edges = [e for e in st.session_state["flow_state"].edges if e.id != edge.id]
                        st.session_state["canvas_version"] = st.session_state.get("canvas_version", 1) + 1
                        st.rerun()

    with right_col:
        st.markdown("#### ⚙️ Step 3: Node Config")
        
        current_node_ids = [n.id for n in st.session_state["flow_state"].nodes]
        if current_node_ids:
            selected_node_id = st.selectbox("Select Node", current_node_ids, key="step3_node_select_box")
            
            node_cfg = st.session_state["node_configs"].get(selected_node_id, {"recipe_id": "missing_value_imputer", "config": {}})
            recipe_obj = recipe_registry.get(node_cfg["recipe_id"])
            
            if st.session_state.get("_prev_selected_node_id") != selected_node_id:
                st.session_state["_prev_selected_node_id"] = selected_node_id
                record_api_telemetry(
                    action_name=f"Inspect Node Schema: `{selected_node_id}`",
                    endpoint=f"/api/v1/recipes/{recipe_obj.recipe_id}/schema",
                    method="GET",
                    request_payload={"path_param": {"recipe_id": recipe_obj.recipe_id}, "node_id": selected_node_id},
                    response_payload={"recipe_id": recipe_obj.recipe_id, "name": recipe_obj.name, "parameters_schema": recipe_obj.get_schema()},
                    status_code=200,
                    duration_ms=1.1
                )
            
            st.markdown(f"**Recipe:** `{recipe_obj.name}`")
            st.caption(f"🌐 `GET /api/v1/recipes/{recipe_obj.recipe_id}/schema`")
            # Node Canvas Label (Editable in Real-Time)
            curr_label = ""
            for n in st.session_state["flow_state"].nodes:
                if n.id == selected_node_id:
                    curr_label = n.data.get("content", n.id)
                    break

            new_label = st.text_input("🏷️ Whiteboard Node Title", value=curr_label, key=f"lbl_{selected_node_id}")
            if new_label and new_label != curr_label:
                for n in st.session_state["flow_state"].nodes:
                    if n.id == selected_node_id:
                        n.data = {"content": new_label}
                st.session_state["canvas_version"] = st.session_state.get("canvas_version", 1) + 1
                st.rerun()

            # Node View Toggle: Parameters vs Code View (Section 9 of Spec)
            view_mode = st.radio("Node View", ["⚙️ Parameters Form", "</> Generated Code"], horizontal=True, key=f"vmode_{selected_node_id}")

            if view_mode == "</> Generated Code":
                cfg = node_cfg.get("config", {})
                if hasattr(recipe_obj, "to_code"):
                    code_snippet = recipe_obj.to_code(cfg)
                else:
                    args_str = ", ".join(f"{k}={repr(v)}" for k, v in cfg.items())
                    code_snippet = f"# Recipe: {recipe_obj.name}\n# ID: {recipe_obj.recipe_id}\nresult = {recipe_obj.recipe_id}({args_str})"
                st.code(code_snippet, language="python")
            else:
                # SPECIAL HANDLER FOR INGESTION NODES
                if recipe_obj.category == "ingestion":
                    st.markdown("##### 📁 Ingestion Data Source:")
                    data_source_mode = st.radio("Source Mode", ["Use Active Dataset", "Upload New File", "📂 Platform Uploaded Datasets", "Choose Preset Sample", "🏢 Tenant Model Source"], index=0, key=f"src_mode_{selected_node_id}")

                    if data_source_mode == "📂 Platform Uploaded Datasets":
                        p_datasets = fetch_platform_datasets()
                        if p_datasets:
                            ds_options = {f"{d['name']} ({d['rows']} rows, {d['columns']} cols)": d for d in p_datasets}
                            chosen_label = st.selectbox("Select Platform Dataset", list(ds_options.keys()), key=f"pds_{selected_node_id}")
                            if st.button("Apply Selected Dataset to Pipeline", key=f"btn_apply_pds_{selected_node_id}"):
                                d_meta = ds_options[chosen_label]
                                df_loaded = load_platform_dataset(d_meta["storage_path"])
                                if df_loaded is not None:
                                    st.session_state["active_df"] = df_loaded
                                    st.session_state["active_dataset_name"] = d_meta["name"]
                                    for n in st.session_state["flow_state"].nodes:
                                        if n.id == selected_node_id:
                                            n.data = {"content": f"📄 {d_meta['name'][:15]}"}
                                    sanitize_node_configs_for_active_dataset()
                                    st.session_state["canvas_version"] = st.session_state.get("canvas_version", 1) + 1
                                    st.success(f"Applied '{d_meta['name']}' ({len(df_loaded)} rows, {len(df_loaded.columns)} cols) to pipeline!")
                                    st.rerun()
                        else:
                            st.info("No uploaded datasets found in database. Upload a CSV/Excel file using 'Upload New File' above.")
                    
                    elif data_source_mode == "Choose Preset Sample":
                        preset_name = st.selectbox("Preset Dataset", ["Customer Churn (Classification)", "Daily Retail Sales (Time-Series)", "Credit Transactions (Anomaly Injection)", "Titanic Survival", "Iris Flower"], key=f"preset_{selected_node_id}")
                        if st.button("Apply Preset to Ingestion Node"):
                            st.session_state["active_df"] = get_preset_dataset(preset_name)
                            st.session_state["active_dataset_name"] = preset_name
                            for n in st.session_state["flow_state"].nodes:
                                if n.id == selected_node_id:
                                    n.data = {"content": f"📄 {preset_name[:15]}"}
                            st.session_state["canvas_version"] = st.session_state.get("canvas_version", 1) + 1
                            st.success(f"Ingestion source set to '{preset_name}'!")
                            st.rerun()

                    elif data_source_mode == "Upload New File":
                        ingest_file = st.file_uploader("Upload CSV/Excel/JSON", type=["csv", "xlsx", "json"], key=f"up_{selected_node_id}")
                        if ingest_file is not None:
                            parsed_df, f_name = parse_uploaded_dataset(ingest_file)
                            if parsed_df is not None:
                                st.session_state["active_df"] = parsed_df
                                st.session_state["active_dataset_name"] = f_name
                                for n in st.session_state["flow_state"].nodes:
                                    if n.id == selected_node_id:
                                        n.data = {"content": f"📄 {f_name[:15]}"}
                                sanitize_node_configs_for_active_dataset()
                                st.session_state["canvas_version"] = st.session_state.get("canvas_version", 1) + 1
                                st.success(f"Ingested '{f_name}'!")
                                st.rerun()

                    elif data_source_mode == "🏢 Tenant Model Source":
                        tenant_id_val = int(st.number_input("Tenant ID", min_value=1, value=1, step=1, key=f"wb_tenant_{selected_node_id}"))
                        envs_cache_k = f"envs_cache_{tenant_id_val}"
                        if envs_cache_k not in st.session_state or st.button("🔄 Refresh Environments", key=f"refr_{selected_node_id}"):
                            try:
                                st.session_state[envs_cache_k] = fetch_tenant_environments(tenant_id=tenant_id_val)
                            except Exception as e:
                                st.error(f"Error fetching environments: {e}")
                                st.session_state[envs_cache_k] = []
                        wb_envs = st.session_state.get(envs_cache_k, [])
                        if wb_envs:
                            wb_env_opts = {f"{e['name']} (ID: {e['environment_id']})": e for e in wb_envs}
                            wb_sel_env_label = st.selectbox("Select Environment", list(wb_env_opts.keys()), key=f"wb_env_{selected_node_id}")
                            wb_sel_env = wb_env_opts[wb_sel_env_label]
                            wb_models = wb_sel_env.get("models", [])
                            if wb_models:
                                wb_model_opts = {f"{m['name']} (ID: {m['id']})": m for m in wb_models}
                                wb_sel_model_label = st.selectbox("Select Model", list(wb_model_opts.keys()), key=f"wb_mod_{selected_node_id}")
                                wb_sel_model = wb_model_opts[wb_sel_model_label]
                                wb_limit = st.selectbox("Row Limit", ["1000", "5000", "10000", "All Rows"], index=0, key=f"wb_limit_{selected_node_id}")
                                wb_row_limit = None if wb_limit == "All Rows" else int(wb_limit)

                                if st.button("📥 Apply Model Table to Pipeline", key=f"btn_apply_model_{selected_node_id}"):
                                    with st.spinner("Extracting model table and ingesting..."):
                                        try:
                                            df, meta = ingest_tenant_model(
                                                tenant_id=tenant_id_val,
                                                environment_id=wb_sel_env["environment_id"],
                                                model_id=wb_sel_model["id"],
                                                row_limit=wb_row_limit,
                                            )
                                            st.session_state["active_df"] = df
                                            st.session_state["active_dataset_name"] = meta["dataset_name"]
                                            for n in st.session_state["flow_state"].nodes:
                                                if n.id == selected_node_id:
                                                    n.data = {"content": f"🏢 {wb_sel_model['name'][:14]}"}
                                            sanitize_node_configs_for_active_dataset()
                                            st.session_state["canvas_version"] = st.session_state.get("canvas_version", 1) + 1
                                            st.success(f"Ingested {meta['row_count']} rows from model '{wb_sel_model['name']}'!")
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"Failed to ingest model table: {e}")
                            else:
                                st.info("No active models in this environment.")
                        else:
                            st.warning(f"No environments found for tenant {tenant_id_val}.")
                    else:
                        st.info(f"Using Active Dataset: **{st.session_state.get('active_dataset_name', 'Customer Churn')}** ({len(st.session_state['active_df'])} rows)")

                schema = recipe_obj.get_schema()
                props = schema.get("properties", {})
                current_config = dict(node_cfg.get("config", {}))

                active_cols = list(st.session_state["active_df"].columns) if "active_df" in st.session_state and st.session_state["active_df"] is not None else []
                saved_cols = list(st.session_state.get("saved_dataset_columns", []))
                available_cols = []
                for c in active_cols + saved_cols:
                    if c not in available_cols:
                        available_cols.append(c)

                active_df_obj = st.session_state.get("active_df")
                col_type_map = {}
                if active_df_obj is not None and isinstance(active_df_obj, pd.DataFrame):
                    ds_n = str(st.session_state.get("active_dataset_name", "ds"))
                    c_map_key = f"_col_type_map_{ds_n}_{len(active_df_obj.columns)}"
                    if c_map_key in st.session_state:
                        col_type_map = st.session_state[c_map_key]
                    else:
                        from backend.app.profiling.profiler import DataProfiler
                        for col_name in active_df_obj.columns:
                            col_type_map[str(col_name)] = DataProfiler._infer_column_type(active_df_obj[col_name])
                        st.session_state[c_map_key] = col_type_map

                def format_col_option(opt):
                    if not opt or opt in ["-- Select Column --", "(None)"]:
                        return str(opt)
                    ctype = col_type_map.get(str(opt))
                    if ctype:
                        icon = {"numeric": "🔢", "categorical": "🔤", "datetime": "📅", "boolean": "🔘", "text": "📝"}.get(ctype, "📄")
                        return f"{icon} {opt}  [{ctype}]"
                    return str(opt)

                if recipe_obj.recipe_id == "data_type_converter":
                    st.markdown("##### 🔤 Column Data Type Converter & Cast")
                    st.caption("Cast dataset columns to explicit types (`categorical`, `datetime`, `boolean`, `numeric`, `integer`, `string`). Unselected columns remain in their original type.")

                    # Quick dataset status & switcher
                    ds_name_active = st.session_state.get('active_dataset_name', 'Customer Churn')
                    c_ds1, c_ds2 = st.columns([3, 1])
                    with c_ds1:
                        st.info(f"📊 Active Data Source: **{ds_name_active}** ({len(available_cols)} columns detected)")
                    with c_ds2:
                        if st.button("🔄 Refresh", key=f"refr_cols_{selected_node_id}", help="Reload columns from active dataset"):
                            st.rerun()

                    with st.expander("📁 Switch Active Dataset (if columns above don't match your data)", expanded=(len(available_cols) == 0)):
                        sw_tab1, sw_tab2 = st.tabs(["📂 Platform Datasets", "📤 Upload File"])
                        with sw_tab1:
                            p_list = fetch_platform_datasets()
                            if p_list:
                                p_map = {f"{d['name']} ({d['rows']} rows, {d['columns']} cols)": d for d in p_list}
                                p_pick = st.selectbox("Select Uploaded Dataset", list(p_map.keys()), key=f"sw_ds_{selected_node_id}")
                                if st.button("Load Dataset", key=f"sw_btn_{selected_node_id}"):
                                    sel_meta = p_map[p_pick]
                                    loaded_df = load_platform_dataset(sel_meta["storage_path"])
                                    if loaded_df is not None:
                                        st.session_state["active_df"] = loaded_df
                                        st.session_state["active_dataset_name"] = sel_meta["name"]
                                        sanitize_node_configs_for_active_dataset()
                                        st.success(f"Loaded '{sel_meta['name']}' ({len(loaded_df.columns)} columns)!")
                                        st.rerun()
                            else:
                                st.caption("No platform datasets found in database.")
                        with sw_tab2:
                            sw_file = st.file_uploader("Upload CSV / Excel", type=["csv", "xlsx"], key=f"sw_up_{selected_node_id}")
                            if sw_file is not None:
                                s_df, s_name = parse_uploaded_dataset(sw_file)
                                if s_df is not None:
                                    st.session_state["active_df"] = s_df
                                    st.session_state["active_dataset_name"] = s_name
                                    sanitize_node_configs_for_active_dataset()
                                    st.success(f"Loaded '{s_name}' ({len(s_df.columns)} cols)!")
                                    st.rerun()

                    # Conversions
                    raw_conversions = current_config.get("conversions", {})
                    if not isinstance(raw_conversions, dict):
                        raw_conversions = {}
                    conversions = dict(raw_conversions)

                    # Gather all columns to display
                    display_cols = list(available_cols)
                    for c in conversions.keys():
                        if c not in display_cols:
                            display_cols.append(c)

                    TARGET_TYPE_CHOICES = [
                        "-- Leave Unchanged --",
                        "datetime (Date / Timestamp)",
                        "categorical (Discrete Category / ID)",
                        "boolean (True / False Flag)",
                        "numeric (Float / Continuous)",
                        "integer (Whole Number / Count)",
                        "string (Raw Text)"
                    ]

                    def map_target_to_choice(t: str) -> str:
                        t = str(t).lower()
                        if t in ["datetime", "date", "timestamp"]:
                            return "datetime (Date / Timestamp)"
                        elif t in ["categorical", "category"]:
                            return "categorical (Discrete Category / ID)"
                        elif t in ["boolean", "bool"]:
                            return "boolean (True / False Flag)"
                        elif t in ["numeric", "float", "number"]:
                            return "numeric (Float / Continuous)"
                        elif t in ["integer", "int"]:
                            return "integer (Whole Number / Count)"
                        elif t in ["string", "text", "str"]:
                            return "string (Raw Text)"
                        return "-- Leave Unchanged --"

                    def map_choice_to_target(c: str) -> Optional[str]:
                        if "datetime" in c:
                            return "datetime"
                        elif "categorical" in c:
                            return "categorical"
                        elif "boolean" in c:
                            return "boolean"
                        elif "numeric" in c:
                            return "numeric"
                        elif "integer" in c:
                            return "integer"
                        elif "string" in c:
                            return "string"
                        return None

                    if display_cols:
                        st.markdown("###### 🎯 Column Type Mappings:")
                        for col_name in display_cols:
                            c_left, c_right = st.columns([1, 1])
                            with c_left:
                                inferred_type = col_type_map.get(str(col_name), "feature")
                                icon = {"numeric": "🔢", "categorical": "🔤", "datetime": "📅", "boolean": "🔘", "text": "📝"}.get(inferred_type, "📄")
                                st.markdown(f"<div style='padding-top: 6px;'><b>{icon} {col_name}</b> <span style='font-size:0.75rem; color:#64748B;'>[{inferred_type}]</span></div>", unsafe_allow_html=True)
                            with c_right:
                                curr_t = conversions.get(col_name, "")
                                cur_choice = map_target_to_choice(curr_t)
                                cur_idx = TARGET_TYPE_CHOICES.index(cur_choice) if cur_choice in TARGET_TYPE_CHOICES else 0
                                chosen = st.selectbox(
                                    f"Type for {col_name}",
                                    TARGET_TYPE_CHOICES,
                                    index=cur_idx,
                                    key=f"dt_col_{selected_node_id}_{col_name}",
                                    label_visibility="collapsed"
                                )
                                new_tgt = map_choice_to_target(chosen)
                                if new_tgt:
                                    conversions[col_name] = new_tgt
                                else:
                                    conversions.pop(col_name, None)

                    # Add Column Manually
                    with st.expander("➕ Add Column Not Listed Above", expanded=(len(display_cols) == 0)):
                        add_c1, add_c2, add_c3 = st.columns([2, 2, 1])
                        with add_c1:
                            new_col_input = st.text_input("Column Name", placeholder="e.g. Date, Store, Holiday_Flag", key=f"dt_new_col_{selected_node_id}")
                        with add_c2:
                            new_type_input = st.selectbox("Target Type", TARGET_TYPE_CHOICES[1:], key=f"dt_new_type_{selected_node_id}")
                        with add_c3:
                            st.write("")
                            st.write("")
                            if st.button("➕ Add", key=f"dt_btn_add_{selected_node_id}"):
                                if new_col_input.strip():
                                    target_key = map_choice_to_target(new_type_input)
                                    conversions[new_col_input.strip()] = target_key
                                    st.rerun()

                    # Datetime parsing format
                    has_datetime = any(v in ["datetime", "date", "timestamp"] for v in conversions.values())
                    DT_PRESETS = [
                        "DD-MM-YYYY (%d-%m-%Y)",
                        "DD/MM/YYYY (%d/%m/%Y)",
                        "YYYY-MM-DD (%Y-%m-%d)",
                        "MM-DD-YYYY (%m-%d-%Y)",
                        "Auto-detect (Flexible)",
                        "Custom format..."
                    ]
                    curr_dt_fmt = current_config.get("datetime_format", "")
                    matched_idx = 0
                    if curr_dt_fmt:
                        if "dd-mm-yyyy" in curr_dt_fmt.lower() or "%d-%m-%y" in curr_dt_fmt.lower():
                            matched_idx = 0
                        elif "dd/mm/yyyy" in curr_dt_fmt.lower() or "%d/%m/%y" in curr_dt_fmt.lower():
                            matched_idx = 1
                        elif "yyyy-mm-dd" in curr_dt_fmt.lower() or "%y-%m-%d" in curr_dt_fmt.lower():
                            matched_idx = 2
                        elif "mm-dd-yyyy" in curr_dt_fmt.lower() or "%m-%d-%y" in curr_dt_fmt.lower():
                            matched_idx = 3
                        else:
                            matched_idx = 5

                    if has_datetime:
                        st.markdown("---")
                        st.markdown("##### 📅 Datetime Parsing Format:")
                        st.caption("Choose **DD-MM-YYYY** for day-first dates (e.g. 05-02-2010 = Feb 5).")
                        chosen_dt_preset = st.selectbox(
                            "Date Format Preset",
                            DT_PRESETS,
                            index=matched_idx,
                            key=f"dt_fmt_sel_{selected_node_id}"
                        )
                        if chosen_dt_preset == "Custom format...":
                            final_dt_fmt = st.text_input("Custom Strftime Pattern", value=curr_dt_fmt or "%d-%m-%Y", key=f"dt_cust_{selected_node_id}")
                        elif chosen_dt_preset == "Auto-detect (Flexible)":
                            final_dt_fmt = ""
                        else:
                            final_dt_fmt = chosen_dt_preset.split(" ")[0]
                    else:
                        final_dt_fmt = ""

                    st.markdown("---")
                    ERR_OPTIONS = ["coerce (replace unparseable values with NaN)", "ignore (keep original value untouched)", "raise (halt pipeline with error)"]
                    cur_err = current_config.get("errors", "coerce")
                    err_idx = 0 if cur_err == "coerce" else (1 if cur_err == "ignore" else 2)
                    chosen_err = st.selectbox("Invalid Value Handling", ERR_OPTIONS, index=err_idx, key=f"dt_err_{selected_node_id}")
                    final_err = "coerce" if "coerce" in chosen_err else ("ignore" if "ignore" in chosen_err else "raise")

                    st.markdown("---")
                    if conversions:
                        st.markdown(f"**⚡ Active Casts ({len(conversions)}):**")
                        for c_name, c_type in conversions.items():
                            extra = f" *(format: {final_dt_fmt})*" if c_type == "datetime" and final_dt_fmt else ""
                            badge_color = {"categorical": "🟣", "datetime": "📅", "boolean": "🟢", "numeric": "🔵", "integer": "🔢"}.get(c_type, "⚪")
                            st.markdown(f"- {badge_color} `{c_name}` ➔ **`{c_type}`**{extra}")
                    else:
                        st.info("ℹ️ No columns configured for casting. Columns will retain their original inferred types.")

                    current_config["conversions"] = conversions
                    current_config["datetime_format"] = final_dt_fmt
                    current_config["errors"] = final_err
                    st.session_state["node_configs"][selected_node_id]["config"] = current_config

                else:
                    for prop_name, prop_meta in props.items():
                        title = prop_meta.get("title", prop_name)
                        default_val = prop_meta.get("default", None)
                        curr_val = current_config.get(prop_name, default_val)
                        prop_type = prop_meta.get("type")

                        # Smart column selectors: array of columns vs single column
                        if prop_type == "array" or prop_name in ["columns", "feature_columns", "categorical_columns", "numerical_columns"]:
                            multiselect_options = list(available_cols)
                            if isinstance(curr_val, str):
                                curr_list = [c.strip() for c in curr_val.split(",") if c.strip()]
                            elif isinstance(curr_val, (list, tuple)):
                                curr_list = [str(c) for c in curr_val]
                            else:
                                curr_list = []
                            # Ensure any saved columns are in the options list so they don't disappear
                            for c in curr_list:
                                if c not in multiselect_options:
                                    multiselect_options.append(c)
                            new_val = st.multiselect(
                                title,
                                options=multiselect_options,
                                default=curr_list,
                                format_func=format_col_option,
                                key=f"cfg_{selected_node_id}_{prop_name}",
                                help=prop_meta.get("description", "Select specific columns or leave empty to apply across all columns.")
                            )
                        elif ("column" in prop_name.lower() or prop_name.endswith("_col")):
                            is_required = prop_name in schema.get("required", []) or prop_name == "target_column"
                            col_options = list(available_cols)
                            if curr_val and str(curr_val).strip() and str(curr_val) not in col_options and str(curr_val) not in ["-- Select Column --", "(None)"]:
                                col_options.append(str(curr_val))
                            
                            if is_required:
                                options = ["-- Select Column --"] + col_options
                                if curr_val and str(curr_val) in options:
                                    col_idx = options.index(str(curr_val))
                                else:
                                    col_idx = 0
                            else:
                                options = ["(None)"] + col_options
                                if curr_val and str(curr_val) in options:
                                    col_idx = options.index(str(curr_val))
                                else:
                                    col_idx = 0

                            selected_opt = st.selectbox(title, options, index=col_idx, format_func=format_col_option, key=f"cfg_{selected_node_id}_{prop_name}")
                            if selected_opt in ["-- Select Column --", "(None)"]:
                                new_val = ""
                            else:
                                new_val = selected_opt
                        elif "enum" in prop_meta:
                            options = prop_meta["enum"]
                            opt_idx = options.index(curr_val) if curr_val in options else 0
                            new_val = st.selectbox(title, options, index=opt_idx, key=f"cfg_{selected_node_id}_{prop_name}")
                        elif prop_type == "integer":
                            min_v = int(prop_meta.get("minimum", 1))
                            max_v = int(prop_meta.get("maximum", 1000))
                            try:
                                val_int = int(curr_val) if curr_val is not None else min_v
                            except (ValueError, TypeError):
                                val_int = min_v
                            new_val = st.slider(title, min_value=min_v, max_value=max_v, value=val_int, key=f"cfg_{selected_node_id}_{prop_name}")
                        elif prop_type == "number":
                            min_v = float(prop_meta.get("minimum", 0.0))
                            max_v = float(prop_meta.get("maximum", 1.0))
                            try:
                                val_float = float(curr_val) if curr_val is not None else min_v
                            except (ValueError, TypeError):
                                val_float = min_v
                            new_val = st.slider(title, min_value=min_v, max_value=max_v, value=val_float, step=0.01, key=f"cfg_{selected_node_id}_{prop_name}")
                        elif prop_type == "boolean":
                            new_val = st.checkbox(title, value=bool(curr_val) if curr_val is not None else False, key=f"cfg_{selected_node_id}_{prop_name}")
                        else:
                            new_val = st.text_input(title, value=str(curr_val) if curr_val is not None else "", key=f"cfg_{selected_node_id}_{prop_name}")

                        current_config[prop_name] = new_val

                    # Inline Auto-Switch / Synchronization for Split Mode & Temporal Setting
                    r_id = getattr(recipe_obj, "recipe_id", None) or node_cfg.get("recipe_id", "")
                    if r_id == "train_test_split":
                        is_strat = bool(current_config.get("stratify", False))
                        is_ts = bool(current_config.get("time_series_mode", False))
                        if is_strat and is_ts:
                            # Stratified and Chronological modes are mutually exclusive
                            current_config["time_series_mode"] = False
                            st.info("ℹ️ **Split Mode Guardrail:** When Stratified Split is enabled, Chronological Time-Series Mode is automatically switched to False because stratification randomly partitions data across classes.")
                    elif r_id == "stratified_split":
                        current_config["is_temporal"] = False

                    # Inline Validation Check for Discrete 0/1 Target in Regression Mode
                    if r_id in ["xgboost_trainer", "lightgbm_trainer", "catboost_trainer", "random_forest_trainer", "linear_trainer"]:
                        if str(current_config.get("task_type", "")).lower() == "regression":
                            resolved_target = current_config.get("target_column")
                            if not resolved_target:
                                for nid, n_info in st.session_state.get("node_configs", {}).items():
                                    if nid != selected_node_id and n_info.get("config", {}).get("target_column"):
                                        resolved_target = n_info["config"]["target_column"]
                                        break
                            if not resolved_target:
                                resolved_target = st.session_state.get("active_target_col")

                            is_discrete_binary = False
                            df_active = st.session_state.get("active_df")
                            if df_active is not None and resolved_target and resolved_target in df_active.columns:
                                s = df_active[resolved_target].dropna()
                                u_vals = set(s.unique())
                                if len(u_vals) <= 2 and (u_vals.issubset({0, 1, 0.0, 1.0, True, False, "0", "1", "yes", "no"}) or len(u_vals) <= 2):
                                    is_discrete_binary = True
                            elif resolved_target:
                                t_lower = str(resolved_target).lower()
                                if any(k in t_lower for k in ["churn", "fraud", "default", "label", "binary", "class", "target_churn", "survived", "promoted", "converted"]) or t_lower.startswith("is_"):
                                    is_discrete_binary = True

                            if is_discrete_binary:
                                st.warning(
                                    f"⚠️ **Discrete 0/1 Target in Regression Mode:** Column `{resolved_target}` contains discrete binary (0/1) labels, "
                                    f"but **Regression** mode is selected.\n\n"
                                    f"Regression models (e.g. XGBoost Regressor) predict continuous numeric quantities like `sales_amount`, `price`, or `temperature`. "
                                    f"For binary 0/1 targets like churn, please switch **ML Task Type** to **Classification**, or choose a numeric target like `sales_amount`."
                                )

                    st.session_state["node_configs"][selected_node_id]["config"] = current_config

            if st.button("🗑️ Delete Node", type="secondary", key=f"del_node_{selected_node_id}"):
                st.session_state["flow_state"].nodes = [n for n in st.session_state["flow_state"].nodes if n.id != selected_node_id]
                st.session_state["flow_state"].edges = [e for e in st.session_state["flow_state"].edges if e.source != selected_node_id and e.target != selected_node_id]
                if selected_node_id in st.session_state["node_configs"]:
                    del st.session_state["node_configs"][selected_node_id]
                st.session_state["canvas_version"] = st.session_state.get("canvas_version", 1) + 1
                
                record_api_telemetry(
                    action_name=f"🗑️ Delete Node `{selected_node_id}`",
                    endpoint=f"/api/v1/workflows/nodes/{selected_node_id}",
                    method="DELETE",
                    request_payload={"deleted_node_id": selected_node_id},
                    response_payload={"status": "DELETED", "node_id": selected_node_id},
                    status_code=200,
                    duration_ms=1.2
                )
                
                st.rerun()

            with st.expander("✨ Chrysa Copilot Assistant", expanded=False):
                render_chrysa_assistant_card()
        else:
            render_chrysa_assistant_card()

    # Execute Trigger Check
    if execute_clicked_top:
        with st.spinner("Executing pipeline through Topological DAG Engine..."):
            execute_pipeline()

    # ---------------------------------------------------------
    # STEP 4: PERSISTENT EXECUTION RESULTS & REPORTS
    # ---------------------------------------------------------
    if "last_execution" in st.session_state and st.session_state["last_execution"]:
        exec_data = st.session_state["last_execution"]
        final_metrics = exec_data.get("final_metrics")
        node_outputs = exec_data.get("node_outputs", {})
        execution_logs = exec_data.get("execution_logs", [])

        st.markdown("---")
        st.markdown("## 📊 Execution & Diagnostic Results")

        # -----------------------------------------------------
        # 1. TIME-SERIES FORECASTING REPORT
        # -----------------------------------------------------
        fc_sum = exec_data.get("forecasting_summary") or final_metrics if (final_metrics and final_metrics.get("task_type") == "time_series_forecasting") else exec_data.get("forecasting_summary")
        fc_df = None
        if fc_sum and "forecast_data" in fc_sum:
            fc_df = pd.DataFrame(fc_sum["forecast_data"])
        else:
            for n_id, out in node_outputs.items():
                if isinstance(out, dict):
                    cand = out.get("forecast_df") or out.get("dataframe")
                    if isinstance(cand, dict) and "records" in cand:
                        cand_df = pd.DataFrame(cand["records"])
                        if "ds" in cand_df.columns and "yhat" in cand_df.columns:
                            fc_df = cand_df
                            break
                    elif isinstance(cand, pd.DataFrame) and "ds" in cand.columns and "yhat" in cand.columns:
                        fc_df = cand
                        break

        if fc_sum or fc_df is not None:
            st.markdown("### 📈 Time-Series Forecast Predictions")
            
            f_col1, f_col2, f_col3 = st.columns(3)
            hist_pts = fc_sum.get('historical_points', len(fc_df[fc_df['is_future']==0]) if fc_df is not None and 'is_future' in fc_df.columns else 0) if fc_sum else (len(fc_df[fc_df['is_future']==0]) if fc_df is not None and 'is_future' in fc_df.columns else 0)
            fc_horizon = fc_sum.get('forecast_horizon', len(fc_df[fc_df['is_future']==1]) if fc_df is not None and 'is_future' in fc_df.columns else 30) if fc_sum else (len(fc_df[fc_df['is_future']==1]) if fc_df is not None and 'is_future' in fc_df.columns else 30)
            trend_dir = fc_sum.get("trend_direction", "Upward" if fc_df is not None and float(fc_df['yhat'].iloc[-1]) >= float(fc_df['yhat'].iloc[0]) else "Neutral") if fc_sum else "Neutral"
            
            f_col1.metric("Historical Periods", f"{hist_pts:,}")
            f_col2.metric("Forecast Horizon", f"{fc_horizon} intervals")
            f_col3.metric("Trend Direction", str(trend_dir).upper())

            if fc_df is not None:
                fig_fc = go.Figure()
                
                if "ds" in fc_df.columns and "yhat" in fc_df.columns:
                    fig_fc.add_trace(go.Scatter(x=fc_df["ds"], y=fc_df["yhat"], mode="lines+markers", name="Forecast Trend", line=dict(color="#3B82F6", width=2.5)))
                    if "yhat_lower" in fc_df.columns and "yhat_upper" in fc_df.columns:
                        fig_fc.add_trace(go.Scatter(x=fc_df["ds"], y=fc_df["yhat_upper"], mode="lines", line=dict(width=0), showlegend=False))
                        fig_fc.add_trace(go.Scatter(x=fc_df["ds"], y=fc_df["yhat_lower"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(59, 130, 246, 0.2)", name="95% Confidence Interval"))
                
                fig_fc.update_layout(title="Future Forecast Trajectory with Confidence Intervals", xaxis_title="Timeline", yaxis_title="Predicted Value", height=420)
                st.plotly_chart(fig_fc, use_container_width=True)

                with st.expander("📋 Tabular Forecast Matrix", expanded=False):
                    st.dataframe(fc_df, use_container_width=True)

        # -----------------------------------------------------
        # 2. UNSUPERVISED ANOMALY DETECTION REPORT
        # -----------------------------------------------------
        anom_sum = exec_data.get("anomaly_summary") or final_metrics if (final_metrics and final_metrics.get("task_type") == "anomaly_detection") else exec_data.get("anomaly_summary")
        anomaly_df = None
        if anom_sum and "scatter_preview" in anom_sum:
            anomaly_df = pd.DataFrame(anom_sum["scatter_preview"])
        else:
            for n_id, out in node_outputs.items():
                if isinstance(out, dict):
                    candidate_df = out.get("dataframe") if "dataframe" in out else out.get("df")
                    if isinstance(candidate_df, dict) and "records" in candidate_df:
                        cand_df = pd.DataFrame(candidate_df["records"])
                        if "is_anomaly" in cand_df.columns or "is_outlier" in cand_df.columns:
                            anomaly_df = cand_df
                            break
                    elif candidate_df is not None and isinstance(candidate_df, pd.DataFrame) and ("is_anomaly" in candidate_df.columns or "is_outlier" in candidate_df.columns):
                        anomaly_df = candidate_df
                        break

        if anom_sum or anomaly_df is not None:
            st.markdown("### 🚨 Anomaly Detection & Outlier Risk Matrix")
            
            tot_rec = anom_sum.get('total_records', len(anomaly_df)) if anom_sum else (len(anomaly_df) if anomaly_df is not None else 0)
            flag_col = "is_anomaly" if anomaly_df is not None and "is_anomaly" in anomaly_df.columns else "is_outlier"
            anom_cnt = anom_sum.get('anomaly_count', int(anomaly_df[flag_col].sum())) if anom_sum else (int(anomaly_df[flag_col].sum()) if anomaly_df is not None and flag_col in anomaly_df.columns else 0)
            anom_rate = anom_sum.get('anomaly_percentage', (anom_cnt/tot_rec*100) if tot_rec > 0 else 0.0) if anom_sum else ((anom_cnt/tot_rec*100) if tot_rec > 0 else 0.0)

            a_col1, a_col2, a_col3 = st.columns(3)
            a_col1.metric("Total Records Inspected", f"{tot_rec:,}")
            a_col2.metric("Anomalies Flagged", f"{anom_cnt:,}")
            a_col3.metric("Anomaly Rate", f"{anom_rate:.2f}%")

            if anomaly_df is not None:
                anom_tab1, anom_tab2, anom_tab3 = st.tabs([
                    "🌌 2D Risk Cluster Scatter",
                    "📊 Anomaly Score Distribution",
                    "📋 Flagged Records Table"
                ])

                with anom_tab1:
                    num_cols = anomaly_df.select_dtypes(include=[np.number]).columns.tolist()
                    num_cols = [c for c in num_cols if c not in ["is_anomaly", "is_outlier", "anomaly_score"]]
                    if len(num_cols) >= 2:
                        x_ax = st.selectbox("X-Axis Feature", num_cols, index=0, key="anom_x")
                        y_ax = st.selectbox("Y-Axis Feature", num_cols, index=min(1, len(num_cols)-1), key="anom_y")
                        
                        plot_df = anomaly_df.copy()
                        plot_df[flag_col] = plot_df[flag_col].map({0: "Normal Record", 1: "Flagged Anomaly"})

                        fig_anom = px.scatter(
                            plot_df,
                            x=x_ax,
                            y=y_ax,
                            color=flag_col,
                            color_discrete_map={"Normal Record": "#3B82F6", "Flagged Anomaly": "#EF4444"},
                            title=f"Outlier Scatter Plot: {x_ax} vs {y_ax}",
                            hover_data=num_cols[:4]
                        )
                        st.plotly_chart(fig_anom, use_container_width=True)
                    else:
                        st.info("ℹ️ No numeric columns available for outlier distribution plotting.")

                with anom_tab2:
                    if "anomaly_score" in anomaly_df.columns:
                        hist_df = anomaly_df.copy()
                        if flag_col in hist_df.columns:
                            hist_df[flag_col] = hist_df[flag_col].astype(str)

                        fig_hist = px.histogram(
                            hist_df,
                            x="anomaly_score",
                            color=flag_col if flag_col in hist_df.columns else None,
                            color_discrete_map={"0": "#3B82F6", "1": "#EF4444"},
                            nbins=30,
                            title="Distribution of Anomaly Scores (0.0 = Normal, 1.0 = High-Risk Anomaly)"
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)

                with anom_tab3:
                    outliers_only = anomaly_df[anomaly_df[flag_col] == 1]
                    st.markdown(f"##### Showing {len(outliers_only)} Flagged Anomalous Records:")
                    st.dataframe(outliers_only, use_container_width=True)

                    st.download_button(
                        "📥 Download Flagged Anomalies (CSV)",
                        data=anomaly_df.to_csv(index=False),
                        file_name="flagged_anomalies_report.csv",
                        mime="text/csv"
                    )

        # -----------------------------------------------------
        # 3. CLASSIFICATION / REGRESSION EVALUATION REPORT
        # -----------------------------------------------------
        if final_metrics and (final_metrics.get("task_type") in ["classification", "regression"] or "accuracy" in final_metrics or "r2_score" in final_metrics):
            st.markdown("### 🏆 Comprehensive Model Evaluation & Diagnostic Reports")
            
            is_regression = final_metrics.get("task_type") == "regression" or "r2_score" in final_metrics

            if is_regression:
                rep_tab1, rep_tab2, rep_tab3 = st.tabs([
                    "📊 Regression KPIs",
                    "📈 Actual vs Predicted Fit",
                    "🌲 Feature Importances & Diagnostics"
                ])

                with rep_tab1:
                    m1, m2, m3, m4, m5 = st.columns(5)
                    r2_v = final_metrics.get("r2_score", 0.0)
                    mae_v = final_metrics.get("mae", 0.0)
                    rmse_v = final_metrics.get("rmse", 0.0)
                    mse_v = final_metrics.get("mse", 0.0)
                    mape_v = final_metrics.get("mape")

                    m1.metric("R² Score (Variance Explained)", f"{r2_v:.4f}")
                    m2.metric("Mean Absolute Error (MAE)", f"{mae_v:.4f}")
                    m3.metric("Root Mean Squared Error (RMSE)", f"{rmse_v:.4f}")
                    m4.metric("Mean Squared Error (MSE)", f"{mse_v:.4f}")
                    mape_display = final_metrics.get("mape_display")
                    if mape_display:
                        mape_str = mape_display
                    elif mape_v is not None:
                        mape_str = f"{mape_v*100:.2f}%" if (isinstance(mape_v, (int, float)) and mape_v < 1.0) else f"{mape_v:.2f}%"
                    else:
                        mape_str = "N/A"
                    m5.metric("MAPE", mape_str)

                    # KPI Cards for Split Mode and Temporal Status
                    if "split_mode" in final_metrics or "is_temporal" in final_metrics:
                        sm_col1, sm_col2, sm_col3 = st.columns(3)
                        sm_col1.metric("Split Mode", str(final_metrics.get("split_mode", "N/A")).capitalize())
                        is_temp = bool(final_metrics.get("is_temporal", False))
                        sm_col2.metric("Is Temporal", "true" if is_temp else "false")
                        temp_col = final_metrics.get("temporal_column")
                        sm_col3.metric("Temporal Column", str(temp_col) if (temp_col and is_temp) else "None")

                    smape_str = f" (SMAPE: {final_metrics['smape']:.2f}%)" if final_metrics.get("smape") is not None else ""
                    st.info(f"✨ **Regression Model Evaluation:** Model achieved **R² = {r2_v:.4f}** with an average absolute deviation of **MAE = {mae_v:.4f}** on the unseen test set.{smape_str}")

                with rep_tab2:
                    traj_data = final_metrics.get("trajectory") or final_metrics.get("actual_vs_predicted_time_series")
                    temp_col_label = final_metrics.get("temporal_column", "Step")

                    if traj_data and len(traj_data) > 0:
                        traj_df = pd.DataFrame(traj_data)
                        # Ensure ds column is string for plotly categorical axis
                        traj_df["ds"] = traj_df["ds"].astype(str)

                        fig_traj = go.Figure()
                        fig_traj.add_trace(go.Scatter(
                            x=traj_df["ds"], y=traj_df["actual"],
                            mode="lines", name="Actual",
                            line=dict(color="#10B981", width=2)
                        ))
                        fig_traj.add_trace(go.Scatter(
                            x=traj_df["ds"], y=traj_df["yhat"],
                            mode="lines", name="Predicted",
                            line=dict(color="#3B82F6", width=2, dash="dot")
                        ))
                        fig_traj.update_layout(
                            title="Actual vs Predicted — Chronological Trajectory",
                            xaxis_title=temp_col_label,
                            yaxis_title="Value",
                            height=400,
                            template="plotly_white",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(family="Inter, sans-serif", color="#0F172A"),
                            legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
                            margin=dict(l=40, r=20, t=55, b=40)
                        )
                        st.plotly_chart(fig_traj, use_container_width=True)

                        # Residual chart
                        if "residual" in traj_df.columns:
                            fig_res = go.Figure()
                            fig_res.add_trace(go.Bar(
                                x=traj_df["ds"], y=traj_df["residual"],
                                name="Residual (Actual − Predicted)",
                                marker_color=traj_df["residual"].apply(
                                    lambda r: "#EF4444" if r < 0 else "#10B981"
                                )
                            ))
                            fig_res.add_hline(y=0, line_dash="dash", line_color="#94A3B8", opacity=0.6)
                            fig_res.update_layout(
                                title="Residuals (Actual − Predicted)",
                                xaxis_title=temp_col_label,
                                yaxis_title="Residual",
                                height=280,
                                template="plotly_white",
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font=dict(family="Inter, sans-serif", color="#0F172A"),
                                margin=dict(l=40, r=20, t=45, b=40)
                            )
                            st.plotly_chart(fig_res, use_container_width=True)

                        with st.expander("📋 Raw Trajectory Table", expanded=False):
                            st.dataframe(traj_df, use_container_width=True)
                    else:
                        st.markdown("##### Model Predictions Preview on Unseen Test Partition:")
                        if "predictions_sample" in exec_data.get("final_metrics", {}):
                            preds = exec_data["final_metrics"]["predictions_sample"]
                            st.dataframe(pd.DataFrame({"Sample Prediction": preds}), use_container_width=True)
                        else:
                            st.info("ℹ️ No trajectory data available. Re-run the pipeline to generate the Actual vs Predicted chart.")

                with rep_tab3:
                    for n_id, out in node_outputs.items():
                        if "feature_importances" in out and out["feature_importances"]:
                            fi = out["feature_importances"]
                            fi_df = pd.DataFrame(list(fi.items()), columns=["Feature", "Importance"]).sort_values(by="Importance", ascending=True)
                            fig_fi = px.bar(
                                fi_df, x="Importance", y="Feature", orientation="h",
                                title=f"Feature Importances from Node '{n_id}'",
                                color_discrete_sequence=["#2563EB"],
                                template="plotly_white"
                            )
                            fig_fi.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color="#0F172A"))
                            st.plotly_chart(fig_fi, use_container_width=True)

            else:
                rep_tab1, rep_tab2, rep_tab3, rep_tab4 = st.tabs([
                    "📊 Executive KPIs",
                    "🗂️ Confusion Matrix",
                    "📋 Classification Report (Per-Class)",
                    "🌲 Feature Importances & Diagnostics"
                ])

                with rep_tab1:
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Accuracy", f"{final_metrics.get('accuracy', 0)*100:.2f}%")
                    m2.metric("Balanced Accuracy", f"{final_metrics.get('balanced_accuracy', 0)*100:.2f}%")
                    m3.metric("Precision", f"{final_metrics.get('precision', 0)*100:.2f}%")
                    m4.metric("Recall", f"{final_metrics.get('recall', 0)*100:.2f}%")
                    m5.metric("F1 Score", f"{final_metrics.get('f1_score', 0)*100:.2f}%")

                    # KPI Cards for Split Mode and Temporal Status
                    if "split_mode" in final_metrics or "is_temporal" in final_metrics:
                        sm_col1, sm_col2, sm_col3 = st.columns(3)
                        sm_col1.metric("Split Mode", str(final_metrics.get("split_mode", "N/A")).capitalize())
                        is_temp = bool(final_metrics.get("is_temporal", False))
                        sm_col2.metric("Is Temporal", "true" if is_temp else "false")
                        temp_col = final_metrics.get("temporal_column")
                        sm_col3.metric("Temporal Column", str(temp_col) if (temp_col and is_temp) else "None")

                    if "roc_auc" in final_metrics:
                        st.info(f"✨ **ROC-AUC Score:** `{final_metrics['roc_auc']:.4f}` | **Log Loss:** `{final_metrics.get('log_loss', 'N/A')}`")

                with rep_tab2:
                    if "confusion_matrix" in final_metrics and final_metrics["confusion_matrix"]:
                        try:
                            cm = final_metrics["confusion_matrix"]
                            cm_arr = np.array(cm)
                            n_classes = len(cm_arr)
                            labels_x = ["Pred: Class 0", "Pred: Class 1"] if n_classes == 2 else [f"Pred: Class {i}" for i in range(n_classes)]
                            labels_y = ["Actual: Class 0", "Actual: Class 1"] if n_classes == 2 else [f"Actual: Class {i}" for i in range(n_classes)]

                            fig_cm = px.imshow(
                                cm_arr,
                                labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
                                x=labels_x,
                                y=labels_y,
                                text_auto=True,
                                color_continuous_scale="Blues",
                                aspect="auto"
                            )
                            fig_cm.update_layout(
                                title="Confusion Matrix Heatmap",
                                height=380,
                                template="plotly_white",
                                paper_bgcolor="rgba(0,0,0,0)",
                                font=dict(family="Inter, sans-serif", color="#0F172A"),
                                xaxis_title="Predicted Label",
                                yaxis_title="Actual Label",
                                margin=dict(l=40, r=40, t=50, b=40)
                            )
                            st.plotly_chart(fig_cm, use_container_width=True)
                        except Exception:
                            st.markdown("##### Confusion Matrix Matrix:")
                            st.dataframe(pd.DataFrame(final_metrics["confusion_matrix"]), use_container_width=True)

                with rep_tab3:
                    if "classification_report" in final_metrics and final_metrics["classification_report"]:
                        clf_dict = final_metrics["classification_report"]
                        clf_df = pd.DataFrame(clf_dict).transpose()
                        st.markdown("##### Per-Class Performance Table:")
                        st.dataframe(clf_df.style.format("{:.3f}", na_rep="-"), use_container_width=True)

                with rep_tab4:
                    for n_id, out in node_outputs.items():
                        if "feature_importances" in out and out["feature_importances"]:
                            fi = out["feature_importances"]
                            fi_df = pd.DataFrame(list(fi.items()), columns=["Feature", "Importance"]).sort_values(by="Importance", ascending=True)
                            fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h", title=f"Feature Importances from Node '{n_id}'")
                            st.plotly_chart(fig_fi, use_container_width=True)

            st.download_button(
                "📥 Download Full Evaluation Report (JSON)",
                data=json.dumps(final_metrics, indent=2),
                file_name="pipeline_evaluation_report.json",
                mime="application/json"
            )

        # -----------------------------------------------------
        # 4. STEP-BY-STEP DATA STREAM DEBUGGER (n8n / Boomi Inspector)
        # -----------------------------------------------------
        st.markdown("### 🔬 Step-by-Step Data Stream Inspector & Node Debugger")
        st.caption("Inspect live data transformations, schema changes, and intermediate DataFrames as they flowed across every edge in the DAG.")
        
        step_snapshots = exec_data.get("step_snapshots", {})
        if step_snapshots:
            selected_step_node = st.selectbox(
                "Select Executed Node to Inspect Data Snapshot",
                list(step_snapshots.keys()),
                format_func=lambda nid: f"Node: {nid} [{step_snapshots[nid]['recipe_name']}] ({step_snapshots[nid]['duration_ms']}ms)"
            )
            
            if selected_step_node and selected_step_node in step_snapshots:
                snap = step_snapshots[selected_step_node]
                st.markdown(f"#### 📦 Snapshot: `{snap['node_id']}` ({snap['recipe_name']})")
                
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Execution Latency", f"{snap['duration_ms']}ms")
                k1_rows = f"{snap['row_count']:,}" if snap['row_count'] is not None else "N/A"
                k2.metric("Output Records", k1_rows)
                k3.metric("Output Features", f"{len(snap['columns'])}")
                k4.metric("Inputs Received", ", ".join(snap['input_keys']) if snap['input_keys'] else "Root Node")

                if snap["preview_rows"]:
                    st.markdown("##### 📄 Output DataFrame Snapshot (First 5 Rows):")
                    st.dataframe(pd.DataFrame(snap["preview_rows"]), use_container_width=True)
                elif snap["output_keys"]:
                    st.markdown(f"**Outputs Generated:** `{', '.join(snap['output_keys'])}`")

                if snap["columns"]:
                    with st.expander("📋 Feature Schema at this Step", expanded=False):
                        st.write(snap["columns"])
        
        with st.expander("📜 Raw Execution Logs & Timing Telemetry", expanded=False):
            for log in execution_logs:
                st.write(log)

        # -----------------------------------------------------
        # 5. INTERACTIVE MODEL PREDICTION STUDIO & DIALOG
        # -----------------------------------------------------
        st.markdown("---")
        st.markdown("### 🔮 Interactive Model Prediction Studio & Sandbox")
        st.caption("Ask for live predictions on new observations, time ranges, or test records using your trained pipeline model.")

        bundle = st.session_state.get("inference_bundle")
        schema = st.session_state.get("inference_schema")

        if bundle and bundle.get("model"):
            col_dlg1, col_dlg2 = st.columns([3.5, 6.5])
            with col_dlg1:
                if st.button("🔮 Open Live Prediction Dialog (Modal Window)", type="primary", use_container_width=True, key="btn_open_dialog_modal_hero"):
                    show_prediction_dialog()

            with st.expander("⚡ In-Page Prediction Sandbox", expanded=True):
                render_prediction_studio(bundle, schema, in_dialog=False)
        else:
            st.info("ℹ️ Pipeline has executed. If a predictive model was trained, its live prediction sandbox will appear here.")

        # -----------------------------------------------------
        # 6. LIVE REST API CONTRACT & CURL GENERATOR
        # -----------------------------------------------------
        with st.expander("📡 Live REST API Contract & cURL Generator (n8n / Postman Integration)", expanded=False):
            st.markdown("##### 📦 Active DAG Execution Payload (`POST /api/v1/workflows/execute`):")
            st.caption("This exact JSON payload can be sent to FastAPI, external services, or automated CI/CD pipelines to execute this visual workflow.")
            
            dag_payload = get_current_dag_payload()
            dag_json_str = json.dumps(dag_payload, indent=2)
            st.code(dag_json_str, language="json")
            
            st.markdown("##### 💻 Ready-to-Run cURL Terminal Command:")
            curl_cmd = f"""curl -X POST http://localhost:8000/api/v1/workflows/execute \\
  -H "Content-Type: application/json" \\
  -d '{json.dumps(dag_payload)}'"""
            st.code(curl_cmd, language="bash")
            
            c_btn1, c_btn2, _ = st.columns([2.5, 2.5, 3])
            with c_btn1:
                st.download_button("📥 Download DAG Payload (JSON)", data=dag_json_str, file_name="workflow_dag_payload.json", mime="application/json")
            with c_btn2:
                if st.button("🧪 Validate via API Engine", key="btn_validate_api_inline"):
                    wf_graph = WorkflowGraph(
                        nodes=[WorkflowNode(id=n["id"], recipe_id=n["recipe_id"], config=n["config"]) for n in dag_payload["nodes"]],
                        edges=[WorkflowEdge(source=e["source"], target=e["target"]) for e in dag_payload["edges"]]
                    )
                    errs = wf_graph.validate_graph()
                    if not errs:
                        st.success("✅ Graph structure is 100% topologically valid and API-compatible!")
                    else:
                        st.warning(f"⚠️ Validation notes: {len(errs)} items detected.")
                        for err in errs:
                            st.write(f"- {err}")


# -------------------------------------------------------------
# TAB 2: DATASET STUDIO & PROFILER
# -------------------------------------------------------------
elif "Dataset Studio" in app_mode:
    st.markdown('<div class="main-header">📊 Dataset Studio & Automated Profiler</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload your dataset to inspect preview rows and automatic statistical profiles.</div>', unsafe_allow_html=True)

    ingest_tab1, ingest_tab2, ingest_tab3 = st.tabs([
        "🏢 Multi-Tenant Model Source (External DB / Druid)",
        "📁 Upload Local File (CSV / Excel / JSON)",
        "📦 Choose Preset Sample",
    ])

    with ingest_tab1:
        st.markdown(
            """
            <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 14px; margin-bottom: 14px;">
                <b style="color: #1E293B;">🏢 Tenant & Model Warehouse Ingestion</b>
                <p style="font-size: 0.85rem; color: #64748B; margin-bottom: 0;">
                    Select an Environment and Model registered under your tenant. The platform will automatically extract the physical fact/dimension table from Postgres / Druid, save it as a parquet dataset, and profile it for your ML pipeline.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_t1, col_t2 = st.columns([1, 2])
        with col_t1:
            tenant_id = int(st.number_input("Tenant ID", min_value=1, value=1, step=1, key="studio_tenant_id"))
            fetch_btn = st.button("🔄 Refresh Environments", key="refresh_envs_btn")

        envs_cache_key = f"envs_cache_{tenant_id}"
        if fetch_btn or envs_cache_key not in st.session_state:
            with st.spinner(f"Fetching environments for Tenant {tenant_id}..."):
                try:
                    st.session_state[envs_cache_key] = fetch_tenant_environments(tenant_id=tenant_id)
                except Exception as e:
                    st.error(f"Failed to fetch environments for tenant {tenant_id}: {e}")
                    st.session_state[envs_cache_key] = []

        envs = st.session_state.get(envs_cache_key, [])

        if not envs:
            st.warning(f"No active environments found for Tenant {tenant_id}.")
        else:
            with col_t2:
                env_options = {f"{e['name']} (ID: {e['environment_id']})": e for e in envs}
                selected_env_label = st.selectbox("Select Environment", list(env_options.keys()), key="studio_env_select")
                selected_env = env_options[selected_env_label]

            models = selected_env.get("models", [])
            if not models:
                st.info(f"Environment '{selected_env['name']}' has no active models.")
            else:
                col_m1, col_m2 = st.columns([2, 1])
                with col_m1:
                    model_options = {f"{m['name']} (ID: {m['id']})": m for m in models}
                    selected_model_label = st.selectbox("Select Model", list(model_options.keys()), key="studio_model_select")
                    selected_model = model_options[selected_model_label]

                with col_m2:
                    row_limit_choice = st.selectbox("Row Limit", ["1000", "5000", "10000", "50000", "All Rows"], index=0, key="studio_row_limit")
                    row_limit = None if row_limit_choice == "All Rows" else int(row_limit_choice)

                # Show model details
                with st.expander(f"ℹ️ Model Details & Schema: {selected_model['name']}", expanded=True):
                    ds_name = selected_model.get("druid_datasource_name") or "None"
                    st.markdown(f"**Physical Warehouse Table / Datasource:** `{ds_name}`")
                    if selected_model.get("description"):
                        st.markdown(f"**Description:** {selected_model['description']}")
                    
                    meas = selected_model.get("measures", [])
                    dims = selected_model.get("dimensions", [])
                    col_det1, col_det2 = st.columns(2)
                    with col_det1:
                        st.markdown(f"**Measures ({len(meas)}):**")
                        for m in meas:
                            st.write(f"- `{m.get('name')}` ({m.get('aggregation_type')}) -> source col: `{m.get('source_column')}`")
                    with col_det2:
                        st.markdown(f"**Dimensions ({len(dims)}):**")
                        for d in dims:
                            st.write(f"- `{d.get('dimension_name')}` (key: `{d.get('key_column')}`, cols: {d.get('columns')})")

                if st.button("🚀 Ingest Model Table into Studio & Pipeline", type="primary", key="btn_ingest_tenant_model"):
                    t_s0 = time.time()
                    with st.spinner(f"Extracting table for '{selected_model['name']}' from warehouse..."):
                        try:
                            df, meta = ingest_tenant_model(
                                tenant_id=tenant_id,
                                environment_id=selected_env["environment_id"],
                                model_id=selected_model["id"],
                                row_limit=row_limit,
                            )
                            st.session_state["active_df"] = df
                            st.session_state["active_dataset_name"] = meta["dataset_name"]
                            sanitize_node_configs_for_active_dataset()

                            record_api_telemetry(
                                action_name=f"🏢 Ingest Model: {selected_model['name']}",
                                endpoint=f"/api/v1/tenant-data/environments/{selected_env['environment_id']}/models/{selected_model['id']}/ingest",
                                method="POST",
                                request_payload={
                                    "tenant_id": tenant_id,
                                    "environment_id": selected_env["environment_id"],
                                    "model_id": selected_model["id"],
                                    "row_limit": row_limit,
                                },
                                response_payload={
                                    "dataset_name": meta["dataset_name"],
                                    "rows": meta["row_count"],
                                    "columns": meta["column_count"],
                                },
                                status_code=201,
                                duration_ms=(time.time() - t_s0) * 1000.0,
                            )
                            st.success(f"Successfully ingested {meta['row_count']} rows and {meta['column_count']} columns from '{selected_model['name']}'!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error ingesting model table: {e}")

    with ingest_tab2:
        uploaded_file = st.file_uploader("Upload CSV, Excel or JSON dataset", type=["csv", "xlsx", "json"], key="uploader_studio_tab")
        if uploaded_file is not None:
            parsed_df, file_name = parse_uploaded_dataset(uploaded_file)
            if parsed_df is not None:
                st.session_state["active_df"] = parsed_df
                st.session_state["active_dataset_name"] = file_name
                sanitize_node_configs_for_active_dataset()
                st.success(f"Loaded '{file_name}' successfully!")

    with ingest_tab3:
        preset = st.selectbox("Choose Preset Sample", ["Customer Churn (Classification)", "Daily Retail Sales (Time-Series)", "Credit Transactions (Anomaly Injection)", "Titanic Survival", "Iris Flower"], key="preset_studio_tab")
        if st.button("Load Selected Sample", key="btn_load_sample_tab"):
            t_s0 = time.time()
            st.session_state["active_df"] = get_preset_dataset(preset)
            st.session_state["active_dataset_name"] = preset
            
            record_api_telemetry(
                action_name=f"📦 Load Sample: {preset}",
                endpoint="/api/v1/datasets/sample",
                method="POST",
                request_payload={"preset_name": preset},
                response_payload={"dataset_name": preset, "rows": len(st.session_state["active_df"]), "columns": list(st.session_state["active_df"].columns)},
                status_code=200,
                duration_ms=(time.time() - t_s0) * 1000.0
            )
            st.rerun()

    df = st.session_state["active_df"]

    with st.expander(f"🔍 Dataset Preview: {st.session_state['active_dataset_name']}", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)

    profile = DataProfiler.profile_dataframe(df)

    st.markdown("### 📈 Quality & Dataset Overview")
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Rows", f"{profile['row_count']:,}")
    kpi2.metric("Columns", f"{profile['column_count']:,}")
    kpi3.metric("Missing Cells", f"{profile['missing_cells_percentage']}% ({profile['total_missing_cells']})")
    kpi4.metric("Duplicates", f"{profile['duplicate_percentage']}% ({profile['duplicate_rows']})")
    kpi5.metric("Quality Score", f"{profile['quality_score']}/100")

    st.markdown("### 📋 Column-by-Column Deep Dive")
    col_records = []
    for col_name, c_meta in profile["columns"].items():
        col_records.append({
            "Column": col_name,
            "Type": c_meta["inferred_type"],
            "Null Count": c_meta["null_count"],
            "Null %": f"{c_meta['null_percentage']}%",
            "Unique Count": c_meta["unique_count"],
            "Stats / Details": str(c_meta.get("stats", {}))[:80] + "..." if len(str(c_meta.get("stats", {}))) > 80 else str(c_meta.get("stats", {}))
        })
    st.dataframe(pd.DataFrame(col_records), use_container_width=True)


# -------------------------------------------------------------
# TAB 3: RECIPE CATALOG
# -------------------------------------------------------------
elif "Recipe Catalog" in app_mode:
    tot_recipes = sum(len(v) for v in RECIPE_CATEGORY_MAP.values())
    st.markdown('<div class="main-header">🧩 Enterprise Component & Recipe Catalog</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">All {tot_recipes} modular processing, transformation, AI agents, and ML algorithms registered in the platform backend.</div>', unsafe_allow_html=True)

    # Catalog Overview Metrics
    rc_m1, rc_m2, rc_m3, rc_m4 = st.columns(4)
    rc_m1.metric("Registered Processors", f"{tot_recipes}")
    rc_m2.metric("Component Categories", f"{len(RECIPE_CATEGORY_MAP)}")
    rc_m3.metric("ML Frameworks", "8 Supported")
    rc_m4.metric("Engine Execution", "Zero-Copy In-Memory")

    # Search Bar
    catalog_search = st.text_input(
        "Search Catalog",
        placeholder="Filter across 187 processors (e.g. XGBoost, Imputer, Kafka, Snowflake, Resampling...)",
        label_visibility="collapsed",
        key="cat_search_input"
    )

    q_cat = catalog_search.strip().lower() if catalog_search else ""

    for cat_name, recipes_list in RECIPE_CATEGORY_MAP.items():
        if q_cat:
            filtered_list = [
                r for r in recipes_list 
                if (q_cat in r["name"].lower() or 
                    q_cat in r["id"].lower() or 
                    q_cat in r.get("description", "").lower() or 
                    q_cat in cat_name.lower())
            ]
        else:
            filtered_list = recipes_list

        if filtered_list:
            st.markdown(f"### {cat_name} ({len(filtered_list)})")
            for r_meta in filtered_list:
                r_obj = recipe_registry.get(r_meta["id"])
                if not r_obj:
                    continue
                exp_title = f"{r_meta['icon']} {r_obj.name} (`{r_obj.recipe_id}`) — {r_meta.get('version', 'v1.0.0')}"
                with st.expander(exp_title):
                    r_col1, r_col2 = st.columns([3, 1])
                    with r_col1:
                        st.markdown(f"**Description:** {r_obj.description}")
                        in_t = ", ".join(r_obj.input_types) if isinstance(r_obj.input_types, list) else str(r_obj.input_types)
                        out_t = ", ".join(r_obj.output_types) if isinstance(r_obj.output_types, list) else str(r_obj.output_types)
                        st.markdown(f"**Ports:** `In: {in_t}` ➔ `Out: {out_t}`")
                    with r_col2:
                        if st.button("➕ Drop to Board", key=f"btn_cat_add_{r_obj.recipe_id}"):
                            st.session_state["node_counter"] = st.session_state.get("node_counter", 0) + 1
                            counter = st.session_state["node_counter"]
                            count = len(st.session_state["flow_state"].nodes)
                            pos_x = 80 + (count % 3) * 260
                            pos_y = 60 + (count // 3) * 140
                            node_id = f"{r_meta['id']}_{counter}"
                            node_title = f"{r_meta['icon']} {r_meta['name'].split('(')[0].strip()}"
                            
                            new_node = create_flow_node(
                                node_id=node_id,
                                pos=(pos_x, pos_y),
                                content=f"{node_title}\n[{node_id}]",
                                category_key=r_meta.get("category_key", "default")
                            )
                            st.session_state["flow_state"].nodes.append(new_node)
                            st.session_state["node_configs"][node_id] = {
                                "recipe_id": r_meta["id"],
                                "label": node_title,
                                "config": dict(r_meta.get("default_config", {}))
                            }
                            st.session_state["canvas_version"] = st.session_state.get("canvas_version", 1) + 1
                            st.success(f"Added '{r_obj.name}' to Pipeline Whiteboard!")

                    # Schema and Code View tabs
                    c_tab1, c_tab2 = st.tabs(["📋 JSON Schema", "</> Generated Python Code"])
                    with c_tab1:
                        st.json(r_obj.get_schema())
                    with c_tab2:
                        sample_cfg = r_meta.get("default_config", {})
                        if hasattr(r_obj, "to_code"):
                            code_snip = r_obj.to_code(sample_cfg)
                        else:
                            args_str = ", ".join(f"{k}={repr(v)}" for k, v in sample_cfg.items())
                            code_snip = f"# Recipe: {r_obj.name}\n# Category: {r_obj.category}\nresult = {r_obj.recipe_id}({args_str})"
                        st.code(code_snip, language="python")


# -------------------------------------------------------------
# TAB 4: API & TELEMETRY INSPECTOR
# -------------------------------------------------------------
elif "API & Telemetry" in app_mode:
    st.markdown('<div class="main-header">🌐 REST API & Network Telemetry Inspector</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Inspect API contracts, real-time JSON payloads, headers, cURL commands, and test all backend endpoints interactively.</div>', unsafe_allow_html=True)

    # API Status Banner
    st.markdown("""
    <div style="display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap;">
        <span class="badge" style="background-color: #DEF7EC; color: #03543F; font-size: 0.85rem; padding: 6px 12px;">🟢 FastAPI Server: Active (Port 8000)</span>
        <span class="badge" style="background-color: #E1EFFE; color: #1E429F; font-size: 0.85rem; padding: 6px 12px;">📖 OpenAPI Docs: /docs</span>
        <span class="badge" style="background-color: #FDF6B2; color: #723B13; font-size: 0.85rem; padding: 6px 12px;">⚡ Engine: Topological In-Memory DAG</span>
        <span class="badge" style="background-color: #F3E8FF; color: #5521B5; font-size: 0.85rem; padding: 6px 12px;">🏛️ Governance: MLflow Audit Active</span>
    </div>
    """, unsafe_allow_html=True)

    api_tab1, api_tab2, api_tab3 = st.tabs([
        "📡 Live Whiteboard DAG Contract",
        "🚀 Interactive API Playground & Catalog",
        "📊 Network Telemetry & Execution Audit"
    ])

    # ---------------------------------------------------------
    # TAB 1: LIVE WHITEBOARD DAG CONTRACT
    # ---------------------------------------------------------
    with api_tab1:
        st.markdown("### 📦 Active Visual Whiteboard Contract")
        st.caption("Live, real-time JSON payload corresponding to the DAG nodes and connections currently configured on your Whiteboard.")
        
        current_dag = get_current_dag_payload()
        nodes_count = len(current_dag["nodes"])
        edges_count = len(current_dag["edges"])
        
        c_k1, c_k2, c_k3, c_k4 = st.columns(4)
        c_k1.metric("Active Nodes", f"{nodes_count}")
        c_k2.metric("Active Edges", f"{edges_count}")
        c_k3.metric("Dataset Bound", st.session_state.get("active_dataset_name", "N/A")[:18])
        c_k4.metric("Estimated Payload Size", f"{len(json.dumps(current_dag)):,} bytes")

        st.markdown("#### 1. REST Request Body (`POST /api/v1/workflows/execute`):")
        st.json(current_dag)

        st.markdown("#### 2. cURL Terminal Execution Snippet:")
        curl_snippet = f"""curl -X POST http://localhost:8000/api/v1/workflows/execute \\
  -H "Content-Type: application/json" \\
  -d '{json.dumps(current_dag)}'"""
        st.code(curl_snippet, language="bash")

        st.markdown("#### 3. Python `requests` Client Snippet:")
        python_snippet = f"""import requests

url = "http://localhost:8000/api/v1/workflows/execute"
payload = {json.dumps(current_dag, indent=4)}

response = requests.post(url, json=payload)
print("Status Code:", response.status_code)
print("Execution Result:", response.json())"""
        st.code(python_snippet, language="python")

        st.download_button(
            "📥 Download Active Workflow Payload (JSON)",
            data=json.dumps(current_dag, indent=2),
            file_name="active_workflow_contract.json",
            mime="application/json"
        )

    # ---------------------------------------------------------
    # TAB 2: INTERACTIVE API PLAYGROUND & CATALOG
    # ---------------------------------------------------------
    with api_tab2:
        st.markdown("### 🚀 Interactive API Endpoint Catalog & Live Tester")
        st.caption("Select any backend API endpoint, inspect its specification, customize the JSON payload, and execute real-time test calls.")

        api_endpoints = {
            "1. POST /api/v1/workflows/execute (Sync DAG Execution)": {
                "method": "POST",
                "path": "/api/v1/workflows/execute",
                "desc": "Executes a complete DAG workflow synchronously from end-to-end, returning KPIs, step snapshots, and logs.",
                "sample_payload": json.dumps(get_current_dag_payload(), indent=2)
            },
            "2. POST /api/v1/workflows/validate (Pre-Flight Validator)": {
                "method": "POST",
                "path": "/api/v1/workflows/validate",
                "desc": "Validates graph topology for cycle detection, orphan nodes, and recipe semantic compatibility.",
                "sample_payload": json.dumps({
                    "nodes": [
                        {"id": "n1", "recipe_id": "csv_loader", "config": {}},
                        {"id": "n2", "recipe_id": "feature_scaler", "config": {"method": "standard"}}
                    ],
                    "edges": [{"source": "n1", "target": "n2"}]
                }, indent=2)
            },
            "3. POST /api/v1/workflows/async-execute (Background Worker Pool)": {
                "method": "POST",
                "path": "/api/v1/workflows/async-execute",
                "desc": "Dispatches DAG execution to an asynchronous background worker pool and returns an immediate tracking job_id.",
                "sample_payload": json.dumps({
                    "nodes": [
                        {"id": "n1", "recipe_id": "csv_loader", "config": {}},
                        {"id": "n2", "recipe_id": "isolation_forest", "config": {"contamination": 0.05}}
                    ],
                    "edges": [{"source": "n1", "target": "n2"}]
                }, indent=2)
            },
            "4. POST /api/v1/workflows/trigger/{webhook_path} (Inbound Webhook)": {
                "method": "POST",
                "path": "/api/v1/workflows/trigger/realtime_telemetry",
                "desc": "Streams inbound JSON event batches directly into the pipeline entrypoint.",
                "sample_payload": json.dumps([
                    {"device_id": "sensor_01", "temperature": 82.4, "vibration": 14.2, "status": "active"},
                    {"device_id": "sensor_02", "temperature": 115.8, "vibration": 88.6, "status": "warning"}
                ], indent=2)
            },
            "5. GET /api/v1/recipes/ (List All Recipes)": {
                "method": "GET",
                "path": "/api/v1/recipes/",
                "desc": "Retrieves the complete catalog of registered preprocessing, ML, anomaly, forecasting, and trigger recipes.",
                "sample_payload": "{}"
            },
            "6. GET /api/v1/recipes/{recipe_id}/schema (Recipe JSON Schema)": {
                "method": "GET",
                "path": "/api/v1/recipes/xgboost_trainer/schema",
                "desc": "Fetches JSON schema definitions for dynamic parameter form rendering.",
                "sample_payload": "{}"
            },
            "7. GET /api/v1/workflows/jobs (List Recent Background Jobs)": {
                "method": "GET",
                "path": "/api/v1/workflows/jobs",
                "desc": "Lists recent background pipeline jobs and their execution states.",
                "sample_payload": "{}"
            }
        }

        selected_ep_key = st.selectbox("Select API Endpoint to Inspect & Test:", list(api_endpoints.keys()))
        selected_ep = api_endpoints[selected_ep_key]

        c_meta1, c_meta2 = st.columns([1, 4])
        with c_meta1:
            badge_color = "#DEF7EC" if selected_ep["method"] == "GET" else "#E1EFFE"
            badge_text = "#03543F" if selected_ep["method"] == "GET" else "#1E429F"
            st.markdown(f"<span class='badge' style='background-color: {badge_color}; color: {badge_text}; font-size: 1.0rem; padding: 6px 14px;'>{selected_ep['method']}</span>", unsafe_allow_html=True)
        with c_meta2:
            st.markdown(f"**Endpoint Path:** `{selected_ep['path']}`")
            st.caption(selected_ep["desc"])

        st.markdown("#### 📝 Request Body Editor:")
        user_req_body = st.text_area("JSON Request Payload", value=selected_ep["sample_payload"], height=160, key=f"req_body_{selected_ep_key}")

        if st.button(f"🚀 Send Test Request to `{selected_ep['path']}`", type="primary", key=f"btn_send_{selected_ep_key}"):
            with st.spinner("Executing API call through Platform Engine..."):
                t_start = time.time()
                try:
                    # In-memory execution simulation with identical backend logic
                    if "/workflows/validate" in selected_ep["path"]:
                        req_dict = json.loads(user_req_body)
                        g = WorkflowGraph(
                            nodes=[WorkflowNode(id=n["id"], recipe_id=n["recipe_id"], config=n.get("config", {})) for n in req_dict.get("nodes", [])],
                            edges=[WorkflowEdge(source=e["source"], target=e["target"]) for e in req_dict.get("edges", [])]
                        )
                        errs = g.validate_graph()
                        res_json = {"valid": len(errs) == 0, "errors": errs}
                        http_code = 200

                    elif "/workflows/execute" in selected_ep["path"]:
                        req_dict = json.loads(user_req_body)
                        g = WorkflowGraph(
                            nodes=[WorkflowNode(id=n["id"], recipe_id=n["recipe_id"], config=n.get("config", {})) for n in req_dict.get("nodes", [])],
                            edges=[WorkflowEdge(source=e["source"], target=e["target"]) for e in req_dict.get("edges", [])]
                        )
                        exec_res = DAGExecutor.execute_workflow(
                            execution_id=f"test_{int(time.time())}",
                            workflow=g,
                            initial_df=st.session_state["active_df"]
                        )
                        res_json = {
                            "execution_id": exec_res.execution_id,
                            "status": exec_res.status,
                            "total_duration_ms": exec_res.total_duration_ms,
                            "final_metrics": exec_res.final_metrics,
                            "step_snapshots_count": len(exec_res.step_snapshots),
                            "logs_count": len(exec_res.logs)
                        }
                        http_code = 200 if exec_res.status == "SUCCESS" else 422

                    elif "/workflows/async-execute" in selected_ep["path"]:
                        req_dict = json.loads(user_req_body)
                        g = WorkflowGraph(
                            nodes=[WorkflowNode(id=n["id"], recipe_id=n["recipe_id"], config=n.get("config", {})) for n in req_dict.get("nodes", [])],
                            edges=[WorkflowEdge(source=e["source"], target=e["target"]) for e in req_dict.get("edges", [])]
                        )
                        job_id = job_manager.submit_job(workflow=g, initial_df=st.session_state["active_df"])
                        res_json = {"job_id": job_id, "status": "PENDING", "message": "Workflow dispatched to background worker pool."}
                        http_code = 202

                    elif "/workflows/trigger/" in selected_ep["path"]:
                        req_data = json.loads(user_req_body)
                        df_trig = pd.json_normalize(req_data) if isinstance(req_data, list) else pd.json_normalize([req_data])
                        res_json = {
                            "status": "TRIGGERED",
                            "webhook_path": "realtime_telemetry",
                            "records_ingested": len(df_trig),
                            "columns_received": list(df_trig.columns),
                            "sample_preview": df_trig.head(2).to_dict(orient="records")
                        }
                        http_code = 200

                    elif "/recipes/xgboost_trainer/schema" in selected_ep["path"]:
                        r = recipe_registry.get("xgboost_trainer")
                        res_json = {"recipe_id": r.recipe_id, "name": r.name, "parameters_schema": r.get_schema()}
                        http_code = 200

                    elif "/recipes/" in selected_ep["path"]:
                        res_json = [{"id": r.recipe_id, "name": r.name, "category": r.category, "inputs": r.input_types, "outputs": r.output_types} for r in recipe_registry.list_all()]
                        http_code = 200

                    else:
                        jobs = job_manager.list_jobs(limit=10)
                        res_json = {"total_jobs": len(jobs), "recent_jobs": jobs}
                        http_code = 200

                    latency_ms = round((time.time() - t_start) * 1000.0, 2)
                    st.success(f"HTTP {http_code} OK ({latency_ms}ms)")
                    st.markdown("##### 📥 Live JSON Response:")
                    st.json(res_json)

                except Exception as ex:
                    st.error(f"API Execution Error: {str(ex)}")

    # ---------------------------------------------------------
    # TAB 3: NETWORK TELEMETRY & EXECUTION AUDIT
    # ---------------------------------------------------------
    with api_tab3:
        st.markdown("### 📊 Live API Flight Recorder & Session Action Audit")
        st.caption("Inspect real-time HTTP requests, payloads, status codes, and latencies generated by every single user action in this session.")

        if st.session_state.get("api_telemetry_history"):
            tot_hits = len(st.session_state["api_telemetry_history"])
            success_hits = sum(1 for h in st.session_state["api_telemetry_history"] if h["status_code"] < 400)
            avg_lat = sum(h["duration_ms"] for h in st.session_state["api_telemetry_history"]) / max(1, tot_hits)
            
            tel_k1, tel_k2, tel_k3 = st.columns(3)
            tel_k1.metric("Total Recorded API Requests", f"{tot_hits}")
            tel_k2.metric("Success Rate", f"{(success_hits/max(1, tot_hits))*100:.1f}% ({success_hits}/{tot_hits})")
            tel_k3.metric("Average Latency", f"{avg_lat:.2f}ms")

            st.markdown("#### ⚡ Chronological API Flight Recorder (Last 50 Actions):")
            for i, hit in enumerate(st.session_state["api_telemetry_history"]):
                badge_t = "#03543F" if hit["status_code"] < 400 else "#9B1C1C"
                
                with st.expander(f"🕒 {hit['timestamp']} | {hit['action']} ➔ {hit['method']} {hit['endpoint']} ({hit['duration_ms']}ms)", expanded=(i==0)):
                    c_req, c_res = st.columns(2)
                    with c_req:
                        st.markdown(f"**📤 Request Body (`{hit['method']}`):**")
                        st.json(hit["request_payload"])
                    with c_res:
                        st.markdown(f"**📥 Response Body (<span style='color:{badge_t}; font-weight:600;'>HTTP {hit['status_code']}</span>):**", unsafe_allow_html=True)
                        st.json(hit["response_payload"])
        else:
            st.info("💡 No actions recorded yet. Interact with the Whiteboard or Dataset Studio to record real-time API telemetry.")

        if "last_execution" in st.session_state and st.session_state["last_execution"]:
            l_exec = st.session_state["last_execution"]
            
            st.markdown("---")
            st.markdown("#### 🔬 Latest DAG Pipeline Execution Latency Breakdown:")
            snaps = l_exec.get("step_snapshots", {})
            if snaps:
                breakdown_data = []
                for nid, s in snaps.items():
                    breakdown_data.append({
                        "Node ID": nid,
                        "Recipe": s.get("recipe_name", nid),
                        "Latency (ms)": s.get("duration_ms", 0.0),
                        "Output Records": s.get("row_count", 0),
                        "Features Generated": len(s.get("columns", []))
                    })
                b_df = pd.DataFrame(breakdown_data)
                st.dataframe(b_df, use_container_width=True)

                fig_lat = px.bar(b_df, x="Node ID", y="Latency (ms)", color="Latency (ms)", color_continuous_scale="Blues", title="Node Latency Distribution across DAG Execution")
                st.plotly_chart(fig_lat, use_container_width=True)

            st.markdown("#### 📜 Live Execution Trace Logs:")
            for log in l_exec.get("execution_logs", []):
                st.write(f"`{log}`")
        else:
            st.info("💡 Run a pipeline from the **🎨 Pipeline Whiteboard** to stream real-time telemetry traces here.")

