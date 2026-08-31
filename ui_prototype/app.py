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

import backend.app.recommendation.recommender as recommender_mod
importlib.reload(recommender_mod)
from backend.app.recommendation.recommender import AIRecommender

import backend.app.engine.execution.executor as executor_mod
importlib.reload(executor_mod)
from backend.app.engine.execution.executor import DAGExecutor

from backend.app.profiling.profiler import DataProfiler
from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes import register_all_recipes
register_all_recipes()

from backend.app.engine.dag.graph import WorkflowGraph, WorkflowNode, WorkflowEdge

st.set_page_config(
    page_title="Visual AI/ML Pipeline Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #475569;
        margin-bottom: 1.0rem;
    }
    .badge {
        display: inline-block;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 4px;
        background-color: #E2E8F0;
        color: #1E293B;
    }
    .ai-card {
        padding: 14px;
        border-radius: 8px;
        background: #F8FAFC;
        border-left: 4px solid #3B82F6;
        margin-bottom: 10px;
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
# RECIPE CATEGORY HIERARCHY
# -------------------------------------------------------------
RECIPE_CATEGORY_MAP = {
    "⚡ Triggers & Inbound Events": [
        {"id": "webhook_trigger", "name": "Webhook Inbound Trigger (HTTP POST)", "icon": "⚡", "default_config": {"webhook_path": "ml_inbound_stream", "auth_header_required": "None (Public)", "payload_format": "JSON Array of Objects"}},
        {"id": "cron_trigger", "name": "Cron Schedule Trigger (Recurring)", "icon": "⏰", "default_config": {"cron_expression": "0 0 * * *", "interval_preset": "Daily at Midnight", "timezone": "UTC"}}
    ],
    "📥 Ingestion & Sources": [
        {"id": "csv_loader", "name": "Dataset Ingestion (CSV/Excel/Upload)", "icon": "📄", "default_config": {}}
    ],
    "🧹 Data Cleaning & Preprocessing": [
        {"id": "missing_value_imputer", "name": "Missing Value Imputer (Mean/Median/Mode)", "icon": "🧹", "default_config": {"strategy": "median"}},
        {"id": "feature_scaler", "name": "Feature Scaler (StandardScaler/MinMax)", "icon": "⚖️", "default_config": {"method": "standard"}},
        {"id": "categorical_encoder", "name": "Categorical Encoder (One-Hot/Label)", "icon": "🔤", "default_config": {"method": "one_hot"}}
    ],
    "✂️ Splitting": [
        {"id": "train_test_split", "name": "Train / Test Splitter", "icon": "✂️", "default_config": {"target_column": "Churn", "test_size": 0.2}}
    ],
    "🤖 Machine Learning Models": [
        {"id": "xgboost_trainer", "name": "XGBoost Classifier / Regressor", "icon": "⚡", "default_config": {"task_type": "classification", "n_estimators": 100, "max_depth": 6}},
        {"id": "lightgbm_trainer", "name": "LightGBM Classifier / Regressor", "icon": "🚀", "default_config": {"task_type": "classification", "n_estimators": 100, "num_leaves": 31}},
        {"id": "catboost_trainer", "name": "CatBoost Classifier / Regressor", "icon": "🐱", "default_config": {"task_type": "classification", "iterations": 100, "depth": 6}},
        {"id": "random_forest_trainer", "name": "Random Forest Classifier", "icon": "🌲", "default_config": {"task_type": "classification", "n_estimators": 50, "max_depth": 10}},
        {"id": "linear_trainer", "name": "Logistic / Ridge Linear Model", "icon": "📈", "default_config": {"task_type": "classification", "max_iter": 200}}
    ],
    "🚨 Anomaly Detection (Unsupervised)": [
        {"id": "isolation_forest", "name": "Isolation Forest Anomaly Detector", "icon": "🌲", "default_config": {"contamination": 0.05, "n_estimators": 100}},
        {"id": "statistical_guardrail", "name": "Statistical Outlier Guardrail (Z-Score / IQR)", "icon": "🛡️", "default_config": {"method": "z_score", "threshold": 3.0, "action": "flag"}}
    ],
    "📈 Time-Series Forecasting": [
        {"id": "prophet_forecaster", "name": "Prophet Time-Series Forecaster", "icon": "🔮", "default_config": {"horizon_periods": 14, "seasonality_mode": "additive"}},
        {"id": "arima_forecaster", "name": "ARIMA Statistical Forecaster", "icon": "📊", "default_config": {"p": 1, "d": 1, "q": 1, "horizon_periods": 14}},
        {"id": "lag_feature_engineering", "name": "Lag & Time Feature Engineer", "icon": "⏱️", "default_config": {"lag_periods": "1, 2, 3, 7, 14", "rolling_windows": "7"}}
    ],
    "🏛️ Governance & Model Registry": [
        {"id": "mlflow_tracker", "name": "MLflow Governance & Registry", "icon": "🏛️", "default_config": {"experiment_name": "Enterprise_ML_Pipelines", "registered_model_name": "Production_Model", "stage": "Production"}}
    ],
    "🎯 Model Evaluation & Reports": [
        {"id": "model_evaluator", "name": "Model Evaluator (Accuracy, F1, Confusion Matrix)", "icon": "🎯", "default_config": {"report_type": "Comprehensive (All Metrics + Confusion Matrix)"}}
    ]
}

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


# -------------------------------------------------------------
# SIDEBAR NAVIGATION & PERSISTENT API INSPECTOR DOCK
# -------------------------------------------------------------
st.sidebar.title("⚡ AI/ML Pipeline Studio")
app_mode = st.sidebar.radio(
    "Navigation",
    ["🎨 Pipeline Whiteboard", "📊 Dataset Studio & Profiler", "🧩 Recipe Catalog", "🌐 API & Telemetry Inspector"]
)

# Track Tab Navigation GET APIs
if st.session_state.get("_prev_app_mode") != app_mode:
    st.session_state["_prev_app_mode"] = app_mode
    if app_mode == "🎨 Pipeline Whiteboard":
        record_api_telemetry("View Pipeline Whiteboard Canvas", "/api/v1/workflows/canvas", "GET", {"view": "whiteboard_canvas"}, {"nodes_count": len(st.session_state.get("flow_state", StreamlitFlowState(nodes=[], edges=[])).nodes)}, 200, 0.8)
    elif app_mode == "📊 Dataset Studio & Profiler":
        record_api_telemetry("View Dataset Profile & Preview", f"/api/v1/datasets/{st.session_state.get('active_dataset_name', 'churn')}/profile", "GET", {"dataset_name": st.session_state.get("active_dataset_name")}, {"rows": len(st.session_state.get("active_df", [])), "columns": list(st.session_state.get("active_df", pd.DataFrame()).columns)}, 200, 1.2)
    elif app_mode == "🧩 Recipe Catalog":
        record_api_telemetry("Fetch All Recipe Components", "/api/v1/recipes/", "GET", {}, {"total_recipes": len(recipe_registry.list_all())}, 200, 0.9)
    elif app_mode == "🌐 API & Telemetry Inspector":
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
def sanitize_node_configs_for_active_dataset():
    """Sanitizes node parameters when active dataset columns change."""
    df = st.session_state.get("active_df")
    if df is None:
        return
    cols = list(df.columns)
    for n_id, cfg_data in st.session_state.get("node_configs", {}).items():
        cfg = cfg_data.get("config", {})
        if "target_column" in cfg and cfg["target_column"] not in cols:
            cfg["target_column"] = cols[-1]
            st.info(f"ℹ️ Auto-aligned target column for `{n_id}` to `{cols[-1]}`.")
        if "date_column" in cfg and cfg["date_column"] not in cols:
            date_candidates = [c for c in cols if "date" in c.lower() or "time" in c.lower()]
            cfg["date_column"] = date_candidates[0] if date_candidates else cols[0]
        if "columns" in cfg:
            if isinstance(cfg["columns"], str):
                cfg["columns"] = [c.strip() for c in cfg["columns"].split(",") if c.strip() in cols]
            elif isinstance(cfg["columns"], list):
                cfg["columns"] = [c for c in cfg["columns"] if c in cols]


def execute_pipeline():
    canvas_nodes = list(st.session_state["flow_state"].nodes)
    canvas_edges = list(st.session_state["flow_state"].edges)

    if not canvas_nodes:
        st.warning("⚠️ The canvas is empty. Please add nodes or load a template first.")
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

    workflow_graph = WorkflowGraph(nodes=backend_nodes, edges=backend_edges)
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
        "status": exec_result.status,
        "final_metrics": exec_result.final_metrics,
        "anomaly_summary": exec_result.anomaly_summary,
        "forecasting_summary": exec_result.forecasting_summary,
        "governance_summary": exec_result.governance_summary,
        "execution_logs": exec_result.logs,
        "node_outputs": exec_result.node_outputs,
        "step_snapshots": exec_result.step_snapshots
    }

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


def create_flow_node(node_id: str, pos: tuple, content: str) -> StreamlitFlowNode:
    """Helper to create fully interactive, connectable, and draggable React Flow nodes."""
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
        deletable=True
    )


def build_recommended_pipeline(rec_dict: dict = None, target_col: str = None, task_type: str = None):
    """Instantiates a full end-to-end DAG based on dataset analysis."""
    df = st.session_state["active_df"]
    rec = rec_dict or AIRecommender.recommend_pipeline(df, target_column=target_col, task_type=task_type)
    
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
    nodes_payload = []
    for n in st.session_state["flow_state"].nodes:
        pos = get_node_position(n)
        content = n.data.get("content", n.id) if hasattr(n, "data") and isinstance(n.data, dict) else n.id
        nodes_payload.append({
            "id": n.id,
            "position": pos,
            "content": content
        })

    edges_payload = [
        {"id": e.id, "source": e.source, "target": e.target}
        for e in st.session_state["flow_state"].edges
    ]

    body = {
        "name": name,
        "description": description,
        "nodes": nodes_payload,
        "edges": edges_payload,
        "node_configs": st.session_state.get("node_configs", {})
    }

    try:
        import httpx
        res = httpx.post("http://localhost:8000/api/v1/workflows/", json=body, timeout=4.0)
        if res.status_code in [200, 201]:
            saved_json = res.json()
            record_api_telemetry("💾 Save Workflow API", "/api/v1/workflows/", "POST", body, saved_json, res.status_code, 2.1)
            return saved_json
    except Exception:
        pass

    # Direct Async DB Fallback
    import uuid
    import asyncio
    from backend.app.infrastructure.database.session import AsyncSessionLocal, init_db
    from backend.app.workflows.models import Workflow

    wf_id = str(uuid.uuid4())
    async def _async_save():
        await init_db()
        async with AsyncSessionLocal() as session:
            wf = Workflow(
                id=wf_id,
                name=name,
                description=description,
                nodes=nodes_payload,
                edges=edges_payload,
                node_configs=st.session_state.get("node_configs", {})
            )
            session.add(wf)
            await session.commit()

    try:
        asyncio.run(_async_save())
        saved_dict = {"id": wf_id, "name": name, "status": "SAVED"}
        record_api_telemetry("💾 Save Workflow DB", "/api/v1/workflows/", "POST", body, saved_dict, 201, 1.8)
        return saved_dict
    except Exception as ex:
        st.error(f"Error saving workflow: {str(ex)}")
        return None


def fetch_saved_workflows_from_backend(include_deleted: bool = False) -> list:
    """Fetches list of all saved pipeline workbooks from backend REST API/DB."""
    try:
        import httpx
        url = f"http://localhost:8000/api/v1/workflows/?include_deleted={'true' if include_deleted else 'false'}"
        res = httpx.get(url, timeout=4.0)
        if res.status_code == 200:
            return res.json()
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
        import httpx
        res = httpx.delete(f"http://localhost:8000/api/v1/workflows/{workflow_id}", timeout=4.0)
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
        import httpx
        res = httpx.post(f"http://localhost:8000/api/v1/workflows/{workflow_id}/restore", timeout=4.0)
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

    try:
        return asyncio.run(_async_restore())
    except Exception:
        return False


def restore_saved_workflow(wf_data: dict):
    """Restores a saved pipeline workbook into active StreamlitFlowState and node_configs."""
    t_nodes = []
    t_edges = []
    
    # Restore Nodes
    for nd in wf_data.get("nodes", []):
        nid = nd["id"]
        pos = nd.get("position", (100, 100))
        content = nd.get("content", nid)
        t_nodes.append(create_flow_node(nid, pos, content))

    # Restore Edges
    for ed in wf_data.get("edges", []):
        eid = ed.get("id", f"e_{ed['source']}_{ed['target']}")
        t_edges.append(StreamlitFlowEdge(id=eid, source=ed["source"], target=ed["target"], animated=True))

    st.session_state["flow_state"] = StreamlitFlowState(nodes=t_nodes, edges=t_edges)
    st.session_state["node_configs"] = wf_data.get("node_configs", {})
    st.session_state["canvas_version"] = st.session_state.get("canvas_version", 1) + 1
    st.session_state["active_saved_workflow_name"] = wf_data.get("name", "Saved Workflow")

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
if app_mode == "🎨 Pipeline Whiteboard":
    st.markdown('<div class="main-header">🎨 Visual Pipeline Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Construct, wire, and execute your machine learning, anomaly detection & forecasting pipelines.</div>', unsafe_allow_html=True)

    # Top Execution & Action Bar
    bar_col1, bar_col2, bar_col3, bar_col4, bar_col5, bar_col6, bar_col7 = st.columns([2.2, 2.0, 1.8, 1.8, 1.8, 1.4, 1.4])
    with bar_col1:
        execute_clicked_top = st.button("▶️ RUN PIPELINE", type="primary", use_container_width=True)

    with bar_col2:
        if st.button("🧠 AI Recommend", type="secondary", use_container_width=True):
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
            if len(curr_nodes) < 2:
                st.warning("⚠️ Auto-Wire requires at least 2 nodes on the canvas. Add components first.")
            else:
                def get_recipe_type(nid: str) -> str:
                    cfg = st.session_state.get("node_configs", {}).get(nid, {})
                    return cfg.get("recipe_id", "")

                def get_category_order(nid: str) -> int:
                    r_id = get_recipe_type(nid)
                    if r_id in ["cron_trigger", "webhook_trigger"]:
                        return 0
                    if r_id in ["csv_loader"]:
                        return 1
                    if r_id in ["missing_value_imputer", "feature_scaler", "categorical_encoder", "statistical_guardrail", "lag_feature_engineering"]:
                        return 2
                    if r_id in ["train_test_split"]:
                        return 3
                    if r_id in ["xgboost_trainer", "lightgbm_trainer", "catboost_trainer", "random_forest_trainer", "linear_trainer", "isolation_forest", "prophet_forecaster", "arima_forecaster"]:
                        return 4
                    if r_id in ["model_evaluator", "mlflow_tracker"]:
                        return 5
                    return 2

                # Categorize nodes
                triggers = [n for n in curr_nodes if get_category_order(n.id) == 0]
                ingestions = [n for n in curr_nodes if get_category_order(n.id) == 1]
                preprocessings = [n for n in curr_nodes if get_category_order(n.id) == 2]
                splitters = [n for n in curr_nodes if get_category_order(n.id) == 3]
                models = [n for n in curr_nodes if get_category_order(n.id) == 4]
                evaluators = [n for n in curr_nodes if get_category_order(n.id) == 5]

                # Sort each group by X position
                triggers.sort(key=get_node_position)
                ingestions.sort(key=get_node_position)
                preprocessings.sort(key=get_node_position)
                splitters.sort(key=get_node_position)
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
                data_candidates = ingestions + preprocessings + splitters + models
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
                    
                    # Splitter connects to all models and evaluators
                    for m in models:
                        add_edge(main_splitter, m.id)
                    for ev in evaluators:
                        add_edge(main_splitter, ev.id)
                    # Models connect to evaluators
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
        rec = AIRecommender.recommend_pipeline(active_df, target_column=custom_target, task_type=task_override)

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
    # STEP 1: COMPONENT PALETTE (Top Toolbar)
    # ---------------------------------------------------------
    with st.expander("📦 Step 1: Add New Component to Whiteboard", expanded=True):
        st.caption("🌐 **Active API:** `GET /api/v1/recipes?category=...` ➔ `GET /api/v1/recipes/{recipe_id}/schema`")
        col_cat, col_recipe, col_add = st.columns([2, 3, 2])
        
        with col_cat:
            selected_cat = st.selectbox("Category", list(RECIPE_CATEGORY_MAP.keys()), index=0, key="step1_cat_select")
            cat_slug = selected_cat.split(" ")[-1].lower()
            if st.session_state.get("_prev_cat_select") != selected_cat:
                st.session_state["_prev_cat_select"] = selected_cat
                record_api_telemetry(
                    action_name=f"Browse Category: {selected_cat}",
                    endpoint=f"/api/v1/recipes?category={cat_slug}",
                    method="GET",
                    request_payload={"query_param": {"category": cat_slug}},
                    response_payload={"recipes_count": len(RECIPE_CATEGORY_MAP[selected_cat]), "recipes": [r['id'] for r in RECIPE_CATEGORY_MAP[selected_cat]]},
                    status_code=200,
                    duration_ms=1.1
                )

        with col_recipe:
            available = RECIPE_CATEGORY_MAP[selected_cat]
            recipe_map = {r["name"]: r for r in available}
            chosen_name = st.selectbox("Subcategory / Recipe", list(recipe_map.keys()), key="step1_recipe_select")
            chosen_meta = recipe_map[chosen_name]
            
            if st.session_state.get("_prev_recipe_select") != chosen_meta["id"]:
                st.session_state["_prev_recipe_select"] = chosen_meta["id"]
                r_obj_temp = recipe_registry.get(chosen_meta["id"])
                record_api_telemetry(
                    action_name=f"Fetch Recipe Schema: {chosen_meta['name']}",
                    endpoint=f"/api/v1/recipes/{chosen_meta['id']}/schema",
                    method="GET",
                    request_payload={"path_param": {"recipe_id": chosen_meta["id"]}},
                    response_payload={"recipe_id": chosen_meta["id"], "name": chosen_meta["name"], "parameters_schema": r_obj_temp.get_schema() if r_obj_temp else {}},
                    status_code=200,
                    duration_ms=0.9
                )

        with col_add:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("➕ Drop Node Onto Board", type="primary", use_container_width=True):
                # Unique collision-free node counter
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
                    content=f"{node_title}\n[{node_id}]"
                )
                
                st.session_state["flow_state"].nodes.append(new_node)
                
                init_cfg = dict(chosen_meta["default_config"])
                if "target_column" in init_cfg:
                    cols = list(st.session_state["active_df"].columns)
                    init_cfg["target_column"] = cols[-1]

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
    canvas_col, right_col = st.columns([3, 1])

    with canvas_col:
        st.markdown("#### 🖱️ Step 2: Interactive Whiteboard")
        
        if not st.session_state["flow_state"].nodes:
            st.info("💡 Whiteboard is empty. Click **'🧠 AI Recommend'**, **'⚡ ML Template'**, **'🔮 Forecast'**, or drop nodes above to start!")

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
                    data_source_mode = st.radio("Source Mode", ["Use Active Dataset", "Upload New File", "Choose Preset Sample"], index=0, key=f"src_mode_{selected_node_id}")
                    
                    if data_source_mode == "Choose Preset Sample":
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
                    else:
                        st.info(f"Using Active Dataset: **{st.session_state.get('active_dataset_name', 'Customer Churn')}** ({len(st.session_state['active_df'])} rows)")

                schema = recipe_obj.get_schema()
                props = schema.get("properties", {})
                current_config = dict(node_cfg.get("config", {}))

                active_cols = list(st.session_state["active_df"].columns) if "active_df" in st.session_state and st.session_state["active_df"] is not None else []

                for prop_name, prop_meta in props.items():
                    title = prop_meta.get("title", prop_name)
                    default_val = prop_meta.get("default", None)
                    curr_val = current_config.get(prop_name, default_val)
                    prop_type = prop_meta.get("type")

                    # Smart column selectors: array of columns vs single column
                    if prop_type == "array" or prop_name in ["columns", "feature_columns", "categorical_columns", "numerical_columns"]:
                        if isinstance(curr_val, str):
                            curr_list = [c.strip() for c in curr_val.split(",") if c.strip() in active_cols] if curr_val else []
                        elif isinstance(curr_val, (list, tuple)):
                            curr_list = [c for c in curr_val if c in active_cols]
                        else:
                            curr_list = []
                        new_val = st.multiselect(
                            title,
                            options=active_cols,
                            default=curr_list,
                            key=f"cfg_{selected_node_id}_{prop_name}",
                            help=prop_meta.get("description", "Select specific columns or leave empty to apply across all columns.")
                        )
                    elif ("column" in prop_name.lower() or prop_name.endswith("_col")) and active_cols:
                        col_idx = active_cols.index(curr_val) if curr_val in active_cols else len(active_cols) - 1
                        new_val = st.selectbox(title, active_cols, index=col_idx, key=f"cfg_{selected_node_id}_{prop_name}")
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
        else:
            st.write("No nodes selected.")

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
        fc_sum = exec_data.get("forecasting_summary")
        fc_df = None
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
        anom_sum = exec_data.get("anomaly_summary")
        anomaly_df = None
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
                    mape_str = f"{mape_v*100:.2f}%" if mape_v is not None else "N/A"
                    m5.metric("MAPE", mape_str)

                    st.info(f"✨ **Regression Model Evaluation:** Model achieved **R² = {r2_v:.4f}** with an average absolute deviation of **MAE = {mae_v:.4f}** on the unseen test set.")

                with rep_tab2:
                    st.markdown("##### Model Predictions Preview on Unseen Test Partition:")
                    if "predictions_sample" in exec_data.get("final_metrics", {}):
                        preds = exec_data["final_metrics"]["predictions_sample"]
                        st.dataframe(pd.DataFrame({"Sample Prediction": preds}), use_container_width=True)
                    else:
                        st.write("Predictions generated successfully.")

                with rep_tab3:
                    for n_id, out in node_outputs.items():
                        if "feature_importances" in out and out["feature_importances"]:
                            fi = out["feature_importances"]
                            fi_df = pd.DataFrame(list(fi.items()), columns=["Feature", "Importance"]).sort_values(by="Importance", ascending=True)
                            fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h", title=f"Feature Importances from Node '{n_id}'")
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
        # 5. LIVE REST API CONTRACT & CURL GENERATOR
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
elif app_mode == "📊 Dataset Studio & Profiler":
    st.markdown('<div class="main-header">📊 Dataset Studio & Automated Profiler</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload your dataset to inspect preview rows and automatic statistical profiles.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader("Upload CSV, Excel or JSON dataset", type=["csv", "xlsx", "json"])
        if uploaded_file is not None:
            parsed_df, file_name = parse_uploaded_dataset(uploaded_file)
            if parsed_df is not None:
                st.session_state["active_df"] = parsed_df
                st.session_state["active_dataset_name"] = file_name
                sanitize_node_configs_for_active_dataset()
                st.success(f"Loaded '{file_name}' successfully!")

    with col2:
        preset = st.selectbox("Choose Preset Sample", ["Customer Churn (Classification)", "Daily Retail Sales (Time-Series)", "Credit Transactions (Anomaly Injection)", "Titanic Survival", "Iris Flower"])
        if st.button("Load Selected Sample"):
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
elif app_mode == "🧩 Recipe Catalog":
    st.markdown('<div class="main-header">🧩 Dynamic Component & Recipe Catalog</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">All modular processing, transformation, and ML algorithms registered in the platform backend.</div>', unsafe_allow_html=True)

    for cat_name, recipes_list in RECIPE_CATEGORY_MAP.items():
        st.markdown(f"### {cat_name}")
        for r_meta in recipes_list:
            r_obj = recipe_registry.get(r_meta["id"])
            with st.expander(f"{r_meta['icon']} {r_obj.name} (`{r_obj.recipe_id}`)"):
                st.markdown(f"**Description:** {r_obj.description}")
                st.markdown(f"**Inputs:** `{r_obj.input_types}` ➔ **Outputs:** `{r_obj.output_types}`")
                st.markdown("#### JSON Schema:")
                st.json(r_obj.get_schema())


# -------------------------------------------------------------
# TAB 4: API & TELEMETRY INSPECTOR
# -------------------------------------------------------------
elif app_mode == "🌐 API & Telemetry Inspector":
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

