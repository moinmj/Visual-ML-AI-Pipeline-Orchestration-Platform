import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import json

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


def parse_uploaded_dataset(up_file):
    """Unified file reader supporting CSV, Excel, and nested JSON with schema validation."""
    if up_file is None:
        return None, None
    try:
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
        return df, up_file.name
    except Exception as e:
        st.error(f"Error parsing uploaded file '{up_file.name}': {str(e)}")
        return None, None

# -------------------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------------------
st.sidebar.title("⚡ AI/ML Pipeline Studio")
app_mode = st.sidebar.radio(
    "Navigation",
    ["🎨 Pipeline Whiteboard", "📊 Dataset Studio & Profiler", "🧩 Recipe Catalog"]
)

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
            label=n.data.get("content", n.id)
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

    exec_result = DAGExecutor.execute_workflow(
        execution_id=f"exec_{int(pd.Timestamp.now().timestamp())}",
        workflow=workflow_graph,
        initial_df=st.session_state["active_df"]
    )

    if exec_result.status == "SUCCESS":
        st.session_state["last_execution"] = {
            "final_metrics": exec_result.final_metrics,
            "anomaly_summary": exec_result.anomaly_summary,
            "forecasting_summary": exec_result.forecasting_summary,
            "governance_summary": exec_result.governance_summary,
            "execution_logs": exec_result.logs,
            "node_outputs": exec_result.node_outputs
        }
        st.success("🎉 Pipeline executed cleanly through DAG Engine!")
    else:
        st.error("### ❌ Pipeline Execution Failed")
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


def build_recommended_pipeline(target_col=None, task_type=None):
    """Builds and wires the AI recommended DAG on the whiteboard."""
    df = st.session_state["active_df"]
    rec = AIRecommender.recommend_pipeline(df, target_column=target_col, task_type=task_type)
    
    t_nodes = [
        create_flow_node("node_csv", (40, 100), f"📄 {st.session_state['active_dataset_name'][:18]}")
    ]
    node_configs = {
        "node_csv": {"recipe_id": "csv_loader", "label": "📄 Dataset Ingestion", "config": {}}
    }

    cur_x = 280
    prev_id = "node_csv"
    t_edges = []

    # 1. Preprocessing Nodes
    for i, step in enumerate(rec.get("preprocessing_recommendations", [])):
        step_id = f"node_prep_{i+1}"
        t_nodes.append(create_flow_node(step_id, (cur_x, 100), step["name"]))
        t_edges.append(StreamlitFlowEdge(id=f"e_{prev_id}_{step_id}", source=prev_id, target=step_id, animated=True))
        node_configs[step_id] = {"recipe_id": step["recipe_id"], "label": step["name"], "config": step["config"]}
        prev_id = step_id
        cur_x += 240

    task = rec.get("task_type")
    target_col = rec.get("target_column")

    # 2. Split or Direct Model
    if task in ["classification", "regression"]:
        split_id = "node_split"
        t_nodes.append(create_flow_node(split_id, (cur_x, 100), "✂️ Train/Test Split"))
        t_edges.append(StreamlitFlowEdge(id=f"e_{prev_id}_{split_id}", source=prev_id, target=split_id, animated=True))
        node_configs[split_id] = {"recipe_id": "train_test_split", "label": "✂️ Split", "config": {"target_column": target_col, "test_size": 0.2}}
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
    execute_pipeline()


def load_ml_template(force_preset: bool = False):
    """Applies ML classification/regression template to current active data or sample preset."""
    if force_preset or "active_df" not in st.session_state or st.session_state["active_df"] is None:
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
        create_flow_node("node_encode", (760, 100), "🔤 One-Hot Encoder"),
        create_flow_node("node_split", (1000, 100), "✂️ Train/Test Split"),
        create_flow_node("node_xgb", (1240, 50), "⚡ XGBoost Model"),
        create_flow_node("node_eval", (1480, 100), "🎯 Evaluation Report"),
    ]
    t_edges = [
        StreamlitFlowEdge(id="e1", source="node_csv", target="node_impute", animated=True),
        StreamlitFlowEdge(id="e2", source="node_impute", target="node_scale", animated=True),
        StreamlitFlowEdge(id="e3", source="node_scale", target="node_encode", animated=True),
        StreamlitFlowEdge(id="e4", source="node_encode", target="node_split", animated=True),
        StreamlitFlowEdge(id="e5", source="node_split", target="node_xgb", animated=True),
        StreamlitFlowEdge(id="e6", source="node_split", target="node_eval", animated=True),
        StreamlitFlowEdge(id="e7", source="node_xgb", target="node_eval", animated=True),
    ]
    st.session_state["flow_state"] = StreamlitFlowState(nodes=t_nodes, edges=t_edges)
    st.session_state["node_configs"] = {
        "node_csv": {"recipe_id": "csv_loader", "label": "📄 Dataset Ingestion", "config": {}},
        "node_impute": {"recipe_id": "missing_value_imputer", "label": "🧹 Imputer", "config": {"strategy": "median"}},
        "node_scale": {"recipe_id": "feature_scaler", "label": "⚖️ Scaler", "config": {"method": "standard"}},
        "node_encode": {"recipe_id": "categorical_encoder", "label": "🔤 Encoder", "config": {"method": "one_hot"}},
        "node_split": {"recipe_id": "train_test_split", "label": "✂️ Split", "config": {"target_column": target_col, "test_size": 0.2}},
        "node_xgb": {"recipe_id": "xgboost_trainer", "label": "⚡ XGBoost", "config": {"task_type": "classification", "n_estimators": 100}},
        "node_eval": {"recipe_id": "model_evaluator", "label": "🎯 Evaluator", "config": {"report_type": "Comprehensive (All Metrics + Confusion Matrix)"}},
    }
    st.session_state["canvas_version"] = st.session_state.get("canvas_version", 1) + 1
    execute_pipeline()


def load_forecast_template(force_preset: bool = False):
    """Applies Forecasting template to current active data or sample preset."""
    if force_preset or "active_df" not in st.session_state or st.session_state["active_df"] is None:
        st.session_state["active_df"] = get_preset_dataset("Daily Retail Sales (Time-Series)")
        st.session_state["active_dataset_name"] = "Daily Retail Sales (Time-Series)"

    df = st.session_state["active_df"]
    cols = list(df.columns)
    date_candidates = [c for c in cols if any(k in c.lower() for k in ["date", "time", "timestamp", "day", "ds"])]
    date_col = date_candidates[0] if date_candidates else cols[0]
    num_candidates = [c for c in cols if pd.api.types.is_numeric_dtype(df[c]) and c != date_col]
    target_col = num_candidates[-1] if num_candidates else cols[-1]

    fc_nodes = [
        create_flow_node("node_csv", (40, 100), f"📄 {st.session_state.get('active_dataset_name', 'Active Data')[:15]}"),
        create_flow_node("node_prophet", (340, 100), "🔮 Prophet Forecaster"),
    ]
    fc_edges = [
        StreamlitFlowEdge(id="fe1", source="node_csv", target="node_prophet", animated=True)
    ]
    st.session_state["flow_state"] = StreamlitFlowState(nodes=fc_nodes, edges=fc_edges)
    st.session_state["node_configs"] = {
        "node_csv": {"recipe_id": "csv_loader", "label": "📄 Dataset Ingestion", "config": {}},
        "node_prophet": {"recipe_id": "prophet_forecaster", "label": "🔮 Prophet Forecaster", "config": {"date_column": date_col, "target_column": target_col, "horizon_periods": 30, "seasonality_mode": "additive"}},
    }
    st.session_state["canvas_version"] = st.session_state.get("canvas_version", 1) + 1
    execute_pipeline()


def load_anomaly_template(force_preset: bool = False):
    """Applies Anomaly detection template to current active data or sample preset."""
    if force_preset or "active_df" not in st.session_state or st.session_state["active_df"] is None:
        st.session_state["active_df"] = get_preset_dataset("Credit Transactions (Anomaly Injection)")
        st.session_state["active_dataset_name"] = "Credit Transactions (Anomaly Injection)"

    anom_nodes = [
        create_flow_node("node_csv", (40, 100), f"📄 {st.session_state.get('active_dataset_name', 'Active Data')[:15]}"),
        create_flow_node("node_guard", (320, 100), "🛡️ Statistical Guardrail"),
        create_flow_node("node_iso", (620, 100), "🌲 Isolation Forest Detector"),
    ]
    anom_edges = [
        StreamlitFlowEdge(id="ae1", source="node_csv", target="node_guard", animated=True),
        StreamlitFlowEdge(id="ae2", source="node_guard", target="node_iso", animated=True),
    ]
    st.session_state["flow_state"] = StreamlitFlowState(nodes=anom_nodes, edges=anom_edges)
    st.session_state["node_configs"] = {
        "node_csv": {"recipe_id": "csv_loader", "label": "📄 Dataset Ingestion", "config": {}},
        "node_guard": {"recipe_id": "statistical_guardrail", "label": "🛡️ Statistical Guardrail", "config": {"method": "z_score", "threshold": 3.0, "action": "flag"}},
        "node_iso": {"recipe_id": "isolation_forest", "label": "🌲 Isolation Forest Detector", "config": {"contamination": 0.05, "n_estimators": 100}},
    }
    st.session_state["canvas_version"] = st.session_state.get("canvas_version", 1) + 1
    execute_pipeline()


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
                load_ml_template()
                st.rerun()

    with bar_col4:
        if st.button("🔮 Forecast", use_container_width=True):
            if len(st.session_state["flow_state"].nodes) > 0:
                st.session_state["pending_action"] = "forecast_template"
            else:
                load_forecast_template()
                st.rerun()

    with bar_col5:
        if st.button("🚨 Anomaly", use_container_width=True):
            if len(st.session_state["flow_state"].nodes) > 0:
                st.session_state["pending_action"] = "anomaly_template"
            else:
                load_anomaly_template()
                st.rerun()

    with bar_col6:
        if st.button("🔗 Auto-Wire", use_container_width=True):
            curr_nodes = st.session_state["flow_state"].nodes
            if len(curr_nodes) < 2:
                st.warning("⚠️ Auto-Wire requires at least 2 nodes on the canvas. Add components first.")
            else:
                # 1. Sort nodes spatially from left to right (X-coordinate, then Y-coordinate)
                def get_pos(n):
                    if isinstance(n.pos, (list, tuple)) and len(n.pos) >= 2:
                        return (float(n.pos[0]), float(n.pos[1]))
                    elif hasattr(n.pos, 'x') and hasattr(n.pos, 'y'):
                        return (float(n.pos.x), float(n.pos.y))
                    elif isinstance(n.pos, dict):
                        return (float(n.pos.get('x', 0)), float(n.pos.get('y', 0)))
                    return (0.0, 0.0)

                sorted_nodes = sorted(curr_nodes, key=get_pos)
                
                # 2. Build clean sequential left-to-right DAG connections
                new_edges = []
                for i in range(len(sorted_nodes) - 1):
                    src = sorted_nodes[i].id
                    tgt = sorted_nodes[i+1].id
                    new_edges.append(StreamlitFlowEdge(
                        id=f"auto_{src}_{tgt}",
                        source=src,
                        target=tgt,
                        animated=True
                    ))

                st.session_state["flow_state"].edges = new_edges
                st.session_state["canvas_version"] = st.session_state.get("canvas_version", 1) + 1
                st.success(f"🔗 Sequentially wired {len(sorted_nodes)} nodes along visual left-to-right path!")
                st.rerun()

    with bar_col7:
        if st.button("🧹 Clear", use_container_width=True):
            if len(st.session_state["flow_state"].nodes) > 0:
                st.session_state["pending_action"] = "clear_canvas"
            else:
                st.info("ℹ️ Whiteboard is already empty.")

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
            custom_target = st.selectbox("🎯 Target / Dependent Feature", df_cols, index=len(df_cols)-1, key="ai_drawer_target")
        with c_t2:
            custom_task = st.selectbox("⚙️ Problem Type Intent", ["Auto-Detect", "classification", "regression", "time_series_forecasting", "anomaly_detection"], index=0, key="ai_drawer_task")

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
        col_cat, col_recipe, col_add = st.columns([2, 3, 2])
        
        with col_cat:
            selected_cat = st.selectbox("Category", list(RECIPE_CATEGORY_MAP.keys()), index=0)

        with col_recipe:
            available = RECIPE_CATEGORY_MAP[selected_cat]
            recipe_map = {r["name"]: r for r in available}
            chosen_name = st.selectbox("Subcategory / Recipe", list(recipe_map.keys()))
            chosen_meta = recipe_map[chosen_name]

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
            selected_node_id = st.selectbox("Select Node", current_node_ids)
            
            node_cfg = st.session_state["node_configs"].get(selected_node_id, {"recipe_id": "missing_value_imputer", "config": {}})
            recipe_obj = recipe_registry.get(node_cfg["recipe_id"])
            
            st.markdown(f"**Recipe:** `{recipe_obj.name}`")
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
                                    n.data = {"content": f"📄 {preset_name[:18]}"}
                            sanitize_node_configs_for_active_dataset()
                            st.session_state["canvas_version"] = st.session_state.get("canvas_version", 1) + 1
                            st.success(f"Loaded and synced '{preset_name}' to whiteboard!")
                            st.rerun()

                    elif data_source_mode == "Upload New File":
                        up_file = st.file_uploader("Upload CSV / Excel / JSON", type=["csv", "xlsx", "json"], key=f"up_{selected_node_id}")
                        if up_file is not None:
                            parsed_df, file_name = parse_uploaded_dataset(up_file)
                            if parsed_df is not None:
                                st.session_state["active_df"] = parsed_df
                                st.session_state["active_dataset_name"] = file_name
                                for n in st.session_state["flow_state"].nodes:
                                    if n.id == selected_node_id:
                                        n.data = {"content": f"📄 {file_name[:18]}"}
                                sanitize_node_configs_for_active_dataset()
                                st.session_state["canvas_version"] = st.session_state.get("canvas_version", 1) + 1
                                st.success(f"Loaded and synced '{file_name}' to whiteboard!")
                                st.rerun()

                    st.caption(f"Active Data: **{st.session_state['active_dataset_name']}** ({len(st.session_state['active_df'])} rows)")

                # Dynamic Parameters Form from Recipe JSON Schema
                schema = recipe_obj.get_schema()
                props = schema.get("properties", {})
                df = st.session_state["active_df"]
                
                st.markdown("##### Parameters:")
                updated_params = dict(node_cfg.get("config", {}))
                
                for prop_name, prop_meta in props.items():
                    title = prop_meta.get("title", prop_name)
                    current_val = updated_params.get(prop_name, prop_meta.get("default"))
                    
                    if "enum" in prop_meta:
                        opts = prop_meta["enum"]
                        idx = opts.index(current_val) if current_val in opts else 0
                        val = st.selectbox(title, opts, index=idx, key=f"p_{selected_node_id}_{prop_name}")
                        updated_params[prop_name] = val
                    elif prop_name in ["target_column", "date_column"]:
                        cols = list(df.columns)
                        idx = cols.index(current_val) if current_val in cols else (0 if prop_name == "date_column" else len(cols)-1)
                        val = st.selectbox(title, cols, index=idx, key=f"p_{selected_node_id}_{prop_name}")
                        updated_params[prop_name] = val
                    elif prop_meta.get("type") == "integer":
                        default_int = int(prop_meta.get("default", 0))
                        init_int = int(current_val if current_val is not None else default_int)
                        val = st.number_input(title, min_value=prop_meta.get("minimum", 0), max_value=prop_meta.get("maximum", 2000), value=init_int, key=f"p_{selected_node_id}_{prop_name}")
                        updated_params[prop_name] = val
                    elif prop_meta.get("type") == "number":
                        min_v = float(prop_meta.get("minimum", 0.0))
                        max_v = float(prop_meta.get("maximum", 1.0))
                        default_num = float(prop_meta.get("default", min_v))
                        cur_v = float(current_val if current_val is not None else default_num)
                        cur_v = max(min_v, min(max_v, cur_v))
                        val = st.slider(title, min_value=min_v, max_value=max_v, value=cur_v, key=f"p_{selected_node_id}_{prop_name}")
                        updated_params[prop_name] = val
                    elif prop_meta.get("type") == "string":
                        val = st.text_input(title, value=str(current_val if current_val is not None else ""), key=f"p_{selected_node_id}_{prop_name}")
                        updated_params[prop_name] = val

                st.session_state["node_configs"][selected_node_id]["config"] = updated_params

            if st.button("🗑️ Delete Node", key=f"btn_del_{selected_node_id}"):
                st.session_state["flow_state"].nodes = [n for n in st.session_state["flow_state"].nodes if n.id != selected_node_id]
                st.session_state["flow_state"].edges = [e for e in st.session_state["flow_state"].edges if e.source != selected_node_id and e.target != selected_node_id]
                if selected_node_id in st.session_state["node_configs"]:
                    del st.session_state["node_configs"][selected_node_id]
                st.session_state["canvas_version"] = st.session_state.get("canvas_version", 1) + 1
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
        anomaly_summary = exec_data.get("anomaly_summary")
        forecasting_summary = exec_data.get("forecasting_summary")
        governance_summary = exec_data.get("governance_summary")
        execution_logs = exec_data.get("execution_logs", [])
        node_outputs = exec_data.get("node_outputs", {})

        st.markdown("---")

        # -----------------------------------------------------
        # 0. MLFLOW MODEL GOVERNANCE & REGISTRY REPORT
        # -----------------------------------------------------
        if governance_summary:
            st.markdown("### 🏛️ MLflow Model Governance & Registry Audit")
            gcol1, gcol2, gcol3, gcol4 = st.columns(4)
            gcol1.metric("Registered Model Name", governance_summary.get("registered_model_name", "N/A"))
            gcol2.metric("Promotion Stage", governance_summary.get("stage", "Staging"))
            gcol3.metric("Metrics Audited", f"{governance_summary.get('metrics_logged', 0)} KPIs")
            gcol4.metric("Experiment", governance_summary.get("experiment_name", "Enterprise_ML_Pipelines"))

            st.success(f"🔒 **Audited MLflow Run ID:** `{governance_summary.get('mlflow_run_id')}` | **Tracking URI:** `{governance_summary.get('tracking_uri')}`")

        # -----------------------------------------------------
        # 1. TIME-SERIES FORECASTING INTELLIGENCE REPORT
        # -----------------------------------------------------
        if forecasting_summary:
            st.markdown("### 🔮 Time-Series Forecasting Intelligence Dashboard")
            
            fkpi1, fkpi2, fkpi3, fkpi4 = st.columns(4)
            fkpi1.metric("Algorithm Applied", forecasting_summary.get("algorithm", "Meta Prophet"))
            fkpi2.metric("Horizon Periods Ahead", f"{forecasting_summary.get('horizon_periods', 14)} steps")
            fkpi3.metric("MAPE (Mean Abs % Error)", f"{forecasting_summary.get('mape', 0.0):.2f}%")
            fkpi4.metric("RMSE", f"{forecasting_summary.get('rmse', 0.0):.2f}")

            fc_df = None
            for out in node_outputs.values():
                if isinstance(out, dict) and "forecast_df" in out:
                    fc_df = out["forecast_df"]
                    break

            if fc_df is not None:
                fc_tab1, fc_tab2 = st.tabs([
                    "📈 Interactive Time-Series Forecast Plot with Confidence Bands",
                    "📋 Future Forecast Predictions Table"
                ])

                with fc_tab1:
                    fig_fc = go.Figure()
                    hist_part = fc_df[fc_df["is_future"] == 0]
                    fut_part = fc_df[fc_df["is_future"] == 1]

                    if "yhat_upper" in fut_part.columns and "yhat_lower" in fut_part.columns:
                        fig_fc.add_trace(go.Scatter(
                            x=pd.concat([fut_part["ds"], fut_part["ds"][::-1]]),
                            y=pd.concat([fut_part["yhat_upper"], fut_part["yhat_lower"][::-1]]),
                            fill='toself',
                            fillcolor='rgba(249, 115, 22, 0.15)',
                            line=dict(color='rgba(255,255,255,0)'),
                            hoverinfo="skip",
                            showlegend=True,
                            name='95% Prediction Confidence Band'
                        ))

                    fig_fc.add_trace(go.Scatter(
                        x=hist_part["ds"],
                        y=hist_part["yhat"],
                        mode='lines',
                        name='Historical In-Sample Series',
                        line=dict(color='#3B82F6', width=2)
                    ))

                    fig_fc.add_trace(go.Scatter(
                        x=fut_part["ds"],
                        y=fut_part["yhat"],
                        mode='lines+markers',
                        name='Future Forecast Predictions',
                        line=dict(color='#F97316', width=3, dash='dash')
                    ))

                    fig_fc.update_layout(
                        title="Time-Series Forecast Trajectory (Blue: Historical | Orange: Future Prediction)",
                        xaxis_title="Date / Timestamp",
                        yaxis_title="Forecast Metric",
                        hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_fc, use_container_width=True)

                with fc_tab2:
                    st.markdown(f"##### Showing {len(fut_part)} Future Prediction Timesteps:")
                    st.dataframe(fut_part[["ds", "yhat", "yhat_lower", "yhat_upper"]], use_container_width=True)

                    st.download_button(
                        "📥 Download Forecast Output (CSV)",
                        data=fc_df.to_csv(index=False),
                        file_name="time_series_forecast.csv",
                        mime="text/csv"
                    )

        # -----------------------------------------------------
        # 2. ANOMALY DETECTION REPORT
        # -----------------------------------------------------
        if anomaly_summary:
            st.markdown("### 🚨 Anomaly Detection & Outlier Intelligence Dashboard")
            
            akpi1, akpi2, akpi3, akpi4 = st.columns(4)
            tot = anomaly_summary.get("total_records", anomaly_summary.get("total_records_before", 0))
            anom_c = anomaly_summary.get("anomaly_count", anomaly_summary.get("outliers_detected", 0))
            anom_p = anomaly_summary.get("anomaly_percentage", anomaly_summary.get("outlier_percentage", 0))
            
            akpi1.metric("Total Records Evaluated", f"{tot:,}")
            akpi2.metric("Anomalies / Outliers Flagged", f"{anom_c:,}")
            akpi3.metric("Anomaly Rate", f"{anom_p:.2f}%")
            akpi4.metric("Algorithm Applied", anomaly_summary.get("algorithm", "Isolation Forest"))

            anomaly_df = None
            for out in node_outputs.values():
                if isinstance(out, dict) and "dataframe" in out:
                    d = out["dataframe"]
                    if "is_anomaly" in d.columns or "is_outlier" in d.columns:
                        anomaly_df = d
                        break

            if anomaly_df is not None:
                anom_tab1, anom_tab2, anom_tab3 = st.tabs([
                    "📈 2D Outlier Scatter Visualizer",
                    "📊 Anomaly Score Distribution",
                    "📋 Flagged Anomalies Inspection Table"
                ])

                with anom_tab1:
                    num_cols = [c for c in anomaly_df.columns if pd.api.types.is_numeric_dtype(anomaly_df[c]) and c not in ["is_anomaly", "is_outlier", "anomaly_score"]]
                    color_col = "is_anomaly" if "is_anomaly" in anomaly_df.columns else "is_outlier"
                    plot_df = anomaly_df.copy()
                    if color_col in plot_df.columns:
                        plot_df[color_col] = plot_df[color_col].astype(str)

                    if len(num_cols) >= 2:
                        sc_col1, sc_col2 = st.columns(2)
                        with sc_col1:
                            x_col = st.selectbox("X-Axis Feature", num_cols, index=0)
                        with sc_col2:
                            y_col = st.selectbox("Y-Axis Feature", num_cols, index=1 if len(num_cols) > 1 else 0)

                        fig_scatter = px.scatter(
                            plot_df,
                            x=x_col,
                            y=y_col,
                            color=color_col if color_col in plot_df.columns else None,
                            color_discrete_map={"0": "#3B82F6", "1": "#EF4444"},
                            hover_data=num_cols[:4],
                            title=f"Outlier Scatter Plot: {x_col} vs {y_col} (Red = Anomaly)"
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)
                    elif len(num_cols) == 1:
                        feat = num_cols[0]
                        fig_1d = px.strip(
                            plot_df,
                            x=feat,
                            color=color_col if color_col in plot_df.columns else None,
                            color_discrete_map={"0": "#3B82F6", "1": "#EF4444"},
                            title=f"1D Anomaly Distribution: {feat} (Red = Anomaly)"
                        )
                        st.plotly_chart(fig_1d, use_container_width=True)
                    else:
                        st.info("ℹ️ No numeric columns available for outlier distribution plotting.")

                with anom_tab2:
                    if "anomaly_score" in anomaly_df.columns:
                        hist_df = anomaly_df.copy()
                        if "is_anomaly" in hist_df.columns:
                            hist_df["is_anomaly"] = hist_df["is_anomaly"].astype(str)

                        fig_hist = px.histogram(
                            hist_df,
                            x="anomaly_score",
                            color="is_anomaly" if "is_anomaly" in hist_df.columns else None,
                            color_discrete_map={"0": "#3B82F6", 1: "#EF4444"},
                            nbins=30,
                            title="Distribution of Anomaly Scores (0.0 = Normal, 1.0 = High-Risk Anomaly)"
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)

                with anom_tab3:
                    flag_col = "is_anomaly" if "is_anomaly" in anomaly_df.columns else "is_outlier"
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
        if final_metrics and final_metrics.get("task_type") in ["classification", "regression"]:
            st.markdown("### 🏆 Comprehensive Model Evaluation & Diagnostic Reports")
            
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
                if "confusion_matrix" in final_metrics:
                    cm = final_metrics["confusion_matrix"]
                    fig_cm = ff.create_annotated_heatmap(
                        z=cm,
                        x=["Pred: Class 0", "Pred: Class 1"] if len(cm) == 2 else [f"Pred {i}" for i in range(len(cm))],
                        y=["Actual: Class 0", "Actual: Class 1"] if len(cm) == 2 else [f"Actual {i}" for i in range(len(cm))],
                        colorscale="Blues"
                    )
                    fig_cm.update_layout(title="Confusion Matrix Heatmap", width=480, height=360)
                    st.plotly_chart(fig_cm, use_container_width=False)

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
        
        step_snapshots = last_execution.get("step_snapshots", {})
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
            st.session_state["active_df"] = get_preset_dataset(preset)
            st.session_state["active_dataset_name"] = preset
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
