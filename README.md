# ⚡ Visual AI/ML Pipeline Orchestration Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Tests](https://img.shields.io/badge/tests-22%20passed%20(100%25)-success.svg)](https://pytest.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-grade, visual no-code/low-code workflow automation and AI orchestration platform modeled on the Symbol-Recipe paradigm (similar to **Boomi** and **n8n**) with rich data science and AutoML abstractions (similar to **Dataiku**).

---

## 🌟 Key Platform Capabilities

### 1. 🎨 Interactive Visual Whiteboard (React Flow)
- **Drag-to-Connect Wiring:** Native 2D canvas with interactive handle-to-handle visual wiring, selectable/draggable nodes, and animated edge flows.
- **🔗 Smart Auto-Wire:** 1-click spatial sequential left-to-right DAG autowiring.
- **Interactive Node Config Drawer:** Dynamic form rendering generated directly from JSON Schemas alongside real-time Python `</> Generated Code` inspectability.
- **Live Whiteboard Renaming:** Real-time editable node labels and collision-free ID management.

### 2. 🧠 Automated Data Profiler & AI Auto-Architect
- **Statistical Health Profiling:** Computes missingness, duplicate rates, cardinality, and automated dataset quality scores (0–100).
- **Auto-Task Inference:** Automatically identifies problem types (**Binary/Multi-Class Classification**, **Continuous Regression**, **Time-Series Forecasting**, or **Unsupervised Anomaly Detection**).
- **1-Click Auto-Architect:** Synthesizes the optimal end-to-end DAG preprocessing chain and model architecture based on dataset characteristics.

### 3. ⚡ Inbound Triggers & Async Worker Pool (n8n / Boomi Parity)
- **🌐 Webhook Inbound Trigger (`POST /api/v1/workflows/trigger/{path}`):** Ingests real-time HTTP JSON batches and streams them downstream into the pipeline DAG.
- **⏰ Cron Schedule Trigger (`Recurring`):** Automated recurring pipeline execution via Unix Cron expressions (`0 0 * * *`) and interval presets.
- **👷 Asynchronous Job Manager:** Dispatches long-running training/inference workloads to a background worker pool (`POST /api/v1/workflows/async-execute`) with instant `job_id` status polling.

### 4. 🔬 Step-by-Step Data Stream Debugger (Node Inspector)
- **Intermediate Data Snapshots:** Captures exact tabular DataFrame snapshots, column schemas, and row counts as data flows across every edge.
- **Execution Telemetry:** Inspects per-node execution latencies in milliseconds (`ms`) and feature throughput.

### 5. 📡 Real-Time API Flight Recorder & Inspector
- **Persistent Sidebar Dock:** Real-time banner on every tab displaying the latest REST API hit, HTTP status code, latency, and expandable request/response JSON payloads.
- **Interactive API Playground:** Full interactive testing suite for all 8 FastAPI endpoints (`/workflows/execute`, `/workflows/validate`, `/workflows/async-execute`, `/recipes/`, `/datasets/upload`, etc.) with customizable request bodies and live response rendering.
- **cURL & Python Snippet Generator:** 1-click generation of terminal commands and code snippets for active DAG payloads.

### 6. 🏆 Comprehensive Task-Aware Evaluation
- **Classification:** Accuracy, Balanced Accuracy, Precision, Recall, F1 Score, ROC-AUC, Interactive Confusion Matrix Heatmap, and Per-Class breakdown tables.
- **Regression:** $R^2$ Score (Variance Explained), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Mean Squared Error (MSE), MAPE, and Actual vs. Predicted sample tables.
- **Time-Series:** Shaded 95% confidence bands ($yhat_{lower}$ to $yhat_{upper}$), historical vs. future predictions, and trend diagnostics.
- **Anomaly Detection:** Outlier risk cluster scatter plots, anomaly score histograms (0.0 to 1.0), and 1-click flagged records CSV export.

### 7. 🛡️ Robust High-Cardinality & Out-of-Memory Protection
- Safe preprocessing pipelines that protect target columns during encoding and apply date decomposition and ordinal encoding to high-cardinality columns ($>50$ categories), preventing $13.2\text{ GiB}$ memory crashes.

---

## 🏛️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FastAPI REST API / Streamlit UI Studio                   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                        ┌──────────────┴──────────────┐
                        ▼                             ▼
        ┌──────────────────────────────┐┌─────────────────────────────┐
        │   Dataset Profiler Engine    ││    AI Auto-Architect Engine │
        │  (Nulls, Types, Health Score)││   (Diagnosis & Model Ranking)│
        └──────────────┬───────────────┘└─────────────┬───────────────┘
                       │                              │
                       └──────────────┬───────────────┘
                                      ▼
        ┌─────────────────────────────────────────────────────────────┐
        │                 DAG Workflow Engine (Kahn's Sort)           │
        ├─────────────────────────────────────────────────────────────┤
        │  • Triggers (Webhook POST, Cron Schedules)                  │
        │  • Ingestion (CSV, Excel, Nested JSON)                      │
        │  • Cleaning & Preprocessing (Imputer, Scaler, Encoder)      │
        │  • ML Training (XGBoost, LightGBM, CatBoost, Random Forest) │
        │  • Time-Series (Prophet, ARIMA, Lag Feature Engineer)       │
        │  • Anomaly Detection (Isolation Forest, Z-Score Guardrail)  │
        │  • Step Data Inspector (DataFrame Snapshots per Node)       │
        │  • Model Evaluation & Diagnostic Visualizations             │
        │  • Model Governance (MLflow Tracking & Model Registry)      │
        └─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quickstart Guide

### 1. Clone Repository & Setup Environment
```bash
git clone https://github.com/moinmj/Visual-ML-AI-Pipeline-Orchestration-Platform.git
cd Visual-ML-AI-Pipeline-Orchestration-Platform

# Checkout active feature branch
git checkout ml-ai-pipeline

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `OPENAI_API_KEY` in `.env` to enable the OpenAI agent endpoints. The
existing non-agent recommender works without an OpenAI key. Never commit or
share the `.env` file.

### 2. Launch Interactive Whiteboard Studio
```bash
streamlit run ui_prototype/app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

### 3. Launch Backend FastAPI Server (Optional)
```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive OpenAPI Swagger docs available at **[http://localhost:8000/docs](http://localhost:8000/docs)**.

If `P_DATABASE_URL` is configured, that database must contain the tenant metadata
tables (`tenant_environments`, `tenant_models`, and related tables). For a new
development database only, set `TENANT_DB_AUTO_CREATE=true` before starting the
server. Keep it `false` in production and provision the schema through the
service that owns the tenant database.

### 4. Run Automated Test Suite
```bash
pytest backend/tests -v
```
In Windows PowerShell, with the virtual environment activated, use
`python -m pytest backend\tests -v`.

---

## 📦 API Reference & REST Contract

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/workflows/execute` | Synchronously executes a DAG end-to-end and returns evaluation metrics, step snapshots, and logs. |
| `POST` | `/api/v1/workflows/validate` | Pre-flight sanity check for topological cycles, orphan nodes, and semantic recipe compatibility. |
| `POST` | `/api/v1/workflows/async-execute` | Dispatches long-running DAG jobs to the background worker pool and returns a tracking `job_id`. |
| `GET` | `/api/v1/workflows/jobs/{job_id}` | Polls the runtime status, duration, and telemetry for an asynchronous workflow job. |
| `POST` | `/api/v1/workflows/trigger/{path}` | Inbound webhook trigger ingesting live JSON event streams into the DAG entrypoint. |
| `POST` | `/api/v1/agent/pipelines/plan` | Uses the registered recipe catalog to draft a classification or regression DAG for review. |
| `POST` | `/api/v1/agent/pipelines/run` | Queues a bounded agent-led classification or regression search and immediately returns a `job_id`. |
| `GET` | `/api/v1/agent/pipelines/jobs/{job_id}` | Polls the agent job and returns its result when complete. |
| `POST` | `/api/v1/datasets/upload` | Ingests CSV, XLSX, or JSON files and automatically computes statistical quality profiles. |
| `GET` | `/api/v1/datasets/{id}/profile` | Retrieves column-by-column missingness, distribution stats, and health scores. |
| `GET` | `/api/v1/recipes/` | Lists the entire catalog of registered AI/ML and data-processing components. |
| `GET` | `/api/v1/recipes/{id}/schema` | Retrieves JSON Schema definitions for dynamic frontend parameter form rendering. |

### OpenAI recipe agent

```json
POST /api/v1/agent/pipelines/plan
{
  "dataset_id": "<uploaded-dataset-id>",
  "request": "Predict house prices and compare suitable regression models",
  "target_column": "price",
  "task_type": "regression"
}
```

`task_type` is optional: the planner can infer classification or regression
from the request and dataset profile, and asks for clarification when it cannot
choose safely. `/plan` returns clarification questions or a validated workflow
for review. Recipe tools are generated from the registry; the planner can use
registered preprocessing, NLP, and training recipes compatible with the
selected supervised task. The complete recipe catalog is also included as
planning context, while ingestion, splitting, and evaluation are controlled by
the server.

For autonomous execution, use `/run`:

```json
POST /api/v1/agent/pipelines/run
{
  "dataset_id": "<uploaded-dataset-id>",
  "request": "Train and improve a regression model to predict house prices",
  "target_column": "price",
  "task_type": "regression",
  "max_attempts": 5
}
```

`/run` returns `202 Accepted` with a job ID; poll the `status_endpoint` in its
response. The agent judges task-specific validation metrics, selects
alternative registered recipes/configurations, and retries up to five times.
Regression candidates use R2 (higher is better), falling back to RMSE (lower
is better); classification candidates use balanced accuracy/F1. The best
candidate is registered in MLflow only when accepted.

Agent jobs run outside the API event loop in a bounded per-process worker pool.
Defaults are two active jobs, sixteen queued jobs, and eight concurrent OpenAI
requests; configure `AGENT_MAX_WORKERS`, `AGENT_MAX_QUEUED`, and
`AGENT_MAX_OPENAI_CONCURRENT`. When capacity is full, the API returns
`429 Too Many Requests` with `Retry-After`, rather than accepting unlimited
work. Job state and queue are currently in process memory, so a server restart
loses them and multiple API replicas do not share job state. Production
horizontal scaling requires a shared durable queue and job store.

The same validation split is used to compare candidates, so its score is not
an unbiased final test score. Preprocessing recipes that learn values before
splitting can also inflate that score; a separate final holdout and
split-aware preprocessing are needed for production-grade evaluation.

OpenAI receives the user's request, column names, aggregate schema/profile
metadata, and scalar evaluation metrics—not dataset rows, category examples,
or the dataset ID.

The run endpoint is orchestrated with a LangGraph `StateGraph`: a planner
agent proposes the first pipeline, the existing workflow executor runs it,
and an evaluator/reviser agent either accepts the result, asks for
clarification, or proposes another validated recipe configuration. This is a
bounded, sequential agent graph—not a hierarchy of autonomous worker agents.
Forecasting and anomaly-detection recipes are visible in the catalog, but
their task-specific planning and validation flows are not yet enabled in this
supervised agent endpoint.

---

## 📂 Project Structure

```
Visual-ML-AI-Pipeline-Orchestration-Platform/
├── backend/
│   ├── app/
│   │   ├── core/                  # Config, Logging, Exceptions
│   │   ├── datasets/              # Dataset Upload, Schemas, Profiling Router
│   │   ├── engine/                # DAG Graph, Kahn's Validation, Executor & Job Manager
│   │   ├── infrastructure/        # DB Session, Storage Manager
│   │   ├── profiling/             # Automated DataProfiler Engine
│   │   ├── recommendation/        # AIRecommender Engine
│   │   ├── recipes/               # 17 Modular Processing & ML Algorithms
│   │   │   ├── triggers/          # Webhook & Cron Schedule Inbound Triggers
│   │   │   ├── ingestion/         # CSV, Excel, JSON Loaders
│   │   │   ├── preprocessing/     # Imputers, Scalers, Categorical Encoders
│   │   │   ├── splitting/         # Train/Test Partition Splitters
│   │   │   ├── training/          # XGBoost, LightGBM, CatBoost, RF, Linear Trainers
│   │   │   ├── anomaly/           # Isolation Forest, Statistical Guardrails
│   │   │   ├── forecasting/       # Prophet, ARIMA, Lag Feature Engineers
│   │   │   ├── governance/        # MLflow Tracking & Registry
│   │   │   └── evaluation/        # Classification & Regression Evaluators
│   │   └── workflows/             # FastAPI REST Routes & Execution Endpoints
│   └── tests/                     # 22 Comprehensive Unit & Integration Tests
├── ui_prototype/
│   └── app.py                     # Streamlit + React Flow Whiteboard Studio & API Inspector
├── docs/                          # Architecture Specifications & API Guides
└── requirements.txt               # Dependencies
```

---

## 📜 License
MIT License. Built for scalable, visual machine learning pipeline orchestration.
