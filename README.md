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

### 4. Run Automated Test Suite
```bash
pytest backend/tests -v
```

---

## 📦 API Reference & REST Contract

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/workflows/execute` | Synchronously executes a DAG end-to-end and returns evaluation metrics, step snapshots, and logs. |
| `POST` | `/api/v1/workflows/validate` | Pre-flight sanity check for topological cycles, orphan nodes, and semantic recipe compatibility. |
| `POST` | `/api/v1/workflows/async-execute` | Dispatches long-running DAG jobs to the background worker pool and returns a tracking `job_id`. |
| `GET` | `/api/v1/workflows/jobs/{job_id}` | Polls the runtime status, duration, and telemetry for an asynchronous workflow job. |
| `POST` | `/api/v1/workflows/trigger/{path}` | Inbound webhook trigger ingesting live JSON event streams into the DAG entrypoint. |
| `POST` | `/api/v1/datasets/upload` | Ingests CSV, XLSX, or JSON files and automatically computes statistical quality profiles. |
| `GET` | `/api/v1/datasets/{id}/profile` | Retrieves column-by-column missingness, distribution stats, and health scores. |
| `GET` | `/api/v1/recipes/` | Lists the entire catalog of registered AI/ML and data-processing components. |
| `GET` | `/api/v1/recipes/{id}/schema` | Retrieves JSON Schema definitions for dynamic frontend parameter form rendering. |

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
