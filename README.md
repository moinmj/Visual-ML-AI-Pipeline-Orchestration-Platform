# ⚡ Visual AI/ML Pipeline Orchestration Platform

An enterprise-grade, visual no-code/low-code workflow automation and AI orchestration platform modeled on the Symbol-Recipe paradigm (similar to Boomi and n8n) with data science abstractions (similar to Dataiku).

---

## 🌟 Key Features

- **🎨 Interactive 2D Whiteboard:** Visual DAG construction powered by React Flow with drag-and-drop nodes, dynamic parameter forms, 1-click connectors, and node-level Python `</> Code View`.
- **🧠 Automated Data Profiler & AI Recommender:** Deterministic statistical profiling engine that automatically detects problem types (Classification, Regression, Time-Series Forecasting, Anomaly Detection) and generates the optimal DAG pipeline.
- **⚡ Gradient Boosting Triumvirate:** Enterprise support for **XGBoost**, **LightGBM**, and **CatBoost** with JSON-safe feature sanitization and native high-cardinality string handling.
- **🔮 Time-Series Forecasting:** Built-in **Meta Prophet**, **ARIMA / SARIMAX**, and **Lag Feature Engineering** with shaded 95% confidence bands.
- **🚨 Anomaly Detection:** Tier-1 **Isolation Forest** unsupervised partitioning and **Statistical Guardrails** (Z-Score / Tukey IQR) with interactive 2D outlier scatter plots and score histograms.
- **🏛️ MLOps & Model Governance:** Native **MLflow Model Registry** integration logging hyperparameters, evaluation metrics, artifacts, and promoting models to `Staging` or `Production`.
- **⚙️ DAG Workflow Engine:** Kahn's algorithm cycle detection, topological sorting, and error isolation with clear diagnostic explanations.

---

## 🚀 Quickstart Guide

### 1. Clone & Setup Environment
```bash
git clone https://github.com/moinmj/Visual-ML-AI-Pipeline-Orchestration-Platform.git
cd Visual-ML-AI-Pipeline-Orchestration-Platform

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

### 3. Run Automated Backend Tests
```bash
pytest backend/tests -v
```

---

## 🏛️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                 FastAPI REST API / Streamlit                │
└──────────────────────────────┬──────────────────────────────┘
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
┌──────────────────────────────┐┌─────────────────────────────┐
│    Dataset Profiler Engine   ││    AI Recommendation Engine │
│  (Nulls, Types, Health Score)││   (Diagnosis & Model Ranking)│
└──────────────┬───────────────┘└─────────────┬───────────────┘
               │                              │
               └──────────────┬───────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 DAG Workflow Engine (Kahn's Sort)           │
├─────────────────────────────────────────────────────────────┤
│  • Ingestion (CSV / Excel / JSON)                           │
│  • Cleaning & Preprocessing (Imputer, Scaler, Encoder)      │
│  • ML Training (XGBoost, LightGBM, CatBoost, Random Forest) │
│  • Forecasting (Prophet, ARIMA, Lag Features)               │
│  • Anomaly Detection (Isolation Forest, Z-Score Guardrail)  │
│  • Model Evaluation & Diagnostic Visualizations             │
│  • Model Governance (MLflow Tracking & Model Registry)      │
└─────────────────────────────────────────────────────────────┘
```

---

## 📂 Repository Structure

```
├── backend/
│   ├── app/
│   │   ├── core/                  # Config, Logging, Exceptions
│   │   ├── engine/                # DAG Graph, Kahn's Validation & Execution
│   │   ├── infrastructure/        # DB Session, Storage Manager (Local/S3)
│   │   ├── profiling/             # Automated DataProfiler Engine
│   │   ├── recommendation/        # AIRecommender Engine
│   │   └── recipes/               # 17 Standard Recipe Modules
│   └── tests/                     # 19 Unit & Integration Tests
├── ui_prototype/
│   └── app.py                     # Streamlit + React Flow Whiteboard Studio
├── docs/                          # Architecture Docs & Master Context
└── requirements.txt               # Dependencies
```

---

## 📜 License
MIT License. Built for scalable, visual machine learning pipeline orchestration.
