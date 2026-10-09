# Datalect ML — Visual AI/ML Pipeline Orchestration Engine
### Platform Overview & Technical Brief

> **Document Type:** Internal Technical Reference
> **Audience:** Engineers, Product Owners, Stakeholders
> **Last Updated:** October 2026

---

## 1. What Is This Platform?

**Datalect ML** is an enterprise-grade, **no-code / low-code Visual AI/ML Pipeline Orchestration Engine** — inspired by the philosophy of Dataiku, n8n, and Boomi.

It lets business analysts, data scientists, and engineers **drag, drop, connect, and execute** full end-to-end machine learning and data transformation workflows through a visual canvas — **without writing a single line of code**.

Think of it as: **"Build an ML model the way you draw a flowchart."**

---

## 2. What Problem Does It Solve?

Traditional ML workflow development requires:
- Writing custom Python scripts for every transformation step
- Manually wiring data between preprocessing stages and model trainers
- No visual audit trail of how data flows through a pipeline
- Teams using different tools with no centralized governance

**This platform eliminates all of that.** Users build pipelines visually, the backend handles all computation, and every execution is versioned, logged, and auditable.

---

## 3. High-Level Architecture

```
+--------------------------------------+
|          FRONTEND CANVAS             |
|   (React Flow / Next.js Drag-Drop)   |
+------------------+-------------------+
                   |  REST API / JSON
                   v
+--------------------------------------+
|       BACKEND PLATFORM ENGINE        |
|         (Python / FastAPI)           |
|                                      |
|  +-------------+  +---------------+  |
|  | Recipe      |  | DAG / Workflow |  |
|  | Engine      |  | Engine        |  |
|  +-------------+  +---------------+  |
|  +-------------+  +---------------+  |
|  | ML / Train  |  | Inference     |  |
|  | Engine      |  | Engine        |  |
|  +-------------+  +---------------+  |
|  +-------------+  +---------------+  |
|  | Data        |  | MLflow Model  |  |
|  | Profiler    |  | Governance    |  |
|  +-------------+  +---------------+  |
+------------------+-------------------+
                   |
        +----------+-----------+
        v                      v
  PostgreSQL/SQLite       Apache Ozone / S3
  (State & Metadata)   (Datasets, Models, Artifacts)
```

**Key Principle:** The backend is the **single source of truth**. The frontend has zero ML business logic — it only renders what the backend tells it to.

---

## 4. Core Concepts

| Concept | Description |
|---|---|
| **Recipe** | A single, self-contained processing step (e.g., Impute Missing Values, Train XGBoost, PDF Chunker). Each recipe is parameterized and independently testable. |
| **Node** | An instance of a Recipe placed on the canvas with user-configured parameters. |
| **Edge** | A connection between two Nodes, defining how data flows (e.g., output of a Scaler to input of a Trainer). |
| **Workflow** | A Directed Acyclic Graph (DAG) made up of interconnected Nodes and Edges. |
| **Execution** | A single runtime run of a Workflow — tracks per-node status, logs, duration, and artifacts. |
| **Dataset** | Raw or intermediate tabular/unstructured data stored in object storage and profiled automatically. |
| **Experiment** | A tracked ML training run capturing hyperparameters, evaluation metrics, and model artifacts via MLflow. |
| **Inference Bundle** | A post-execution snapshot containing the fitted model + all transformers — enabling live prediction without retraining. |

---

## 5. What the Backend Handles

### 5.1 Recipe Engine
- Hosts 20+ built-in recipe types as self-contained Python classes inheriting from BaseRecipe
- Every recipe exposes a JSON **configuration schema** (used by frontend to auto-render input forms)
- Every recipe exposes a to_code() method producing reproducible Python snippets
- Categories: **Ingestion, Preprocessing, NLP, Splitting, Training, Forecasting, Anomaly, Evaluation, Governance, Integrations**

### 5.2 DAG / Workflow Engine
- Parses the canvas (nodes + edges) submitted from the frontend
- Runs **cycle detection** (Kahn's algorithm) to ensure no circular dependencies
- Performs **topological sort** to determine safe execution order
- Executes each node in sequence, passing DataFrames between nodes via handle-aware routing
  (e.g., left / right ports for joins, matched / unmatched for filters)
- Tracks execution state per-node (pending -> running -> success / failed)

### 5.3 Data Ingestion & Storage
- Accepts CSV, Excel (XLSX/XLS), JSON Lines, Parquet uploads
- Stores raw and intermediate datasets in **Apache Ozone** (S3-compatible object storage)
- Uses **Pandas** for in-memory, low-latency node-to-node data passing during execution

### 5.4 Data Profiler
- Automatically runs on every uploaded dataset
- Outputs: schema inference, missing value rates, cardinality, value distributions, outlier flags, a 0-100 data quality score
- Results are consumed by the AI Recommendation Engine

### 5.5 AI Recommendation Engine
- Analyzes dataset profiling metrics to diagnose the ML problem type:
  Classification / Regression / Forecasting / Anomaly Detection
- Recommends an optimal preprocessing chain and ranks algorithms with rationale
- Powered by **Groq LLM** (openai/gpt-oss-120b) with heuristic fallback if LLM is unavailable

### 5.6 ML Training & Evaluation
- **Classic supervised:** Logistic Regression, Random Forest
- **Gradient boosting triumvirate:** XGBoost, LightGBM, CatBoost (with auto-categorical fallbacks)
- **Time-series forecasting:** Meta Prophet, ARIMA/SARIMAX, Lag Feature Engineering
- **Anomaly detection:** Isolation Forest-based outlier scoring
- **Evaluation metrics:** Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix, MAE, MSE, R-squared

### 5.7 Universal Inference Engine (PipelineInferencer)
After a workflow finishes training, the platform automatically preserves an **Inference Bundle**
(model + transformers + feature schema) in memory. This enables:

| Inference Type | What It Does |
|---|---|
| **Classification** | Returns predicted class label (e.g., Iris-setosa) + per-class probability scores |
| **Regression** | Returns predicted numeric target value |
| **Forecasting** | Accepts start_date to end_date or horizon steps; returns future trajectories with 95% confidence intervals |
| **Anomaly Scoring** | Returns NORMAL_RECORD / ANOMALOUS_OUTLIER verdict + risk score (0.0-1.0) + risk level (LOW / MEDIUM / HIGH / CRITICAL) |
| **Batch CSV Scoring** | Accepts bulk record lists, scores all rows simultaneously |

REST Endpoints:
```
POST  /api/v1/workflows/{execution_id}/predict   -> Live inference on JSON input
GET   /api/v1/workflows/{execution_id}/schema    -> Dynamic feature schema discovery
POST  /api/v1/workflows/predict                  -> Convenience endpoint for API gateways
```

### 5.8 MLflow Model Governance
- All training runs are tracked via **MLflow Model Registry**
- Captured: hyperparameters, evaluation metrics, artifact paths, model stage (Staging / Production)
- Enables full model audit trail and rollback

### 5.9 Authentication & Multi-Tenancy
- **JWT-based authentication** — every API request carries a Bearer token
- **Multi-tenant data isolation** — tenant ID is enforced at the SQLAlchemy ORM layer via before_flush hooks in tenant_guard.py
- Each tenant's data, workflows, and experiments are fully isolated from other tenants

### 5.10 Visual Data Prep
Data preparation is a first-class citizen. Users visually connect prep nodes on the canvas:
- **Join:** Merge two datasets (inner/left/right/outer) by key columns, with separate output handles for matched/unmatched rows
- **Filter / Split:** Row-level filtering; train-test split by ratio or time boundary
- **Imputation:** Fill missing values (mean/median/mode/constant/KNN)
- **Encoding:** Label, One-Hot, Target, Ordinal, Frequency, Binary encoding
- **Scaling:** StandardScaler, MinMaxScaler, RobustScaler
- **Outlier Handling:** IQR capping, Z-score removal
- **Feature Selection:** Correlation threshold, mutual information, top-K importance
- **NLP:** TF-IDF / Count vectorizer, text cleaning, chunking

---

## 6. Execution Flow (End to End)

```
1. User builds workflow on canvas (nodes + edges)
2. Frontend POSTs JSON payload to:
      POST /api/v1/workflows/{workflow_id}/execute
3. Backend parses DAG -> validates -> topological sort
4. Root nodes (CSV loaders / dataset loaders) fetch data from Ozone/DB
5. Each node executes its Recipe in topological order:
      output_df = recipe.execute(input_df, config)
6. Output DataFrames are passed to the next connected node via handle routing
7. Trained models + transformers are bundled into InferenceBundle
8. Execution result (per-node status, metrics) returned to frontend
9. Frontend renders node-level status, logs, and output previews
10. User hits /predict endpoint to perform live inference
```

---

## 7. Storage Architecture

| Layer | Technology | Purpose |
|---|---|---|
| **Relational State** | PostgreSQL (prod) / SQLite (dev) | Workflows, datasets metadata, execution history, user/tenant data |
| **Object Storage** | Apache Ozone (S3-compatible) | Raw dataset files, intermediate artifacts, serialized model binaries |
| **In-Memory Execution** | Pandas DataFrames | Live node-to-node data passing during DAG execution |
| **ML Experiment Tracking** | MLflow + mlflow.db | Hyperparameters, metrics, model registry |

---

## 8. Integration Layer

| Integration | Description |
|---|---|
| **Gmail Notifier** | Sends email notifications with execution results via Gmail API |
| **Webhook Trigger** | Starts a workflow execution from an external HTTP event |
| **Cron / Scheduled Trigger** | Time-based automated pipeline execution |
| **Tenant Data Connectors** | Druid, Iceberg, SFTP, Azure Graph, Twilio |

---

## 9. Technology Stack

| Layer | Technology |
|---|---|
| **API Framework** | FastAPI (Python 3.12) |
| **Data Processing** | Pandas, NumPy |
| **ML Libraries** | Scikit-learn, XGBoost, LightGBM, CatBoost, Prophet, statsmodels |
| **NLP** | NLTK, Sentence-Transformers, TF-IDF |
| **Model Registry** | MLflow |
| **Database ORM** | SQLAlchemy (async) |
| **Authentication** | JWT (python-jose), Fernet encryption |
| **Object Storage** | Apache Ozone (boto3 S3-compatible client) |
| **LLM Integration** | Groq API (openai/gpt-oss-120b) |
| **Process Manager** | PM2 (production server) |
| **CI/CD** | GitHub Actions, Azure Pipelines |
| **Frontend (separate)** | React Flow + Next.js |

---

## 10. API Surface Summary

| Endpoint Group | Description |
|---|---|
| POST /api/v1/auth/login | Obtain JWT token |
| GET/POST /api/v1/datasets | Upload, list, preview, profile datasets |
| GET/POST /api/v1/workflows | Create, list, fetch workflow definitions |
| POST /api/v1/workflows/{id}/execute | Execute a workflow DAG |
| GET /api/v1/workflows/{execution_id}/schema | Discover inference input schema |
| POST /api/v1/workflows/{execution_id}/predict | Run live model inference |
| GET /api/v1/recipes | Browse available recipe catalog |
| GET /api/v1/recipes/{id}/schema | Get recipe parameter schema (for dynamic UI) |
| GET /api/v1/experiments | Browse MLflow experiment runs |
| GET /api/v1/profiling/{dataset_id} | Retrieve dataset profile/health report |
| POST /api/v1/recommendation | Get AI-recommended pipeline for a dataset |

---

## 11. Deployment

- **Server:** Ubuntu VPS managed via PM2 (ecosystem.config.js)
- **Domain:** https://ml.datalect.app
- **CI/CD:** Push to pipeline_salman branch -> GitHub Actions -> SSH deploy -> pm2 restart ML-Pipeline-Platform --update-env
- **Environment:** Configured via .env with DEBUG=False in production
- **Storage:** Apache Ozone at https://s3-oz.datalect.app with bucket mybucket, prefix ml-workflow

---

## 12. Summary Table

| What | Detail |
|---|---|
| **Type** | Enterprise Visual AI/ML Orchestration Platform |
| **Core Value** | No-code ML pipeline building via drag-and-drop canvas |
| **Backend Role** | 100% of ML logic, DAG execution, data processing, inference, auth, multi-tenancy |
| **Frontend Role** | Canvas rendering, form generation (schema-driven), result visualization |
| **Supported ML Tasks** | Classification, Regression, Forecasting, Anomaly Detection, NLP/RAG |
| **Supported Data** | CSV, Excel, JSON, Parquet; Tabular + Unstructured |
| **Users** | Data scientists, business analysts, ML engineers inside enterprise tenants |
| **Production URL** | https://ml.datalect.app |
