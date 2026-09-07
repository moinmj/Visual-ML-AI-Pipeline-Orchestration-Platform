# Project Master Context: Visual AI/ML Pipeline Orchestration Platform

## 1. Executive Summary & Vision

The **Visual AI/ML Pipeline Orchestration Platform** is an enterprise-grade, extensible data science, machine learning, and AI workflow automation system inspired by capabilities found in **Dataiku**, **n8n**, and **Boomi**.

The platform enables technical and non-technical practitioners to visually build, configure, validate, execute, and monitor end-to-end data processing pipelines, traditional predictive ML workflows, and GenAI / RAG applications through a drag-and-drop canvas.

---

## 2. Engineering Boundaries & Architectural Roles

```
               +-------------------------------------------------------+
               |                  FRONTEND DEVELOPER                   |
               |      (Next.js / React Flow / Visual Canvas / Forms)   |
               +-------------------------------------------------------+
                                          |
                                          | REST APIs / WebSocket Streams / OpenAPI Contract
                                          v
+-----------------------------------------------------------------------------------------+
|                              BACKEND PLATFORM & AI ENGINE                               |
|                                (Python / FastAPI / MLOps)                                |
|                                                                                         |
|  +-----------------------------------------------------------------------------------+  |
|  | Application Layer (FastAPI)                                                       |  |
|  | - /auth, /projects, /datasets, /recipes, /workflows, /executions, /experiments     |  |
|  +-----------------------------------------------------------------------------------+  |
|                                          |                                              |
|  +-----------------------------------------------------------------------------------+  |
|  | Core Engines                                                                      |  |
|  | - Recipe Engine: Schema-driven metadata, parameter validation, modular executors   |  |
|  | - DAG Engine: Cycle detection, topological sort, compatibility verification        |  |
|  | - Data Profiler: Automated schema inference, distributions, health checks         |  |
|  | - Execution Engine: Deterministic context, artifact passing, state tracking       |  |
|  | - ML Engine: Feature transformation, Scikit-learn, XGBoost, Model evaluation       |  |
|  +-----------------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------------+
                               |                         |
                               v                         v
                   PostgreSQL / SQLite               MinIO / S3
                   (Relational State)             (Artifacts & Data)
```

### Key Principles:
1. **Backend is the Single Source of Truth:**
   The frontend has zero business logic for ML algorithms, data transformations, or DAG validation.
2. **Schema-Driven Dynamic UI:**
   Every recipe exposes its own JSON configuration schema so the frontend renders forms automatically without custom coding per node.
3. **Strict API Contracts:**
   All endpoints use Pydantic models with explicit request/response schemas.
4. **Independent Recipe Units:**
   Every recipe implements a standardized `BaseRecipe` interface and can be executed and unit-tested in complete isolation from HTTP handlers.

---

## 3. Core Domain Entities

* **Project:** Workspaces isolating datasets, workflows, experiments, and model artifacts.
* **Dataset:** Tabular or unstructured data uploaded by users, stored in object storage, and indexed with automated statistical profiles.
* **Recipe:** A self-contained, parameterized transformation or modeling step (e.g., *Imputer*, *StandardScaler*, *XGBoost*, *PDFChunker*).
* **Workflow:** A Directed Acyclic Graph (DAG) consisting of parameterized **Nodes** and connective **Edges**.
* **Execution:** A single runtime instance of a workflow tracking node-by-node status, logs, duration, and intermediate output artifacts.
* **Experiment & Model:** Model tracking metadata (hyperparameters, metrics like ROC-AUC / F1, confusion matrices, and serialized model binaries).

---

## 4. Supported Data Types & Capabilities

### Tabular / Structured:
* CSV, Excel (XLSX/XLS), JSON lines, Parquet.
* Preprocessing: Missing value imputation, duplicate removal, outlier handling, categorical encoding, feature scaling.
* Feature Engineering & Selection: Correlation filtering, mutual information, mathematical expressions.
* Modeling: Logistic Regression, Random Forest, XGBoost, LightGBM.
* Evaluation: Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix, MAE, MSE, R².

### Unstructured / GenAI (Phase 2):
* PDF, DOCX, TXT, OCR.
* Text chunking, vector embeddings, similarity retrieval, and LLM inference.
