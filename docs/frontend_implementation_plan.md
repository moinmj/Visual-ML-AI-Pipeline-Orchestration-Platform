# Visual ML/AI Pipeline Platform — Frontend Implementation & Integration Plan

This document serves as the complete, end-to-end technical implementation plan for frontend developers integrating with the **Visual ML/AI Pipeline Orchestration Platform** FastAPI backend.

---

## 1. System Architecture & Component Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                 FRONTEND APPLICATION                                   │
├─────────────────────────┬─────────────────────────────┬─────────────────────────────────┤
│  1. DATASET STUDIO      │  2. VISUAL DAG STUDIO       │  3. RESULTS & DIAGNOSTICS       │
│  - Dataset Uploader     │  - Flow Canvas (React Flow) │  - Executive KPI Cards          │
│  - Table Preview & Pager│  - Recipe Palette / Catalog │  - Confusion Matrix (Plotly)    │
│  - Statistical Profiler │  - Node Inspector (JSONForm)│  - Feature Importances Bar Chart│
│  - Quality Health Badge │  - Auto-Wire Engine         │  - Time-Series Trajectory Plot  │
│  - AI Dataset Advisor   │  - AI One-Click Pipeline    │  - 2D Anomaly Scatter & Hist    │
│                         │  - Pre-flight Guardrails    │  - Step-by-Step Data Stream HUD │
├─────────────────────────┴─────────────────────────────┴─────────────────────────────────┤
│  4. LIVE API FLIGHT RECORDER & TELEMETRY                                                │
│  - Real-time HTTP Request/Response Inspector, Payloads, Status Codes, Execution Latency │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                          │  REST API / JSON
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                             FASTAPI BACKEND (Port 8000)                                 │
│  ├── /api/v1/datasets   (Upload, Metadata, Preview, Statistical Profile)               │
│  ├── /api/v1/recipes    (19+ Catalog Components, Categories, JSON Schemas)             │
│  ├── /api/v1/workflows  (DAG Validation, Sync/Async Execute, Job Worker Pool, Triggers)│
│  └── /docs & /openapi.json (Interactive Swagger Specification)                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Recommended Frontend Technology Stack

| Layer | Recommended Technologies |
|---|---|
| **Framework** | Next.js 14+ (App Router) / React 18+ / Vite + React |
| **DAG Flow Canvas** | `@xyflow/react` (React Flow v12+) or `@vue-flow/core` |
| **UI Components & Icons** | Shadcn UI / Radix UI, TailwindCSS, Lucide React Icons |
| **Interactive Charts** | `plotly.js-dist-min` + `react-plotly.js` or `echarts-for-react` |
| **Data Fetching & Caching** | `@tanstack/react-query` (TanStack Query v5) + `axios` |
| **Dynamic Form Generator** | `@rjsf/core` (React JSON Schema Form) or `@hookform/resolvers` + `zod` |
| **State Management** | `zustand` (Canvas nodes/edges, active dataset, telemetry logs) |

---

## 3. TypeScript Interfaces & Data Contracts

### 3.1. Workflow & DAG Models

```typescript
export interface WorkflowNode {
  id: string;                      // e.g. "node_csv_1", "node_impute_2"
  recipe_id: string;               // e.g. "csv_loader", "missing_value_imputer", "xgboost_trainer"
  label?: string;                  // Display title on the canvas
  config: Record<string, any>;     // Key-value configuration matching recipe JSON schema
  position?: { x: number; y: number }; // Canvas coordinate (managed by React Flow)
}

export interface WorkflowEdge {
  id?: string;
  source: string;                  // Source node id
  target: string;                  // Target node id
  source_handle?: string;
  target_handle?: string;
  animated?: boolean;
}

export interface WorkflowGraph {
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
}

export interface DAGValidationResponse {
  valid: boolean;
  errors: string[];                // Cycle warnings, orphan node notices, type mismatches
}
```

### 3.2. Execution Results & Diagnostic Snapshots

```typescript
export interface NodeExecutionResult {
  node_id: string;
  recipe_id: string;
  status: "SUCCESS" | "FAILED" | "SKIPPED";
  duration_ms: number;
  error_message?: string | null;
  output_summary: Record<string, any>;
}

export interface StepSnapshot {
  node_id: string;
  recipe_id: string;
  duration_ms: number;
  row_count: number;
  column_count: number;
  columns: string[];
  sample_records: Record<string, any>[]; // Top 5 preview rows
  schema_types: Record<string, string>;  // e.g. { "Age": "float64", "Churn": "int64" }
}

export interface WorkflowExecutionResult {
  execution_id: string;
  status: "SUCCESS" | "FAILED";
  total_duration_ms: number;
  node_results: NodeExecutionResult[];
  final_metrics?: Record<string, any> | null;
  anomaly_summary?: Record<string, any> | null;
  forecasting_summary?: Record<string, any> | null;
  governance_summary?: Record<string, any> | null;
  logs: string[];
  node_outputs: Record<string, any>;
  step_snapshots: Record<string, StepSnapshot>;
}
```

---

## 4. End-to-End REST API Integration Specifications

### 4.1. Dataset Management (`/api/v1/datasets`)

#### 1. Upload Dataset
- **Endpoint**: `POST /api/v1/datasets/upload`
- **Content-Type**: `multipart/form-data`
- **Request Body**:
  - `file`: Binary file (CSV, XLSX, Parquet, JSON)
  - `name`: String (optional)
  - `description`: String (optional)
- **Response**: `201 Created`
  ```json
  {
    "id": "ds_8f21bc9e",
    "name": "Customer Churn Data",
    "file_format": "csv",
    "row_count": 1000,
    "column_count": 12,
    "file_size_bytes": 45120,
    "created_at": "2026-08-27T12:00:00Z"
  }
  ```

#### 2. List All Datasets
- **Endpoint**: `GET /api/v1/datasets/`
- **Response**: `200 OK` → `DatasetResponse[]`

#### 3. Dataset Table Preview
- **Endpoint**: `GET /api/v1/datasets/{dataset_id}/preview?limit=20`
- **Response**: `200 OK`
  ```json
  {
    "dataset_id": "ds_8f21bc9e",
    "total_rows": 1000,
    "columns": ["CustomerID", "Age", "MonthlyCharges", "Churn"],
    "data": [
      { "CustomerID": 1, "Age": 42, "MonthlyCharges": 70.35, "Churn": 1 },
      { "CustomerID": 2, "Age": 28, "MonthlyCharges": 29.85, "Churn": 0 }
    ]
  }
  ```

#### 4. Statistical Profiling & Quality Health
- **Endpoint**: `GET /api/v1/datasets/{dataset_id}/profile`
- **Response**: `200 OK`
  ```json
  {
    "dataset_id": "ds_8f21bc9e",
    "row_count": 1000,
    "column_count": 12,
    "total_missing_cells": 14,
    "missing_percentage": 0.12,
    "duplicate_rows": 0,
    "columns": {
      "Age": {
        "inferred_type": "numeric",
        "missing_count": 0,
        "unique_count": 52,
        "mean": 38.4,
        "std": 12.1,
        "min": 18,
        "max": 80
      },
      "Churn": {
        "inferred_type": "categorical",
        "unique_count": 2,
        "top_values": { "0": 700, "1": 300 }
      }
    },
    "data_quality_score": 98.8
  }
  ```

---

### 4.2. Recipe Catalog & Dynamic Inspector (`/api/v1/recipes`)

#### 1. List Registered Components
- **Endpoint**: `GET /api/v1/recipes/?category=training` (category optional)
- **Response**: `200 OK` → Array of:
  ```json
  [
    {
      "recipe_id": "xgboost_trainer",
      "name": "XGBoost Classifier / Regressor",
      "version": "1.0.0",
      "category": "training",
      "description": "High-performance gradient boosting tree model.",
      "input_types": ["split_dataset"],
      "output_types": ["trained_model", "metrics", "feature_importances"]
    },
    {
      "recipe_id": "prophet_forecaster",
      "name": "Prophet Forecaster",
      "version": "1.0.0",
      "category": "forecasting",
      "description": "Additive regression time-series forecasting with trend & seasonality.",
      "input_types": ["dataframe"],
      "output_types": ["forecast_df", "forecasting_summary"]
    }
  ]
  ```

#### 2. Get Recipe JSON Schema for Dynamic Node Inspector
- **Endpoint**: `GET /api/v1/recipes/{recipe_id}/schema`
- **Response**: `200 OK`
  ```json
  {
    "recipe_id": "xgboost_trainer",
    "name": "XGBoost Classifier / Regressor",
    "parameters_schema": {
      "type": "object",
      "properties": {
        "task_type": {
          "type": "string",
          "title": "Task Type",
          "enum": ["classification", "regression"],
          "default": "classification"
        },
        "n_estimators": {
          "type": "integer",
          "title": "Number of Estimators (Trees)",
          "default": 100,
          "minimum": 10,
          "maximum": 2000
        },
        "learning_rate": {
          "type": "number",
          "title": "Learning Rate (Eta)",
          "default": 0.1,
          "minimum": 0.001,
          "maximum": 1.0
        },
        "max_depth": {
          "type": "integer",
          "title": "Max Tree Depth",
          "default": 6,
          "minimum": 1,
          "maximum": 20
        }
      },
      "required": ["task_type"]
    }
  }
  ```

---

### 4.3. Workflow Graph Validation & Execution (`/api/v1/workflows`)

#### 1. Pre-Flight Graph Validation
- **Endpoint**: `POST /api/v1/workflows/validate`
- **Request Body**: `WorkflowGraph`
- **Response**: `200 OK`
  ```json
  {
    "valid": false,
    "errors": [
      "⚠️ Incompatible Connection for 'xgboost_trainer_5': Model trainers expect split partitions (X_train, y_train). Currently connected directly to 'csv_loader_1'."
    ]
  }
  ```

#### 2. Synchronous DAG Execution
- **Endpoint**: `POST /api/v1/workflows/execute`
- **Request Body**: `WorkflowGraph`
- **Response**: `200 OK` → `WorkflowExecutionResult` (Includes all node outputs, KPI metrics, charts, step snapshots).

#### 3. Asynchronous Background Execution (Worker Pool Pattern)
- **Endpoint**: `POST /api/v1/workflows/async-execute`
- **Request Body**: `WorkflowGraph`
- **Response**: `202 Accepted`
  ```json
  {
    "job_id": "job_9b3c4a21",
    "status": "PENDING",
    "message": "Workflow execution dispatched to background worker pool."
  }
  ```

#### 4. Poll Job Status
- **Endpoint**: `GET /api/v1/workflows/jobs/{job_id}`
- **Response**: `200 OK`
  ```json
  {
    "job_id": "job_9b3c4a21",
    "status": "COMPLETED",
    "created_at": "2026-08-27T12:00:00Z",
    "completed_at": "2026-08-27T12:00:02Z",
    "result": { /* Full WorkflowExecutionResult */ }
  }
  ```

#### 5. Inbound Webhook Trigger
- **Endpoint**: `POST /api/v1/workflows/trigger/{webhook_path}`
- **Request Body**: JSON array or single object
- **Response**: `200 OK`

---

### 4.4. AI Recommendation & Auto-Architect (`/api/v1/recommend`)

#### 1. One-Click AI Pipeline Recommendation
- **Endpoint**: `POST /api/v1/recommend/pipeline`
- **Request Body**:
  ```json
  {
    "dataset_id": "ds_8f21bc9e",
    "target_column": "churn",
    "task_type": "classification",
    "preset": "balanced"
  }
  ```
- **Response**: `200 OK`
  ```json
  {
    "task_type": "classification",
    "explanation": "Detected discrete classification on `churn` (2 unique classes).",
    "target_column": "churn",
    "profile_summary": {
      "rows": 1000,
      "columns": 12,
      "missing_cells": 15,
      "categorical_columns": 4,
      "numeric_columns": 8,
      "date_columns": 0
    },
    "preprocessing_recommendations": [
      {
        "recipe_id": "missing_value_imputer",
        "name": "🧹 Missing Value Imputer",
        "config": { "strategy": "median" },
        "reason": "Dataset contains 15 missing cells requiring imputation."
      }
    ],
    "model_rankings": [
      {
        "recipe_id": "xgboost_trainer",
        "name": "⚡ XGBoost Classification",
        "score": 9.8,
        "tier": "Tier-1 Gold Standard",
        "reason": "Highest accuracy and regularized gradient boosting for tabular datasets."
      },
      {
        "recipe_id": "lightgbm_trainer",
        "name": "🚀 LightGBM Classification",
        "score": 9.5,
        "tier": "Tier-1 High Speed",
        "reason": "Optimal for ultra-fast training with histogram-based leaf growth."
      }
    ],
    "recommended_dag": {
      "nodes": [
        { "id": "node_csv", "recipe_id": "csv_loader", "label": "📄 Data Ingestion", "position": { "x": 40, "y": 100 }, "config": { "dataset_id": "ds_8f21bc9e" } },
        { "id": "node_prep_1", "recipe_id": "missing_value_imputer", "label": "🧹 Missing Value Imputer", "position": { "x": 280, "y": 100 }, "config": { "strategy": "median" } },
        { "id": "node_split", "recipe_id": "train_test_split", "label": "✂️ Train/Test Split", "position": { "x": 520, "y": 100 }, "config": { "target_column": "churn", "test_size": 0.2 } },
        { "id": "node_model", "recipe_id": "xgboost_trainer", "label": "⚡ XGBoost Trainer", "position": { "x": 760, "y": 50 }, "config": { "task_type": "classification", "n_estimators": 100 } },
        { "id": "node_eval", "recipe_id": "classification_evaluator", "label": "🎯 Model Evaluator", "position": { "x": 1000, "y": 100 }, "config": { "report_type": "Comprehensive" } }
      ],
      "edges": [
        { "id": "e_node_csv_node_prep_1", "source": "node_csv", "target": "node_prep_1", "animated": true },
        { "id": "e_node_prep_1_node_split", "source": "node_prep_1", "target": "node_split", "animated": true },
        { "id": "e_node_split_node_model", "source": "node_split", "target": "node_model", "animated": true },
        { "id": "e_node_split_node_eval", "source": "node_split", "target": "node_eval", "animated": true },
        { "id": "e_node_model_node_eval", "source": "node_model", "target": "node_eval", "animated": true }
      ],
      "node_configs": { ... }
    }
  }
  ```

#### 2. Auto-Wiring Endpoint
- **Endpoint**: `POST /api/v1/recommend/autowire` (or `POST /api/v1/workflows/autowire`)
- **Request Body**:
  ```json
  {
    "nodes": [
      { "id": "node_csv", "recipe_id": "csv_loader", "position": { "x": 40, "y": 100 } },
      { "id": "node_scale", "recipe_id": "feature_scaler", "position": { "x": 280, "y": 100 } },
      { "id": "node_model", "recipe_id": "xgboost_trainer", "position": { "x": 520, "y": 100 } }
    ]
  }
  ```
- **Response**: `200 OK` → `{ "status": "AUTOWIRED", "nodes_count": 3, "edges_count": 2, "edges": [ ... ] }`

---

### 4.5. Pipeline Templates & Blueprints (`/api/v1/templates`)

#### 1. List Available Templates
- **Endpoint**: `GET /api/v1/templates/`
- **Response**: `200 OK`
  ```json
  [
    { "id": "ml_supervised", "name": "Supervised ML Classification / Regression", "category": "Machine Learning", "icon": "⚡", "node_count": 6 },
    { "id": "time_series_forecasting", "name": "Time-Series Seasonality & Trend Forecasting", "category": "Time-Series", "icon": "🔮", "node_count": 3 },
    { "id": "anomaly_detection", "name": "Unsupervised Anomaly & Outlier Detection", "category": "Anomaly Detection", "icon": "🚨", "node_count": 3 },
    { "id": "enterprise_governance", "name": "Enterprise CatBoost with Model Governance Card", "category": "Governance & Compliance", "icon": "🛡️", "node_count": 7 }
  ]
  ```

#### 2. Get Specific Blueprint DAG
- **Endpoint**: `GET /api/v1/templates/{template_id}`
- **Examples**: `/api/v1/templates/ml_supervised`, `/api/v1/templates/time_series_forecasting`, `/api/v1/templates/anomaly_detection`
- **Response**: `200 OK` → Returns the full ready-to-render DAG (`dag.nodes`, `dag.edges`, `dag.node_configs`).

---

### 4.6. Saved Workflow Workbooks & Soft-Delete CRUD (`/api/v1/workflows`)

- `POST /api/v1/workflows/` — Save / create pipeline workbook.
- `GET /api/v1/workflows/` — List saved workbooks (`include_deleted=true` to retrieve trash).
- `GET /api/v1/workflows/{id}` — Load specific workbook with exact configurations.
- `PUT /api/v1/workflows/{id}` — Upsert / update existing workbook.
- `DELETE /api/v1/workflows/{id}` — Soft-delete workbook (archives it).
- `POST /api/v1/workflows/{id}/restore` — Restore soft-deleted workbook.

---

## 5. UI Views & Visual Component Specifications

### 5.1. View 1: 🎨 Visual Pipeline Studio (Canvas)

1. **Left Component Palette**:
   - Categorized accordion: `Ingestion`, `Preprocessing`, `Feature Engineering`, `Splitting`, `Training (The Triumvirate)`, `Evaluation`, `Anomaly Detection`, `Time-Series Forecasting`, `Triggers`.
   - Drag-and-drop or click-to-spawn nodes onto canvas.
2. **Top Action Toolbar**:
   - `▶️ Run Pipeline`: Triggers pre-flight validation and execution.
   - `🧠 AI Recommend`: Inspects active dataset and generates 1-click end-to-end DAG.
   - `⚡ ML Supervised`: Ingestion ➔ Imputer ➔ Scaler ➔ Split ➔ XGBoost ➔ Evaluator.
   - `🔮 Time-Series Forecast`: Ingestion ➔ Time Imputer ➔ Prophet Forecaster.
   - `🚨 Anomaly Detection`: Ingestion ➔ Imputer ➔ Isolation Forest.
   - `🔗 Auto-Wire`: Automatically links sequential nodes from left to right.
   - `🧹 Clear Canvas`: Clears whiteboard with confirmation modal.
3. **Canvas Node Design (React Flow Custom Node)**:
   - Header with category color token:
     - Preprocessing: `#3B82F6` (Blue)
     - Training: `#8B5CF6` (Purple)
     - Anomaly: `#EF4444` (Red)
     - Forecasting: `#10B981` (Emerald)
     - Triggers: `#F59E0B` (Amber)
   - Status Badge: `IDLE` (Gray), `RUNNING` (Spinning Amber), `SUCCESS` (Green Check), `FAILED` (Red Cross).
   - Input Handle (Left) and Output Handle (Right).
4. **Right Node Inspector Panel**:
   - Displays selected node configuration.
   - Dynamically renders inputs according to `/api/v1/recipes/{recipe_id}/schema`.
   - For `type: "array"`, render multi-select tags populated with active dataset column names.

---

### 5.2. View 2: 📊 Execution & Diagnostic Results Dashboards

Rendered below canvas upon pipeline run:

#### 1. Machine Learning Supervised Report
- **Executive KPIs**: 5 metric cards (`Accuracy`, `Balanced Accuracy`, `Precision`, `Recall`, `F1 Score` or `R2 Score`, `MAE`, `RMSE`).
- **Confusion Matrix Heatmap**: Interactive Plotly heatmap with custom labels and actual vs predicted counts.
- **Classification Report**: Per-class precision, recall, and support table.
- **Feature Importances**: Horizontal bar chart sorted by feature contribution.

#### 2. Time-Series Forecasting Dashboard
- **Metric Cards**: `Historical Periods`, `Forecast Horizon`, `Trend Direction` (Upward/Downward), `MAE`, `RMSE`, `MAPE`.
- **Plotly Trajectory Plot**: Line graph combining historical actuals, forecasted trend (`yhat`), and shaded 95% confidence bands (`yhat_lower` to `yhat_upper`).
- **Tabular Forecast Matrix**: Data table of forecasted future intervals with CSV download.

#### 3. Anomaly Detection & Outlier Risk Matrix
- **Metric Cards**: `Total Records Inspected`, `Anomalies Flagged`, `Anomaly Percentage`.
- **2D Risk Cluster Scatter Plot**: Scatter plot with interactive X/Y feature select dropdowns; data points color-coded into `Normal Record` (Blue) vs `Flagged Anomaly` (Red).
- **Anomaly Score Histogram**: Distribution of risk scores (0.0 to 1.0).
- **Flagged Anomalies Table & CSV Download**: Data grid showing exclusively anomalous rows.

#### 4. Step-by-Step Data Stream Inspector & Node Debugger (n8n / Boomi HUD)
- Dropdown selector to pick any executed DAG node.
- Schema Diff view (Columns added/removed, dtypes transformed).
- Intermediate 5-row preview table of the exact DataFrame emitted at that stage.
- Millisecond execution duration badge.

---

### 5.3. View 3: 📡 Live API Flight Recorder HUD
- Floating drawer or bottom panel.
- Intercepts all outgoing backend HTTP calls.
- Shows timestamp, HTTP method, endpoint URL, latency in ms, payload preview, and status code (200, 201, 422, 500).

---

## 6. Implementation Milestones for Frontend Developer

| Phase | Deliverables | Est. Timeline |
|---|---|---|
| **Phase 1: Project Setup & API Client** | Axios/TanStack client, TypeScript interfaces, Tailwind theme, Layout shell | Day 1 |
| **Phase 2: Dataset Studio & Profiler** | File uploader, table preview, column health indicators, AI Advisor banner | Day 2 |
| **Phase 3: React Flow Canvas & Catalog**| Node palette, custom recipe nodes, auto-wire engine, pre-flight validation | Day 3–4 |
| **Phase 4: Node Inspector & JSON Forms** | Dynamic schema form renderer, column multiselects, preset template buttons | Day 5 |
| **Phase 5: Diagnostic Dashboards & Plots**| Plotly charts (Confusion matrix, feature importances, forecast line, anomaly scatter) | Day 6–7 |
| **Phase 6: Data Stream Inspector & Telemetry**| Intermediate DataFrame debugger, live API Flight Recorder drawer, polishing | Day 8 |

---

## 7. Verification & Testing Checklist

- [ ] Uploading a CSV, XLSX, or Parquet dataset triggers automatic profiling and column metric calculation.
- [ ] Dragging recipe nodes and connecting incompatible handles displays pre-flight warning banners without crashing.
- [ ] One-click templates (**⚡ ML Supervised**, **🔮 Forecast**, **🚨 Anomaly**) populate the canvas and auto-execute.
- [ ] Supervised classification renders executive KPIs, confusion matrix, classification report, and feature importances.
- [ ] Forecasting template renders confidence intervals (`yhat_lower`/`yhat_upper`) and future projection line graphs.
- [ ] Anomaly template renders 2D scatter plots, score distributions, and filtered anomalous records table.
- [ ] Step-by-Step Data Stream inspector accurately displays intermediate DataFrames and latency metrics.
- [ ] Live API Flight Recorder logs all backend REST transactions in real-time.
