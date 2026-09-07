# Roadmap and Task Tracker

All development progress is categorized into structured phases. Each task is marked with `[x]` (Completed) or `[ ]` (Pending).

---

## Phase 1: Environment & Context Setup
- [x] Create project `.gitignore`, `.env.example`, `.env`, and `requirements.txt`.
- [x] Create `docs/PROJECT_MASTER_CONTEXT.md`.
- [x] Create `docs/ROADMAP_AND_TASKS.md`.
- [x] Create `docs/CURRENT_STATE.md`.
- [x] Create `docs/api/api-overview.md`.

---

## Phase 2: Backend Core & Infrastructure Setup
- [x] Implement `backend/app/core/config.py` (Pydantic Settings).
- [x] Implement `backend/app/core/logging.py` (Structured Logging).
- [x] Implement `backend/app/core/exceptions.py` (Custom Exception Handlers).
- [x] Implement `backend/app/infrastructure/database/session.py` (Async DB Engine & Base).
- [x] Implement `backend/app/infrastructure/storage/storage_manager.py` (Local File + S3/MinIO Storage).

---

## Phase 3: Data Management & Automated Data Profiler (Data Layer First)
- [x] Implement Dataset database models (`Dataset`).
- [x] Implement Dataset Pydantic schemas (`DatasetCreate`, `DatasetResponse`, `DatasetPreviewResponse`, `DatasetProfileResponse`).
- [x] Implement **Automated Data Profiler**:
  - [x] Column type detection (numeric, categorical, datetime, text, boolean).
  - [x] Row and column count computation.
  - [x] Missing values analysis (counts & percentages).
  - [x] Duplicate row detection.
  - [x] Statistical summary (mean, std, min, 25%, 50%, 75%, max).
  - [x] Categorical cardinality and top value frequencies.
  - [x] Data Quality Health Score calculation (0-100%).
- [x] Implement Dataset REST API Router (`/api/v1/datasets`):
  - [x] `POST /upload` (Upload CSV, XLSX, JSON).
  - [x] `GET /` (List datasets).
  - [x] `GET /{id}` (Dataset metadata).
  - [x] `GET /{id}/preview` (Top N preview rows).
  - [x] `GET /{id}/profile` (Detailed statistical profile).
  - [x] `DELETE /{id}` (Remove dataset and files).

---

## Phase 4: Machine Learning Recipe Engine & Task Families

### 4.1 Tabular Prediction (Classification & Regression)
- [x] `CSVLoaderRecipe` (Ingestion).
- [x] `MissingValueImputerRecipe` (Mean, median, mode, constant).
- [x] `FeatureScalerRecipe` (StandardScaler, MinMaxScaler, RobustScaler with auto-target protection).
- [x] `CategoricalEncoderRecipe` (One-Hot, Label encoding with target preservation).
- [x] `TrainTestSplitRecipe` (Train/Test partitioning).
- [x] `LogisticRegressionTrainerRecipe` (Linear classification & regression).
- [x] `RandomForestTrainerRecipe` (Ensemble trees).
- [x] `XGBoostTrainerRecipe` (Extreme Gradient Boosting with auto-categorical fallback).
- [x] `LightGBMTrainerRecipe` (Histogram-based leaf-wise gradient boosting with JSON-safe feature sanitization).
- [x] `CatBoostTrainerRecipe` (Oblivious decision trees with native high-cardinality categorical processing).
- [x] `ModelEvaluatorRecipe` (Accuracy, Balanced Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix heatmap, and per-class reports).

### 4.2 Anomaly Detection (Unsupervised & Statistical)
- [x] `IsolationForestRecipe` (Unsupervised tree isolation, continuous anomaly score 0-1, inlier/outlier classification).
- [x] `StatisticalGuardrailRecipe` (Z-Score std-dev & Tukey IQR outlier detection and filtering).
- [x] Anomaly Visualization & Dashboard (Scatter plots, score distribution histograms, outlier inspection table).

### 4.3 Time-Series Forecasting
- [x] `LagFeatureEngineeringRecipe` (Rolling statistics, lag steps `t-1` to `t-n`, calendar features).
- [x] `ProphetForecasterRecipe` (Meta Prophet trend, weekly/yearly seasonality, and holidays).
- [x] `ARIMAForecasterRecipe` (Statsmodels SARIMAX univariate forecasting with AIC/BIC).
- [x] Time-Series Visualization Dashboard (Historical actuals vs future predictions + 95% shaded confidence bands).

---

## Phase 5: AI Recommendation Engine (Section 8 of Architecture Spec)
- [x] `AIRecommender` engine: Problem diagnosis, rule-based filtering, algorithm ranking scorecards.
- [x] 1-Click `🧠 AI Recommend` whiteboard DAG builder.
- [x] Dedicated `🧠 AI Pipeline Recommender` Studio tab.

---

## Phase 6: DAG Engine & Visual Whiteboard
- [x] Cycle detection (Kahn's algorithm) and topological sorting.
- [x] In-memory fault-tolerant pipeline executor with shared context.
- [x] Streamlit + React Flow (`streamlit-flow`) interactive 2D whiteboard.
- [x] Node-level Code View (`to_code()`) toggle in Inspector.
- [x] 1-Click Line Connector & Auto-Wire functionality.
- [x] Dynamic Canvas Versioning (`canvas_version`) for seamless node additions.
- [x] 19/19 Automated unit and integration tests passing.

---

## Phase 7: Model & Data Governance (MLflow & Serving)
- [x] `MLflowTrackerRecipe` (Experiment logging for parameters, evaluation metrics, and artifacts).
- [x] Model Registry integration (`models:/name/Production` and `Staging` stage tags).
- [x] Governance Audit Card in Streamlit UI.
- [ ] `PretrainedModelInferenceRecipe` (Batch inference on registered model URIs without retraining).
