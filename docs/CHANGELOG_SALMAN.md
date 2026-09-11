# Master Development Portfolio & Changelog — Author: `mighatenosak`

This document provides a comprehensive, unified record of all recipes, features, bug fixes, DAG engine enhancements, and architectural contributions authored by **`mighatenosak`** across the ML Pipeline Platform repository (auditable via GitLens).

---

## 1. Executive Summary

| Category | Highlights & Added Modules |
| :--- | :--- |
| **New NLP & Text Suite** | Built `TextPreprocessorRecipe` and `TextVectorizerRecipe` with TF-IDF, Count/BoW, n-grams, stopword filtering, and token normalization. |
| **Data Cleaning & Redundancy Suite** | Built `DuplicateRowsRemoverRecipe`, `CategorySanitizerRecipe`, `HighCorrelationFilterRecipe`, and `LowVarianceFilterRecipe`. |
| **Advanced Feature Encoding** | Expanded `CategoricalEncoderRecipe` to support **7 distinct encoding algorithms** (`one_hot`, `label`, `ordinal`, `target`, `frequency`, `binary`, `woe`). |
| **ML Model Training & Hyperparameters** | Full hyperparameter suites and business logic across LightGBM, XGBoost, Random Forest, and Logistic Regression. Fixed task type handling (classification vs regression). |
| **Model Evaluation & DAG Context** | Auto-encoding and alignment of `X_test` / `y_test` in `ModelEvaluator`; auto-propagation of models & test splits across DAG context. |
| **Time Series Forecasters** | Hardened `ProphetForecasterRecipe` and `ArimaForecasterRecipe` with strict column validation and typing fixes. |
| **Workflow Persistence & Execution Lifecycle** | Persisted `last_execution` reports across run & save, smart auto-adoption fallback, dedicated `/save-execution` endpoint, `workflow_id` aliasing, and dataset attribution (`dataset_id` / `dataset_name`). |
| **Dataset Engine & Preview API** | Real preview pagination (`offset` / `page`) with `df.iloc[start:end]` and automatic column data type inference (`column_types`, `columns_schema`). |
| **DAG Validation & Auto-Wiring** | Sequential Kahn's topological error ordering; recipe-aware Auto-Wire DAG generation with intelligent NLP and preprocessing ordering. |
| **Platform CRUD & Governance** | Full workbook CRUD, in-place updates, upsert endpoints, soft-delete & restore, and role-based access control (RBAC). |

---

## 2. Complete GitLens Commit Trail (`author: mighatenosak`)

```text
8995d43 feat(workflows): persist last_execution reports on workflow save and upsert
1eada37 fix(workflows): support workflow_id and pipeline_id aliases during execute to ensure reports attach to current workflow
548b67b fix(datasets): add offset and page pagination support to dataset preview API
1e4a486 fix(workflows): persist dataset_id, dataset_name, and last_execution reports on save and execute
906a3cf feat(datasets): return column_types and columns_schema in preview API and UI column selectors
f9e7a84 fix(dag): validate node configurations in sequential topological order
1bb376e fix(typing): import List in prophet_forecaster and arima_forecaster
d1491fe fix(recipes): eliminate silent column fallbacks in forecasting, NLP, and UI palette
c5f56a9 feat(validation, auth): enforce strict target_column validation and role authorization on workflow APIs
949870f feat(workflows): persist execution outputs and diagnostics with saved workbooks
9acff2f Add auto-upsert to execute endpoint and direct workflow execution by ID
239a451 Fix workflow parameter persistence, required field defaults, and Auto-Wire validation
5d2ef80 fix(training): respect task_type classification and remove silent regression overrides
faf6d7e fix(autowire): order NLP vectorization before feature scaling and refine topological weights
25d730a Add NLP & Text Processing category and comprehensive hyperparameter tuning & business logic suite across trainers and evaluator
7817604 Expand CategoricalEncoderRecipe to support 7 encoding methods (one_hot, label, ordinal, target, frequency, binary, woe)
579ab36 Add enterprise Data Duplicacy & Redundancy Cleaning Recipe Suite (Duplicate Rows, Category Sanitizer, Correlation Filter, Variance Filter)
de13783 Implement soft-delete and restore capabilities for pipeline workbooks with UI integration
2a6f9a1 Implement Upsert capability in PUT and POST workflow endpoints
f1af223 Add POST /api/v1/workflows/autowire REST API endpoint for frontend DAG autowiring
5246690 Add Update Loaded Pipeline feature in UI and PUT REST API integration
5b62291 Add persistent pipeline workbook CRUD REST APIs and UI Workbook Manager
c0e508b Upgrade Auto-Wire logic to perform smart recipe-aware DAG pipeline autowiring
```

---

## 3. Recipes Added & Enhanced

### A. New NLP & Text Processing Recipes (`backend/app/recipes/nlp/`)
*Commit: `25d730a`*

#### 1. `TextPreprocessorRecipe` (`text_preprocessor.py`)
- **Purpose:** Cleans and normalizes raw text columns for downstream machine learning and vectorization.
- **Capabilities & Parameters:**
  - Case normalization (lowercasing, uppercasing).
  - Punctuation and symbol removal.
  - Stopword filtering (English, custom stopword lists).
  - Lemmatization and Stemming (WordNet, Porter/Snowball).
  - HTML tag stripping and whitespace normalization.
  - Regex pattern matching and custom text replacements.
  - Contraction expansion (e.g., *"don't"* $\rightarrow$ *"do not"*).

#### 2. `TextVectorizerRecipe` (`text_vectorizer.py`)
- **Purpose:** Converts textual tokens into numerical feature matrices suitable for ML model training.
- **Capabilities & Parameters:**
  - **Methods:** `tfidf` (TF-IDF Vectorizer) and `count` (Bag-of-Words).
  - `max_features`: Restricts vocabulary size to the most informative tokens.
  - `ngram_range`: Supports unigrams `(1, 1)`, bigrams `(1, 2)`, and trigrams `(1, 3)`.
  - `min_df` & `max_df`: Frequency thresholds to filter out ultra-rare noise or ubiquitous corpus terms.
  - `sublinear_tf`: Sublinear term frequency scaling for skewed text corpora.
  - Integrated into Auto-Wire to automatically position text vectorization before feature scaling.

---

### B. Enterprise Data Duplicacy & Redundancy Cleaning Suite (`backend/app/recipes/preprocessing/`)
*Commit: `579ab36`*

#### 1. `DuplicateRowsRemoverRecipe` (`duplicates.py`)
- **Purpose:** Identifies and cleans redundant duplicate records.
- **Parameters:**
  - `subset`: Optional list of key identifier columns to evaluate duplicates on.
  - `keep`: Strategy for duplicate handling (`"first"`, `"last"`, `False` to drop all occurrences).
  - Generates before/after row count diagnostics in `output_summary`.

#### 2. `CategorySanitizerRecipe` (`duplicates.py`)
- **Purpose:** Normalizes messy string category values across high-variance columns.
- **Parameters:**
  - `columns`: Target columns to sanitize.
  - `strip_whitespace`: Trims leading/trailing whitespace.
  - `case_mode`: Normalizes casing (`"lower"`, `"upper"`, `"title"`).
  - `replace_map`: Exact dictionary mapping for typo correction (e.g. `{"NYC": "New York", "N.Y.": "New York"}`).
  - `rare_threshold`: Groups infrequent categories below a count threshold into an `"Other"` category.

#### 3. `HighCorrelationFilterRecipe` (`duplicates.py`)
- **Purpose:** Prevents severe multicollinearity in linear models and neural networks.
- **Parameters:**
  - Computes Pearson or Spearman correlation matrix across all numeric features.
  - `threshold`: Cutoff correlation coefficient (e.g., `0.85` or `0.90`).
  - Iteratively identifies and drops collinear redundant features while retaining the most informative column.

#### 4. `LowVarianceFilterRecipe` (`duplicates.py`)
- **Purpose:** Strips non-informative constant and quasi-constant columns.
- **Parameters:**
  - `threshold`: Minimum variance required for a column to be retained.
  - Drops zero-variance columns automatically, shrinking memory overhead and speeding up model convergence.

---

### C. Advanced Categorical Feature Encoding (7 Algorithms)
*Commit: `7817604`*

#### `CategoricalEncoderRecipe` (`categorical_encoder.py`)
Significantly expanded from basic one-hot encoding into an enterprise suite of **7 encoding strategies**:
1. **`one_hot`**: Standard dummy variable creation with drop-first option to prevent dummy trap.
2. **`label`**: Ordinal integer labeling (0 to $N-1$) for tree-based algorithms.
3. **`ordinal`**: User-defined hierarchical mapping (e.g. `Low: 0`, `Medium: 1`, `High: 2`).
4. **`target`**: Target Mean Encoding with m-estimate smoothing to prevent overfitting on high-cardinality variables.
5. **`frequency`**: Replaces category levels with their percentage/count frequency in the dataset.
6. **`binary`**: Binary digit encoding across multiple columns (prevents column explosion on 1,000+ categories).
7. **`woe`**: Weight of Evidence encoding for credit risk and financial scoring models.

---

### D. Class Imbalance Resampling Suite (SMOTE & Re-balancing)
*File: `backend/app/recipes/preprocessing/class_imbalance.py`*

#### `ClassImbalanceResamplerRecipe` (`class_imbalance.py`)
- **Purpose:** Solves acute target class skewness in classification problems (fraud detection, churn prediction, healthcare triage).
- **Dual-Mode Architectural Flexibility:**
  - **Mode A (Post-Split - ML Gold Standard):** Resamples training partitions (`X_train`, `y_train`) while preserving untouched testing partitions (`X_test`, `y_test`) to prevent data leakage into evaluation.
  - **Mode B (Pre-Split):** Direct full-dataframe balancing prior to train/test splitting.
- **Supported Strategies:**
  - **`smote`**: Synthetic Minority Over-sampling Technique using $k$-nearest neighbors interpolation. Built with zero-dependency pure Scikit-Learn/Numpy fallback and automatic $k$ neighbor adaptation.
  - **`random_oversample`**: Random over-sampling with replacement; works across both numeric and categorical string features.
  - **`random_undersample`**: Downsamples majority class instances for ultra-fast training on massive datasets.
- **Parameters:** `strategy`, `sampling_ratio`, `k_neighbors`, `target_column`, `random_state`.

---

### E. Model Training & Hyperparameter Tuning Suite
*Commits: `25d730a`, `5d2ef80`*

Expanded and exposed parameter controls across all core model trainers:
- **`LightGBMTrainerRecipe`**: `n_estimators`, `learning_rate`, `max_depth`, `num_leaves`, `min_child_samples`, `reg_alpha`, `reg_lambda`, `objective`.
- **`XGBoostTrainerRecipe`**: `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `gamma`, `eval_metric`.
- **`RandomForestTrainerRecipe`**: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `criterion`.
- **`LogisticRegressionTrainerRecipe`**: `C`, `penalty`, `solver`, `max_iter`, `class_weight`.
- **Classification vs Regression Integrity:** Fixed issue where classification models were silently overwritten with regression defaults (`5d2ef80`).

---

### E. Model Evaluator & DAG Context Artifact Propagation
*Commits: `c4a0d39`, `a465493`, `d85020c`*

- **Automated Fallback Encoding in Evaluator:** When a user builds a pipeline without an explicit categorical encoder node, `ModelEvaluator` automatically encodes and aligns `X_test` and `y_test` to match training column dimensions.
- **Context Artifact Auto-Propagation:** Ensures trained model objects, test splits (`X_test`, `y_test`), and metadata flow seamlessly across branches so evaluators never fail with `"Missing Model"` or `"Missing Dataset Split"`.

---

### F. Time Series Forecasters Hardening
*Commits: `d1491fe`, `1bb376e`*

- **`ProphetForecasterRecipe` & `ArimaForecasterRecipe`**:
  - Eliminated silent column fallbacks (e.g. guessing date/target columns).
  - Added strict validation requiring explicit `date_column` and `target_column`.
  - Fixed Python `typing.List` import issues.

---

## 4. Workflow Persistence & Platform Architecture

*Commits: `8995d43`, `1eada37`, `1e4a486`, `949870f`, `9acff2f`, `de13783`, `2a6f9a1`, `5b62291`*

1. **Run-Then-Save Report Persistence:**
   - Designed `resolve_or_normalize_last_execution(...)` in `backend/app/workflows/router.py`.
   - Saves metrics, confusion matrix, logs, and summaries whether passed directly, linked via `workflow_id` or `execution_id`, or auto-adopted from unsaved runs.
2. **Dedicated `/save-execution` Endpoint:**
   - `POST /api/v1/workflows/{workflow_id}/save-execution` for direct report updates.
3. **First-Class Dataset Attribution:**
   - Added `dataset_id` and `dataset_name` fields to `Workflow` model with database auto-migration and automatic node-config inspection.
4. **Workbook CRUD & Governance:**
   - In-place updates, upsert (`PUT /{id}` and `POST /`), soft-deletion (`is_active=False`, `deleted_at`), and restoration (`POST /{id}/restore`).

---

## 5. Dataset Preview API & DAG Diagnostics

*Commits: `548b67b`, `906a3cf`, `f9e7a84`, `c0e508b`, `faf6d7e`*

1. **Preview Pagination:**
   - Slice-based pagination (`offset`, `page`, `limit`) via `df.iloc[start_idx:end_idx]`.
2. **Column Data Type Badges:**
   - `column_types: Dict[str, str]` and `columns_schema: List[ColumnSchemaItem]` classifying columns into `Numeric`, `Categorical`, `Datetime`, `Text`.
3. **Sequential Kahn's Topological Sort:**
   - Upgraded DAG validation in `graph.py` to sort nodes topologically so upstream configuration issues are reported before downstream nodes.
4. **Intelligent Auto-Wire (Backend & UI Canvas):**
   - Implemented hierarchical topological weight ordering across all 26 recipes:
     $$\text{Ingestion (1.0)} \rightarrow \text{NLP (1.4-1.6)} \rightarrow \text{Deduplication (2.0)} \rightarrow \text{Outlier Guardrail (2.02)} \rightarrow \text{Sanitizer (2.05)} \rightarrow \text{Imputation (2.1)} \rightarrow \text{Corr/Var Filters (2.15-2.18)} \rightarrow \text{Encoding (2.2)} \rightarrow \text{Scaling (2.4)} \rightarrow \text{Lag Features (2.5)} \rightarrow \text{Train/Test Split (3.0)} \rightarrow \mathbf{\text{SMOTE Resampler (3.5)}} \rightarrow \text{Model Training (4.0)} \rightarrow \text{Evaluation (5.0)} \rightarrow \text{MLflow Governance (6.0)}$$
   - Automatic secondary branch edge generation: `train_test_split` $\rightarrow$ `model_evaluator` preserves unbiased $X_{\text{test}}, y_{\text{test}}$ partitions while the resampler balances only $X_{\text{train}}, y_{\text{train}}$.
   - Supported both in the FastAPI endpoint (`POST /api/v1/recommend/autowire`) and the interactive Streamlit Whiteboard canvas.
