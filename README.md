# Datalect ML — Pipeline Platform (Backend)

Backend of **Datalect ML**, a visual no-code / low-code platform for building data and machine-learning
pipelines. Users draw a workflow on a canvas (nodes = *recipes*, edges = data flow); this service validates
it, runs it as a DAG, stores every run, and serves the trained models.

This repository is the **FastAPI backend**. The canvas UI is a separate React Flow / Next.js frontend
(`FE_POC`). `ui_prototype/app.py` is the original Streamlit prototype and is no longer the product UI.

---

## Contents

- [Architecture](#architecture)
- [Core concepts](#core-concepts)
- [What the platform does](#what-the-platform-does)
- [Recipe catalog](#recipe-catalog)
- [Getting started](#getting-started)
- [Configuration](#configuration)
- [API overview](#api-overview)
- [Project structure](#project-structure)
- [Tests](#tests)
- [Branches and deployment](#branches-and-deployment)
- [Further documentation](#further-documentation)

---

## Architecture

```
            Frontend canvas (FE_POC, React Flow)
                          │  REST · WebSocket · cookie session (sid)
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│ FastAPI backend (this repo)              backend/app/main.py     │
│                                                                  │
│  Auth (session lookup in UM DB) ── Tenant guard (row isolation)  │
│                                                                  │
│  Workflows API ─► DAG engine ─► Recipes (224)                    │
│      │              │  validation · topological run · test step  │
│      │              │  sub-processes · resume · live logs        │
│      │              └► Inference (predict / schema)              │
│  Datasets API · Credentials vault · Templates · AI recommender   │
│  Cron scheduler (background task) · Webhook / API-endpoint flows │
└──────┬───────────────┬───────────────┬───────────────┬───────────┘
       ▼               ▼               ▼               ▼
  Platform DB      UM DB           Object storage   MLflow
  PostgreSQL       user_sessions   Apache Ozone     tracking &
  (SQLite in dev)  (read-only)     (S3 API)         model registry
```

- **Platform DB** (`DATABASE_URL`): workflows, executions, sub-process versions, datasets, credentials.
  Tables are created and migrated on startup (`init_db`).
- **UM DB** (`UM_DATABASE_URL`): the User-Management service's database. The backend only reads
  `user_sessions` to authenticate requests; it never issues tokens itself.
- **Tenant metadata DB** (`P_DATABASE_URL`, optional): source environments and models for tenant data ingestion.
- **Ozone** stores uploaded datasets, node outputs and exports. **MLflow** tracks training runs.
- **LLM providers** (OpenAI, Groq, Gemini) are optional and only used by the AI recommender, the
  pipeline agent and LLM/agent recipes.

## Core concepts

| Concept | Meaning |
|---|---|
| **Recipe** | One processing step (e.g. *Impute Missing Values*, *Train XGBoost*, *Salesforce Connector*). A `BaseRecipe` subclass with a JSON config schema (the frontend renders its form from it), typed input/output ports and a `to_code()` preview. |
| **Workflow** | A DAG of nodes (recipe instances with their settings) and edges. Stored in `workflows`. |
| **Execution** | One run of a workflow: per-node status, timings, logs, data snapshots and metrics. Stored in `workflow_executions` as an immutable snapshot of the graph that ran. |
| **Sub-process** | A workflow with a **Start Anchor** and an **End Anchor** that other workflows call as a single step through the **Add Workflow** recipe. Each saved change creates a version in `workflow_versions`; callers run `latest` or pin a version. |
| **Tenant** | Every workflow, execution, version and dataset belongs to a tenant. A database-level guard (`infrastructure/database/tenant_guard.py`) adds the tenant filter to every ORM query, so other tenants' rows behave as if they don't exist. |
| **Credential** | A saved connection secret (API key, OAuth token, password) in the `credentials` vault, referenced by nodes through `credential_id`. |

## What the platform does

**Workflow engine**
- Validates graphs (cycles, orphan nodes, port compatibility) and runs them in topological order,
  routing data through named ports (e.g. left/right for joins, matched/unmatched for filters).
- Runs synchronously (`/execute`), in a background job pool (`/async-execute`, with polling and
  streamed logs) or over a WebSocket (`/ws/execute`) for live progress on the canvas.
- **Test Step**: runs a single node against cached upstream data while editing.
- **Resume**: a failed run can continue from the last successful node.
- Each node's output can be previewed, downloaded as CSV or saved to Ozone.

**Sub-processes (Add Workflow)**
- The parent's table is passed to the child's Start Anchor; the End Anchor's table comes back as the
  node's output.
- The child's Start Anchor can declare **expected input columns** (checked before the run) and
  **parameters** (passed by the caller, used in the child as `{{params.name}}`).
- The child runs in an isolated context with its own timeout; loops (A → B → A) and nesting deeper
  than 5 levels are refused.
- *Halt Pipeline on Sub-Workflow Error*: **on** stops the parent; **off** marks the node WARNING
  and skips the steps after it, while other branches keep running.

**Triggers and automation**
- Cron schedules (a scheduler runs inside the API process), inbound webhooks
  (`/workflows/trigger/{path}`), IMAP email triggers, and workflows exposed as API endpoints
  (`/api/v1/workflows/endpoints/{path}`, public or API-key protected per endpoint node).

**History and versioning**
- Every run is stored as an immutable snapshot; runs can be listed, compared, opened and pruned.
- Workflows are soft-deleted and can be restored.

**Models and inference**
- Trained pipelines are served with `POST /workflows/{execution_id}/predict`; `/schema` describes
  the expected input. Forecasting runs can generate future datasets.
- MLflow tracking and model registry, model cards, explainability (SHAP/LIME), fairness and drift monitoring.

**AI assistance**
- **Recommender** (`/recommend/pipeline`, `/recommend/autowire`): profiles a dataset, infers the task
  (classification, regression, forecasting, anomaly detection) and proposes a pipeline.
- **Pipeline agent** (`/agent/pipelines/plan` and `/run`): a LangGraph planner / evaluator loop that
  builds, runs and refines classification or regression pipelines from a natural-language request.
  It sends the request, column names, aggregate profile statistics and scalar metrics to the LLM,
  never dataset rows. Jobs run in a bounded in-process pool, so job state is lost on restart and not
  shared between replicas.

**Security**
- Session authentication shared with User-Management (`sid` cookie or `X-Session-Id` header; CSRF
  check on unsafe methods).
- Tenant isolation enforced in the database layer (see *Tenant* above).
- Inline secrets in node settings are **encrypted before they are stored** (AES-256, key derived from
  `SECRET_KEY`), including nested settings and the graph snapshots in run history, and masked
  (`••••••••`) in API responses. Nodes that use a vaulted credential don't store their own copy.
- CSV exports are limited by role and IP (`EXPORT_ALLOWED_ROLES`, `EXPORT_ALLOWED_IPS`).

## Recipe catalog

224 recipes in `backend/app/recipes/`, by category:

| Category | Count | Examples |
|---|---:|---|
| `preprocessing` | 44 | Imputer, scaler, encoders, aggregate, combine/join, SMOTE resampler, PII tokenization, address parser |
| `integrations` | 42 | Salesforce, HubSpot, Airtable, Asana, GitHub, Freshdesk, Discord, Dropbox, Google Ads, Shopify |
| `ingestion` | 27 | CSV/Excel/JSON loaders, PostgreSQL/MySQL/MongoDB, BigQuery, Azure Data Lake, S3, CDC, SFTP, API endpoint |
| `agents` | 18 | Document Q&A, chatbot endpoint, guardrails, conversational memory, cost-aware model router, agent hand-off |
| `training` | 14 | XGBoost, LightGBM, CatBoost, random forest, linear models, AutoML tournament, Optuna tuner, ensemble stacker |
| `flow_control` | 13 | Add Workflow, Start/End Anchor, IF/condition, loop/batch iterator, merge, delay, dead-letter queue |
| `canvas` | 10 | Notes, groups, comment threads, charts, parameter sweep, pipeline diff, shareable snapshot |
| `vision` | 8 | Image classifier, object detection, OCR, image embeddings, speech-to-text, text-to-speech |
| `evaluation` | 7 | Model evaluator, SHAP/LIME explainability, A/B and Bayesian tests, champion/challenger, fairness audit |
| `nlp` | 7 | Text preprocessing, vectorizers, sentiment, summarization, translation, document parsing, PII masking |
| `monitoring` | 6 | Data drift, model drift, cost/latency monitor, serving endpoint, rollback trigger |
| `unsupervised` | 5 | Clustering (KMeans/DBSCAN), PCA, t-SNE, UMAP, association rules |
| `reporting` | 4 | Scheduled PDF/Excel reports, BI export, dashboard publish, threshold alerts |
| `security` | 4 | Audit trail, credential rotation, encryption toggle, row-level access control |
| `splitting` | 4 | Train/test, stratified, chronological, walk-forward |
| `forecasting` | 3 | Prophet, ARIMA/SARIMAX, lag & time features |
| `governance` | 3 | MLflow registry, model card generator, compliance trail export |
| `triggers` | 3 | Cron schedule, webhook, email (IMAP) |
| `anomaly` | 2 | Isolation Forest, statistical outlier guardrail (Z-score/IQR) |

`GET /api/v1/recipes/` returns the live catalog with each recipe's schema and ports.

---

## Getting started

**Requirements:** Python 3.11+ (tested on 3.14), PostgreSQL for a shared setup (SQLite works for a
quick local start), access to the User-Management database for authentication, and optionally
Apache Ozone / S3 and MLflow.

```bash
git clone https://github.com/AIIOTs-Geek/ML_Pipeline_platform-O.git
cd ML_Pipeline_platform-O

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the repository root (see [Configuration](#configuration)). It is git-ignored.
**Never commit it.**

Start the API from the repository root:

```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

On startup the service creates and migrates its tables, then starts the cron scheduler.
Interactive API docs are at <http://localhost:8000/docs>.

API routes require a valid User-Management session (`sid` cookie or `X-Session-Id` header). For local
work without UM, see `DEV_AUTH_BYPASS` below (development only).

## Configuration

Settings are read from environment variables / `.env` (`backend/app/core/config.py`). The main ones:

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | Platform database, e.g. `postgresql+asyncpg://user:pass@host:5432/db`. Defaults to SQLite in `./data`. |
| `UM_DATABASE_URL` | User-Management database (asyncpg DSN; URL-encode special characters in the password). |
| `SESSION_SECRET_KEY` | Must equal UM's `SECRET_KEY`; used to check CSRF tokens. |
| `SECRET_KEY` | Encryption key for stored credentials. **Use the same value in every instance that shares a database**; values encrypted with one key can't be read with another. |
| `ENVIRONMENT`, `DEBUG` | Runtime mode. |
| `DEV_AUTH_BYPASS` | With `ENVIRONMENT=development`/`local` (or `DEBUG=true`), accepts the session id `dev` without UM. Local development only, never on a server. |
| `OZONE_ENDPOINT`, `OZONE_ACCESS_KEY`, `OZONE_SECRET_KEY`, `OZONE_BUCKET`, `OZONE_PREFIX` | Object storage for datasets and outputs. |
| `MLFLOW_TRACKING_URI` | MLflow server or local store. |
| `OPENAI_API_KEY` / `OPENAI_MODEL`, `GROQ_API_KEY` / `GROQ_MODEL`, `GEMINI_API_KEY` | LLM providers (optional). The agent uses OpenAI if set, otherwise Groq. |
| `AGENT_MAX_WORKERS`, `AGENT_MAX_QUEUED`, `AGENT_MAX_GROQ_CONCURRENT` | Pipeline-agent capacity (defaults 2 / 16 / 8). When full, the API answers `429` with `Retry-After`. |
| `P_DATABASE_URL`, `TENANT_DB_AUTO_CREATE` | Tenant metadata database. Set auto-create to `true` only for a new development database. |
| `EXPORT_ALLOWED_ROLES`, `EXPORT_ALLOWED_IPS`, `TRUSTED_PROXY_IPS` | Who may export CSVs. |
| `DISCORD_*`, `TEAMS_*`, `AZURE_*`, `TWILIO_*`, `SALESFORCE_*`, `HUBSPOT_*`, … | Credentials for individual integrations (optional). |

---

## API overview

Routes are under `/api/v1` and need an authenticated session, except flow endpoints
(`/workflows/endpoints/{path}`), whose access (public or API key) is set on the workflow's endpoint
node. Full request/response schemas: `/docs`.

| Area | Main routes |
|---|---|
| **Workflows** `/workflows` | `GET /` list · `POST /` save · `GET`/`PUT`/`DELETE /{id}` · `POST /{id}/restore` · `GET /{id}/export` · `POST /import`, `/import/file` |
| **Run** `/workflows` | `POST /validate` · `POST /execute`, `/{id}/execute` · `POST /async-execute` · `GET /jobs/{job_id}`, `/logs`, `/stream` · `WS /ws/execute` · `GET /executions/{id}/resume-info` |
| **Editing** `/workflows` | `POST /nodes/{node_id}/test-step` · `POST /autowire` · `POST /infer-schema` |
| **Sub-processes** | `GET /workflows?sub_workflow_eligible=true` · `GET /workflows/{id}/sub-workflow-versions` |
| **Run outputs** `/workflows/{execution_id}/nodes/{node_id}` | `GET /data` · `GET /artifacts`, `/artifacts/{name}/download` · `GET /export-csv` · `POST /save-to-ozone` |
| **History** `/workflows/{id}/history` | `GET` list · `GET /{execution_id}` · `GET /compare` · `DELETE` prune |
| **Inference** `/workflows` | `GET /{execution_id}/schema` · `POST /{execution_id}/predict`, `/predict` · `POST /{execution_id}/generate-future-dataset` |
| **Triggers** | `POST /workflows/trigger/{path}` · `POST /workflows/{id}/trigger-cron` · `GET /workflows/scheduler/status` · flow endpoints `/workflows/endpoints/{path}` |
| **Datasets** `/datasets` | `POST /upload` · `GET /`, `/{id}`, `/{id}/preview`, `/{id}/profile` · `DELETE /{id}` · `GET /ozone/browse` · `POST /ozone/register` |
| **Recipes** `/recipes` | `GET /` · `GET /categories` · `GET /{id}`, `/{id}/schema`, `/{id}/icon` · `POST /{id}/run`, `/{id}/inspect` |
| **Credentials** | `GET`/`POST /credentials` · `GET`/`PUT`/`DELETE /credentials/{id}` · OAuth: `/oauth/discord/*`, `/oauth/msteams/*` |
| **AI** | `POST /recommend/pipeline`, `/recommend/autowire` · `POST /agent/pipelines/plan`, `/run` · `GET /agent/pipelines/jobs/{job_id}` |
| **Other** | `GET /templates/`, `/templates/{id}` · `GET /tenant-data/environments`, `…/models`, `POST …/ingest` · `GET /auth/whoami` |

---

## Project structure

```
ML_Pipeline_platform-O/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app, routers, startup (DB init, scheduler)
│   │   ├── auth/                # /auth/whoami
│   │   ├── core/                # settings, session auth (security.py), crypto, tenant context, logging
│   │   ├── credentials/         # credential vault and OAuth flows
│   │   ├── datasets/            # upload, Ozone browse/register, preview, profiling
│   │   ├── engine/              # DAG graph & validation, executor, step executor, job manager, inference
│   │   ├── infrastructure/      # DB sessions, tenant guard, storage (Ozone/S3/local)
│   │   ├── profiling/           # dataset profiler
│   │   ├── recipes/             # BaseRecipe, registry, recipe router, 224 recipes by category
│   │   ├── recommendation/      # heuristic + LLM recommender, LangGraph pipeline agent
│   │   ├── templates/           # pipeline templates
│   │   ├── tenant_data/         # tenant environments / models ingestion
│   │   ├── workflows/           # workflow & execution API, models, scheduler, sub-processes, flow endpoints
│   │   └── static/
│   ├── scripts/                 # one-off maintenance scripts (e.g. encrypt stored secrets)
│   └── tests/                   # pytest suite
├── docs/                        # architecture, API overview, agent guide, roadmap
├── ui_prototype/app.py          # legacy Streamlit prototype
├── azure-pipelines.yml          # CI/CD: deploys the development branch
├── ecosystem.config.js          # PM2 process definition on the server
└── requirements.txt
```

## Tests

```bash
pytest backend/tests -v
```

Most tests use in-memory SQLite. A few integration tests need the PostgreSQL / UM databases from
`.env`, and fail when those aren't reachable.

## Branches and deployment

- **`development` is the deployed branch.** A push to it triggers `azure-pipelines.yml`, which
  deploys the service to the server, where PM2 runs it (`ecosystem.config.js`, process
  `ML-Pipeline-Platform`, uvicorn on `127.0.0.1:9369`).
- Work happens on feature branches and reaches `development` only through reviewed pull requests.
  Don't push to `development` directly.

## Further documentation

- [`docs/platform_overview.md`](docs/platform_overview.md) — product and technical overview
- [`docs/api/api-overview.md`](docs/api/api-overview.md) — API notes
- [`docs/AGENT_PIPELINE_GUIDE.md`](docs/AGENT_PIPELINE_GUIDE.md) — pipeline agent
- [`docs/PROJECT_MASTER_CONTEXT.md`](docs/PROJECT_MASTER_CONTEXT.md), [`docs/ROADMAP_AND_TASKS.md`](docs/ROADMAP_AND_TASKS.md)
- `/docs` on a running server — OpenAPI reference
