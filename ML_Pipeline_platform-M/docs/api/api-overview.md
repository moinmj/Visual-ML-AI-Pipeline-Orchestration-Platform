# API Contract & Integration Overview

This document provides frontend developers with the specifications for integrating with the backend platform.

---

## 1. Base URL & Common Conventions

* **Base URL:** `/api/v1`
* **Content-Type:** `application/json` (except file uploads which use `multipart/form-data`)
* **OpenAPI Documentation:** Interactive Swagger UI available at `http://localhost:8000/docs`
* **Response Wrapper:** Standard JSON payloads with consistent error formats.

---

## 2. Dataset APIs (`/api/v1/datasets`)

| Method | Endpoint | Description | Request Type |
| :--- | :--- | :--- | :--- |
| `POST` | `/upload` | Upload a dataset (CSV, XLSX, JSON) | `multipart/form-data` |
| `GET` | `/` | List all uploaded datasets | Query params (optional) |
| `GET` | `/{id}` | Get dataset metadata | - |
| `GET` | `/{id}/preview` | Preview top N rows of the dataset | `limit` (int, default 10) |
| `GET` | `/{id}/profile` | Get automated data profile & stats | - |
| `DELETE` | `/{id}` | Delete a dataset and its storage files | - |

---

## 3. Recipe APIs (`/api/v1/recipes`)

| Method | Endpoint | Description | Request Type |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | Get full catalog of registered recipes | Query: `category` (optional) |
| `GET` | `/{id}` | Get recipe metadata | - |
| `GET` | `/{id}/schema` | Get JSON schema for dynamic form rendering | - |

---

## 4. Workflow APIs (`/api/v1/workflows`)

| Method | Endpoint | Description | Request Type |
| :--- | :--- | :--- | :--- |
| `POST` | `/` | Save/create a workflow DAG | JSON (nodes + edges) |
| `GET` | `/{id}` | Retrieve workflow definition | - |
| `POST` | `/{id}/validate` | Validate DAG for cycles and type errors | - |
| `POST` | `/{id}/run` | Execute the workflow pipeline | JSON (optional runtime overrides) |
| `GET` | `/executions/{id}` | Get execution status, logs & results | - |
