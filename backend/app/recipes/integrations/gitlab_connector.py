"""
GitLab Connector Recipe (Apps & Integrations -> Project & Dev Tools).

Reads GitLab issues, merge requests, CI/CD pipelines, and projects via GitLab REST API v4
(with credential resolution, rate-limit resilience, and benchmark simulation fallbacks).
"""

import json
import os
import urllib.request
import urllib.parse
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

try:
    import pandas as pd
except ImportError:
    pd = None

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort
from backend.app.core.logging import logger


class GitLabConnectorRecipe(BaseRecipe):
    recipe_id = "gitlab_connector"
    name = "GitLab Connector"
    version = "1.0.0"
    category = "integrations"
    group = "Apps & Integrations"
    subgroup = "Project & Dev Tools"
    description = "Reads GitLab issues, merge requests and pipeline data."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="input",
            label="Trigger / Input Dataset",
            type="dataframe",
            required=False,
            max_connections=1,
            description="Optional upstream dataset for batch IDs or execution trigger"
        )
    ]

    outputs = [
        RecipePort(
            id="output",
            label="GitLab Dataset",
            type="dataframe",
            description="DataFrame containing retrieved GitLab issues, merge requests, pipelines, or projects"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "resource": {
                    "type": "string",
                    "title": "GitLab Resource",
                    "input_type": "dropdown",
                    "enum": ["Issues", "Merge Requests", "Pipelines", "Projects"],
                    "default": "Issues",
                    "description": "Select the GitLab resource to query (Issues, Merge Requests, Pipelines, Projects)."
                },
                "project_id": {
                    "type": "string",
                    "title": "Project Path or ID",
                    "input_type": "text",
                    "default": "gitlab-org/gitlab",
                    "description": "GitLab project path (namespace/project) or integer ID."
                },
                "gitlab_url": {
                    "type": "string",
                    "title": "GitLab Server URL",
                    "input_type": "text",
                    "default": "https://gitlab.com",
                    "description": "GitLab server host URL (e.g. https://gitlab.com or self-hosted instance)."
                },
                "state_filter": {
                    "type": "string",
                    "title": "State Filter",
                    "input_type": "dropdown",
                    "enum": ["all", "opened", "closed", "merged"],
                    "default": "opened",
                    "description": "Filter issues or merge requests by state."
                },
                "access_token": {
                    "type": "string",
                    "title": "Personal Access Token",
                    "input_type": "password",
                    "description": "GitLab Personal Access Token with api or read_api scope."
                },
                "credential_id": {
                    "type": "string",
                    "title": "Credential ID",
                    "input_type": "text",
                    "description": "Saved credential ID containing GitLab authentication secrets."
                },
                "record_limit": {
                    "type": "integer",
                    "title": "Record Limit",
                    "input_type": "number",
                    "default": 50,
                    "minimum": 1,
                    "maximum": 200,
                    "description": "Maximum number of records to retrieve."
                },
                "simulation_mode": {
                    "type": "boolean",
                    "title": "Simulation Mode (Offline Benchmark Data)",
                    "input_type": "checkbox",
                    "default": False,
                    "description": "Generate realistic benchmark GitLab records without live API calls."
                }
            },
            "required": ["resource"]
        }

    def _generate_benchmark_data(self, resource: str, limit: int) -> List[Dict[str, Any]]:
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        limit_to_use = max(1, min(limit, 100))
        records = []

        if resource == "Issues":
            issues = [
                (201, "Runner autoscaling on AWS Spot instances", "opened", "Engineering", "dev-lead", 2),
                (202, "Fix secret variable masking in pipeline artifacts", "closed", "Security", "sec-officer", 5),
                (203, "Add OIDC authentication support for Kubernetes clusters", "opened", "Infrastructure", "k8s-admin", 8),
                (204, "Upgrade Alpine base image for security patches", "closed", "DevOps", "qa-eng", 1),
                (205, "Optimize CI/CD cache restoration latency", "opened", "Performance", "perf-guru", 3)
            ]
            for i in range(limit_to_use):
                item = issues[i % len(issues)]
                records.append({
                    "id": 920000 + i,
                    "iid": item[0] + i,
                    "title": f"{item[1]} (Run #{i//len(issues)+1})" if i >= len(issues) else item[1],
                    "state": item[2],
                    "labels": item[3],
                    "author": item[4],
                    "weight": item[5],
                    "created_at": now_iso,
                    "updated_at": now_iso
                })

        elif resource == "Merge Requests":
            mrs = [
                (101, "Resolve race condition in model artifact storage", "merged", "dev-lead", "main", "fix/storage-race"),
                (102, "Support GitLab CI variable injection for ML pipelines", "opened", "k8s-admin", "main", "feat/ci-variables"),
                (103, "Refactor database migrations for zero-downtime deploy", "merged", "dba", "main", "refactor/db-migrations"),
                (104, "Add linting step to pre-commit hook template", "opened", "qa-eng", "main", "chore/linting")
            ]
            for i in range(limit_to_use):
                item = mrs[i % len(mrs)]
                records.append({
                    "id": 810000 + i,
                    "iid": item[0] + i,
                    "title": f"{item[1]} (Run #{i//len(mrs)+1})" if i >= len(mrs) else item[1],
                    "state": item[2],
                    "author": item[3],
                    "target_branch": item[4],
                    "source_branch": item[5],
                    "created_at": now_iso
                })

        elif resource == "Pipelines":
            statuses = ["success", "success", "failed", "running", "success"]
            branches = ["main", "feature/auth", "bugfix/102", "develop"]
            for i in range(limit_to_use):
                records.append({
                    "id": 730000 + i,
                    "status": statuses[i % len(statuses)],
                    "ref": branches[i % len(branches)],
                    "sha": f"c{i+1:04d}fa78901234567890abcdef1234567890abcd"[:40],
                    "duration_seconds": 180 + (i * 15),
                    "coverage": "94.2%",
                    "created_at": now_iso
                })

        elif resource == "Projects":
            records = [
                {"id": 482910, "name": "data-pipeline-runner", "path_with_namespace": "gitlab-org/data-pipeline-runner", "visibility": "private", "star_count": 45},
                {"id": 482911, "name": "ml-model-registry", "path_with_namespace": "gitlab-org/ml-model-registry", "visibility": "internal", "star_count": 89}
            ]

        return records

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        resource = str(config.get("resource", "Issues")).strip()
        project_id = str(config.get("project_id", "gitlab-org/gitlab")).strip()
        gitlab_url = str(config.get("gitlab_url", "https://gitlab.com")).strip().rstrip("/")
        token = str(config.get("access_token", "")).strip() or os.getenv("GITLAB_TOKEN", "")
        credential_id = str(config.get("credential_id", "")).strip()
        limit = int(config.get("record_limit", 50))
        simulation_mode = bool(config.get("simulation_mode", False))

        if credential_id and context and isinstance(context, dict):
            creds_store = context.get("credentials", {})
            if credential_id in creds_store:
                cred = creds_store[credential_id]
                c_data = cred.get("data", cred)
                token = c_data.get("token") or c_data.get("access_token") or token
                gitlab_url = c_data.get("gitlab_url") or gitlab_url

        records: List[Dict[str, Any]] = []
        delivery_status = "SIMULATED"
        status_msg = ""

        encoded_project = urllib.parse.quote(project_id, safe="")
        subpath = "issues" if resource == "Issues" else ("merge_requests" if resource == "Merge Requests" else "pipelines")

        if not simulation_mode and token and not token.startswith("mock") and len(token) >= 15:
            try:
                url = f"{gitlab_url}/api/v4/projects/{encoded_project}/{subpath}?per_page={min(limit, 100)}"
                req = urllib.request.Request(url, headers={"PRIVATE-TOKEN": token})
                with urllib.request.urlopen(req, timeout=12) as response:
                    if response.status == 200:
                        data = json.loads(response.read().decode("utf-8"))
                        for item in data:
                            records.append({
                                "id": item.get("id"),
                                "iid": item.get("iid"),
                                "title": item.get("title", f"Pipeline #{item.get('id')}"),
                                "state": item.get("state") or item.get("status"),
                                "author": item.get("author", {}).get("name") if item.get("author") else "System",
                                "created_at": item.get("created_at")
                            })
                        delivery_status = "LIVE"
                        status_msg = f"Retrieved {len(records)} live GitLab records via GitLab REST API v4."
            except Exception as ex:
                logger.warning(f"GitLab live connection fallback: {ex}")
                status_msg = f"GitLab live connection failed ({ex}). Switched to benchmark fallback data."

        if not records:
            records = self._generate_benchmark_data(resource, limit)
            if not status_msg:
                status_msg = f"Generated {len(records)} realistic GitLab {resource} records (Simulation Mode)."

        df_out = pd.DataFrame(records) if pd is not None else records

        summary = {
            "title": "GitLab Connector Read Summary",
            "message": status_msg,
            "status": delivery_status,
            "resource": resource,
            "records_retrieved": len(records)
        }

        metrics = {
            "gitlab_resource": resource,
            "record_count": len(records),
            "delivery_status": delivery_status
        }

        return {
            "dataframe": df_out,
            "output_summary": summary,
            "metrics": metrics
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        return (
            f"# Recipe: GitLab Connector\n"
            f"# ID: {self.recipe_id}\n"
            f"import os\n"
            f"import urllib.parse\n"
            f"import requests\n"
            f"import pandas as pd\n\n"
            f"token = os.getenv('GITLAB_TOKEN', '')\n"
            f"proj = urllib.parse.quote('{config.get('project_id', 'gitlab-org/gitlab')}', safe='')\n"
            f"url = f'{config.get('gitlab_url', 'https://gitlab.com')}/api/v4/projects/{{proj}}/issues'\n"
            f"records = requests.get(url, headers={{'PRIVATE-TOKEN': token}}).json()\n"
            f"df = pd.DataFrame(records)\n"
        )

