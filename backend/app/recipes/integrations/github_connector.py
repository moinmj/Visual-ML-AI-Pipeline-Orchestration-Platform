"""
GitHub Connector Recipe (Apps & Integrations -> Project & Dev Tools).

Reads GitHub issues, pull requests, commits, releases, and repository activity via GitHub REST API v3
(with credential resolution, rate-limit resilience, and benchmark simulation fallbacks).
"""

import json
import os
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

try:
    import pandas as pd
except ImportError:
    pd = None

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort
from backend.app.core.logging import logger
from backend.app.core.config import settings


class GitHubConnectorRecipe(BaseRecipe):
    recipe_id = "github_connector"
    name = "GitHub Connector"
    version = "1.1.0"
    category = "integrations"
    group = "Apps & Integrations"
    subgroup = "Project & Dev Tools"
    description = "Reads GitHub issues, pull requests and repository activity via GitHub REST API v3."
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
            label="GitHub Dataset",
            type="dataframe",
            description="DataFrame containing retrieved GitHub issues, pull requests, commits, or releases"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "resource": {
                    "type": "string",
                    "title": "GitHub Resource",
                    "input_type": "dropdown",
                    "enum": ["Issues", "Pull Requests", "Commits", "Releases", "Repositories"],
                    "default": "Issues",
                    "description": "Select the GitHub entity to query (Issues, Pull Requests, Commits, Releases, Repositories)."
                },
                "repository": {
                    "type": "string",
                    "title": "Repository (owner/repo)",
                    "input_type": "text",
                    "default": "octocat/Hello-World",
                    "description": "GitHub target repository in 'owner/repo' format (e.g. pytorch/pytorch or octocat/Hello-World)."
                },
                "state_filter": {
                    "type": "string",
                    "title": "State Filter",
                    "input_type": "dropdown",
                    "enum": ["all", "open", "closed"],
                    "default": "open",
                    "description": "Filter issues or pull requests by state (all, open, closed)."
                },
                "access_token": {
                    "type": "string",
                    "title": "Personal Access Token",
                    "input_type": "password",
                    "description": "Optional GitHub Personal Access Token (classic or fine-grained) for higher rate limits or private repos."
                },
                "credential_id": {
                    "type": "string",
                    "title": "Connected GitHub Credential",
                    "input_type": "credential_dropdown",
                    "provider": "github",
                    "description": "Select a stored GitHub credential or enter an access token above."
                },
                "record_limit": {
                    "type": "integer",
                    "title": "Record Limit",
                    "input_type": "number",
                    "default": 50,
                    "minimum": 1,
                    "maximum": 200,
                    "description": "Maximum number of records to retrieve (1-200)."
                },
                "simulation_mode": {
                    "type": "boolean",
                    "title": "Simulation Mode (Offline Benchmark Data)",
                    "input_type": "checkbox",
                    "default": False,
                    "description": "Generate realistic benchmark GitHub records without live API calls."
                }
            },
            "required": ["resource"]
        }

    def _generate_benchmark_data(self, resource: str, limit: int, state_filter: str = "open") -> List[Dict[str, Any]]:
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        limit_to_use = max(1, min(limit, 200))
        records: List[Dict[str, Any]] = []

        if resource == "Issues":
            issues = [
                (1042, "Support for PyTorch 2.4 compiler graph capture", "open", "torvalds", 14, 5, "bug,enhancement"),
                (1043, "Fix dead-letter queue backoff exponential jitter", "closed", "octocat", 3, 2, "bug"),
                (1044, "Refactor DAG topological cycle validation routine", "open", "antigravity-bot", 8, 3, "performance"),
                (1045, "Add Azure Blob Storage chunked upload pipeline node", "open", "cloud-dev", 21, 8, "feature"),
                (1046, "CI pipeline intermittent timeout on Windows runner", "closed", "qa-lead", 6, 1, "ci/cd"),
                (1047, "Update scikit-learn dependency constraint to >= 1.5", "closed", "dependabot[bot]", 2, 0, "dependencies")
            ]
            for i in range(limit_to_use):
                item = issues[i % len(issues)]
                num = item[0] + i
                item_state = item[2] if state_filter == "all" else state_filter
                records.append({
                    "id": 1400000 + i,
                    "number": num,
                    "title": f"{item[1]} (Run #{i//len(issues)+1})" if i >= len(issues) else item[1],
                    "state": item_state,
                    "user": item[3],
                    "comments": item[4],
                    "reactions": item[5],
                    "labels": item[6],
                    "created_at": now_iso,
                    "updated_at": now_iso
                })

        elif resource == "Pull Requests":
            prs = [
                (512, "feat(recipes): add Linear and Jira live connectors", "open", "engineer-salman", "main", "feature/integrations"),
                (513, "fix(engine): resolve WebSocket reconnect flood", "closed", "octocat", "main", "fix/ws-flood"),
                (514, "perf(canvas): memoize connection handle coordinate lookups", "closed", "ui-dev", "main", "perf/canvas-coords"),
                (515, "docs: add interactive recipe parameter guide", "open", "tech-writer", "main", "docs/recipes")
            ]
            for i in range(limit_to_use):
                item = prs[i % len(prs)]
                item_state = item[2] if state_filter == "all" else state_filter
                records.append({
                    "id": 2400000 + i,
                    "number": item[0] + i,
                    "title": f"{item[1]} (Run #{i//len(prs)+1})" if i >= len(prs) else item[1],
                    "state": item_state,
                    "user": item[3],
                    "base_branch": item[4],
                    "head_branch": item[5],
                    "created_at": now_iso
                })

        elif resource == "Commits":
            authors = ["Linus Torvalds", "Ada Lovelace", "Alan Turing", "Grace Hopper"]
            for i in range(limit_to_use):
                records.append({
                    "sha": f"a{i+1:04d}ef98bc437190d8a5621374b5c6d7e8f90123"[:40],
                    "author": authors[i % len(authors)],
                    "message": f"feat: optimize pipeline execution cycle #{i+1}",
                    "commit_date": now_iso,
                    "files_changed": (i % 6) + 1
                })

        elif resource == "Releases":
            versions = ["v2.4.0", "v2.3.2", "v2.3.1", "v2.3.0", "v2.2.0"]
            for i in range(min(limit_to_use, len(versions))):
                records.append({
                    "id": 90001 + i,
                    "tag_name": versions[i],
                    "name": f"Production Release {versions[i]}",
                    "draft": False,
                    "prerelease": False,
                    "published_at": now_iso,
                    "download_count": 14500 - (i * 2000)
                })

        elif resource == "Repositories":
            records = [
                {"name": "ML_Pipeline_platform-O", "full_name": "enterprise/ML_Pipeline_platform-O", "private": True, "stars": 340, "forks": 52},
                {"name": "Datarept_Forge", "full_name": "enterprise/Datarept_Forge", "private": True, "stars": 128, "forks": 19}
            ]

        return records

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        resource = str(config.get("resource", "Issues")).strip()
        repo = str(config.get("repository", "octocat/Hello-World")).strip()
        state = str(config.get("state_filter", "open")).strip()
        token = str(config.get("access_token", "")).strip() or getattr(settings, "GITHUB_TOKEN", None) or os.getenv("GITHUB_TOKEN", "")
        credential_id = str(config.get("credential_id", "")).strip()
        limit = int(config.get("record_limit", config.get("limit", 50)))
        simulation_mode = bool(config.get("simulation_mode", config.get("simulate_if_no_auth", False)))

        if credential_id and context and isinstance(context, dict):
            creds_store = context.get("credentials", {})
            if credential_id in creds_store:
                cred = creds_store[credential_id]
                c_data = cred.get("data", cred)
                token = c_data.get("token") or c_data.get("access_token") or token

        records: List[Dict[str, Any]] = []
        delivery_status = "SIMULATED"
        status_msg = ""

        # Live GitHub REST API v3 query attempt
        if not simulation_mode and repo and "/" in repo:
            try:
                per_page = max(1, min(limit, 100))
                headers = {
                    "Accept": "application/vnd.github.v3+json",
                    "User-Agent": "Datarept-ML-Pipeline/1.0"
                }
                if token and len(token) >= 10 and not token.startswith("mock"):
                    headers["Authorization"] = f"Bearer {token}"

                if resource == "Issues":
                    url = f"https://api.github.com/repos/{repo}/issues?state={state}&per_page={per_page}"
                elif resource == "Pull Requests":
                    url = f"https://api.github.com/repos/{repo}/pulls?state={state}&per_page={per_page}"
                elif resource == "Commits":
                    url = f"https://api.github.com/repos/{repo}/commits?per_page={per_page}"
                elif resource == "Releases":
                    url = f"https://api.github.com/repos/{repo}/releases?per_page={per_page}"
                elif resource == "Repositories":
                    owner = repo.split("/")[0]
                    url = f"https://api.github.com/users/{owner}/repos?per_page={per_page}"
                else:
                    url = f"https://api.github.com/repos/{repo}/issues?state={state}&per_page={per_page}"

                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=12) as response:
                    if response.status == 200:
                        data = json.loads(response.read().decode("utf-8"))

                        if resource == "Issues":
                            for item in data:
                                # GitHub issues endpoint includes PRs unless filtered
                                if "pull_request" not in item:
                                    records.append({
                                        "id": item.get("id"),
                                        "number": item.get("number"),
                                        "title": item.get("title"),
                                        "state": item.get("state"),
                                        "user": item.get("user", {}).get("login"),
                                        "comments": item.get("comments", 0),
                                        "created_at": item.get("created_at")
                                    })
                        elif resource == "Pull Requests":
                            for item in data:
                                records.append({
                                    "id": item.get("id"),
                                    "number": item.get("number"),
                                    "title": item.get("title"),
                                    "state": item.get("state"),
                                    "user": item.get("user", {}).get("login"),
                                    "base_branch": item.get("base", {}).get("ref"),
                                    "head_branch": item.get("head", {}).get("ref"),
                                    "created_at": item.get("created_at")
                                })
                        elif resource == "Commits":
                            for item in data:
                                commit_obj = item.get("commit", {})
                                records.append({
                                    "sha": item.get("sha", "")[:40],
                                    "author": commit_obj.get("author", {}).get("name") or item.get("author", {}).get("login"),
                                    "message": commit_obj.get("message", "").split("\n")[0],
                                    "commit_date": commit_obj.get("author", {}).get("date"),
                                    "html_url": item.get("html_url")
                                })
                        elif resource == "Releases":
                            for item in data:
                                records.append({
                                    "id": item.get("id"),
                                    "tag_name": item.get("tag_name"),
                                    "name": item.get("name"),
                                    "draft": item.get("draft", False),
                                    "prerelease": item.get("prerelease", False),
                                    "published_at": item.get("published_at")
                                })
                        elif resource == "Repositories":
                            for item in data:
                                records.append({
                                    "name": item.get("name"),
                                    "full_name": item.get("full_name"),
                                    "private": item.get("private", False),
                                    "stars": item.get("stargazers_count", 0),
                                    "forks": item.get("forks_count", 0),
                                    "open_issues": item.get("open_issues_count", 0)
                                })

                        if records:
                            delivery_status = "LIVE"
                            status_msg = f"Retrieved {len(records)} live GitHub {resource} from '{repo}' via GitHub REST API v3."
            except urllib.error.HTTPError as he:
                err_body = ""
                try:
                    err_body = he.read().decode("utf-8")
                except Exception:
                    pass
                logger.warning(f"GitHub API HTTP {he.code}: {err_body[:200]}")
                status_msg = f"GitHub API query returned HTTP {he.code}. Using realistic benchmark simulation fallback."
            except Exception as ex:
                logger.warning(f"GitHub connection failed: {ex}")
                status_msg = f"GitHub connection failed ({ex}). Using realistic benchmark simulation fallback."

        # Offline / Benchmark Fallback
        if not records:
            records = self._generate_benchmark_data(resource, limit, state)
            if not status_msg:
                status_msg = f"Generated {len(records)} realistic GitHub {resource} benchmark records (Simulation Mode)."

        df_out = pd.DataFrame(records) if pd is not None else records

        summary = {
            "title": "GitHub Connector Read Summary",
            "message": status_msg,
            "status": delivery_status,
            "resource": resource,
            "repository": repo,
            "records_retrieved": len(records)
        }

        metrics = {
            "github_resource": resource,
            "repository": repo,
            "record_count": len(records),
            "delivery_status": delivery_status
        }

        return {
            "dataframe": df_out,
            "output": df_out,
            "output_summary": summary,
            "metrics": metrics
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        repo = config.get("repository", "octocat/Hello-World")
        res = config.get("resource", "Issues")
        lim = config.get("record_limit", 50)
        subpath = "issues" if res == "Issues" else ("pulls" if res == "Pull Requests" else "commits")
        return (
            f"# Recipe: GitHub Connector\n"
            f"# ID: {self.recipe_id}\n"
            f"import os\n"
            f"import requests\n"
            f"import pandas as pd\n\n"
            f"token = os.getenv('GITHUB_TOKEN', '')\n"
            f"url = 'https://api.github.com/repos/{repo}/{subpath}?per_page={lim}'\n"
            f"headers = {{'User-Agent': 'Datarept-ML-Pipeline', 'Accept': 'application/vnd.github.v3+json'}}\n"
            f"if token:\n"
            f"    headers['Authorization'] = f'Bearer {{token}}'\n"
            f"response = requests.get(url, headers=headers)\n"
            f"df = pd.DataFrame(response.json())\n"
        )
