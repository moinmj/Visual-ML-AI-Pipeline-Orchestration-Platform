import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from collections import defaultdict, deque
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from backend.app.datasets.models import Dataset
from backend.app.infrastructure.storage.storage_manager import storage_manager
from backend.app.workflows.schemas import (
    NodeColumnInfo,
    NodePortSchema,
    NodeInferredSchema,
    WorkflowInferSchemaResponse
)

logger = logging.getLogger(__name__)


class WorkflowSchemaInferencer:
    """
    Infers expected column schemas and per-port column distributions
    for all nodes in a DAG workflow before execution.
    Powers frontend smart column pickers and handle auto-configuration.
    """

    @classmethod
    async def infer_schema(
        cls,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        node_configs: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        db: Optional[AsyncSession] = None
    ) -> WorkflowInferSchemaResponse:
        node_configs = node_configs or {}
        node_map: Dict[str, Dict[str, Any]] = {str(n.get("id")): n for n in nodes if n.get("id")}
        node_schemas: Dict[str, NodeInferredSchema] = {}
        errors: List[str] = []

        # Build Graph Dependency Structure
        in_degree: Dict[str, int] = {nid: 0 for nid in node_map}
        adj_list: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        in_edges_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for edge in edges:
            src = str(edge.get("source", ""))
            tgt = str(edge.get("target", ""))
            if src in node_map and tgt in node_map:
                adj_list[src].append(edge)
                in_edges_map[tgt].append(edge)
                in_degree[tgt] += 1

        # Topological Sort via Kahn's algorithm
        queue = deque([nid for nid, deg in in_degree.items() if deg == 0])
        topo_order: List[str] = []

        while queue:
            curr = queue.popleft()
            topo_order.append(curr)
            for edge in adj_list[curr]:
                tgt = str(edge.get("target", ""))
                in_degree[tgt] -= 1
                if in_degree[tgt] == 0:
                    queue.append(tgt)

        # Fallback if cycles or disconnected components exist
        if len(topo_order) < len(node_map):
            for nid in node_map:
                if nid not in topo_order:
                    topo_order.append(nid)

        # Infer schema for each node in order
        for node_id in topo_order:
            node = node_map.get(node_id, {})
            recipe_id = str(node.get("recipe_id", "")).lower()
            incoming_edges = in_edges_map.get(node_id, [])
            config = {**node.get("config", {}), **node_configs.get(node_id, {})}

            try:
                if recipe_id in ["csv_loader", "dataset_loader", "file_ingestion"]:
                    schema = await cls._infer_dataset_loader(node_id, recipe_id, config, node, db)
                elif recipe_id == "dataset_join":
                    schema = cls._infer_dataset_join(node_id, recipe_id, config, incoming_edges, node_schemas)
                else:
                    schema = cls._infer_generic_node(
                        node_id,
                        recipe_id,
                        config,
                        incoming_edges,
                        node_schemas,
                        session_id=session_id
                    )

                node_schemas[node_id] = schema
            except Exception as e:
                logger.warning(f"Schema inference warning for node '{node_id}' ({recipe_id}): {e}")
                errors.append(f"Node '{node_id}' ({recipe_id}): {str(e)}")
                # Provide empty fallback so frontend never breaks
                node_schemas[node_id] = NodeInferredSchema(
                    node_id=node_id,
                    recipe_id=recipe_id,
                    columns=[],
                    column_details=[],
                    available_left_columns=[],
                    available_right_columns=[],
                    ports={}
                )

        return WorkflowInferSchemaResponse(
            success=len(errors) == 0,
            node_schemas=node_schemas,
            errors=errors
        )

    @classmethod
    async def _infer_dataset_loader(
        cls,
        node_id: str,
        recipe_id: str,
        config: Dict[str, Any],
        node: Dict[str, Any],
        db: Optional[AsyncSession]
    ) -> NodeInferredSchema:
        cols: List[str] = []
        col_details: List[NodeColumnInfo] = []

        dataset_id = config.get("dataset_id") or node.get("dataset_id")
        if not dataset_id and "dataset" in config:
            dataset_id = config["dataset"]

        if dataset_id and db:
            result = await db.execute(select(Dataset).where(Dataset.id == str(dataset_id)))
            dataset = result.scalar_one_or_none()
            if dataset:
                if dataset.profile and "columns" in dataset.profile:
                    cols_dict = dataset.profile["columns"]
                    if isinstance(cols_dict, dict):
                        cols = list(cols_dict.keys())
                        for cname, cinfo in cols_dict.items():
                            ctype = cinfo.get("inferred_type", "string") if isinstance(cinfo, dict) else "string"
                            raw_dtype = cinfo.get("raw_dtype") if isinstance(cinfo, dict) else None
                            col_details.append(NodeColumnInfo(name=str(cname), type=ctype, raw_dtype=raw_dtype))
                elif dataset.storage_path:
                    try:
                        df_header = storage_manager.read_dataframe(dataset.storage_path)
                        cols = list(df_header.columns)
                        col_details = [NodeColumnInfo(name=str(c), type="string") for c in cols]
                    except Exception as ex:
                        logger.debug(f"Failed to read dataset header: {ex}")

        # Port definition
        port_out = NodePortSchema(port_id="output", columns=cols, column_details=col_details)

        return NodeInferredSchema(
            node_id=node_id,
            recipe_id=recipe_id,
            columns=cols,
            column_details=col_details,
            available_left_columns=[],
            available_right_columns=[],
            ports={"output": port_out, "dataframe": port_out}
        )

    @classmethod
    def _infer_dataset_join(
        cls,
        node_id: str,
        recipe_id: str,
        config: Dict[str, Any],
        incoming_edges: List[Dict[str, Any]],
        node_schemas: Dict[str, NodeInferredSchema]
    ) -> NodeInferredSchema:
        left_cols: List[str] = []
        right_cols: List[str] = []
        left_details: List[NodeColumnInfo] = []
        right_details: List[NodeColumnInfo] = []

        # 1. Identify left and right upstream edges
        left_edge: Optional[Dict[str, Any]] = None
        right_edge: Optional[Dict[str, Any]] = None

        for edge in incoming_edges:
            th = (
                edge.get("targetHandle")
                or edge.get("target_handle")
                or edge.get("target_port")
                or ""
            ).lower().strip()
            if th in ["left", "left_dataset", "df_left", "upstream_left"]:
                left_edge = edge
            elif th in ["right", "right_dataset", "df_right", "upstream_right"]:
                right_edge = edge

        # Fallback to positional incoming edges if handles are omitted
        if left_edge is None and len(incoming_edges) >= 1:
            left_edge = incoming_edges[0]
        if right_edge is None and len(incoming_edges) >= 2:
            right_edge = incoming_edges[1]

        # Fallback to config parent IDs if edges are unrouted
        if left_edge is None and config.get("left_parent_id"):
            left_edge = {"source": config["left_parent_id"]}
        if right_edge is None and config.get("right_parent_id"):
            right_edge = {"source": config["right_parent_id"]}

        # Helper to extract columns from upstream schema
        def _get_upstream_columns(edge: Optional[Dict[str, Any]]) -> Tuple[List[str], List[NodeColumnInfo]]:
            if not edge:
                return [], []
            src = str(edge.get("source", ""))
            parent_schema = node_schemas.get(src)
            if not parent_schema:
                return [], []
            sh = (
                edge.get("sourceHandle")
                or edge.get("source_handle")
                or edge.get("source_port")
                or ""
            ).lower().strip()
            if sh and sh in parent_schema.ports:
                return parent_schema.ports[sh].columns, parent_schema.ports[sh].column_details
            return parent_schema.columns, parent_schema.column_details

        left_cols, left_details = _get_upstream_columns(left_edge)
        right_cols, right_details = _get_upstream_columns(right_edge)

        available_left = list(left_cols)
        available_right = list(right_cols)

        # 2. Apply column selections and renaming
        sel_left = config.get("selected_columns_left") or []
        sel_right = config.get("selected_columns_right") or []
        ren_left = config.get("rename_columns_left") or {}
        ren_right = config.get("rename_columns_right") or {}
        suffixes = config.get("suffixes") or ["_left", "_right"]
        sfx_l = suffixes[0] if len(suffixes) > 0 else "_left"
        sfx_r = suffixes[1] if len(suffixes) > 1 else "_right"

        left_kept = [c for c in left_cols if not sel_left or c in sel_left]
        right_kept = [c for c in right_cols if not sel_right or c in sel_right]

        left_renamed = [ren_left.get(c, c) for c in left_kept]
        right_renamed = [ren_right.get(c, c) for c in right_kept]

        # 3. Resolve Join Keys
        left_keys: List[str] = []
        right_keys: List[str] = []

        conditions = config.get("conditions")
        if conditions and isinstance(conditions, list):
            for c in conditions:
                if isinstance(c, dict):
                    lk = c.get("left") or c.get("left_column")
                    rk = c.get("right") or c.get("right_column")
                    if lk:
                        left_keys.append(str(lk).strip())
                    if rk:
                        right_keys.append(str(rk).strip())
        if config.get("on"):
            on_keys = [s.strip() for s in str(config["on"]).split(",") if s.strip()]
            left_keys.extend(on_keys)
            right_keys.extend(on_keys)
        if config.get("left_on"):
            left_keys.extend([s.strip() for s in str(config["left_on"]).split(",") if s.strip()])
        if config.get("right_on"):
            right_keys.extend([s.strip() for s in str(config["right_on"]).split(",") if s.strip()])

        # Map original keys to renamed keys
        left_keys_renamed = {ren_left.get(k, k) for k in left_keys}
        right_keys_renamed = {ren_right.get(k, k) for k in right_keys}
        same_key_names = left_keys_renamed & right_keys_renamed

        # 4. Compute Joined Output Schema
        joined_columns: List[str] = []
        joined_details: List[NodeColumnInfo] = []

        left_detail_map = {ren_left.get(d.name, d.name): d for d in left_details}
        right_detail_map = {ren_right.get(d.name, d.name): d for d in right_details}

        # Add left columns
        for c in left_renamed:
            if c in right_renamed and c not in same_key_names:
                final_name = f"{c}{sfx_l}"
            else:
                final_name = c
            joined_columns.append(final_name)
            d = left_detail_map.get(c)
            joined_details.append(NodeColumnInfo(name=final_name, type=d.type if d else "string"))

        # Add right columns
        for c in right_renamed:
            if c in same_key_names:
                # Merged into left key column
                continue
            elif c in left_renamed:
                final_name = f"{c}{sfx_r}"
            else:
                final_name = c
            joined_columns.append(final_name)
            d = right_detail_map.get(c)
            joined_details.append(NodeColumnInfo(name=final_name, type=d.type if d else "string"))

        # Add _merge column if indicator is enabled
        raw_indicator = config.get("indicator", False)
        indicator = (raw_indicator.strip().lower() in ("true", "1", "yes")) if isinstance(raw_indicator, str) else bool(raw_indicator)
        if indicator:
            joined_columns.append("_merge")
            joined_details.append(NodeColumnInfo(name="_merge", type="categorical"))

        # Build ports
        joined_port = NodePortSchema(port_id="joined", columns=joined_columns, column_details=joined_details)
        unmatched_left_port = NodePortSchema(
            port_id="unmatched_left",
            columns=left_renamed,
            column_details=[NodeColumnInfo(name=c, type=left_detail_map[c].type if c in left_detail_map else "string") for c in left_renamed]
        )
        unmatched_right_port = NodePortSchema(
            port_id="unmatched_right",
            columns=right_renamed,
            column_details=[NodeColumnInfo(name=c, type=right_detail_map[c].type if c in right_detail_map else "string") for c in right_renamed]
        )

        return NodeInferredSchema(
            node_id=node_id,
            recipe_id=recipe_id,
            columns=joined_columns,
            column_details=joined_details,
            available_left_columns=available_left,
            available_right_columns=available_right,
            ports={
                "joined": joined_port,
                "unmatched_left": unmatched_left_port,
                "unmatched_right": unmatched_right_port,
                "output": joined_port,
                "dataframe": joined_port
            }
        )

    @classmethod
    def _infer_generic_node(
        cls,
        node_id: str,
        recipe_id: str,
        config: Dict[str, Any],
        incoming_edges: List[Dict[str, Any]],
        node_schemas: Dict[str, NodeInferredSchema],
        session_id: Optional[str] = None
    ) -> NodeInferredSchema:
        parent_cols: List[str] = []
        parent_details: List[NodeColumnInfo] = []
        parent_recipe: Optional[str] = None

        if incoming_edges:
            first_edge = incoming_edges[0]
            src = str(first_edge.get("source", ""))
            parent_schema = node_schemas.get(src)
            if parent_schema:
                parent_recipe = parent_schema.recipe_id
                raw_sh = (
                    first_edge.get("sourceHandle")
                    or first_edge.get("source_handle")
                    or first_edge.get("source_port")
                    or ""
                ).lower().strip()
                sh_candidates = [
                    raw_sh,
                    raw_sh.replace(" ", "_"),
                    raw_sh.replace("_", " "),
                    raw_sh.replace("-", "_"),
                    raw_sh.replace(" ", ""),
                ]
                matched_port = None
                for c in sh_candidates:
                    if c and c in parent_schema.ports:
                        matched_port = parent_schema.ports[c]
                        break
                if matched_port and matched_port.columns:
                    parent_cols = list(matched_port.columns)
                    parent_details = list(matched_port.column_details)
                else:
                    parent_cols = list(parent_schema.columns)
                    parent_details = list(parent_schema.column_details)

        # Check if node is an independent generator or preceded by a trigger
        is_trigger_parent = parent_recipe in ["cron_trigger", "webhook_trigger", None]
        if not parent_cols or is_trigger_parent:
            if recipe_id == "stock_market_api":
                mode = str(config.get("mode", "historical")).lower().strip()
                if mode == "quote":
                    parent_cols = ["symbol", "price", "change", "change_percent", "open", "high_24h", "low_24h", "volume_24h", "market_cap_est", "last_updated"]
                else:
                    parent_cols = ["date", "open", "high", "low", "close", "adj_close", "volume", "daily_return_pct", "sma_20", "sma_50", "symbol"]
                parent_details = [NodeColumnInfo(name=c, type="numeric" if c not in ["date", "symbol", "last_updated"] else "string") for c in parent_cols]
            elif recipe_id == "currency_exchange_api":
                parent_cols = ["timestamp", "base_currency", "quote_currency", "exchange_rate", "converted_amount", "last_updated_utc"]
                parent_details = [NodeColumnInfo(name=c, type="numeric" if "rate" in c or "amount" in c else "string") for c in parent_cols]
            elif recipe_id in ["news_api", "rss_news_reader"]:
                parent_cols = ["title", "link", "published_at", "description", "author", "guid", "category", "feed_title", "feed_url"]
                parent_details = [NodeColumnInfo(name=c, type="string") for c in parent_cols]
            elif recipe_id == "open_data_api":
                parent_cols = ["country_code", "country_name", "indicator_id", "indicator_name", "year", "value"]
                parent_details = [NodeColumnInfo(name=c, type="numeric" if c in ["year", "value"] else "string") for c in parent_cols]
            elif recipe_id in ["twitter_connector", "twitter", "x_connector"]:
                parent_cols = ["tweet_id", "text", "created_at", "author_id", "author_username", "like_count", "retweet_count", "reply_count", "quote_count", "impression_count", "bookmark_count", "lang", "source"]
                parent_details = [NodeColumnInfo(name=c, type="numeric" if "count" in c else "string") for c in parent_cols]
            elif recipe_id in ["reddit_connector", "reddit"]:
                parent_cols = ["post_id", "title", "author", "subreddit", "score", "upvote_ratio", "num_comments", "created_utc", "url", "permalink", "selftext", "is_self", "over_18"]
                parent_details = [NodeColumnInfo(name=c, type="numeric" if c in ["score", "upvote_ratio", "num_comments"] else "string") for c in parent_cols]
            elif recipe_id in ["linkedin_connector", "linkedin"]:
                parent_cols = ["post_urn", "author_urn", "published_at", "commentary_text", "content_format", "like_count", "comment_count", "share_count", "impression_count", "click_count", "engagement_rate"]
                parent_details = [NodeColumnInfo(name=c, type="numeric" if "count" in c or "rate" in c else "string") for c in parent_cols]
            elif recipe_id in ["cron_trigger", "webhook_trigger"]:
                parent_cols = ["timestamp", "trigger_time", "payload"]
                parent_details = [NodeColumnInfo(name=c, type="string") for c in parent_cols]

        # Feature selection / dropping / modeling transformations
        active_cols = list(parent_cols)
        detail_map = {d.name: d for d in parent_details}

        if recipe_id == "isolation_forest":
            for col_name in ["is_anomaly", "anomaly_score"]:
                if col_name not in active_cols:
                    active_cols.append(col_name)
                    detail_map[col_name] = NodeColumnInfo(name=col_name, type="numeric")
        elif recipe_id in ["arima_forecaster", "prophet_forecaster"]:
            for col_name in ["forecast", "lower_bound", "upper_bound"]:
                if col_name not in active_cols:
                    active_cols.append(col_name)
                    detail_map[col_name] = NodeColumnInfo(name=col_name, type="numeric")
        elif recipe_id == "aggregate":
            gb_cols = config.get("group_by") or []
            if isinstance(gb_cols, str):
                gb_cols = [c.strip() for c in gb_cols.split(",") if c.strip()]
            aggs = config.get("aggregations") or []
            if gb_cols:
                active_cols = [c for c in active_cols if c in gb_cols]
            for a in aggs:
                if isinstance(a, dict):
                    c = a.get("column", "")
                    f = a.get("func", "sum")
                    if c:
                        new_col = f"{c}_{f}"
                        if new_col not in active_cols:
                            active_cols.append(new_col)
                            detail_map[new_col] = NodeColumnInfo(name=new_col, type="numeric")
        elif recipe_id == "column_selector":
            mode = str(config.get("mode", "keep")).lower().strip()
            cols = config.get("columns") or []
            if isinstance(cols, str):
                cols = [c.strip() for c in cols.split(",") if c.strip()]
            if cols:
                if mode == "keep":
                    active_cols = [c for c in active_cols if c in cols]
                elif mode == "drop":
                    active_cols = [c for c in active_cols if c not in cols]
        elif config.get("selected_features") and isinstance(config["selected_features"], list):
            sel = set(config["selected_features"])
            active_cols = [c for c in active_cols if c in sel]
        elif config.get("columns_to_drop") and isinstance(config["columns_to_drop"], list):
            drop_set = set(config["columns_to_drop"])
            active_cols = [c for c in active_cols if c not in drop_set]

        # Check session test-step cache if available for runtime truth
        if session_id:
            try:
                from backend.app.engine.execution.step_executor import NodeStepExecutor
                cached = NodeStepExecutor.load_node_output(session_id=session_id, node_id=node_id)
                if cached and isinstance(cached.get("dataframe"), pd.DataFrame):
                    df_c = cached["dataframe"]
                    if len(df_c.columns) > 0:
                        active_cols = list(df_c.columns)
                        detail_map = {
                            c: NodeColumnInfo(
                                name=c,
                                type="numeric" if pd.api.types.is_numeric_dtype(df_c[c]) else "string"
                            ) for c in active_cols
                        }
            except Exception:
                pass

        active_details = [detail_map.get(c, NodeColumnInfo(name=c, type="string")) for c in active_cols]

        port_out = NodePortSchema(port_id="output", columns=active_cols, column_details=active_details)
        ports = {
            "output": port_out,
            "dataframe": port_out,
            "output 1": port_out,
            "output_1": port_out,
            "output1": port_out,
            "output-1": port_out,
            "1": port_out,
            "out": port_out,
            "input": port_out,
            "in": port_out
        }
        if recipe_id == "isolation_forest":
            model_port = NodePortSchema(port_id="model", columns=[], column_details=[])
            metrics_port = NodePortSchema(port_id="metrics", columns=[], column_details=[])
            ports.update({
                "model": model_port,
                "output 2": model_port,
                "output_2": model_port,
                "output2": model_port,
                "metrics": metrics_port,
                "output 3": metrics_port,
                "output_3": metrics_port,
                "output3": metrics_port
            })

        return NodeInferredSchema(
            node_id=node_id,
            recipe_id=recipe_id,
            columns=active_cols,
            column_details=active_details,
            available_left_columns=list(parent_cols),
            available_right_columns=[],
            ports=ports
        )