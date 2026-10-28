import math
from collections import Counter
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union, Set
import networkx as nx

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort
from backend.app.recipes.training.encoder_utils import pass_through_metadata


class GraphNetworkAnalysisRecipe(BaseRecipe):
    """
    Graph & Network Analysis Recipe (NetworkX Centrality & Communities).
    Constructs graph networks from edge lists, detects modular communities (Louvain),
    and computes PageRank, Betweenness, Closeness, and Degree centrality to identify
    influencers, bottleneck bridges, and cluster topologies.
    """

    recipe_id = "graph_network_analysis"
    name = "Graph & Network Analysis (NetworkX Centrality & Communities)"
    version = "1.0.0"
    category = "training"
    group = "Modeling"
    subgroup = "Specialized Models"
    description = (
        "Constructs graph networks from edge lists, detects modular communities (Louvain), and "
        "calculates PageRank, Betweenness, Closeness, and Degree centrality."
    )
    input_types = ["dataframe"]
    output_types = ["dataframe", "model", "metrics", "report"]

    inputs: List[RecipePort] = [
        RecipePort(
            id="edges",
            label="Edge List / Network Interactions",
            type="dataframe",
            required=True,
            max_connections=1,
            description="Table with source node, target node, and optional edge weight."
        )
    ]

    outputs: List[RecipePort] = [
        RecipePort(
            id="nodes",
            label="Node Metrics Table",
            type="dataframe",
            description="Dataframe mapping each node to Degree, PageRank, Betweenness, Closeness, and Community ID."
        ),
        RecipePort(
            id="graph",
            label="NetworkX Graph Object",
            type="model",
            description="Constructed NetworkX Graph / DiGraph object for downstream graph algorithms."
        ),
        RecipePort(
            id="metrics",
            label="Network Topology Metrics",
            type="metrics",
            description="Graph density, node count, edge count, modularity, and top influencers."
        ),
        RecipePort(
            id="report",
            label="Executive Network Report",
            type="report",
            description="Markdown report containing network summary, community breakdown, and bridge nodes."
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "source_col": {
                    "type": "string",
                    "title": "Source Node Column",
                    "input_type": "text",
                    "default": "source",
                    "description": "Column defining origin / starting node of each edge."
                },
                "target_col": {
                    "type": "string",
                    "title": "Target Node Column",
                    "input_type": "text",
                    "default": "target",
                    "description": "Column defining destination / ending node of each edge."
                },
                "weight_col": {
                    "type": "string",
                    "title": "Optional Edge Weight Column",
                    "input_type": "text",
                    "default": "",
                    "description": "Numeric column representing connection strength, frequency, or capacity."
                },
                "is_directed": {
                    "type": "boolean",
                    "title": "Directed Network (DiGraph)",
                    "input_type": "toggle",
                    "default": False,
                    "description": "True for directed graphs (e.g. following, transactions); False for mutual relationships."
                },
                "community_algorithm": {
                    "type": "string",
                    "title": "Community Detection Method",
                    "input_type": "dropdown",
                    "enum": ["louvain", "greedy_modularity", "none"],
                    "default": "louvain",
                    "description": "Louvain modularity optimization detects natural clusters and dense sub-graphs."
                },
                "compute_betweenness": {
                    "type": "boolean",
                    "title": "Compute Betweenness Centrality",
                    "input_type": "toggle",
                    "default": True,
                    "description": "Measures how often a node acts as a bridge along the shortest path between other nodes."
                },
                "compute_closeness": {
                    "type": "boolean",
                    "title": "Compute Closeness Centrality",
                    "input_type": "toggle",
                    "default": True,
                    "description": "Measures average distance from a node to all other reachable nodes."
                },
                "top_k_influencers": {
                    "type": "integer",
                    "title": "Top-K Influencers to Report",
                    "input_type": "slider",
                    "default": 5,
                    "minimum": 3,
                    "maximum": 20
                }
            },
            "required": []
        }

    def _auto_detect_columns(self, df: pd.DataFrame, src_cfg: str, tgt_cfg: str, wt_cfg: str) -> Tuple[str, str, str]:
        cols_lower = {c.lower(): c for c in df.columns}

        # Resolve source
        src_col = ""
        if src_cfg in df.columns:
            src_col = src_cfg
        elif src_cfg.lower() in cols_lower:
            src_col = cols_lower[src_cfg.lower()]
        else:
            candidates = ["source", "src", "from", "node1", "user1", "origin", "sender", "start", "u"]
            for cand in candidates:
                if cand in cols_lower:
                    src_col = cols_lower[cand]
                    break

        # Resolve target
        tgt_col = ""
        if tgt_cfg in df.columns:
            tgt_col = tgt_cfg
        elif tgt_cfg.lower() in cols_lower:
            tgt_col = cols_lower[tgt_cfg.lower()]
        else:
            candidates = ["target", "dst", "to", "node2", "user2", "destination", "receiver", "end", "v"]
            for cand in candidates:
                if cand in cols_lower:
                    tgt_col = cols_lower[cand]
                    break

        # Resolve weight
        wt_col = ""
        if wt_cfg and wt_cfg in df.columns:
            wt_col = wt_cfg
        elif wt_cfg and wt_cfg.lower() in cols_lower:
            wt_col = cols_lower[wt_cfg.lower()]
        else:
            candidates = ["weight", "strength", "count", "capacity", "frequency", "score", "value"]
            for cand in candidates:
                if cand in cols_lower:
                    wt_col = cols_lower[cand]
                    break

        return src_col, tgt_col, wt_col

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        # 1. Resolve Dataframe
        df = inputs.get("dataframe")
        if df is None:
            df = inputs.get("edges")
        if df is None:
            df = inputs.get("data")
        if df is None and isinstance(context, dict):
            df = context.get("dataframe")
            if df is None:
                df = context.get("edges")

        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("GraphNetworkAnalysis expects a non-empty pandas DataFrame in inputs.")

        df = df.copy()

        # 2. Detect & Validate Columns
        src_cfg = config.get("source_col", "source")
        tgt_cfg = config.get("target_col", "target")
        wt_cfg = config.get("weight_col", "")

        src_col, tgt_col, wt_col = self._auto_detect_columns(df, src_cfg, tgt_cfg, wt_cfg)

        if not src_col or src_col not in df.columns:
            raise ValueError(
                f"GraphNetworkAnalysis could not locate source column '{src_cfg}'. Available columns: {list(df.columns)}"
            )
        if not tgt_col or tgt_col not in df.columns:
            raise ValueError(
                f"GraphNetworkAnalysis could not locate target column '{tgt_cfg}'. Available columns: {list(df.columns)}"
            )

        df = df.dropna(subset=[src_col, tgt_col])
        if len(df) < 1:
            raise ValueError("Edge list contains no valid edges after dropping null values.")

        # 3. Construct NetworkX Graph
        is_directed = bool(config.get("is_directed", False))
        G = nx.DiGraph() if is_directed else nx.Graph()

        has_weights = bool(wt_col and wt_col in df.columns)
        if has_weights:
            try:
                df["_weight"] = pd.to_numeric(df[wt_col]).fillna(1.0)
            except Exception:
                df["_weight"] = 1.0

            for _, r in df[[src_col, tgt_col, "_weight"]].iterrows():
                u = str(r[src_col])
                v = str(r[tgt_col])
                w = float(r["_weight"])
                G.add_edge(u, v, weight=w)
        else:
            for _, r in df[[src_col, tgt_col]].iterrows():
                u = str(r[src_col])
                v = str(r[tgt_col])
                G.add_edge(u, v, weight=1.0)

        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()

        if n_nodes < 2:
            raise ValueError(f"Graph requires at least 2 distinct nodes, found {n_nodes}.")

        # 4. Topology Metrics
        density = float(round(nx.density(G), 4))
        G_undirected = G.to_undirected() if is_directed else G
        n_components = nx.number_connected_components(G_undirected)

        try:
            avg_clustering = float(round(nx.average_clustering(G_undirected), 4))
        except Exception:
            avg_clustering = 0.0

        # 5. Compute Centrality Measures
        degree_dict = dict(G.degree())
        degree_cent = nx.degree_centrality(G)

        # PageRank
        try:
            pagerank_dict = nx.pagerank(G, weight="weight" if has_weights else None)
        except Exception:
            pagerank_dict = {n: 1.0 / n_nodes for n in G.nodes()}

        # Betweenness Centrality
        compute_betweenness = bool(config.get("compute_betweenness", True))
        if compute_betweenness:
            # For large graphs (N > 500), use k-sample approximation for speed
            k_samples = 100 if n_nodes > 500 else None
            betweenness_dict = nx.betweenness_centrality(G, k=k_samples, weight="weight" if has_weights else None)
        else:
            betweenness_dict = {n: 0.0 for n in G.nodes()}

        # Closeness Centrality
        compute_closeness = bool(config.get("compute_closeness", True))
        if compute_closeness:
            closeness_dict = nx.closeness_centrality(G)
        else:
            closeness_dict = {n: 0.0 for n in G.nodes()}

        # 6. Community Detection
        comm_algo = config.get("community_algorithm", "louvain")
        node_community: Dict[str, int] = {}
        modularity_score = None
        communities_list: List[Set[str]] = []

        if comm_algo == "louvain":
            try:
                communities_list = list(nx.community.louvain_communities(G_undirected, seed=42))
                for c_idx, comm in enumerate(communities_list, start=1):
                    for node in comm:
                        node_community[node] = c_idx
                if len(communities_list) > 1:
                    modularity_score = float(round(nx.community.modularity(G_undirected, communities_list), 4))
            except Exception:
                communities_list = []
        elif comm_algo == "greedy_modularity":
            try:
                communities_list = list(nx.community.greedy_modularity_communities(G_undirected))
                for c_idx, comm in enumerate(communities_list, start=1):
                    for node in comm:
                        node_community[node] = c_idx
                if len(communities_list) > 1:
                    modularity_score = float(round(nx.community.modularity(G_undirected, communities_list), 4))
            except Exception:
                communities_list = []

        if not node_community:
            node_community = {n: 1 for n in G.nodes()}

        n_communities = len(set(node_community.values()))

        # 7. Construct Node Metrics Table
        node_rows = []
        for n in G.nodes():
            node_rows.append({
                "node_id": n,
                "degree": degree_dict.get(n, 0),
                "degree_centrality": float(round(degree_cent.get(n, 0.0), 4)),
                "pagerank": float(round(pagerank_dict.get(n, 0.0), 6)),
                "betweenness_centrality": float(round(betweenness_dict.get(n, 0.0), 4)),
                "closeness_centrality": float(round(closeness_dict.get(n, 0.0), 4)),
                "community_id": node_community.get(n, 1)
            })

        nodes_df = pd.DataFrame(node_rows)
        nodes_df = nodes_df.sort_values(by="pagerank", ascending=False).reset_index(drop=True)

        # 8. Top Influencers & Hubs
        top_k = int(config.get("top_k_influencers", 5))
        top_hubs = nodes_df.nlargest(top_k, "pagerank")[["node_id", "pagerank", "degree", "community_id"]].to_dict(orient="records")
        top_bridges = nodes_df.nlargest(top_k, "betweenness_centrality")[["node_id", "betweenness_centrality", "community_id"]].to_dict(orient="records")

        # Community distribution
        comm_counts = Counter(node_community.values())
        top_comm_summary = [{"community_id": cid, "node_count": cnt, "pct": round(cnt / n_nodes * 100.0, 1)} for cid, cnt in comm_counts.most_common(5)]

        metrics = {
            "network_type": "Directed (DiGraph)" if is_directed else "Undirected (Graph)",
            "total_nodes": n_nodes,
            "total_edges": n_edges,
            "graph_density": density,
            "connected_components": n_components,
            "average_clustering_coefficient": avg_clustering,
            "communities_detected": n_communities,
            "modularity_score": modularity_score,
            "top_pagerank_hubs": top_hubs,
            "top_betweenness_bridges": top_bridges,
            "community_distribution": top_comm_summary
        }

        # 9. Format Executive Network Report
        report_lines = [
            f"# === Graph & Network Analysis Report ({metrics['network_type']}) ===",
            f"- **Network Scale**: {n_nodes:,} nodes and {n_edges:,} edges",
            f"- **Graph Density**: `{density:.4f}` | **Clustering Coefficient**: `{avg_clustering:.4f}`",
            f"- **Connected Sub-Components**: {n_components} component(s)",
            f"- **Modular Communities**: {n_communities} clusters (Modularity: **{modularity_score if modularity_score is not None else 'N/A'}**)",
            "",
            f"## Top {top_k} Network Hubs (Ranked by PageRank Influence):",
            "| Node Identifier | PageRank Score | Degree (Connections) | Community ID |",
            "| :--- | :--- | :--- | :--- |"
        ]
        for hub in top_hubs:
            report_lines.append(
                f"| `{hub['node_id']}` | **{hub['pagerank']:.6f}** | {hub['degree']} connections | Community #{hub['community_id']} |"
            )

        report_lines.append("")
        report_lines.append(f"## Top {top_k} Information Brokers & Bridges (Ranked by Betweenness):")
        report_lines.append("| Node Identifier | Betweenness Centrality | Community ID |")
        report_lines.append("| :--- | :--- | :--- |")
        for b in top_bridges:
            report_lines.append(
                f"| `{b['node_id']}` | **{b['betweenness_centrality']:.4f}** | Community #{b['community_id']} |"
            )

        report_lines.append("")
        report_lines.append("## Community Cluster Breakdown:")
        report_lines.append("| Community # | Nodes in Cluster | Share of Network |")
        report_lines.append("| :--- | :--- | :--- |")
        for c in top_comm_summary:
            bar = "█" * max(1, int(c["pct"] / 5))
            report_lines.append(f"| Cluster {c['community_id']} | {c['node_count']:,} nodes | **{c['pct']:.1f}%** ({bar}) |")

        report_markdown = "\n".join(report_lines)

        output: Dict[str, Any] = {
            "model": G,
            "graph": G,
            "nodes": nodes_df,
            "dataframe": nodes_df,
            "metrics": metrics,
            "report": report_markdown
        }

        return pass_through_metadata(inputs, output, context)

    def to_code(self, config: Dict[str, Any]) -> str:
        s_col = config.get("source_col", "source")
        t_col = config.get("target_col", "target")
        is_dir = config.get("is_directed", False)
        cls_name = "DiGraph" if is_dir else "Graph"
        return (
            f"import networkx as nx\n"
            f"import pandas as pd\n\n"
            f"# 1. Build NetworkX graph from edge list\n"
            f"G = nx.{cls_name}()\n"
            f"for _, r in df[['{s_col}', '{t_col}']].iterrows():\n"
            f"    G.add_edge(r['{s_col}'], r['{t_col}'])\n\n"
            "# 2. Calculate PageRank and Louvain communities\n"
            "pr = nx.pagerank(G)\n"
            "communities = nx.community.louvain_communities(G.to_undirected())\n"
            "print(f'Graph constructed with {G.number_of_nodes()} nodes and {len(communities)} communities')"
        )
