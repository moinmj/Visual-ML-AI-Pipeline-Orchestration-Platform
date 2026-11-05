"""
Chart / Visualization Operator Recipe (Canvas Tools -> Visualization).

Creates interactive visualizations (bar, line, scatter, histogram, box, heatmap, pie)
from incoming tabular data with automated column resolution, aggregation options,
downsampling safeguards for large datasets, and formatted Plotly figure definitions.
"""

import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

try:
    import pandas as pd
    import numpy as np
except ImportError:
    pd = None
    np = None

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class ChartVisualizationRecipe(BaseRecipe):
    recipe_id = "chart_visualization"
    name = "Chart / Visualization Operator"
    version = "1.0.0"
    category = "canvas"
    group = "Canvas Tools"
    subgroup = "Visualization"
    description = "Creates charts (bar, line, scatter, histogram) from the incoming data."
    input_types = ["dataframe"]
    output_types = ["report", "dataframe"]

    inputs: List[RecipePort] = [
        RecipePort(
            id="input",
            label="Input Table",
            type="dataframe",
            required=True,
            max_connections=1,
            description="Upstream dataset to analyze and visualize."
        )
    ]

    outputs: List[RecipePort] = [
        RecipePort(
            id="output",
            label="Chart Report",
            type="report",
            description="Interactive Plotly JSON specification and visual data distributions."
        ),
        RecipePort(
            id="dataframe",
            label="Passthrough Table",
            type="dataframe",
            description="Preserved input dataset forwarded unchanged to downstream nodes."
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "chart_type": {
                    "type": "string",
                    "title": "Visualization Chart Type",
                    "input_type": "dropdown",
                    "enum": ["bar", "line", "scatter", "histogram", "box", "heatmap", "pie"],
                    "default": "bar",
                    "description": "Visual chart representation: distribution, correlation, or trend."
                },
                "x_column": {
                    "type": "string",
                    "title": "X-Axis Feature",
                    "input_type": "column_select",
                    "default": "",
                    "description": "Primary attribute plotted along the horizontal axis."
                },
                "y_column": {
                    "type": "string",
                    "title": "Y-Axis Metric (Optional for Histogram)",
                    "input_type": "column_select",
                    "default": "",
                    "description": "Numerical measurement plotted on the vertical axis (optional for histograms/counts)."
                },
                "color_column": {
                    "type": "string",
                    "title": "Color / Category Grouping",
                    "input_type": "column_select",
                    "default": "",
                    "description": "Categorical feature used to split series or color points."
                },
                "title": {
                    "type": "string",
                    "title": "Chart Title",
                    "input_type": "text",
                    "default": "Dataset Distribution & Metrics",
                    "description": "Headline text displayed atop the rendered chart."
                },
                "agg_func": {
                    "type": "string",
                    "title": "Aggregation Function",
                    "input_type": "dropdown",
                    "enum": ["none", "mean", "sum", "count", "median", "min", "max"],
                    "default": "none",
                    "description": "Statistical aggregation applied to Y values grouped by X categories."
                },
                "max_points": {
                    "type": "integer",
                    "title": "Max Points Sampled",
                    "input_type": "number",
                    "default": 2000,
                    "minimum": 50,
                    "maximum": 50000,
                    "description": "Safety threshold: downsamples large datasets to prevent browser lag."
                },
                "theme": {
                    "type": "string",
                    "title": "Visual Theme",
                    "input_type": "dropdown",
                    "enum": ["plotly_white", "plotly_dark", "seaborn", "ggplot2"],
                    "default": "plotly_white",
                    "description": "Aesthetic styling theme for colors, gridlines, and backgrounds."
                }
            }
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        # Edge Case 1: Extract upstream DataFrame safely
        df = inputs.get("dataframe") if "dataframe" in inputs else inputs.get("input")
        if df is None:
            raw_data = inputs.get("data")
            if isinstance(raw_data, list) and pd is not None:
                df = pd.DataFrame(raw_data)
            elif pd is not None:
                df = pd.DataFrame()
            else:
                df = None

        now_iso = datetime.now(timezone.utc).isoformat()
        chart_type = str(config.get("chart_type", "bar")).lower()
        if chart_type not in ["bar", "line", "scatter", "histogram", "box", "heatmap", "pie"]:
            chart_type = "bar"

        title = str(config.get("title", "Dataset Distribution & Metrics")).strip() or "Dataset Metrics"
        agg_func = str(config.get("agg_func", "none")).lower()
        theme = str(config.get("theme", "plotly_white"))
        try:
            max_points = max(50, min(50000, int(config.get("max_points", 2000))))
        except (ValueError, TypeError):
            max_points = 2000

        # Edge Case 2: DataFrame is empty or None
        if df is None or (pd is not None and isinstance(df, pd.DataFrame) and df.empty):
            empty_fig = {
                "data": [],
                "layout": {
                    "title": f"{title} (No Data Available)",
                    "template": theme,
                    "annotations": [{
                        "text": "Input table is empty. Connect a populated data source.",
                        "xref": "paper", "yref": "paper",
                        "showarrow": False, "font": {"size": 14, "color": "#94A3B8"}
                    }]
                }
            }
            empty_echarts = {
                "title": {"text": f"{title} (No Data)", "left": "center"},
                "xAxis": {"type": "category", "data": []},
                "yAxis": {"type": "value"},
                "series": []
            }
            empty_report = {
                "title": title,
                "chart_type": chart_type,
                "figure": empty_fig,
                "figure_json": empty_fig,
                "echarts_option": empty_echarts,
                "chart_option": empty_echarts,
                "markdown": f"### 📊 {title}\n*Warning: Input table was empty. No data points plotted.*",
                "timestamp": now_iso
            }
            return {
                "chart_type": chart_type,
                "title": title,
                "figure_json": empty_fig,
                "echarts_option": empty_echarts,
                "chart_option": empty_echarts,
                "metrics": {
                    "rows_evaluated": 0,
                    "columns_available": 0,
                    "downsampled": 0
                },
                "report": empty_report,
                "output": empty_report,
                "dataframe": df if df is not None else pd.DataFrame(),
                "output_summary": {"status": "EMPTY_DATA", "rows": 0, "chart_type": chart_type, "echarts_option": empty_echarts},
                "suppress_preview_rows": True
            }

        cols = list(df.columns)

        # Edge Case 3: Auto-resolve x_column and y_column if missing or invalid
        raw_x = str(config.get("x_column", "")).strip()
        raw_y = str(config.get("y_column", "")).strip()
        color_col = str(config.get("color_column", "")).strip()
        color_col = color_col if color_col in cols else None

        # Resolve X
        if raw_x and raw_x in cols:
            x_col = raw_x
        else:
            # Fallback: pick first categorical or datetime column, else first column
            cat_candidates = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
            x_col = cat_candidates[0] if cat_candidates else cols[0]

        # Resolve Y
        if raw_y and raw_y in cols:
            y_col = raw_y
        else:
            num_candidates = [c for c in cols if c != x_col and pd.api.types.is_numeric_dtype(df[c])]
            if num_candidates:
                y_col = num_candidates[0]
            elif chart_type in ["histogram", "pie"]:
                y_col = None
            else:
                y_col = cols[-1] if len(cols) > 1 else x_col

        # Edge Case 4: Handle large datasets with systematic downsampling
        total_rows = len(df)
        is_downsampled = total_rows > max_points
        if is_downsampled:
            sample_df = df.sample(n=max_points, random_state=42).copy()
        else:
            sample_df = df.copy()

        # Edge Case 5: Clean nulls in plotted features
        plot_cols = [c for c in [x_col, y_col, color_col] if c]
        clean_plot_df = sample_df.dropna(subset=[x_col])

        # Compute Aggregation if requested
        if agg_func != "none" and y_col and pd.api.types.is_numeric_dtype(clean_plot_df[y_col]):
            try:
                group_cols = [x_col]
                if color_col:
                    group_cols.append(color_col)
                if agg_func == "mean":
                    clean_plot_df = clean_plot_df.groupby(group_cols, as_index=False)[y_col].mean()
                elif agg_func == "sum":
                    clean_plot_df = clean_plot_df.groupby(group_cols, as_index=False)[y_col].sum()
                elif agg_func == "median":
                    clean_plot_df = clean_plot_df.groupby(group_cols, as_index=False)[y_col].median()
                elif agg_func == "count":
                    clean_plot_df = clean_plot_df.groupby(group_cols, as_index=False)[y_col].count()
                elif agg_func == "min":
                    clean_plot_df = clean_plot_df.groupby(group_cols, as_index=False)[y_col].min()
                elif agg_func == "max":
                    clean_plot_df = clean_plot_df.groupby(group_cols, as_index=False)[y_col].max()
            except Exception:
                pass

        # Build Plotly Traces
        traces = []
        x_vals = clean_plot_df[x_col].astype(str).tolist() if not pd.api.types.is_numeric_dtype(clean_plot_df[x_col]) else clean_plot_df[x_col].tolist()
        y_vals = clean_plot_df[y_col].tolist() if y_col else None

        if chart_type == "bar":
            traces.append({
                "type": "bar",
                "x": x_vals,
                "y": y_vals if y_vals else [1] * len(x_vals),
                "name": y_col or "Count",
                "marker": {"color": "#3B82F6"}
            })
        elif chart_type == "line":
            traces.append({
                "type": "scatter",
                "mode": "lines+markers",
                "x": x_vals,
                "y": y_vals if y_vals else list(range(len(x_vals))),
                "name": y_col or "Metric",
                "line": {"color": "#2563EB", "width": 2}
            })
        elif chart_type == "scatter":
            traces.append({
                "type": "scatter",
                "mode": "markers",
                "x": x_vals,
                "y": y_vals if y_vals else list(range(len(x_vals))),
                "name": f"{x_col} vs {y_col}",
                "marker": {"size": 7, "color": "#6366F1", "opacity": 0.75}
            })
        elif chart_type == "histogram":
            traces.append({
                "type": "histogram",
                "x": clean_plot_df[x_col].tolist(),
                "name": x_col,
                "marker": {"color": "#10B981"}
            })
        elif chart_type == "box":
            traces.append({
                "type": "box",
                "x": x_vals if len(set(x_vals)) <= 20 else None,
                "y": y_vals if y_vals else clean_plot_df[x_col].tolist(),
                "name": y_col or x_col,
                "marker": {"color": "#8B5CF6"}
            })
        elif chart_type == "pie":
            val_counts = clean_plot_df[x_col].value_counts().head(10)
            traces.append({
                "type": "pie",
                "labels": [str(l) for l in val_counts.index],
                "values": val_counts.values.tolist(),
                "name": x_col
            })
        else: # heatmap
            traces.append({
                "type": "histogram2d",
                "x": x_vals,
                "y": y_vals if y_vals else [1]*len(x_vals),
                "colorscale": "Blues"
            })

        figure_spec = {
            "data": traces,
            "layout": {
                "title": title,
                "xaxis": {"title": x_col},
                "yaxis": {"title": y_col or "Count"},
                "template": theme,
                "margin": {"l": 50, "r": 30, "t": 50, "b": 50}
            }
        }

        # Build Apache ECharts Option Specification (Matching Frontend Chart Engine)
        echarts_spec: Dict[str, Any] = {
            "title": {"text": title, "left": "center", "textStyle": {"fontSize": 14, "fontWeight": "bold"}},
            "tooltip": {"trigger": "axis"},
            "grid": {"left": "4%", "right": "4%", "bottom": "12%", "containLabel": True}
        }

        if chart_type == "bar":
            echarts_spec["tooltip"] = {"trigger": "axis", "axisPointer": {"type": "shadow"}}
            echarts_spec["xAxis"] = {"type": "category", "data": [str(x) for x in x_vals], "name": x_col}
            echarts_spec["yAxis"] = {"type": "value", "name": y_col or "Count"}
            echarts_spec["series"] = [{
                "name": y_col or "Count",
                "type": "bar",
                "data": y_vals if y_vals else [1] * len(x_vals),
                "itemStyle": {"color": "#3B82F6", "borderRadius": [4, 4, 0, 0]}
            }]
        elif chart_type == "line":
            echarts_spec["xAxis"] = {"type": "category", "data": [str(x) for x in x_vals], "name": x_col}
            echarts_spec["yAxis"] = {"type": "value", "name": y_col or "Metric"}
            echarts_spec["series"] = [{
                "name": y_col or "Metric",
                "type": "line",
                "data": y_vals if y_vals else list(range(len(x_vals))),
                "smooth": True,
                "lineStyle": {"color": "#2563EB", "width": 2},
                "itemStyle": {"color": "#2563EB"}
            }]
        elif chart_type == "scatter":
            scatter_pts = [[x, y] for x, y in zip(x_vals, y_vals if y_vals else range(len(x_vals)))]
            echarts_spec["tooltip"] = {"trigger": "item"}
            echarts_spec["xAxis"] = {"name": x_col, "type": "value" if x_vals and isinstance(x_vals[0], (int, float)) else "category"}
            echarts_spec["yAxis"] = {"name": y_col or "Value", "type": "value"}
            echarts_spec["series"] = [{
                "name": f"{x_col} vs {y_col}",
                "type": "scatter",
                "data": scatter_pts,
                "symbolSize": 8,
                "itemStyle": {"color": "#6366F1"}
            }]
        elif chart_type == "histogram":
            echarts_spec["tooltip"] = {"trigger": "axis", "axisPointer": {"type": "shadow"}}
            echarts_spec["xAxis"] = {"type": "category", "data": [str(x) for x in clean_plot_df[x_col].tolist()[:30]], "name": x_col}
            echarts_spec["yAxis"] = {"type": "value", "name": "Frequency"}
            echarts_spec["series"] = [{
                "name": "Frequency",
                "type": "bar",
                "data": y_vals if y_vals else [1] * len(clean_plot_df),
                "itemStyle": {"color": "#10B981", "borderRadius": [4, 4, 0, 0]}
            }]
        elif chart_type == "box":
            num_y = [v for v in (y_vals if y_vals else clean_plot_df[x_col].tolist()) if isinstance(v, (int, float)) and not np.isnan(v)]
            b_data = []
            if num_y:
                b_data = [[float(min(num_y)), float(np.percentile(num_y, 25)), float(np.median(num_y)), float(np.percentile(num_y, 75)), float(max(num_y))]]
            echarts_spec["tooltip"] = {"trigger": "item"}
            echarts_spec["xAxis"] = {"type": "category", "data": [x_col]}
            echarts_spec["yAxis"] = {"type": "value", "name": y_col or x_col}
            echarts_spec["series"] = [{
                "name": y_col or x_col,
                "type": "boxplot",
                "data": b_data,
                "itemStyle": {"borderColor": "#8B5CF6", "borderWidth": 2}
            }]
        elif chart_type == "pie":
            val_counts = clean_plot_df[x_col].value_counts().head(10)
            echarts_spec["tooltip"] = {"trigger": "item", "formatter": "{a} <br/>{b}: {c} ({d}%)"}
            echarts_spec["legend"] = {"orient": "horizontal", "bottom": "3%", "left": "center"}
            echarts_spec["series"] = [{
                "name": x_col,
                "type": "pie",
                "radius": "55%",
                "center": ["50%", "45%"],
                "data": [{"name": str(k), "value": int(v)} for k, v in val_counts.items()],
                "emphasis": {"itemStyle": {"shadowBlur": 10, "shadowOffsetX": 0, "shadowColor": "rgba(0, 0, 0, 0.5)"}}
            }]
        else: # heatmap
            x_uniq = list(dict.fromkeys(str(x) for x in x_vals))[:15]
            y_uniq = list(dict.fromkeys(str(y) for y in (y_vals if y_vals else [1])))[:15]
            h_data = [[i, j, 1] for i in range(len(x_uniq)) for j in range(len(y_uniq))]
            echarts_spec["tooltip"] = {"position": "top"}
            echarts_spec["xAxis"] = {"type": "category", "data": x_uniq, "name": x_col}
            echarts_spec["yAxis"] = {"type": "category", "data": y_uniq, "name": y_col or "Value"}
            echarts_spec["visualMap"] = {"min": 0, "max": 10, "calculable": True, "orient": "horizontal", "left": "center", "bottom": "0%"}
            echarts_spec["series"] = [{
                "name": "Density",
                "type": "heatmap",
                "data": h_data,
                "label": {"show": False}
            }]

        md_summary = (
            f"### 📊 {title} ({chart_type.upper()})\n"
            f"**Plotted:** `{x_col}`" + (f" ➔ `{y_col}`" if y_col else "") + "\n\n"
            f"- **Source Rows:** `{total_rows:,}`\n"
            f"- **Rendered Points:** `{len(clean_plot_df):,}`" + (f" *(downsampled from {total_rows:,})*" if is_downsampled else "") + "\n"
            f"- **Aggregation:** `{agg_func}` | **Theme:** `{theme}`"
        )

        chart_report = {
            "title": title,
            "chart_type": chart_type,
            "x_column": x_col,
            "y_column": y_col,
            "figure_json": figure_spec,
            "echarts_option": echarts_spec,
            "chart_option": echarts_spec,
            "markdown": md_summary,
            "metrics": {
                "total_rows": total_rows,
                "points_rendered": len(clean_plot_df),
                "downsampled": int(is_downsampled),
                "aggregation_applied": int(agg_func != "none")
            },
            "timestamp": now_iso
        }

        output_summary = {
            "title": f"Chart: {title} ({chart_type.upper()})",
            "chart_type": chart_type.upper(),
            "chart_title": title,
            "x_axis": x_col,
            "y_axis": y_col or "Count",
            "aggregation": agg_func,
            "points_rendered": len(clean_plot_df),
            "downsampled": "Yes" if is_downsampled else "No",
            "theme": theme,
            "status": "CHART_RENDERED",
            "echarts_option": echarts_spec,
            "figure_json": figure_spec
        }

        return {
            "output": chart_report,       # Fixed: Output port "output" is the Chart Report dict
            "report": chart_report,       # Full visual chart report
            "figure_json": figure_spec,   # Plotly specification
            "echarts_option": echarts_spec, # Apache ECharts specification (Company UI standard)
            "chart_option": echarts_spec, # Alias
            "chart_type": chart_type,
            "title": title,
            "x_column": x_col,
            "y_column": y_col,
            "dataframe": df,              # Passthrough original dataframe for downstream nodes
            "plot_data": clean_plot_df,   # Plotted coordinates table
            "metrics": {
                "total_rows": total_rows,
                "points_rendered": len(clean_plot_df),
                "downsampled": int(is_downsampled),
                "aggregation_applied": int(agg_func != "none")
            },
            "output_summary": output_summary,
            "suppress_preview_rows": True
        }
