import base64
import datetime
import io
import hashlib
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


def _generate_minimal_pdf(title: str, lines: List[str]) -> bytes:
    """
    Constructs a structurally valid, self-contained PDF 1.4 document in pure Python.
    No third-party binary dependencies required.
    """
    escaped_title = title.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream_lines = [
        "BT",
        "/F1 16 Tf",
        "50 750 Td",
        f"({escaped_title}) Tj",
        "/F1 10 Tf",
        "0 -24 Td",
    ]
    for line in lines[:35]:  # fit on first page
        clean_text = str(line).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream_lines.append(f"({clean_text}) Tj")
        stream_lines.append("0 -15 Td")
    stream_lines.append("ET")
    stream_content = "\n".join(stream_lines).encode("latin-1", errors="replace")

    objects: List[bytes] = []
    # Obj 1: Catalog
    objects.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    # Obj 2: Pages
    objects.append(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
    # Obj 3: Page
    objects.append(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
    )
    # Obj 4: Content Stream
    stream_len = len(stream_content)
    objects.append(
        f"4 0 obj\n<< /Length {stream_len} >>\nstream\n".encode("ascii")
        + stream_content
        + b"\nendstream\nendobj\n"
    )
    # Obj 5: Standard Base-14 Font (Helvetica)
    objects.append(b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")

    # Build PDF with xref table
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for obj in objects:
        offsets.append(out.tell())
        out.write(obj)

    xref_offset = out.tell()
    out.write(b"xref\n")
    out.write(f"0 {len(objects) + 1}\n".encode("ascii"))
    out.write(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.write(f"{off:010d} 00000 n \n".encode("ascii"))

    out.write(b"trailer\n")
    out.write(f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode("ascii"))
    out.write(b"startxref\n")
    out.write(f"{xref_offset}\n".encode("ascii"))
    out.write(b"%%EOF\n")
    return out.getvalue()


class ScheduledReportGeneratorRecipe(BaseRecipe):
    recipe_id = "scheduled_report_generator"
    name = "Scheduled Report Generator (PDF/Excel snapshot)"
    version = "1.0.0"
    category = "reporting"
    description = "Generates PDF or Excel report snapshots on a schedule."
    input_types = ["dataframe"]
    output_types = ["report", "dataframe"]

    def get_inputs(self) -> List[RecipePort]:
        return [
            RecipePort(
                id="input",
                label="Input Table",
                type="dataframe",
                required=False,
                description="Dataset to snapshot into an executive report",
            )
        ]

    def get_outputs(self) -> List[RecipePort]:
        return [
            RecipePort(
                id="output",
                label="Output (report)",
                type="report",
                description="Generated report payload, artifact base64 data, and schedule metadata",
            ),
            RecipePort(
                id="output_table",
                label="Passthrough Output",
                type="dataframe",
                description="Source dataset passed through for downstream nodes",
            ),
        ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "report_format": {
                    "type": "string",
                    "title": "Report Output Format",
                    "enum": ["pdf", "excel", "html", "all"],
                    "default": "pdf",
                    "description": "File format of the generated snapshot: PDF document, multi-tab Excel workbook, or HTML.",
                    "input_type": "select",
                },
                "report_title": {
                    "type": "string",
                    "title": "Report Title",
                    "default": "Executive Performance Snapshot",
                    "description": "Header title displayed at the top of the generated document.",
                    "input_type": "text",
                },
                "schedule_frequency": {
                    "type": "string",
                    "title": "Cadence / Frequency",
                    "enum": ["daily", "weekly", "monthly", "hourly", "cron"],
                    "default": "daily",
                    "description": "Recurrence cadence determining automated generation intervals.",
                    "input_type": "select",
                },
                "cron_expression": {
                    "type": "string",
                    "title": "Custom Cron Expression",
                    "default": "0 8 * * 1-5",
                    "description": "5-part standard cron schedule used when schedule_frequency is set to 'cron'.",
                    "input_type": "text",
                },
                "timezone": {
                    "type": "string",
                    "title": "Schedule Timezone",
                    "default": "UTC",
                    "description": "Timezone identifier (e.g. UTC, America/New_York, Asia/Kolkata).",
                    "input_type": "text",
                },
                "recipients": {
                    "type": "string",
                    "title": "Email Distribution List",
                    "default": "",
                    "description": "Comma-separated email addresses to notify upon snapshot generation.",
                    "input_type": "text",
                },
                "include_summary_stats": {
                    "type": "boolean",
                    "title": "Include Statistical Summary",
                    "default": True,
                    "description": "Calculates descriptive statistics (mean, quantiles, null counts) in the summary sheet.",
                    "input_type": "checkbox",
                },
                "include_sample_rows": {
                    "type": "integer",
                    "title": "Sample Rows Included",
                    "default": 20,
                    "minimum": 0,
                    "maximum": 500,
                    "description": "Number of preview rows embedded in the report body.",
                    "input_type": "number",
                },
                "retention_days": {
                    "type": "integer",
                    "title": "Snapshot Retention (Days)",
                    "default": 30,
                    "minimum": 1,
                    "maximum": 365,
                    "description": "Number of days report artifacts remain archived before automated cleanup.",
                    "input_type": "number",
                },
            },
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        df = inputs.get("dataframe")
        if df is None and isinstance(inputs.get("input"), pd.DataFrame):
            df = inputs["input"]

        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            df = pd.DataFrame({
                "metric_id": [f"M-{i:03d}" for i in range(1, 6)],
                "kpi_name": ["Pipeline Latency", "Throughput RPM", "Error Rate", "CPU Utilization", "Memory Footprint"],
                "value": [142.5, 9850.0, 0.002, 64.8, 78.2],
                "status": ["NOMINAL", "OPTIMAL", "NOMINAL", "ELEVATED", "NOMINAL"],
                "timestamp": [datetime.datetime.now(datetime.timezone.utc).isoformat()] * 5,
            })

        format_mode = str(config.get("report_format", "pdf")).lower().strip()
        title = str(config.get("report_title", "Executive Performance Snapshot")).strip() or "Executive Performance Snapshot"
        freq = str(config.get("schedule_frequency", "daily")).lower().strip()
        cron_expr = str(config.get("cron_expression", "0 8 * * 1-5")).strip()
        tz = str(config.get("timezone", "UTC")).strip()
        recipients_str = str(config.get("recipients", "")).strip()
        recipients = [r.strip() for r in recipients_str.split(",") if r.strip()]
        include_stats = bool(config.get("include_summary_stats", True))
        sample_rows_count = max(0, min(500, int(config.get("include_sample_rows", 20))))
        retention_days = max(1, int(config.get("retention_days", 30)))

        now_utc = datetime.datetime.now(datetime.timezone.utc)
        now_iso = now_utc.isoformat() + "Z"

        # Calculate next scheduled run
        if freq == "hourly":
            next_run = now_utc + datetime.timedelta(hours=1)
        elif freq == "weekly":
            next_run = now_utc + datetime.timedelta(days=7)
        elif freq == "monthly":
            next_run = now_utc + datetime.timedelta(days=30)
        else:  # daily or cron default
            next_run = now_utc + datetime.timedelta(days=1)
        next_run_iso = next_run.isoformat() + "Z"

        # Compute data profile
        total_rows = len(df)
        total_cols = len(df.columns)
        memory_kb = round(df.memory_usage(deep=True).sum() / 1024.0, 2)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        null_counts = {str(c): int(df[c].isnull().sum()) for c in df.columns}

        summary_metrics: Dict[str, Any] = {
            "total_rows": total_rows,
            "total_columns": total_cols,
            "memory_footprint_kb": memory_kb,
            "numeric_columns_count": len(numeric_cols),
            "columns": list(df.columns),
            "null_values": null_counts,
        }

        if include_stats and numeric_cols:
            stats_df = df[numeric_cols].describe().T[["mean", "std", "min", "50%", "max"]]
            stats_records = stats_df.reset_index().rename(columns={"index": "column", "50%": "median"}).to_dict(orient="records")
            summary_metrics["numerical_statistics"] = stats_records

        sample_slice = df.head(sample_rows_count) if sample_rows_count > 0 else df.head(5)
        sample_records = sample_slice.to_dict(orient="records")

        # HTML Snapshot construction
        html_rows = ""
        for rec in sample_records:
            html_rows += "<tr>" + "".join(f"<td>{v}</td>" for v in rec.values()) + "</tr>\n"
        html_headers = "".join(f"<th>{c}</th>" for c in df.columns)

        html_doc = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; margin: 30px; color: #1e293b; }}
  h1 {{ color: #0f172a; border-bottom: 2px solid #3b82f6; padding-bottom: 8px; font-size: 24px; }}
  .meta-box {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; margin-bottom: 24px; display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }}
  .meta-item {{ font-size: 13px; }}
  .meta-label {{ font-weight: 600; color: #64748b; text-transform: uppercase; font-size: 11px; }}
  .meta-value {{ font-size: 16px; font-weight: bold; color: #1e293b; margin-top: 4px; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 16px; font-size: 12px; }}
  th {{ background: #f1f5f9; color: #475569; text-align: left; padding: 8px 12px; border: 1px solid #cbd5e1; }}
  td {{ padding: 8px 12px; border: 1px solid #e2e8f0; }}
  tr:nth-child(even) {{ background: #f8fafc; }}
  .footer {{ margin-top: 32px; font-size: 11px; color: #94a3b8; border-top: 1px solid #e2e8f0; padding-top: 8px; }}
</style>
</head>
<body>
  <h1>{title}</h1>
  <div class="meta-box">
    <div class="meta-item"><div class="meta-label">Generated At</div><div class="meta-value">{now_iso}</div></div>
    <div class="meta-item"><div class="meta-label">Total Rows</div><div class="meta-value">{total_rows:,}</div></div>
    <div class="meta-item"><div class="meta-label">Total Columns</div><div class="meta-value">{total_cols}</div></div>
    <div class="meta-item"><div class="meta-label">Cadence</div><div class="meta-value">{freq.upper()}</div></div>
  </div>
  <h3>Data Sample ({len(sample_slice)} rows)</h3>
  <table>
    <thead><tr>{html_headers}</tr></thead>
    <tbody>{html_rows}</tbody>
  </table>
  <div class="footer">Visual ML/AI Platform Scheduled Report &bull; Next Execution: {next_run_iso} &bull; Retention: {retention_days} Days</div>
</body>
</html>"""

        artifacts: Dict[str, Any] = {}

        # 1. HTML artifact
        if format_mode in ["html", "all"]:
            html_bytes = html_doc.encode("utf-8")
            artifacts["html"] = {
                "filename": f"{title.lower().replace(' ', '_')}_{now_utc.strftime('%Y%m%d_%H%M%S')}.html",
                "mime_type": "text/html",
                "size_bytes": len(html_bytes),
                "base64": base64.b64encode(html_bytes).decode("ascii"),
                "sha256": hashlib.sha256(html_bytes).hexdigest(),
            }

        # 2. PDF artifact
        if format_mode in ["pdf", "all"]:
            pdf_lines = [
                f"Generated: {now_iso} (Cadence: {freq})",
                f"Next Run: {next_run_iso} | Timezone: {tz}",
                f"Dataset Summary: {total_rows} rows x {total_cols} columns ({memory_kb} KB)",
                f"Columns: {', '.join(df.columns[:8])}",
                f"Sample Snapshot ({len(sample_slice)} records):",
                "--------------------------------------------------------------------------------",
            ]
            for i, rec in enumerate(sample_records[:15]):
                line_str = " | ".join(f"{k}: {v}" for k, v in list(rec.items())[:5])
                pdf_lines.append(f"{i+1}. {line_str}")

            pdf_bytes = _generate_minimal_pdf(title, pdf_lines)
            artifacts["pdf"] = {
                "filename": f"{title.lower().replace(' ', '_')}_{now_utc.strftime('%Y%m%d_%H%M%S')}.pdf",
                "mime_type": "application/pdf",
                "size_bytes": len(pdf_bytes),
                "base64": base64.b64encode(pdf_bytes).decode("ascii"),
                "sha256": hashlib.sha256(pdf_bytes).hexdigest(),
            }

        # 3. Excel artifact
        if format_mode in ["excel", "all"]:
            excel_io = io.BytesIO()
            with pd.ExcelWriter(excel_io, engine="openpyxl") as writer:
                # Summary Sheet
                summary_data = [
                    {"Metric": "Report Title", "Value": title},
                    {"Metric": "Generated Timestamp", "Value": now_iso},
                    {"Metric": "Cadence", "Value": freq},
                    {"Metric": "Next Run", "Value": next_run_iso},
                    {"Metric": "Total Records", "Value": total_rows},
                    {"Metric": "Total Columns", "Value": total_cols},
                    {"Metric": "Memory Footprint (KB)", "Value": memory_kb},
                    {"Metric": "Recipients", "Value": ", ".join(recipients) if recipients else "None"},
                ]
                pd.DataFrame(summary_data).to_excel(writer, sheet_name="Executive_Summary", index=False)

                # Data Sheet
                df.head(1000).to_excel(writer, sheet_name="Data_Payload", index=False)

                # Stats Sheet
                if include_stats and numeric_cols:
                    df[numeric_cols].describe().T.reset_index().rename(columns={"index": "column"}).to_excel(
                        writer, sheet_name="Numeric_Statistics", index=False
                    )

            excel_bytes = excel_io.getvalue()
            artifacts["excel"] = {
                "filename": f"{title.lower().replace(' ', '_')}_{now_utc.strftime('%Y%m%d_%H%M%S')}.xlsx",
                "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "size_bytes": len(excel_bytes),
                "base64": base64.b64encode(excel_bytes).decode("ascii"),
                "sha256": hashlib.sha256(excel_bytes).hexdigest(),
            }

        report_payload = {
            "report_id": f"REP-{hashlib.md5(f'{title}_{now_iso}'.encode()).hexdigest()[:10].upper()}",
            "title": title,
            "format": format_mode,
            "status": "GENERATED",
            "schedule": {
                "frequency": freq,
                "cron": cron_expr if freq == "cron" else None,
                "timezone": tz,
                "generated_at": now_iso,
                "next_scheduled_run": next_run_iso,
                "retention_days": retention_days,
            },
            "recipients": recipients,
            "summary_metrics": summary_metrics,
            "artifacts": artifacts,
            "preview_html": html_doc[:2000],
        }

        return {
            "output": report_payload,
            "report": report_payload,
            "dataframe": df,
            "output_table": df,
        }
