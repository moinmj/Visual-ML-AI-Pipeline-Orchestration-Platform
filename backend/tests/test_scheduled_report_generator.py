import base64
import io
import pandas as pd
import pytest

from backend.app.recipes.reporting.scheduled_report_generator import ScheduledReportGeneratorRecipe


def test_scheduled_report_generator_metadata_and_schema():
    recipe = ScheduledReportGeneratorRecipe()
    assert recipe.recipe_id == "scheduled_report_generator"
    assert recipe.category == "reporting"
    assert "report" in recipe.output_types

    schema = recipe.get_schema()
    assert schema["type"] == "object"
    props = schema["properties"]
    assert "report_format" in props
    assert "schedule_frequency" in props
    assert "recipients" in props

    # Verify input_type and description contract on all properties
    for prop_name, prop_data in props.items():
        assert "description" in prop_data, f"Property {prop_name} missing description"
        assert "input_type" in prop_data, f"Property {prop_name} missing input_type"


def test_scheduled_report_generator_fallback_and_default_pdf():
    recipe = ScheduledReportGeneratorRecipe()
    result = recipe.execute({}, {})
    assert "output" in result
    assert "report" in result
    assert "dataframe" in result

    report = result["report"]
    assert report["status"] == "GENERATED"
    assert report["format"] == "pdf"
    assert "pdf" in report["artifacts"]
    pdf_info = report["artifacts"]["pdf"]
    assert pdf_info["mime_type"] == "application/pdf"
    assert pdf_info["size_bytes"] > 0
    # Decode base64 to ensure it has valid PDF header
    raw_pdf = base64.b64decode(pdf_info["base64"])
    assert raw_pdf.startswith(b"%PDF-1.4")
    assert b"%%EOF" in raw_pdf


def test_scheduled_report_generator_excel_and_html_generation():
    recipe = ScheduledReportGeneratorRecipe()
    df = pd.DataFrame({
        "customer_id": [101, 102, 103, 104],
        "mrr": [1200.5, 3400.0, 890.25, 4500.0],
        "churn_risk": ["LOW", "HIGH", "MEDIUM", "LOW"],
    })

    result = recipe.execute(
        {"input": df},
        {
            "report_format": "all",
            "report_title": "Monthly SaaS Cohort Report",
            "schedule_frequency": "weekly",
            "recipients": "exec@company.com, ops@company.com",
            "include_summary_stats": True,
        },
    )

    report = result["report"]
    assert report["title"] == "Monthly SaaS Cohort Report"
    assert len(report["recipients"]) == 2
    assert "pdf" in report["artifacts"]
    assert "excel" in report["artifacts"]
    assert "html" in report["artifacts"]

    excel_info = report["artifacts"]["excel"]
    assert excel_info["mime_type"] == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    raw_excel = base64.b64decode(excel_info["base64"])
    excel_file = io.BytesIO(raw_excel)
    excel_dfs = pd.read_excel(excel_file, sheet_name=None)
    assert "Executive_Summary" in excel_dfs
    assert "Data_Payload" in excel_dfs
    assert len(excel_dfs["Data_Payload"]) == 4


def test_scheduled_report_generator_cron_and_passthrough():
    recipe = ScheduledReportGeneratorRecipe()
    df = pd.DataFrame({"sensor": ["A", "B"], "temp": [21.5, 23.0]})
    result = recipe.execute(
        {"dataframe": df},
        {
            "report_format": "html",
            "schedule_frequency": "cron",
            "cron_expression": "0 12 * * 1-5",
            "timezone": "America/New_York",
        },
    )

    report = result["output"]
    assert report["schedule"]["frequency"] == "cron"
    assert report["schedule"]["cron"] == "0 12 * * 1-5"
    assert report["schedule"]["timezone"] == "America/New_York"
    assert result["dataframe"].equals(df)

