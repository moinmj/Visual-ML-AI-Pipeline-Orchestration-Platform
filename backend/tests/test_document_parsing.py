import os
import io
import json
import tempfile
import pytest
import pandas as pd
from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.nlp.document_parsing import (
    DocumentParsingRecipe,
    clean_extracted_text,
    parse_markdown_tables,
    parse_html_tables,
    parse_docx_bytes,
    parse_pdf_bytes,
)

# Optional helper packages for testing real file formats
try:
    import docx
except ImportError:
    docx = None

try:
    import pypdf
except ImportError:
    pypdf = None


def test_document_parsing_recipe_registration():
    """Verifies that DocumentParsingRecipe is correctly registered with metadata."""
    recipe = recipe_registry.get("document_parsing")
    assert recipe is not None
    assert recipe.recipe_id == "document_parsing"
    assert recipe.name == "Document Parsing"
    assert recipe.category == "nlp"
    assert recipe.subgroup == "NLP/Text"
    assert "PDF" in recipe.description or "documents" in recipe.description

    meta = recipe_registry.get_metadata("document_parsing")
    assert meta is not None
    assert meta.recipe_id == "document_parsing"
    assert "extraction_target" in meta.parameters_schema["properties"]
    assert "output_format" in meta.parameters_schema["properties"]

    # Verify ports
    inputs = recipe.get_inputs()
    assert len(inputs) == 1
    assert inputs[0].id == "dataframe"

    outputs = recipe.get_outputs()
    assert len(outputs) >= 2
    out_ids = [p.id for p in outputs]
    assert "dataframe" in out_ids
    assert "tables" in out_ids


def test_document_parsing_demo_generation():
    """Verifies default demonstration extraction when run without inputs or files."""
    recipe = DocumentParsingRecipe()
    result = recipe.execute(inputs={}, config={})

    assert "dataframe" in result
    assert "tables" in result
    assert "metrics" in result
    assert "output_summary" in result

    df = result["dataframe"]
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "document_name" in df.columns
    assert "extracted_text" in df.columns
    assert "tables_count" in df.columns

    tables_df = result["tables"]
    assert isinstance(tables_df, pd.DataFrame)
    assert not tables_df.empty
    assert "document_name" in tables_df.columns
    assert "Quarter" in tables_df.columns or "Employee Name" in tables_df.columns

    metrics = result["metrics"]
    assert metrics["total_documents_processed"] == 2
    assert metrics["total_tables_extracted"] == 2
    assert result["output_summary"]["status"] == "success"


def test_document_parsing_markdown_and_plain_text():
    """Tests parsing markdown and text documents containing embedded tables."""
    raw_md = (
        "# System Architecture Specification\n\n"
        "This service provides real-time document extraction and indexing.\n\n"
        "| Service | Port | Protocol | Status |\n"
        "|---|---|---|---|\n"
        "| Ingestion Gateway | 8000 | HTTP | Active |\n"
        "| Vector Store | 6333 | gRPC | Ready |\n"
        "| Worker Pool | 9092 | TCP | Standby |\n\n"
        "End of specification document."
    )

    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(raw_md)
        f_path = f.name

    try:
        recipe = DocumentParsingRecipe()
        result = recipe.execute(
            inputs={},
            config={"document_path": f_path}
        )
        df = result["dataframe"]
        assert len(df) == 1
        assert "System Architecture" in df.iloc[0]["extracted_text"]
        assert df.iloc[0]["tables_count"] == 1

        tables_df = result["tables"]
        assert len(tables_df) == 3
        assert "Service" in tables_df.columns
        assert "Ingestion Gateway" in tables_df["Service"].values
    finally:
        if os.path.exists(f_path):
            os.remove(f_path)


def test_document_parsing_html_table():
    """Tests parsing HTML files containing prose and <table> structures."""
    html_content = (
        "<html><body>"
        "<h1>Q3 Inventory Summary</h1>"
        "<p>Current warehouse stock levels are detailed below.</p>"
        "<table border='1'>"
        "<tr><th>SKU</th><th>Item</th><th>Stock</th></tr>"
        "<tr><td>SKU-101</td><td>Widget A</td><td>450</td></tr>"
        "<tr><td>SKU-102</td><td>Widget B</td><td>820</td></tr>"
        "</table>"
        "</body></html>"
    )

    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as f:
        f.write(html_content)
        f_path = f.name

    try:
        recipe = DocumentParsingRecipe()
        result = recipe.execute(
            inputs={},
            config={"document_path": f_path}
        )
        df = result["dataframe"]
        assert len(df) == 1
        assert "Q3 Inventory Summary" in df.iloc[0]["extracted_text"]
        assert df.iloc[0]["tables_count"] == 1

        tables_df = result["tables"]
        assert len(tables_df) == 2
        assert "SKU" in tables_df.columns
        assert "SKU-101" in tables_df["SKU"].values
    finally:
        if os.path.exists(f_path):
            os.remove(f_path)


def test_document_parsing_from_dataframe_text_column():
    """Tests parsing text and markdown directly from an upstream DataFrame column."""
    input_df = pd.DataFrame({
        "document_name": ["Doc_Alpha", "Doc_Beta"],
        "body": [
            "Customer feedback notes for Q1:\n\n| Issue | Count |\n|---|---|\n| Login failure | 12 |\n| Timeout | 5 |",
            "Routine system maintenance logs without any tabular data."
        ]
    })

    recipe = DocumentParsingRecipe()
    result = recipe.execute(
        inputs={"dataframe": input_df},
        config={"text_column": "body"}
    )

    df = result["dataframe"]
    assert len(df) == 2
    assert df.iloc[0]["tables_count"] == 1
    assert df.iloc[1]["tables_count"] == 0

    tables_df = result["tables"]
    assert len(tables_df) == 2
    assert "Issue" in tables_df.columns
    assert "Login failure" in tables_df["Issue"].values


def test_document_parsing_docx_file():
    """Tests parsing a Word (.docx) document with paragraphs and tables."""
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
        docx_path = f.name

    try:
        if docx:
            doc = docx.Document()
            doc.add_heading("Product Specification Document", level=1)
            doc.add_paragraph("This document outlines core system specifications.")
            
            table = doc.add_table(rows=3, cols=2)
            table.cell(0, 0).text = "Module"
            table.cell(0, 1).text = "Version"
            table.cell(1, 0).text = "Auth Service"
            table.cell(1, 1).text = "2.4.1"
            table.cell(2, 0).text = "Data Pipeline"
            table.cell(2, 1).text = "3.1.0"
            doc.save(docx_path)

            recipe = DocumentParsingRecipe()
            result = recipe.execute(
                inputs={},
                config={"document_path": docx_path}
            )

            df = result["dataframe"]
            assert len(df) == 1
            assert "Product Specification" in df.iloc[0]["extracted_text"]
            assert df.iloc[0]["content_type"] == "docx"
            assert df.iloc[0]["tables_count"] == 1

            tables_df = result["tables"]
            assert len(tables_df) == 2
            assert "Module" in tables_df.columns
            assert "Auth Service" in tables_df["Module"].values
    finally:
        if os.path.exists(docx_path):
            os.remove(docx_path)


def test_document_parsing_pdf_file():
    """Tests parsing a synthetic PDF with text and metadata."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        pdf_path = f.name

    try:
        if pypdf:
            writer = pypdf.PdfWriter()
            page = writer.add_blank_page(width=612, height=792)
            writer.add_metadata({
                "/Title": "Automated ML Pipeline Report",
                "/Author": "Lead Scientist"
            })
            with open(pdf_path, "wb") as f_out:
                writer.write(f_out)

            recipe = DocumentParsingRecipe()
            result = recipe.execute(
                inputs={},
                config={"document_path": pdf_path}
            )

            df = result["dataframe"]
            assert len(df) == 1
            assert df.iloc[0]["content_type"] == "pdf"
            meta = json.loads(df.iloc[0]["metadata"])
            assert meta.get("title") == "Automated ML Pipeline Report" or "total_pages" in meta
    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)


def test_document_parsing_output_formats():
    """Tests the different output_format modes: Per Document, Extracted Tables as Rows, Append to Input."""
    recipe = DocumentParsingRecipe()

    # 1. Output format: Extracted Tables as Rows
    res_tables = recipe.execute(
        inputs={},
        config={"output_format": "Extracted Tables as Rows"}
    )
    df_tables = res_tables["dataframe"]
    assert "table_index" in df_tables.columns
    assert len(df_tables) >= 5

    # 2. Output format: Per Document (Combined)
    res_combined = recipe.execute(
        inputs={},
        config={"output_format": "Per Document (Combined)"}
    )
    df_combined = res_combined["dataframe"]
    assert "total_pages" in df_combined.columns
    assert len(df_combined) == 2

    # 3. Output format: Append to Input DataFrame
    in_df = pd.DataFrame({
        "id": [1, 2],
        "content": [
            "Report 1: All systems operational.",
            "Report 2:\n| Metric | Value |\n|---|---|\n| Latency | 25ms |"
        ]
    })
    res_append = recipe.execute(
        inputs={"dataframe": in_df},
        config={"output_format": "Append to Input DataFrame", "text_column": "content"}
    )
    df_append = res_append["dataframe"]
    assert "extracted_text" in df_append.columns
    assert "tables_count" in df_append.columns
    assert df_append.loc[1, "tables_count"] == 1


def test_document_parsing_extraction_targets():
    """Tests extraction_target modes: 'Tables Only' and 'Text Only'."""
    recipe = DocumentParsingRecipe()

    # Tables Only: prose text should be suppressed
    res_tables_only = recipe.execute(
        inputs={},
        config={"extraction_target": "Tables Only"}
    )
    df_t = res_tables_only["dataframe"]
    assert all(df_t["extracted_text"] == "")
    assert res_tables_only["metrics"]["total_tables_extracted"] == 2

    # Text Only: table count should be zero
    res_text_only = recipe.execute(
        inputs={},
        config={"extraction_target": "Text Only"}
    )
    df_txt = res_text_only["dataframe"]
    assert all(df_txt["tables_count"] == 0)
    assert any("EXECUTIVE SUMMARY" in txt for txt in df_txt["extracted_text"])


@pytest.mark.asyncio
async def test_document_parsing_in_workflow_execution():
    """Verifies that a pipeline containing a document_parsing node runs successfully via the workflow API."""
    import uuid
    from httpx import AsyncClient, ASGITransport
    from backend.app.main import app
    from backend.app.infrastructure.database.session import init_db
    from backend.app.core.security import get_current_user, TokenData
    from backend.app.core.tenant_context import set_current_tenant_id, set_current_user_id

    async def _test_user():
        set_current_tenant_id(1)
        set_current_user_id("test")
        return TokenData("test", 1, ["Tenant Admin", "Data Scientist", "ML Engineer"], ["*"])

    app.dependency_overrides[get_current_user] = _test_user
    await init_db()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        wf_id = f"wf_doc_{uuid.uuid4().hex[:8]}"

        dag = {
            "id": wf_id,
            "name": "Doc Parsing Pipeline",
            "nodes": [
                {
                    "id": "node_doc_parser",
                    "recipe_id": "document_parsing",
                    "config": {
                        "extraction_target": "Text & Tables",
                        "output_format": "Per Page / Section"
                    }
                }
            ],
            "edges": []
        }

        resp = await client.post(f"/api/v1/workflows/execute?workflow_id={wf_id}", json=dag)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "SUCCESS"
        assert "execution_id" in data
        assert "node_doc_parser" in data.get("step_snapshots", {})
        snap = data["step_snapshots"]["node_doc_parser"]
        assert snap["row_count"] >= 1
        assert "extracted_text" in snap["columns"]
