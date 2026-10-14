import os
import io
import re
import json
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort

# Optional external parsing libraries with graceful fallbacks
try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    pypdf = None
    PYPDF_AVAILABLE = False

try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    docx = None
    DOCX_AVAILABLE = False


# ----------------------------------------------------------------------
# HELPER DATA STRUCTURES & FORMATTERS
# ----------------------------------------------------------------------

def clean_extracted_text(text: str, normalize_whitespace: bool = True) -> str:
    """Cleans extracted document text by removing null bytes, carriage returns, and excess whitespace."""
    if not text:
        return ""
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    if normalize_whitespace:
        # Normalize 3+ newlines to double newline
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Normalize multiple spaces or tabs
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
        text = "\n".join(lines).strip()
    return text


def parse_markdown_tables(text: str) -> List[Dict[str, Any]]:
    """Detects and extracts markdown tables from text blocks."""
    tables = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Look for markdown table header e.g. | col1 | col2 |
        if line.startswith("|") and line.endswith("|") and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            # Look for delimiter row e.g. |---|---| or |:---|---:|
            if next_line.startswith("|") and next_line.endswith("|") and re.match(r"^\|(?:\s*:?-+:?\s*\|)+$", next_line):
                headers = [c.strip() for c in line.strip("|").split("|")]
                row_records = []
                j = i + 2
                while j < len(lines):
                    r_line = lines[j].strip()
                    if r_line.startswith("|") and r_line.endswith("|"):
                        cells = [c.strip() for c in r_line.strip("|").split("|")]
                        if len(cells) < len(headers):
                            cells.extend([""] * (len(headers) - len(cells)))
                        row_dict = {headers[k]: cells[k] for k in range(min(len(headers), len(cells)))}
                        row_records.append(row_dict)
                        j += 1
                    else:
                        break
                if row_records:
                    tables.append({
                        "headers": headers,
                        "rows": row_records,
                        "num_rows": len(row_records),
                        "num_cols": len(headers),
                        "source_format": "markdown"
                    })
                i = j
                continue
        i += 1
    return tables


def parse_html_tables(html_content: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Extracts tables and clean text from HTML content."""
    tables = []
    # Extract <table>...</table>
    table_pattern = re.compile(r"<table[^>]*>(.*?)</table>", re.IGNORECASE | re.DOTALL)
    for t_match in table_pattern.finditer(html_content):
        t_html = t_match.group(1)
        row_pattern = re.compile(r"<tr[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
        cell_pattern = re.compile(r"<t[hd][^>]*>(.*?)</t[hd]>", re.IGNORECASE | re.DOTALL)
        
        headers = []
        rows = []
        for r_match in row_pattern.finditer(t_html):
            r_html = r_match.group(1)
            cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cell_pattern.findall(r_html)]
            if not cells:
                continue
            if not headers:
                headers = cells
            else:
                if len(cells) < len(headers):
                    cells.extend([""] * (len(headers) - len(cells)))
                row_dict = {headers[k]: cells[k] for k in range(min(len(headers), len(cells)))}
                rows.append(row_dict)
        if headers and rows:
            tables.append({
                "headers": headers,
                "rows": rows,
                "num_rows": len(rows),
                "num_cols": len(headers),
                "source_format": "html"
            })

    # Strip all HTML tags to get plain prose text
    plain_text = re.sub(r"<style[^>]*>.*?</style>", "", html_content, flags=re.DOTALL | re.IGNORECASE)
    plain_text = re.sub(r"<script[^>]*>.*?</script>", "", plain_text, flags=re.DOTALL | re.IGNORECASE)
    plain_text = re.sub(r"<[^>]+>", " ", plain_text)
    plain_text = clean_extracted_text(plain_text)
    return plain_text, tables


def parse_docx_bytes(file_bytes: bytes) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Extracts text paragraphs, structured tables, and metadata from a Word (.docx) file.
    Uses python-docx if installed, with a robust zipfile/xml fallback.
    """
    text_parts: List[str] = []
    tables: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {"format": "docx"}

    if DOCX_AVAILABLE:
        try:
            doc = docx.Document(io.BytesIO(file_bytes))
            # Extract paragraphs
            for p in doc.paragraphs:
                p_text = p.text.strip()
                if p_text:
                    text_parts.append(p_text)
            
            # Extract tables
            for t_idx, table in enumerate(doc.tables):
                headers = []
                t_rows = []
                for r_idx, row in enumerate(table.rows):
                    row_vals = [cell.text.strip() for cell in row.cells]
                    # De-duplicate adjacent identical cells caused by column spans
                    dedup_vals = []
                    for c_val in row_vals:
                        if not dedup_vals or c_val != dedup_vals[-1]:
                            dedup_vals.append(c_val)
                    if not dedup_vals or all(v == "" for v in dedup_vals):
                        continue
                    if r_idx == 0 and not headers:
                        headers = [v if v else f"col_{k+1}" for k, v in enumerate(dedup_vals)]
                    else:
                        if not headers:
                            headers = [f"col_{k+1}" for k in range(len(dedup_vals))]
                        if len(dedup_vals) < len(headers):
                            dedup_vals.extend([""] * (len(headers) - len(dedup_vals)))
                        r_dict = {headers[k]: dedup_vals[k] for k in range(min(len(headers), len(dedup_vals)))}
                        t_rows.append(r_dict)
                if headers and t_rows:
                    tables.append({
                        "table_index": t_idx + 1,
                        "headers": headers,
                        "rows": t_rows,
                        "num_rows": len(t_rows),
                        "num_cols": len(headers),
                        "source_format": "docx"
                    })

            # Core properties
            try:
                cp = doc.core_properties
                if cp.title: metadata["title"] = cp.title
                if cp.author: metadata["author"] = cp.author
                if cp.created: metadata["created"] = str(cp.created)
            except Exception:
                pass

            combined_text = "\n\n".join(text_parts)
            return combined_text, tables, metadata
        except Exception:
            pass  # Fall through to XML parser below

    # Built-in zero-dependency DOCX Parser (zipfile + XML)
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
            if "word/document.xml" in z.namelist():
                xml_content = z.read("word/document.xml")
                root = ET.fromstring(xml_content)
                ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
                
                # Extract paragraphs
                for p in root.findall(".//w:p", ns):
                    p_texts = [node.text for node in p.findall(".//w:t", ns) if node.text]
                    full_p = "".join(p_texts).strip()
                    if full_p:
                        text_parts.append(full_p)

                # Extract tables
                for t_idx, tbl in enumerate(root.findall(".//w:tbl", ns)):
                    headers = []
                    t_rows = []
                    for r_idx, tr in enumerate(tbl.findall(".//w:tr", ns)):
                        row_vals = []
                        for tc in tr.findall(".//w:tc", ns):
                            c_texts = [node.text for node in tc.findall(".//w:t", ns) if node.text]
                            row_vals.append("".join(c_texts).strip())
                        if not row_vals or all(v == "" for v in row_vals):
                            continue
                        if r_idx == 0:
                            headers = [v if v else f"col_{k+1}" for k, v in enumerate(row_vals)]
                        else:
                            if not headers:
                                headers = [f"col_{k+1}" for k in range(len(row_vals))]
                            if len(row_vals) < len(headers):
                                row_vals.extend([""] * (len(headers) - len(row_vals)))
                            r_dict = {headers[k]: row_vals[k] for k in range(min(len(headers), len(row_vals)))}
                            t_rows.append(r_dict)
                    if headers and t_rows:
                        tables.append({
                            "table_index": t_idx + 1,
                            "headers": headers,
                            "rows": t_rows,
                            "num_rows": len(t_rows),
                            "num_cols": len(headers),
                            "source_format": "docx"
                        })
    except Exception:
        pass

    return "\n\n".join(text_parts), tables, metadata


def parse_pdf_bytes(file_bytes: bytes, max_pages: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Extracts text, tables, and metadata page-by-page from a PDF file.
    Uses pypdf if available, with regex tabular parsing and plain text extraction.
    """
    pages_data: List[Dict[str, Any]] = []

    if PYPDF_AVAILABLE:
        try:
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            total_pages = len(reader.pages)
            limit = min(total_pages, max_pages) if max_pages else total_pages

            meta_dict = {}
            if reader.metadata:
                for k, v in reader.metadata.items():
                    if v and isinstance(v, (str, int, float)):
                        clean_k = str(k).lstrip("/").lower()
                        meta_dict[clean_k] = str(v)

            for page_idx in range(limit):
                page = reader.pages[page_idx]
                page_text = clean_extracted_text(page.extract_text() or "")
                
                # Check for markdown or aligned text tables inside page text
                detected_tables = parse_markdown_tables(page_text)
                
                # Fallback: check for tab/multi-space aligned columnar text
                if not detected_tables:
                    col_table = parse_aligned_columns_as_table(page_text)
                    if col_table:
                        detected_tables.append(col_table)

                pages_data.append({
                    "page_number": page_idx + 1,
                    "text": page_text,
                    "tables": detected_tables,
                    "metadata": {**meta_dict, "total_pages": total_pages, "format": "pdf"}
                })
            return pages_data
        except Exception:
            pass

    # Pure Python PDF stream fallback (scans /Filter /FlateDecode or literal text blocks)
    try:
        raw_str = file_bytes.decode("latin1", errors="ignore")
        # Extract stream objects
        streams = re.findall(r"stream[\r\n]+(.*?)[\r\n]+endstream", raw_str, re.DOTALL)
        extracted_text_chunks = []
        for s in streams:
            # Try plain text in stream
            t_matches = re.findall(r"\(([^\)]+)\)\s*Tj", s)
            if t_matches:
                extracted_text_chunks.append(" ".join(t_matches))
            t_array = re.findall(r"\[(.*?)\]\s*TJ", s)
            for arr in t_array:
                arr_texts = re.findall(r"\(([^\)]+)\)", arr)
                if arr_texts:
                    extracted_text_chunks.append("".join(arr_texts))

        full_text = clean_extracted_text("\n".join(extracted_text_chunks))
        if not full_text:
            # As last resort, extract printable words
            words = re.findall(r"[A-Za-z0-9,.:;'\-\(\)\$]{2,}", raw_str[:50000])
            full_text = clean_extracted_text(" ".join(words[:500]))

        pages_data.append({
            "page_number": 1,
            "text": full_text,
            "tables": parse_markdown_tables(full_text),
            "metadata": {"format": "pdf", "total_pages": 1}
        })
    except Exception:
        pass

    return pages_data


def parse_aligned_columns_as_table(text: str) -> Optional[Dict[str, Any]]:
    """Detects multi-column tabular blocks separated by 2+ spaces or tabs."""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    candidate_rows = []
    
    for ln in lines:
        parts = re.split(r"\s{2,}|\t", ln)
        if len(parts) >= 2:
            candidate_rows.append(parts)

    if len(candidate_rows) >= 3:
        # Check consistent column counts
        col_lens = [len(r) for r in candidate_rows]
        most_common_len = max(set(col_lens), key=col_lens.count)
        valid_rows = [r for r in candidate_rows if len(r) == most_common_len]
        if len(valid_rows) >= 3:
            headers = [f"col_{c+1}" if not valid_rows[0][c] else valid_rows[0][c] for c in range(most_common_len)]
            rows = []
            for r in valid_rows[1:]:
                rows.append({headers[c]: r[c] for c in range(most_common_len)})
            return {
                "headers": headers,
                "rows": rows,
                "num_rows": len(rows),
                "num_cols": len(headers),
                "source_format": "aligned_text"
            }
    return None


# ----------------------------------------------------------------------
# DOCUMENT PARSER RECIPE CLASS
# ----------------------------------------------------------------------

class DocumentParsingRecipe(BaseRecipe):
    """
    Document Parsing & Table Extraction Recipe (NLP / Text).
    Extracts structured prose, page sections, and tables from PDF, Word (.docx),
    Markdown, HTML, and plain text files.
    """

    recipe_id = "document_parsing"
    name = "Document Parsing"
    version = "1.0.0"
    category = "nlp"
    group = "Data Preparation & Transformation"
    subgroup = "NLP/Text"
    description = "Extracts text and tables from PDF, Word and other documents."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="dataframe",
            label="Input DataFrame",
            type="dataframe",
            required=False,
            max_connections=1,
            description="Optional input table containing document paths, URLs, or raw document content"
        )
    ]

    outputs = [
        RecipePort(
            id="dataframe",
            label="Parsed Documents / Text",
            type="dataframe",
            description="DataFrame containing extracted document text, metadata, and structural info"
        ),
        RecipePort(
            id="tables",
            label="Extracted Tables",
            type="dataframe",
            description="DataFrame containing structured tabular data extracted across documents"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "extraction_target": {
                    "type": "string",
                    "title": "Extraction Target",
                    "default": "Text & Tables",
                    "enum": ["Text & Tables", "Text Only", "Tables Only"],
                    "input_type": "dropdown",
                    "description": "Choose whether to extract full prose, structured tables, or both."
                },
                "output_format": {
                    "type": "string",
                    "title": "Output Format",
                    "default": "Per Page / Section",
                    "enum": [
                        "Per Page / Section",
                        "Per Document (Combined)",
                        "Extracted Tables as Rows",
                        "Append to Input DataFrame"
                    ],
                    "input_type": "dropdown",
                    "description": "Granularity of rows in the primary output DataFrame."
                },
                "document_path": {
                    "type": "string",
                    "title": "Document Path or Folder",
                    "default": "",
                    "input_type": "text",
                    "description": "Optional direct local path, Ozone s3:// URL, or directory of documents to parse."
                },
                "dataset_id": {
                    "type": "string",
                    "title": "Dataset ID",
                    "input_type": "dataset_select",
                    "description": "Optional ID of an uploaded document file or dataset to parse."
                },
                "file_path_column": {
                    "type": "string",
                    "title": "File Path Column",
                    "default": "",
                    "input_type": "column_select",
                    "description": "Column in upstream DataFrame containing file paths or filenames (auto-detected if blank)."
                },
                "text_column": {
                    "type": "string",
                    "title": "Text / Content Column",
                    "default": "",
                    "input_type": "column_select",
                    "description": "Column in upstream DataFrame containing raw document text, Markdown, or HTML."
                },
                "clean_whitespace": {
                    "type": "boolean",
                    "title": "Clean Whitespace",
                    "default": True,
                    "input_type": "checkbox",
                    "description": "Normalize redundant spaces, tabs, and line breaks in extracted text."
                },
                "max_pages": {
                    "type": "integer",
                    "title": "Max Pages per Document",
                    "default": 100,
                    "input_type": "number",
                    "description": "Maximum number of pages to extract per multi-page document."
                }
            },
            "required": []
        }

    def execute(
        self,
        inputs: Dict[str, Any],
        config: Dict[str, Any],
        context: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Executes document text and table extraction."""
        in_df = inputs.get("dataframe")
        if in_df is None and context and isinstance(context, dict):
            in_df = context.get("dataframe")

        extraction_target = config.get("extraction_target", "Text & Tables")
        output_format = config.get("output_format", "Per Page / Section")
        doc_path = config.get("document_path") or config.get("file_path") or ""
        dataset_id = config.get("dataset_id")
        file_path_col = config.get("file_path_column") or ""
        text_col = config.get("text_column") or ""
        clean_ws = config.get("clean_whitespace", True)
        max_pages = config.get("max_pages") or 100

        # Collected document parsing records
        parsed_docs: List[Dict[str, Any]] = []
        all_extracted_tables: List[Dict[str, Any]] = []

        # -------------------------------------------------------------
        # SOURCE 1: Upstream DataFrame with document paths or text
        # -------------------------------------------------------------
        if in_df is not None and isinstance(in_df, pd.DataFrame) and not in_df.empty:
            cols = list(in_df.columns)
            
            # Detect text/content column if not specified
            if not text_col:
                for c in cols:
                    c_lower = str(c).lower()
                    if c_lower in ["content", "text", "document", "doc_text", "body", "markdown", "html", "raw_text"]:
                        text_col = c
                        break

            # Detect file path column if not specified
            if not file_path_col and not text_col:
                for c in cols:
                    c_lower = str(c).lower()
                    if c_lower in ["path", "file_path", "filepath", "filename", "document_path", "doc_path", "url", "uri"]:
                        file_path_col = c
                        break

            # Case 1A: Parsing text/markdown/html from a DataFrame column
            if text_col and text_col in in_df.columns:
                for row_idx, val in in_df[text_col].dropna().items():
                    raw_text = str(val)
                    doc_name = f"Row_{row_idx + 1}"
                    if "document_name" in in_df.columns:
                        doc_name = str(in_df.loc[row_idx, "document_name"])
                    elif "id" in in_df.columns:
                        doc_name = f"Doc_{in_df.loc[row_idx, 'id']}"

                    text_content, tables = self._parse_text_payload(raw_text, clean_ws)
                    parsed_docs.append({
                        "document_name": doc_name,
                        "page_number": 1,
                        "content_type": "text/markdown",
                        "extracted_text": text_content if extraction_target != "Tables Only" else "",
                        "tables_count": len(tables),
                        "tables_preview": json.dumps([t["headers"] for t in tables]) if tables else "[]",
                        "char_count": len(text_content),
                        "word_count": len(text_content.split()),
                        "metadata": json.dumps({"source": "dataframe_column", "column": text_col}),
                        "tables": tables,
                        "_orig_row_idx": row_idx
                    })
                    for t_idx, tbl in enumerate(tables):
                        tbl["document_name"] = doc_name
                        tbl["table_index"] = t_idx + 1
                        all_extracted_tables.append(tbl)

            # Case 1B: Parsing files from paths in DataFrame column
            elif file_path_col and file_path_col in in_df.columns:
                for row_idx, path_val in in_df[file_path_col].dropna().items():
                    p_str = str(path_val).strip()
                    doc_records, doc_tables = self._parse_single_file_or_data(
                        p_str, clean_ws, max_pages, extraction_target
                    )
                    for rec in doc_records:
                        rec["_orig_row_idx"] = row_idx
                        parsed_docs.append(rec)
                    all_extracted_tables.extend(doc_tables)

        # -------------------------------------------------------------
        # SOURCE 2: Direct document path, directory, or dataset_id
        # -------------------------------------------------------------
        if not parsed_docs and (doc_path or dataset_id):
            target_path = doc_path
            # If dataset_id was given, resolve storage path
            if dataset_id and not target_path:
                target_path = self._resolve_dataset_storage_path(dataset_id, config)

            if target_path:
                p = Path(target_path)
                if p.is_dir():
                    # Parse all documents in directory
                    supported_exts = {".pdf", ".docx", ".txt", ".md", ".html", ".htm", ".csv", ".json"}
                    doc_files = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in supported_exts]
                    for f in doc_files[:50]:
                        doc_records, doc_tables = self._parse_single_file_or_data(
                            str(f), clean_ws, max_pages, extraction_target
                        )
                        parsed_docs.extend(doc_records)
                        all_extracted_tables.extend(doc_tables)
                else:
                    doc_records, doc_tables = self._parse_single_file_or_data(
                        target_path, clean_ws, max_pages, extraction_target
                    )
                    parsed_docs.extend(doc_records)
                    all_extracted_tables.extend(doc_tables)

        # -------------------------------------------------------------
        # SOURCE 3: Demonstration Document Data (When no input is provided)
        # -------------------------------------------------------------
        if not parsed_docs:
            demo_docs, demo_tables = self._generate_demo_parsed_documents()
            parsed_docs.extend(demo_docs)
            all_extracted_tables.extend(demo_tables)

        # Apply extraction_target filter on parsed_docs and tables
        if extraction_target == "Tables Only":
            for d in parsed_docs:
                d["extracted_text"] = ""
                d["char_count"] = 0
                d["word_count"] = 0
        elif extraction_target == "Text Only":
            for d in parsed_docs:
                d["tables_count"] = 0
                d["tables_preview"] = "[]"
                if "tables" in d:
                    d["tables"] = []
            all_extracted_tables = []

        # -------------------------------------------------------------
        # FORMAT OUTPUTS ACCORDING TO USER CONFIG
        # -------------------------------------------------------------
        tables_df = self._build_tables_dataframe(all_extracted_tables)

        if output_format == "Extracted Tables as Rows" and not tables_df.empty:
            out_df = tables_df
        elif output_format == "Per Document (Combined)":
            out_df = self._build_combined_document_dataframe(parsed_docs)
        elif output_format == "Append to Input DataFrame" and in_df is not None and not in_df.empty:
            out_df = in_df.copy()
            text_by_row: Dict[int, str] = {}
            tbl_count_by_row: Dict[int, int] = {}
            for d in parsed_docs:
                r_idx = d.get("_orig_row_idx")
                if r_idx is not None:
                    text_by_row[r_idx] = text_by_row.get(r_idx, "") + "\n" + d.get("extracted_text", "")
                    tbl_count_by_row[r_idx] = tbl_count_by_row.get(r_idx, 0) + d.get("tables_count", 0)
            out_df["extracted_text"] = [text_by_row.get(idx, "").strip() for idx in out_df.index]
            out_df["tables_count"] = [tbl_count_by_row.get(idx, 0) for idx in out_df.index]
        else:
            # Default: Per Page / Section
            display_rows = []
            for d in parsed_docs:
                row = {k: v for k, v in d.items() if not k.startswith("_") and k != "tables"}
                display_rows.append(row)
            out_df = pd.DataFrame(display_rows)

        # Build execution metrics
        total_docs = len(set(d["document_name"] for d in parsed_docs))
        total_pages = sum(1 for d in parsed_docs)
        total_chars = sum(d.get("char_count", 0) for d in parsed_docs)
        total_tables = len(all_extracted_tables)
        doc_types = dict(pd.Series([d.get("content_type", "unknown") for d in parsed_docs]).value_counts())

        metrics = {
            "total_documents_processed": int(total_docs),
            "total_pages_extracted": int(total_pages),
            "total_characters_extracted": int(total_chars),
            "total_tables_extracted": int(total_tables),
            "document_types": doc_types,
        }

        # Build sample previews
        sample_snippets = []
        for d in parsed_docs[:3]:
            txt = d.get("extracted_text", "")
            preview_snippet = txt[:180] + ("..." if len(txt) > 180 else "")
            sample_snippets.append({
                "document": d.get("document_name"),
                "page": d.get("page_number", 1),
                "preview": preview_snippet,
                "tables_found": d.get("tables_count", 0)
            })

        output_summary = {
            "status": "success",
            "message": f"Successfully parsed {total_docs} document(s) ({total_pages} page/section entries) and extracted {total_tables} table(s).",
            "extraction_target": extraction_target,
            "output_format": output_format,
            "documents_count": total_docs,
            "tables_count": total_tables,
            "sample_preview": sample_snippets
        }

        return {
            "dataframe": out_df,
            "tables": tables_df,
            "metrics": metrics,
            "output_summary": output_summary
        }

    # ------------------------------------------------------------------
    # INTERNAL PARSING HELPERS
    # ------------------------------------------------------------------

    def _parse_single_file_or_data(
        self,
        file_path_str: str,
        clean_ws: bool,
        max_pages: int,
        extraction_target: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Parses a single file path, reading binary contents and dispatching by extension."""
        doc_records = []
        extracted_tables = []
        doc_name = Path(file_path_str).name or file_path_str
        ext = Path(file_path_str).suffix.lower()

        # Handle Ozone/S3 or Local Path
        try:
            if file_path_str.startswith("s3://"):
                from backend.app.infrastructure.storage.storage_manager import storage_manager
                file_bytes = storage_manager.download_file_bytes(file_path_str)
            elif os.path.exists(file_path_str):
                with open(file_path_str, "rb") as f:
                    file_bytes = f.read()
            else:
                # File path not found; record as plain text note
                return [{
                    "document_name": doc_name,
                    "page_number": 1,
                    "content_type": ext.lstrip(".") or "unknown",
                    "extracted_text": f"File '{file_path_str}' could not be located on storage.",
                    "tables_count": 0,
                    "tables_preview": "[]",
                    "char_count": 0,
                    "word_count": 0,
                    "metadata": json.dumps({"error": "file_not_found", "path": file_path_str})
                }], []
        except Exception as e:
            return [{
                "document_name": doc_name,
                "page_number": 1,
                "content_type": ext.lstrip(".") or "unknown",
                "extracted_text": f"Error loading file: {str(e)}",
                "tables_count": 0,
                "tables_preview": "[]",
                "char_count": 0,
                "word_count": 0,
                "metadata": json.dumps({"error": str(e)})
            }], []

        # Dispatch by file extension
        if ext == ".pdf":
            pdf_pages = parse_pdf_bytes(file_bytes, max_pages=max_pages)
            for p_info in pdf_pages:
                p_text = p_info["text"] if extraction_target != "Tables Only" else ""
                p_tables = p_info.get("tables", [])
                doc_records.append({
                    "document_name": doc_name,
                    "page_number": p_info.get("page_number", 1),
                    "content_type": "pdf",
                    "extracted_text": p_text,
                    "tables_count": len(p_tables),
                    "tables_preview": json.dumps([t["headers"] for t in p_tables]) if p_tables else "[]",
                    "char_count": len(p_text),
                    "word_count": len(p_text.split()),
                    "metadata": json.dumps(p_info.get("metadata", {}))
                })
                for t_idx, tbl in enumerate(p_tables):
                    tbl["document_name"] = doc_name
                    tbl["page_number"] = p_info.get("page_number", 1)
                    tbl["table_index"] = t_idx + 1
                    extracted_tables.append(tbl)

        elif ext in [".docx", ".doc"]:
            docx_text, docx_tables, meta = parse_docx_bytes(file_bytes)
            p_text = docx_text if extraction_target != "Tables Only" else ""
            doc_records.append({
                "document_name": doc_name,
                "page_number": 1,
                "content_type": "docx",
                "extracted_text": p_text,
                "tables_count": len(docx_tables),
                "tables_preview": json.dumps([t["headers"] for t in docx_tables]) if docx_tables else "[]",
                "char_count": len(p_text),
                "word_count": len(p_text.split()),
                "metadata": json.dumps(meta)
            })
            for tbl in docx_tables:
                tbl["document_name"] = doc_name
                extracted_tables.append(tbl)

        elif ext in [".html", ".htm", ".xhtml"]:
            html_str = file_bytes.decode("utf-8", errors="ignore")
            h_text, h_tables = parse_html_tables(html_str)
            p_text = h_text if extraction_target != "Tables Only" else ""
            doc_records.append({
                "document_name": doc_name,
                "page_number": 1,
                "content_type": "html",
                "extracted_text": p_text,
                "tables_count": len(h_tables),
                "tables_preview": json.dumps([t["headers"] for t in h_tables]) if h_tables else "[]",
                "char_count": len(p_text),
                "word_count": len(p_text.split()),
                "metadata": json.dumps({"format": "html"})
            })
            for t_idx, tbl in enumerate(h_tables):
                tbl["document_name"] = doc_name
                tbl["table_index"] = t_idx + 1
                extracted_tables.append(tbl)

        else:
            # Plain Text, Markdown, CSV, or unknown format
            text_str = file_bytes.decode("utf-8", errors="ignore")
            p_text, m_tables = self._parse_text_payload(text_str, clean_ws)
            if extraction_target == "Tables Only":
                p_text = ""
            doc_records.append({
                "document_name": doc_name,
                "page_number": 1,
                "content_type": ext.lstrip(".") or "txt",
                "extracted_text": p_text,
                "tables_count": len(m_tables),
                "tables_preview": json.dumps([t["headers"] for t in m_tables]) if m_tables else "[]",
                "char_count": len(p_text),
                "word_count": len(p_text.split()),
                "metadata": json.dumps({"format": ext.lstrip(".") or "txt"})
            })
            for t_idx, tbl in enumerate(m_tables):
                tbl["document_name"] = doc_name
                tbl["table_index"] = t_idx + 1
                extracted_tables.append(tbl)

        return doc_records, extracted_tables

    def _parse_text_payload(self, text: str, clean_ws: bool) -> Tuple[str, List[Dict[str, Any]]]:
        """Extracts text content and any markdown/html tables from raw text string."""
        if "<table" in text.lower() and "</table>" in text.lower():
            return parse_html_tables(text)

        cleaned = clean_extracted_text(text, normalize_whitespace=clean_ws)
        tables = parse_markdown_tables(cleaned)
        if not tables:
            col_tbl = parse_aligned_columns_as_table(cleaned)
            if col_tbl:
                tables.append(col_tbl)
        return cleaned, tables

    def _build_tables_dataframe(self, tables_list: List[Dict[str, Any]]) -> pd.DataFrame:
        """Constructs a consolidated DataFrame of all extracted table rows."""
        if not tables_list:
            return pd.DataFrame(columns=["document_name", "table_index", "row_index"])

        all_rows = []
        for tbl in tables_list:
            d_name = tbl.get("document_name", "Document")
            t_idx = tbl.get("table_index", 1)
            for r_idx, r_dict in enumerate(tbl.get("rows", [])):
                row_record = {
                    "document_name": d_name,
                    "table_index": t_idx,
                    "row_index": r_idx + 1,
                    **r_dict
                }
                all_rows.append(row_record)

        if not all_rows:
            return pd.DataFrame(columns=["document_name", "table_index", "row_index"])

        return pd.DataFrame(all_rows)

    def _build_combined_document_dataframe(self, parsed_docs: List[Dict[str, Any]]) -> pd.DataFrame:
        """Groups page-level extracts into combined document-level rows."""
        docs_map: Dict[str, Dict[str, Any]] = {}
        for d in parsed_docs:
            d_name = d["document_name"]
            if d_name not in docs_map:
                docs_map[d_name] = {
                    "document_name": d_name,
                    "total_pages": 0,
                    "content_type": d.get("content_type", "document"),
                    "extracted_text": [],
                    "tables_count": 0,
                    "char_count": 0,
                    "word_count": 0,
                    "metadata": d.get("metadata", "{}")
                }
            entry = docs_map[d_name]
            entry["total_pages"] += 1
            if d.get("extracted_text"):
                entry["extracted_text"].append(d["extracted_text"])
            entry["tables_count"] += d.get("tables_count", 0)
            entry["char_count"] += d.get("char_count", 0)
            entry["word_count"] += d.get("word_count", 0)

        combined_rows = []
        for d_name, entry in docs_map.items():
            combined_rows.append({
                "document_name": d_name,
                "total_pages": entry["total_pages"],
                "content_type": entry["content_type"],
                "extracted_text": "\n\n--- Page Break ---\n\n".join(entry["extracted_text"]),
                "tables_count": entry["tables_count"],
                "char_count": entry["char_count"],
                "word_count": entry["word_count"],
                "metadata": entry["metadata"]
            })
        return pd.DataFrame(combined_rows)

    def _resolve_dataset_storage_path(self, dataset_id: str, config: Dict[str, Any]) -> str:
        """Looks up the dataset storage path from the datasets database table."""
        try:
            from backend.app.core.config import settings
            from sqlalchemy import create_engine, text
            from sqlalchemy.pool import NullPool
            sync_url = settings.DATABASE_URL.replace("+aiosqlite", "").replace("+asyncpg", "")
            if sync_url.startswith(("postgres://", "postgresql://")):
                sync_url = "postgresql+psycopg2://" + sync_url.split("://", 1)[1]
            eng = create_engine(sync_url, poolclass=NullPool)
            with eng.connect() as conn:
                row = conn.execute(
                    text("SELECT storage_path FROM datasets WHERE id = :id LIMIT 1"),
                    {"id": str(dataset_id)}
                ).fetchone()
                if row and row[0]:
                    return row[0]
        except Exception:
            pass
        return ""

    def _generate_demo_parsed_documents(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Provides rich demonstration document entries with prose text and tables for initial canvas testing."""
        table1 = {
            "table_index": 1,
            "document_name": "Quarterly_Financial_Report.pdf",
            "headers": ["Quarter", "Revenue (USD)", "Operating Expenses", "Net Margin (%)"],
            "rows": [
                {"Quarter": "Q1 2026", "Revenue (USD)": "$4,250,000", "Operating Expenses": "$2,980,000", "Net Margin (%)": "29.8%"},
                {"Quarter": "Q2 2026", "Revenue (USD)": "$4,810,000", "Operating Expenses": "$3,150,000", "Net Margin (%)": "34.5%"},
                {"Quarter": "Q3 2026", "Revenue (USD)": "$5,190,000", "Operating Expenses": "$3,420,000", "Net Margin (%)": "34.1%"},
                {"Quarter": "Q4 2026", "Revenue (USD)": "$5,920,000", "Operating Expenses": "$3,710,000", "Net Margin (%)": "37.3%"}
            ],
            "num_rows": 4,
            "num_cols": 4,
            "source_format": "markdown"
        }

        table2 = {
            "table_index": 1,
            "document_name": "Employee_Directory.docx",
            "headers": ["Employee Name", "Department", "Role", "Tenure (Years)"],
            "rows": [
                {"Employee Name": "Sarah Connor", "Department": "Engineering", "Role": "Lead Architect", "Tenure (Years)": "6.2"},
                {"Employee Name": "Miles Dyson", "Department": "R&D", "Role": "Director of Cybernetics", "Tenure (Years)": "8.5"},
                {"Employee Name": "John Connor", "Department": "Operations", "Role": "Operations Specialist", "Tenure (Years)": "3.1"}
            ],
            "num_rows": 3,
            "num_cols": 4,
            "source_format": "docx"
        }

        doc1_text = (
            "EXECUTIVE SUMMARY - FISCAL PERFORMANCE OVERVIEW\n\n"
            "During the fiscal period, total enterprise revenues demonstrated steady sequential expansion driven by "
            "increased adoption of automated AI data pipelines. Cloud infrastructure cost optimization initiatives "
            "helped expand overall gross margins across core business units.\n\n"
            "| Quarter | Revenue (USD) | Operating Expenses | Net Margin (%) |\n"
            "|---|---|---|---|\n"
            "| Q1 2026 | $4,250,000 | $2,980,000 | 29.8% |\n"
            "| Q2 2026 | $4,810,000 | $3,150,000 | 34.5% |\n"
            "| Q3 2026 | $5,190,000 | $3,420,000 | 34.1% |\n"
            "| Q4 2026 | $5,920,000 | $3,710,000 | 37.3% |\n\n"
            "Management expects continued operational momentum into the upcoming operating cycles."
        )

        doc2_text = (
            "HUMAN RESOURCES DEPARTMENT ROSTER & ORGANIZATIONAL MAPPING\n\n"
            "The following team members are verified as active personnel across technical development and strategy groups.\n\n"
            "Team members have successfully completed data privacy compliance and security onboarding protocols."
        )

        docs = [
            {
                "document_name": "Quarterly_Financial_Report.pdf",
                "page_number": 1,
                "content_type": "pdf",
                "extracted_text": doc1_text,
                "tables_count": 1,
                "tables_preview": json.dumps([table1["headers"]]),
                "char_count": len(doc1_text),
                "word_count": len(doc1_text.split()),
                "metadata": json.dumps({"title": "Fiscal Overview", "author": "Finance Team", "format": "pdf"})
            },
            {
                "document_name": "Employee_Directory.docx",
                "page_number": 1,
                "content_type": "docx",
                "extracted_text": doc2_text,
                "tables_count": 1,
                "tables_preview": json.dumps([table2["headers"]]),
                "char_count": len(doc2_text),
                "word_count": len(doc2_text.split()),
                "metadata": json.dumps({"title": "Team Directory", "author": "HR Ops", "format": "docx"})
            }
        ]

        return docs, [table1, table2]
