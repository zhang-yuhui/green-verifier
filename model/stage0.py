# stage0_preprocess.py  (deterministic IDs, structured PDF/HTML, optional OCR)

import io
import uuid
import boto3
import logging
from typing import List, Dict, Any, Tuple
import os
import sys
import hashlib
import re

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import CFG

import pdfplumber
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stage0")

# -----------------------
# CONFIG
# -----------------------
AWS_REGION = CFG["aws"]["region"]
S3_BUCKET = CFG["aws"]["s3_bucket_raw"]
PREFIX_EDGAR = CFG["aws"]["prefix_edgar"]
PREFIX_SITE = CFG["aws"]["prefix_site"]

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

TEXT_COL = "reports_text"
KPI_COL = "reports_kpi"

EMBED_MODEL_NAME = os.getenv(
    "EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)

# batching params
TEXT_BATCH_SIZE = 500
KPI_BATCH_SIZE = 500
MAX_KPI_SENTENCES_PER_REPORT = 5000  # safety cap

# OCR / Textract
ENABLE_TEXTRACT_OCR = os.getenv("ENABLE_TEXTRACT_OCR", "0") == "1"
# if avg chars/page is below this, assume image-heavy & run OCR (when enabled)
TEXTRACT_MIN_CHARS_PER_PAGE = int(os.getenv("TEXTRACT_MIN_CHARS_PER_PAGE", "500"))
# Textract's synchronous AnalyzeDocument has a 5MB doc size limit
TEXTRACT_MAX_BYTES = 5 * 1024 * 1024


# -----------------------
# HELPERS
# -----------------------
def get_s3_client():
    return boto3.client("s3", region_name=AWS_REGION)


def get_textract_client():
    return boto3.client("textract", region_name=AWS_REGION)


def list_s3_objects(prefix: str) -> List[str]:
    s3 = get_s3_client()
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            if not key.endswith("/"):
                keys.append(key)
    return keys


def download_s3_object(key: str) -> bytes:
    s3 = get_s3_client()
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return obj["Body"].read()


def tokenize(text: str) -> List[str]:
    return text.split()


def chunk_tokens(tokens: List[str], chunk_size=400, overlap=100) -> List[str]:
    chunks = []
    start = 0
    n = len(tokens)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = " ".join(tokens[start:end])
        chunks.append(chunk)
        start = start + chunk_size - overlap
    return chunks


# -----------------------
# NUMERIC / KPI HELPERS
# -----------------------

NUMERIC_LINE_RE = re.compile(
    r".*(\d{4}|\d+(\.\d+)?%|\d{1,3}(,\d{3})+).*"
)  # years, %, thousands


def extract_numeric_sentences_from_text(text: str, max_sentences: int = 80) -> List[str]:
    """
    Fallback: grab any line with numbers / % etc.
    """
    lines = text.splitlines()
    out = []
    for ln in lines:
        ln_clean = ln.strip()
        if not ln_clean:
            continue
        if NUMERIC_LINE_RE.match(ln_clean):
            ln_clean = re.sub(r"\s+", " ", ln_clean)
            out.append(f"Reported value: {ln_clean}")
        if len(out) >= max_sentences:
            break
    return out


# -----------------------
# STRUCTURED PDF HANDLING
# -----------------------

def _pdf_tables_to_sentences(
    pdf: pdfplumber.PDF,
) -> List[str]:
    """
    Convert pdfplumber tables into simple 'row/column/value' KPI sentences.
    """
    table_sents: List[str] = []

    pages = list(pdf.pages)
    for page_idx, page in enumerate(pages):
        try:
            tables = page.extract_tables()
        except Exception as e:
            logger.warning(f"Table extraction failed on p{page_idx+1}: {e}")
            tables = []

        for t_idx, table in enumerate(tables):
            if not table or len(table) < 2:
                continue

            header = table[0]
            rows = table[1:]

            for r_idx, row in enumerate(rows):
                if not row:
                    continue
                row_label = (row[0] or "").strip() if row[0] else f"Row {r_idx+1}"
                for c_idx, cell in enumerate(row[1:], start=1):
                    if cell is None:
                        continue
                    cell_str = str(cell).strip()
                    if not cell_str:
                        continue
                    # keep only values with digits
                    if re.search(r"\d", cell_str):
                        col_label = (
                            (header[c_idx] or "").strip()
                            if header and c_idx < len(header)
                            else f"Col {c_idx+1}"
                        )
                        sent = (
                            f"Table p{page_idx+1}, {row_label}, {col_label}: {cell_str}"
                        )
                        table_sents.append(sent)

    return table_sents


def _textract_ocr(pdf_bytes: bytes) -> Tuple[str, List[str]]:
    """
    Use Textract (optional) to OCR PDFs with little selectable text.
    Returns (ocr_text, ocr_table_sents).
    """
    textract = get_textract_client()
    response = textract.analyze_document(
        Document={"Bytes": pdf_bytes},
        FeatureTypes=["TABLES"],
    )

    lines: List[str] = []
    table_sents: List[str] = []

    for block in response.get("Blocks", []):
        btype = block.get("BlockType")
        if btype == "LINE":
            txt = block.get("Text") or ""
            txt = txt.strip()
            if txt:
                lines.append(txt)
        elif btype == "CELL":
            txt = (block.get("Text") or "").strip()
            if not txt or not re.search(r"\d", txt):
                continue
            row = block.get("RowIndex")
            col = block.get("ColumnIndex")
            sent = f"Textract table r{row} c{col}: {txt}"
            table_sents.append(sent)

    ocr_text = "\n".join(lines)
    return ocr_text, table_sents


def pdf_to_structured_text(pdf_bytes: bytes) -> Tuple[str, List[str]]:
    """
    Returns:
        text: full text for text index (pdfplumber + optional OCR)
        table_sents: KPI-friendly sentences from tables (pdfplumber + optional OCR)
    """
    text_parts: List[str] = []
    table_sents: List[str] = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        pages = list(pdf.pages)
        n_pages = len(pages)

        for page in pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)

        # tables from pdfplumber
        try:
            table_sents.extend(_pdf_tables_to_sentences(pdf))
        except Exception as e:
            logger.warning(f"PDF table parsing failed: {e}")

    base_text = "\n".join(text_parts)

    # optional OCR for image-heavy / scanned PDFs
    if ENABLE_TEXTRACT_OCR and len(pdf_bytes) <= TEXTRACT_MAX_BYTES:
        avg_chars_per_page = len(base_text) / max(n_pages, 1)
        if avg_chars_per_page < TEXTRACT_MIN_CHARS_PER_PAGE:
            logger.info(
                f"PDF appears image-heavy (avg {avg_chars_per_page:.1f} chars/page); "
                f"running Textract OCR."
            )
            try:
                ocr_text, ocr_table_sents = _textract_ocr(pdf_bytes)
                if ocr_text:
                    base_text = base_text + "\n" + ocr_text
                table_sents.extend(ocr_table_sents)
            except Exception as e:
                logger.warning(f"Textract OCR failed: {e}")
    elif ENABLE_TEXTRACT_OCR and len(pdf_bytes) > TEXTRACT_MAX_BYTES:
        logger.info(
            "PDF larger than Textract sync limit; skipping OCR (you can add async "
            "Textract if needed)."
        )

    return base_text, table_sents


# -----------------------
# STRUCTURED HTML HANDLING
# -----------------------

def simple_html_to_text(html: str) -> str:
    """
    Fallback text extraction using regex (no structure).
    """
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def html_to_structured_text(html: str) -> Tuple[str, List[str]]:
    """
    Parse HTML into plain text + table KPI sentences.
    """
    table_sents: List[str] = []
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "lxml")

        # plain text (for general text index + numeric fallback)
        text = soup.get_text(separator="\n")

        # tables
        for table in soup.find_all("table"):
            rows = table.find_all("tr")
            if not rows:
                continue
            header_cells = [
                c.get_text(strip=True) for c in rows[0].find_all(["th", "td"])
            ]
            body_rows = rows[1:]
            for r_idx, row in enumerate(body_rows):
                cells = [c.get_text(strip=True) for c in row.find_all(["th", "td"])]
                if not cells:
                    continue
                row_label = cells[0] or f"Row {r_idx+1}"
                for c_idx, cell_val in enumerate(cells[1:], start=1):
                    if not cell_val or not re.search(r"\d", cell_val):
                        continue
                    col_label = (
                        header_cells[c_idx]
                        if header_cells and c_idx < len(header_cells)
                        else f"Col {c_idx+1}"
                    )
                    sent = f"HTML table {row_label}, {col_label}: {cell_val}"
                    table_sents.append(sent)
    except Exception as e:
        logger.warning(f"HTML structured parse failed; fallback to regex text only: {e}")
        text = simple_html_to_text(html)

    return text, table_sents


# -----------------------
# KPI SENTENCE BUILDER
# -----------------------

def build_kpi_sentences(raw_meta: Dict[str, Any]) -> List[str]:
    """
    Combines several sources into KPI-like sentences:
      1) iXBRL tags (if present in HTML)
      2) table-based sentences (PDF + HTML)
      3) numeric-rich lines from full text
    """
    sentences: List[str] = []

    html_str = raw_meta.get("html")
    text = raw_meta.get("text", "") or ""
    table_sents = raw_meta.get("table_sents") or []

    # 1) iXBRL facts from HTML (EDGAR-like filings)
    if html_str:
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html_str, "lxml")
            ix_tags = soup.find_all(["ix:nonfraction", "ix:nonnumeric"])
            for tag in ix_tags:
                metric = tag.get("name") or tag.get("concept") or "Reported fact"
                value = (tag.text or "").strip()
                unit = tag.get("unitref") or ""
                ctx = tag.get("contextref") or ""
                if value:
                    if unit and ctx:
                        sent = f"{metric} ({ctx}): {unit} {value}"
                    elif ctx:
                        sent = f"{metric} ({ctx}): {value}"
                    else:
                        sent = f"{metric}: {value}"
                    sentences.append(sent)
        except Exception as e:
            logger.warning(f"iXBRL parse failed for {raw_meta.get('s3_key')}: {e}")

    # 2) structured table values (PDF + HTML + OCR)
    for ts in table_sents:
        sentences.append(f"Reported table value: {ts}")

    # 3) generic numeric lines from full text
    numeric_sents = extract_numeric_sentences_from_text(text)
    sentences.extend(numeric_sents)

    # safety cap
    if len(sentences) > MAX_KPI_SENTENCES_PER_REPORT:
        logger.info(
            f"KPI sentences for {raw_meta.get('s3_key')} capped "
            f"from {len(sentences)} to {MAX_KPI_SENTENCES_PER_REPORT}"
        )
        sentences = sentences[:MAX_KPI_SENTENCES_PER_REPORT]

    return sentences


# -----------------------
# QDRANT
# -----------------------

def init_qdrant():
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    model = SentenceTransformer(EMBED_MODEL_NAME)
    dim = model.get_sentence_embedding_dimension()

    existing = [c.name for c in client.get_collections().collections]
    if TEXT_COL not in existing:
        client.recreate_collection(
            collection_name=TEXT_COL,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
    if KPI_COL not in existing:
        client.recreate_collection(
            collection_name=KPI_COL,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
    return client, model


# -----------------------
# DETERMINISTIC ID
# -----------------------

def stable_id(s3_key: str, source: str, ordinal: int, content: str) -> str:
    """
    Stable across runs; changes only if the content changes.
    """
    h = hashlib.md5(content.encode("utf-8")).hexdigest()[:8]
    name = f"{s3_key}#{source}#{ordinal}#{h}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, name))


# -----------------------
# MAIN STAGE 0
# -----------------------

def process_report(key: str, qdrant: QdrantClient, model: SentenceTransformer):
    logger.info(f"Processing {key}")
    data = download_s3_object(key)

    html_raw: str | None = None
    table_sents: List[str] = []

    if key.lower().endswith(".pdf"):
        text, table_sents = pdf_to_structured_text(data)
    elif key.lower().endswith((".html", ".htm")):
        html_raw = data.decode("utf-8", errors="ignore")
        text, table_sents = html_to_structured_text(html_raw)
    else:
        logger.warning(f"Unknown format for {key}, skipping.")
        return

    # --- TEXT INDEX ---
    tokens = tokenize(text)
    chunks = chunk_tokens(tokens, chunk_size=400, overlap=100)
    if not chunks:
        logger.warning(f"No text chunks produced for {key}")
        return

    text_vectors = model.encode(
        chunks, convert_to_numpy=True, show_progress_bar=False
    )

    text_points: List[PointStruct] = []
    for idx, (chunk, vec) in enumerate(zip(chunks, text_vectors)):
        pid = stable_id(key, "text", idx, chunk)  # deterministic ID
        payload = {"s3_key": key, "chunk_id": idx, "source": "text"}
        text_points.append(
            PointStruct(id=pid, vector=vec.tolist(), payload=payload)
        )

    for i in range(0, len(text_points), TEXT_BATCH_SIZE):
        batch = text_points[i : i + TEXT_BATCH_SIZE]
        qdrant.upsert(collection_name=TEXT_COL, points=batch)
    logger.info(f"Upserted {len(text_points)} text chunks for {key}")

    # --- KPI INDEX ---
    kpi_sentences = build_kpi_sentences(
        {
            "s3_key": key,
            "text": text,
            "html": html_raw,
            "table_sents": table_sents,
        }
    )

    if kpi_sentences:
        kpi_vectors = model.encode(
            kpi_sentences, convert_to_numpy=True, show_progress_bar=False
        )
        kpi_points: List[PointStruct] = []
        for idx, (sent, vec) in enumerate(zip(kpi_sentences, kpi_vectors)):
            pid = stable_id(key, "kpi", idx, sent)  # deterministic ID
            kpi_points.append(
                PointStruct(
                    id=pid,
                    vector=vec.tolist(),
                    payload={"s3_key": key, "sentence": sent, "source": "kpi"},
                )
            )

        for i in range(0, len(kpi_points), KPI_BATCH_SIZE):
            batch = kpi_points[i : i + KPI_BATCH_SIZE]
            qdrant.upsert(collection_name=KPI_COL, points=batch)
        logger.info(f"Upserted {len(kpi_points)} KPI sentences for {key}")
    else:
        logger.info(f"No KPI-like sentences extracted for {key}")


def run_stage0():
    qdrant, model = init_qdrant()
    edgar_keys = list_s3_objects(PREFIX_EDGAR)
    site_keys = list_s3_objects(PREFIX_SITE)
    all_keys = edgar_keys + site_keys
    logger.info(f"Found {len(all_keys)} reports in S3.")
    for key in all_keys:
        process_report(key, qdrant, model)


if __name__ == "__main__":
    run_stage0()
