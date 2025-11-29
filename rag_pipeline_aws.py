import os
import io
import re
import json
import logging
from typing import Dict, Any, List, Optional

import boto3
import pdfplumber
from huggingface_hub import InferenceClient

import sys
from ingestion.config import CFG

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llm_claim_gen")

# -------------------------
# Config from ENV + CFG
# -------------------------

# Hugging Face token (must be set: HF_TOKEN="hf_xxx")
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN is None:
    raise RuntimeError("Please set HF_TOKEN environment variable to your Hugging Face token.")

# HF chat model name (router-compatible)
HF_MODEL = os.environ.get("HF_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")

AWS_REGION = CFG["aws"]["region"]
S3_BUCKET = CFG["aws"]["s3_bucket_raw"]
PREFIX_EDGAR = CFG["aws"]["prefix_edgar"].rstrip("/") + "/"
PREFIX_SITE = CFG["aws"]["prefix_site"].rstrip("/") + "/"
OUTPUT_PREFIX = CFG["aws"]["output_prefix"].rstrip("/") + "/"  # "results/"

# Safety: truncate very long docs before sending to LLM
MAX_TEXT_CHARS = int(os.environ.get("MAX_TEXT_CHARS", "8000"))

# claims per report (minimum target)
N_CLAIMS = 6

# -------------------------
# S3 helpers
# -------------------------

def s3_client():
    return boto3.client("s3", region_name=AWS_REGION)


def list_objects_under_prefix(prefix: str) -> List[str]:
    """
    List all non-folder objects under a given prefix.
    """
    keys: List[str] = []
    s3 = s3_client()
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            if key.endswith("/"):
                continue
            keys.append(key)

    return keys


def download_bytes(key: str) -> bytes:
    s3 = s3_client()
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return obj["Body"].read()


def s3_write_json(key: str, data: Dict[str, Any]):
    """
    Upload JSON to s3://S3_BUCKET/key
    """
    s3 = s3_client()
    body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=body,
        ContentType="application/json",
    )
    logger.info(f"Uploaded JSON to s3://{S3_BUCKET}/{key}")


# -------------------------
# Text extraction
# -------------------------

HTML_TAG_RE = re.compile(r"<[^>]+>")

def html_to_text(raw_bytes: bytes) -> str:
    text = raw_bytes.decode("utf-8", errors="ignore")
    text = HTML_TAG_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def pdf_to_text(raw_bytes: bytes) -> str:
    parts: List[str] = []
    with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            parts.append(page_text)
    text = "\n".join(parts)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_text_for_key(key: str) -> Optional[str]:
    """
    Download a file and convert it to plain text.
    Supports HTML / HTM / TXT / PDF.
    """
    try:
        raw = download_bytes(key)
        lower = key.lower()

        if lower.endswith(".pdf"):
            text = pdf_to_text(raw)
        else:
            # treat any non-PDF as HTML-ish / text
            text = html_to_text(raw)

        if not text:
            logger.warning(f"No text extracted from {key}")
            return None

        # truncate for safety
        if len(text) > MAX_TEXT_CHARS:
            text = text[:MAX_TEXT_CHARS]

        return text
    except Exception as e:
        logger.exception(f"Failed to extract text for {key}: {e}")
        return None


# -------------------------
# LLM prompt + call (Hugging Face)
# -------------------------

SYSTEM_PROMPT = """You are a fact-checking assistant for financial and ESG disclosures.

You receive the full text of a company disclosure (10-K, 20-F, sustainability report, or similar).
Your task is to propose FACTUAL CLAIMS and classify each claim as either:

- "support": clearly supported by the disclosure, with a specific evidence snippet.
- "not support": not supported or contradicted by the disclosure, or the disclosure gives
  insufficient information to verify the claim.

Requirements:
- Generate AT LEAST 6 claims for each document (ideally between 6 and 10).
- Across the ENTIRE dataset (all documents combined), the number of "support" and "not support"
  labels should be as balanced as possible (close to 50/50).
- For any SINGLE document, you still MUST label each claim correctly based ONLY on the text.
  Never mislabel a claim just to balance labels.
- Claims must be concise (1 sentence) and specific enough that they could be checked in the text.
- For each claim, also output a short evidence snippet (1–3 sentences) copied or very closely
  paraphrased from the disclosure.

Output format:
Return a valid JSON array (and nothing else) of objects like:
[
  {"claim": "...", "answer": "support", "evidence": "..."},
  {"claim": "...", "answer": "not support", "evidence": "..."},
  ...
]

Where:
- "answer" is exactly "support" or "not support".
"""


def call_llm_generate_claims(
    client: InferenceClient,
    doc_text: str,
    total_support: int,
    total_not_support: int,
) -> List[Dict[str, str]]:
    """
    Call the HF model with the document text and parse its JSON response.

    We pass in global label counts so the model can *aim* for balance across the dataset,
    while still labeling correctly for each document.
    """

    imbalance = total_support - total_not_support

    if imbalance > 0:
        # More support than not_support globally
        balance_hint = (
            f"There are currently {total_support} 'support' labels and "
            f"{total_not_support} 'not support' labels in the existing dataset, "
            f"so 'support' is over-represented by {imbalance} labels.\n"
            "When you are genuinely uncertain whether a claim is fully supported, "
            "you may lean slightly towards 'not support'. However, never mislabel: "
            "if the text clearly supports the claim, still use 'support'."
        )
    elif imbalance < 0:
        # More not_support than support globally
        balance_hint = (
            f"There are currently {total_support} 'support' labels and "
            f"{total_not_support} 'not support' labels in the existing dataset, "
            f"so 'not support' is over-represented by {-imbalance} labels.\n"
            "When you are genuinely uncertain, you may lean slightly towards 'support'. "
            "However, never mislabel: if the text does NOT support the claim, "
            "or is insufficient, still use 'not support'."
        )
    else:
        balance_hint = (
            f"The current global label distribution is perfectly balanced: "
            f"{total_support} 'support' and {total_not_support} 'not support'. "
            "You do not need to prefer one label over the other; just label correctly."
        )

    user_prompt = f"""
We are constructing a GLOBAL dataset of fact-checking claims from many documents.

GLOBAL LABEL COUNTS SO FAR:
- support: {total_support}
- not support: {total_not_support}

Guidance:
{balance_hint}

Now, consider ONLY the following document text:

\"\"\"{doc_text}\"\"\"


Generate a JSON array of at least {N_CLAIMS} factual claims for this document, each with:
- "claim": the statement being made,
- "answer": "support" or "not support", strictly based on the document,
- "evidence": a short snippet (1–3 sentences) from the document that justifies your label.

Remember:
- Always prioritize correctness of each label for this document.
- Use the global label counts only as a mild tie-breaker when you are truly unsure.
"""

    resp = client.chat_completion(
        model=HF_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=1500,
        temperature=0.3,
    )

    text = resp.choices[0].message["content"].strip()

    # Try direct JSON parsing first
    def try_parse_json(s: str) -> Optional[List[Dict[str, str]]]:
        try:
            data = json.loads(s)
            if isinstance(data, list):
                return data
        except Exception:
            return None
        return None

    claims = try_parse_json(text)
    if claims is None:
        # Attempt to extract the JSON array substring
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            snippet = text[start: end + 1]
            claims = try_parse_json(snippet)

    if claims is None:
        logger.warning("Failed to parse LLM output as JSON. Raw output:\n%s", text[:500])
        return []

    # light cleanup: ensure fields exist and answers are only support/not support
    cleaned: List[Dict[str, str]] = []
    for c in claims:
        claim = c.get("claim")
        ans_raw = (c.get("answer") or "").strip().lower()
        evid = c.get("evidence")

        if not claim or not evid:
            continue

        if ans_raw == "support":
            answer = "support"
        else:
            # anything else we collapse to "not support"
            answer = "not support"

        cleaned.append(
            {
                "claim": claim.strip(),
                "answer": answer,
                "evidence": evid.strip(),
            }
        )

    # ENFORCE MINIMUM: if fewer than N_CLAIMS after cleaning, treat as failure
    if len(cleaned) < N_CLAIMS:
        logger.warning(
            "Only %d cleaned claims (min required %d). Treating as generation failure.",
            len(cleaned),
            N_CLAIMS,
        )
        return []

    # Otherwise return ALL cleaned claims (can be >= N_CLAIMS)
    return cleaned


# -------------------------
# Main pipeline
# -------------------------

def main():
    logger.info("Using HF model: %s", HF_MODEL)
    client = InferenceClient(model=HF_MODEL, token=HF_TOKEN)

    # 1) Collect all report keys from edgar/ and site/
    edgar_keys = list_objects_under_prefix(PREFIX_EDGAR)
    site_keys = list_objects_under_prefix(PREFIX_SITE)

    all_keys = edgar_keys + site_keys
    logger.info(
        "Found %d EDGAR keys, %d site keys, total %d",
        len(edgar_keys), len(site_keys), len(all_keys)
    )

    results: Dict[str, Any] = {}
    num_success = 0

    # global label counters
    total_support = 0
    total_not_support = 0

    for key in all_keys:
        s3_url = f"s3://{S3_BUCKET}/{key}"
        logger.info("Processing %s", s3_url)

        text = extract_text_for_key(key)
        if not text:
            results[s3_url] = {
                "status": "error",
                "error": "text_extraction_failed",
                "answer": "",
                "sources": [s3_url],
            }
            continue

        claims = call_llm_generate_claims(
            client=client,
            doc_text=text,
            total_support=total_support,
            total_not_support=total_not_support,
        )
        if not claims:
            results[s3_url] = {
                "status": "error",
                "error": "llm_generation_failed",
                "answer": "",
                "sources": [s3_url],
            }
            continue

        # update global counts
        for c in claims:
            if c["answer"] == "support":
                total_support += 1
            else:
                total_not_support += 1

        # Wrap claims in the same inner JSON string format as your old pipeline
        inner = {
            "claims": [
                {
                    "claim": c["claim"],
                    "evidence": c["evidence"],
                    "answer": c["answer"],
                }
                for c in claims
            ]
        }
        inner_str = json.dumps(inner, ensure_ascii=False, indent=2)

        results[s3_url] = {
            "status": "success",
            "answer": inner_str,
            "sources": [s3_url],
        }
        num_success += 1

    # 2) Final batch_results-style object
    out_obj = {
        "total_files": len(all_keys),
        "results": results,
    }

    # 3) Save to S3 in /results/ folder from config
    output_key = OUTPUT_PREFIX + "batch_results.json"
    s3_write_json(output_key, out_obj)

    logger.info("Finished. Successful files: %d / %d", num_success, len(all_keys))
    logger.info(
        "Label distribution in generated claims: support=%d, not_support=%d",
        total_support, total_not_support
    )
    logger.info("batch_results written to: s3://%s/%s", S3_BUCKET, output_key)


if __name__ == "__main__":
    main()
