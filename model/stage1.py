# stage1_retrieve.py
import os
import json
import boto3
import logging
from typing import List, Dict, Any, Tuple, Optional

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import ScoredPoint
from qdrant_client.models import Filter, FieldCondition, MatchValue

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import CFG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stage1")

# ------------ CONFIG ------------
AWS_REGION = CFG["aws"]["region"]
S3_BUCKET = CFG["aws"]["s3_bucket_raw"]
S3_PREFIX_RESULTS = CFG["aws"]["output_prefix"].rstrip("/") + "/"
S3_PREFIX_RETRIEVAL = S3_PREFIX_RESULTS + "retrieval/"

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

TEXT_COLLECTION = "reports_text"
KPI_COLLECTION = "reports_kpi"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5

# ------------ S3 HELPERS ------------
def get_s3_client():
    return boto3.client("s3", region_name=AWS_REGION)


def list_results_files() -> List[str]:
    """
    Return ONLY the Stage-0 claim-generation JSON(s), e.g.:
      results/batch_results.json

    It explicitly ignores:
      - results/retrieval/...
      - results/splits/...
      - results/verification/...
    """
    s3 = get_s3_client()
    keys: List[str] = []

    exclude_prefixes = [
        S3_PREFIX_RETRIEVAL,                   # results/retrieval/
        S3_PREFIX_RESULTS + "splits/",         # results/splits/
        S3_PREFIX_RESULTS + "verification/",   # results/verification/
    ]

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX_RESULTS):
        for item in page.get("Contents", []):
            key = item["Key"]

            # Skip retrieval/splits/verification subfolders
            if any(key.startswith(p) for p in exclude_prefixes):
                continue

            # We only expect Stage-0 to produce "*_results.json"
            if key.endswith("_results.json"):
                keys.append(key)

    return keys


def s3_read_json(key: str) -> Dict[str, Any]:
    s3 = get_s3_client()
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return json.loads(obj["Body"].read())


def s3_delete_if_exists(key: str):
    """Delete an S3 file if it already exists."""
    s3 = get_s3_client()
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=key)
        s3.delete_object(Bucket=S3_BUCKET, Key=key)
        logger.info(f"Deleted existing file: s3://{S3_BUCKET}/{key}")
    except s3.exceptions.ClientError:
        # file does not exist — safe to ignore
        pass


def s3_write_json(key: str, data: Dict[str, Any]):
    """Write JSON file to S3 (after deleting existing one)."""
    s3_delete_if_exists(key)
    s3 = get_s3_client()
    body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=body, ContentType="application/json")
    logger.info(f"Saved new retrieval file to s3://{S3_BUCKET}/{key}")


def s3_url_to_bucket_key(s3_url: str) -> Tuple[str, str]:
    assert s3_url.startswith("s3://")
    no_scheme = s3_url[len("s3://"):]
    bucket, _, key = no_scheme.partition("/")
    return bucket, key

# ------------ LABEL NORMALIZATION ------------

def normalize_label(raw: Optional[str]) -> Optional[str]:
    """
    Map the raw 'answer' field from batch_results into
    one of {'support', 'notsupport', 'dontknow'}.

    Your data uses exactly:
      - "support"
      - "not support"
      - "don't know"

    We normalise spelling/spacing/case and return:
      - "support"
      - "notsupport"
      - "dontknow"
    """
    if not raw:
        return None

    t = raw.strip().lower()

    # 1) support
    if t == "support":
        return "support"

    # 2) not support (with possible spaces/underscores)
    t_nospace = t.replace(" ", "").replace("_", "")
    if t_nospace == "notsupport":
        return "notsupport"

    # 3) don't know (handle apostrophes and spaces)
    t_noapos = t.replace("’", "").replace("'", "")
    t_noapos_nospace = t_noapos.replace(" ", "").replace("_", "")
    if t_noapos_nospace == "dontknow":
        return "dontknow"

    # anything else -> we don't use it as gold label
    logger.warning(f"Unknown gold label value in batch_results: {raw}")
    return None

# ------------ CLAIM LOADING ------------

def load_claims_with_sources(blob: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract (claim, report_s3_key, optional label) from batch_results-style blob.

    Input shape (simplified):

      {
        "results": {
          "s3://bucket/path/to/report.html": {
            "status": "...",
            "answer": "{ \"claims\": [{\"claim\": ..., \"answer\": ...}, ...]}"
          },
          ...
        }
      }

    Output: list of dicts each like
      {
        "claim": "...",
        "report_s3_key": "edgar/AAPL/...",
        "label": "support" | "notsupport" | "dontknow"   # optional
      }
    """
    out: List[Dict[str, Any]] = []

    for s3_url, item in blob.get("results", {}).items():
        try:
            bucket, key = s3_url_to_bucket_key(s3_url)
            if bucket != S3_BUCKET:
                continue

            answer_json = json.loads(item["answer"])
            for c in answer_json.get("claims", []):
                claim_text = c.get("claim")
                if not claim_text:
                    continue

                raw_label = c.get("answer")
                label = normalize_label(raw_label)

                rec: Dict[str, Any] = {
                    "claim": claim_text,
                    "report_s3_key": key,
                }
                if label is not None:
                    rec["label"] = label  # canonical: support/notsupport/dontknow

                out.append(rec)

        except Exception as e:
            logger.warning(f"Failed to parse claims for {s3_url}: {e}")

    return out

# ------------ RETRIEVAL ------------

def point_to_record(sp: ScoredPoint, collection: str) -> Dict[str, Any]:
    payload = sp.payload or {}
    return {
        "id": str(sp.id),
        "collection": collection,
        "score": float(sp.score),
        "source": payload.get("s3_key"),
        "chunk_id": payload.get("chunk_id"),
        "sentence": payload.get("sentence"),
    }


def retrieve_for_claim_in_report(
    claim: str,
    report_s3_key: str,
    qdrant: QdrantClient,
    model: SentenceTransformer,
    k: int = TOP_K,
):
    vec = model.encode(claim, convert_to_numpy=True)
    qfilter = Filter(must=[FieldCondition(key="s3_key", match=MatchValue(value=report_s3_key))])

    res_text = qdrant.search(TEXT_COLLECTION, query_vector=vec, limit=k, query_filter=qfilter)
    res_text = [("reports_text", r) for r in res_text]

    res_kpi = qdrant.search(KPI_COLLECTION, query_vector=vec, limit=k, query_filter=qfilter)
    res_kpi = [("reports_kpi", r) for r in res_kpi]

    combined = res_text + res_kpi
    combined.sort(key=lambda p: p[1].score, reverse=True)
    return combined[:k]

# ------------ MAIN ------------

def run_stage1():
    logger.info("Initializing model and Qdrant client…")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    files = list_results_files()
    logger.info(
        f"Found {len(files)} claim-generation JSONs in s3://{S3_BUCKET}/{S3_PREFIX_RESULTS}"
    )

    for key in files:
        logger.info(f"Processing {key}")
        blob = s3_read_json(key)
        claims = load_claims_with_sources(blob)
        

        output: Dict[str, Any] = {
            "source_results_key": key,
            "model": EMBED_MODEL_NAME,
            "top_k": TOP_K,
            "items": [],
        }

        for item in claims:
            claim = item["claim"]
            report_key = item["report_s3_key"]
            label = item.get("label")  # already normalized to support/notsupport/dontknow

            results = retrieve_for_claim_in_report(claim, report_key, qdrant, model, TOP_K)
            records = [point_to_record(sp, col) for col, sp in results]

            logger.info(f"CLAIM: {claim} ({report_key})")
            for r in records:
                logger.info(f"  → {r['collection']} | {r['score']:.3f} | {r['source']}")

            claim_record: Dict[str, Any] = {
                "claim": claim,
                "report_s3_key": report_key,
                "candidates": records,
            }
            if label is not None:
                claim_record["label"] = label

            output["items"].append(claim_record)

        base = os.path.basename(key)
        out_key = f"{S3_PREFIX_RETRIEVAL}{os.path.splitext(base)[0]}_retrieval.json"
        s3_write_json(out_key, output)

if __name__ == "__main__":
    run_stage1()
