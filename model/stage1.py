import os
import json
import boto3
import logging
from typing import List, Dict, Any, Tuple, Optional

os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, ScoredPoint

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ingestion.config import CFG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stage1")

# ------------ CONFIG ------------
AWS_REGION = CFG["aws"]["region"]
S3_BUCKET = CFG["aws"]["s3_bucket_raw"]
S3_PREFIX_RESULTS = CFG["aws"]["output_prefix"].rstrip("/") + "/"
S3_PREFIX_RETRIEVAL = S3_PREFIX_RESULTS + "retrieval/"

# Qdrant Cloud
QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "https://73f8efd6-dd77-481b-8549-7b40e8a6ea7c.europe-west3-0.gcp.cloud.qdrant.io",
)
QDRANT_API_KEY = os.getenv(
    "QDRANT_API_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.qweUU1RlgvYA4RLxS7Lkv5GBtzlNz0bDIVzOtVcaH-M",
)

TEXT_COLLECTION = "reports_text"
KPI_COLLECTION = "reports_kpi"

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "intfloat/e5-base-v2")
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
        #file does not exist — safe to ignore
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
    one of {'support', 'notsupport'}.
    """
    if not raw:
        return None

    t = raw.strip().lower()

    # support
    if t == "support":
        return "support"

    # not support 
    t_nospace = t.replace(" ", "").replace("_", "")
    if t_nospace == "notsupport":
        return "notsupport"

    #anything else- don't use it as gold label
    logger.warning(f"Unknown gold label value in batch_results: {raw}")
    return None


# ------------ CLAIM LOADING ------------
def load_claims_with_sources(blob: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract (claim, report_s3_key, optional label, optional gold evidence)
    from batch_results-style blob.
    """
    out: List[Dict[str, Any]] = []

    for s3_url, item in blob.get("results", {}).items():
        try:
            bucket, key = s3_url_to_bucket_key(s3_url)
            if bucket != S3_BUCKET:
                continue

            # item["answer"] with format:
            # "{ "claims\":
            answer_json = json.loads(item["answer"])
            for c in answer_json.get("claims", []):
                claim_text = c.get("claim")
                if not claim_text:
                    continue

                raw_label = c.get("answer")
                label = normalize_label(raw_label)
                gold_ev = c.get("evidence")

                rec: Dict[str, Any] = {
                    "claim": claim_text,
                    "report_s3_key": key,
                }
                if label is not None:
                    rec["label"] = label  # canonical: support / notsupport
                if gold_ev:
                    rec["gold_evidence"] = gold_ev

                out.append(rec)

        except Exception as e:
            # e.g. if item["answer"] is empty or invalid JSON
            logger.warning(f"Failed to parse claims for {s3_url}: {e}")

    return out


# ------------ QDRANT HELPERS ------------
def ensure_s3_key_index(client: QdrantClient, collection: str):
    """
    Ensure that a payload index exists on `s3_key` as keyword for the given collection.
    Safe to call multiple times.
    """
    try:
        client.create_payload_index(
            collection_name=collection,
            field_name="s3_key",
            field_schema="keyword",  
            wait=True,
        )
        logger.info(f"Created payload index on 's3_key' for collection '{collection}'")
    except Exception as e:
        # index already exists
        logger.info(
            f"Payload index on 's3_key' for collection '{collection}' may already exist: {e}"
        )


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
    query_text = f"query: {claim}"
    vec = model.encode(query_text, convert_to_numpy=True).tolist()

    qfilter = Filter(
        must=[FieldCondition(key="s3_key", match=MatchValue(value=report_s3_key))]
    )

    # TEXT collection
    text_resp = qdrant.query_points(
        collection_name=TEXT_COLLECTION,
        query=vec,
        query_filter=qfilter,
        limit=k,
    )
    res_text = [("reports_text", p) for p in text_resp.points]

    # KPI collection
    kpi_resp = qdrant.query_points(
        collection_name=KPI_COLLECTION,
        query=vec,
        query_filter=qfilter,
        limit=k,
    )
    res_kpi = [("reports_kpi", p) for p in kpi_resp.points]

    combined = res_text + res_kpi
    combined.sort(key=lambda p: p[1].score, reverse=True)
    return combined[:k]


# ------------ MAIN ------------
def run_stage1():
    logger.info("Initializing model and Qdrant client…")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    #Make sure we can filter on s3_key in both collections
    ensure_s3_key_index(qdrant, TEXT_COLLECTION)
    ensure_s3_key_index(qdrant, KPI_COLLECTION)

    files = list_results_files()
    if not files:
        logger.warning(
            f"No *_results.json files found under s3://{S3_BUCKET}/{S3_PREFIX_RESULTS}"
        )
        return

    files.sort()
    source_key = files[0]
    if len(files) > 1:
        logger.warning(
            f"Multiple results files found; using only {source_key} for retrieval."
        )

    logger.info(f"Processing claim-generation file: {source_key}")
    blob = s3_read_json(source_key)
    claims = load_claims_with_sources(blob)

    output: Dict[str, Any] = {
        "source_results_key": source_key,     # e.g."results/batch_results.json"
        "model": EMBED_MODEL_NAME,
        "top_k": TOP_K,
        "items": [],
    }

    for item in claims:
        claim = item["claim"]
        report_key = item["report_s3_key"]
        label = item.get("label")            # "support" / "notsupport" or None
        gold_ev = item.get("gold_evidence")  # golden snippet from Stage-0

        results = retrieve_for_claim_in_report(
            claim, report_key, qdrant, model, TOP_K
        )
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
        if gold_ev is not None:
            claim_record["gold_evidence"] = gold_ev

        output["items"].append(claim_record)

    #Always write to: results/retrieval/batch_results_retrieval.json
    out_key = f"{S3_PREFIX_RETRIEVAL}batch_results_retrieval.json"
    s3_write_json(out_key, output)


if __name__ == "__main__":
    run_stage1()
