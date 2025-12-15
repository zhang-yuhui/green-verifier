import os
import io
import json
import logging
from typing import Dict, Any, List, Optional

import boto3
import pdfplumber
import torch
import torch.nn.functional as F

from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ingestion.config import CFG

# --------------------------
# Logging & basic config
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stage2_evaluate_mil")

AWS_REGION     = CFG["aws"]["region"]
S3_BUCKET      = CFG["aws"]["s3_bucket_raw"]
RESULTS_PREFIX = CFG["aws"]["output_prefix"].rstrip("/") + "/"
SPLIT_PREFIX   = RESULTS_PREFIX + "splits/"
VERI_PREFIX    = RESULTS_PREFIX + "verification/"

# Where the fine-tuned model lives on S3
DEFAULT_MODEL_S3_PREFIX = RESULTS_PREFIX + "models/deberta_finetuned_mil/"

MODEL_S3_PREFIX = os.getenv("MODEL_S3_PREFIX", DEFAULT_MODEL_S3_PREFIX)

# Where to store/download the model locally
LOCAL_MODEL_DIR = os.getenv("LOCAL_MODEL_DIR", "./deberta_finetuned_mil")

# Split to evaluate on test
SPLIT_NAME = os.getenv("SPLIT_NAME", "test")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Labels (2-way, same as training)
# -------------------------
ID2LABEL = {
    0: "support",
    1: "not support",
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
NUM_LABELS = len(ID2LABEL)

# Map raw labels in claims_*.json to normalized labels
RAW_TO_NORM_LABEL = {
    "support": "support",
    "notsupport": "not support",
    "not support": "not support",
    "not_support": "not support",
    "no support": "not support",
}

# -------------------------
# Chunking config (aligned with MIL training)
# -------------------------
CHUNK_SIZE      = 400      
CHUNK_OVERLAP   = 100
MAX_SEQ_LEN     = 512


TEXT_CACHE: Dict[str, List[str]] = {}
JSON_CACHE: Dict[str, Any] = {}

# -------------------------
# S3 helpers
# -------------------------
def s3():
    return boto3.client("s3", region_name=AWS_REGION)


def s3_read_json(key: str) -> Dict[str, Any]:
    if key in JSON_CACHE:
        return JSON_CACHE[key]
    obj = s3().get_object(Bucket=S3_BUCKET, Key=key)
    data = json.loads(obj["Body"].read())
    JSON_CACHE[key] = data
    return data


def s3_delete_if_exists(key: str):
    client = s3()
    try:
        client.head_object(Bucket=S3_BUCKET, Key=key)
        client.delete_object(Bucket=S3_BUCKET, Key=key)
        logger.info(f"Deleted existing file: s3://{S3_BUCKET}/{key}")
    except client.exceptions.ClientError:
        # not present
        pass


def s3_write_json(key: str, data: Dict[str, Any]):
    s3_delete_if_exists(key)
    body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    s3().put_object(Bucket=S3_BUCKET, Key=key, Body=body, ContentType="application/json")
    logger.info(f"Saved s3://{S3_BUCKET}/{key}")


def download_model_dir_from_s3(s3_prefix: str, local_dir: str):
    """
    Download all files under s3_prefix into local_dir.
    Assumes training uploaded a standard HuggingFace folder there.
    """
    client = s3()
    paginator = client.get_paginator("list_objects_v2")

    logger.info(f"Downloading fine-tuned model from s3://{S3_BUCKET}/{s3_prefix} to {local_dir}")
    os.makedirs(local_dir, exist_ok=True)

    found = False
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            found = True
            rel_path = key[len(s3_prefix):]
            rel_path = rel_path.lstrip("/")
            local_path = os.path.join(local_dir, rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            client.download_file(S3_BUCKET, key, local_path)
            logger.info(f"Downloaded s3://{S3_BUCKET}/{key} -> {local_path}")

    if not found:
        raise RuntimeError(
            f"No files found under s3://{S3_BUCKET}/{s3_prefix}. "
            f"Did you run the MIL training and upload the model?"
        )


def ensure_local_model_dir() -> str:
    """
    Ensure the fine-tuned model is available locally.
    If LOCAL_MODEL_DIR already has a config.json, use it;
    otherwise download from S3.
    """
    config_path = os.path.join(LOCAL_MODEL_DIR, "config.json")
    if os.path.exists(config_path):
        logger.info(f"Using existing local model directory: {LOCAL_MODEL_DIR}")
        return LOCAL_MODEL_DIR

    download_model_dir_from_s3(MODEL_S3_PREFIX, LOCAL_MODEL_DIR)
    return LOCAL_MODEL_DIR


# --------------------------
# Text extraction & chunking
# -------------------------
def simple_html_to_text(html_bytes: bytes) -> str:
    html = html_bytes.decode("utf-8", errors="ignore")
    import re
    text = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", text).strip()


def pdf_to_text(pdf_bytes: bytes) -> str:
    parts = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    return "\n".join(parts)


def download_s3_bytes(key: str) -> bytes:
    obj = s3().get_object(Bucket=S3_BUCKET, Key=key)
    return obj["Body"].read()


def tokenize_ws(text: str) -> List[str]:
    return text.split()


def chunk_tokens(tokens: List[str],
                 chunk_size: int = CHUNK_SIZE,
                 overlap: int = CHUNK_OVERLAP) -> List[str]:
    chunks = []
    start = 0
    n = len(tokens)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(" ".join(tokens[start:end]))
        start = start + chunk_size - overlap
    return chunks


def get_chunks_for_report(report_key: str) -> Optional[List[str]]:
    if report_key in TEXT_CACHE:
        return TEXT_CACHE[report_key]

    try:
        raw = download_s3_bytes(report_key)
        if report_key.lower().endswith(".pdf"):
            text = pdf_to_text(raw)
        elif report_key.lower().endswith((".html", ".htm")):
            text = simple_html_to_text(raw)
        else:
            text = simple_html_to_text(raw)
        toks = tokenize_ws(text)
        chunks = chunk_tokens(toks, CHUNK_SIZE, CHUNK_OVERLAP)
        TEXT_CACHE[report_key] = chunks
        return chunks
    except Exception as e:
        logger.warning(f"Failed to load chunks for {report_key}: {e}")
        return None


def get_text_chunk(report_key: str, chunk_id: int) -> Optional[str]:
    chunks = get_chunks_for_report(report_key)
    if not chunks:
        return None
    if 0 <= chunk_id < len(chunks):
        return chunks[chunk_id]
    logger.warning(f"chunk_id {chunk_id} out of range for {report_key} (len={len(chunks)})")
    return None


# -----------------------------------
# Label normalization & inference helpers
# --------------------------------
def normalize_gold_label(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    key = raw.lower()
    return RAW_TO_NORM_LABEL.get(key)


def load_finetuned_model_and_tokenizer():
    local_dir = ensure_local_model_dir()
    logger.info(f"Loading fine-tuned model from {local_dir}")
    tokenizer = DebertaV2Tokenizer.from_pretrained(local_dir)
    model = DebertaV2ForSequenceClassification.from_pretrained(
        local_dir,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(DEVICE)
    model.eval()
    return tokenizer, model, local_dir


def predict_pair_probs(tokenizer,
                       model,
                       claim: str,
                       evidence: str,
                       max_length: int = MAX_SEQ_LEN) -> torch.Tensor:
    enc = tokenizer(
        claim,
        evidence,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    ).to(DEVICE)
    with torch.no_grad():
        logits = model(**enc).logits  # [1, 2]
        probs = F.softmax(logits, dim=-1)  # [1, 2]
    return probs.squeeze(0).cpu()  # [2]


def aggregate_over_candidates(claim_item: Dict[str, Any],
                              tokenizer,
                              model) -> Dict[str, Any]:
    """
    claim_item:
      {
        "claim": str,
        "report_s3_key": str,
        "candidates": [{id, collection, score, source, chunk_id, sentence}, ...],
        "label": raw_label   # optional gold label
        "gold_evidence": str # optional from Stage 0
      }
    """
    claim      = claim_item["claim"]
    report_key = claim_item["report_s3_key"]
    candidates = claim_item.get("candidates", [])

    max_probs = torch.zeros(NUM_LABELS)
    best_ev_idx_for_label = [-1] * NUM_LABELS
    all_evidence_scored = []

    for idx, cand in enumerate(candidates):
        # Evidence text
        if cand["collection"] == "reports_kpi" and cand.get("sentence"):
            evidence_text = cand["sentence"]
        else:
            cid = cand.get("chunk_id")
            if cid is None:
                continue
            evidence_text = get_text_chunk(report_key=cand["source"], chunk_id=cid)
            if not evidence_text:
                continue

        probs = predict_pair_probs(tokenizer, model, claim, evidence_text)  #[2]
        all_evidence_scored.append((idx, probs.tolist()))

        # update max per label
        for y in range(NUM_LABELS):
            if probs[y] > max_probs[y]:
                max_probs[y] = probs[y]
                best_ev_idx_for_label[y] = idx

    # final label = argmax_y max_i P(y|c,e_i)
    y_star = int(torch.argmax(max_probs).item())
    final_label = ID2LABEL[y_star]

    best_idx = best_ev_idx_for_label[y_star]
    best_evidence = candidates[best_idx] if best_idx >= 0 else None

    if best_evidence is not None:
        for idx, p in all_evidence_scored:
            if idx == best_idx:
                best_evidence = {**best_evidence, "probs": p}
                break

    out = {
        "claim": claim,
        "report_s3_key": report_key,
        "final_label": final_label,
        "label_confidences": {
            ID2LABEL[i]: float(max_probs[i].item()) for i in range(NUM_LABELS)
        },
        "best_evidence": best_evidence,
    }

    #keep gold label if present
    if "label" in claim_item:
        out["gold_label"] = claim_item["label"]

    if "gold_evidence" in claim_item:
        out["gold_evidence"] = claim_item["gold_evidence"]

    return out


# ---------------------------
# Evaluation loop
# -----------------------------
def evaluate_split():
    tokenizer, model, model_dir = load_finetuned_model_and_tokenizer()

    split_key = f"{SPLIT_PREFIX}claims_{SPLIT_NAME}.json"
    logger.info(f"Loading split from s3://{S3_BUCKET}/{split_key}")
    data = s3_read_json(split_key)
    items = data.get("items", [])
    logger.info(f"Loaded {len(items)} claim items for split '{SPLIT_NAME}'.")

    outputs = {
        "split": SPLIT_NAME,
        "source_split_key": split_key,
        "cross_encoder": f"{model_dir} (MIL-finetuned)",
        "aggregation": "argmax_y max_i P(y|c,e_i)",
        "results": [],
        "metrics": {},
    }

    # Metrics- binary, treat support as positive
    total_with_gold = 0
    total_correct   = 0
    tp = fp = tn = fn = 0

    for claim_item in items:
        out = aggregate_over_candidates(claim_item, tokenizer, model)
        outputs["results"].append(out)

        gold_raw = out.get("gold_label")
        gold_norm = normalize_gold_label(gold_raw)

        if gold_norm is None:
            continue

        total_with_gold += 1
        pred = out["final_label"]

        if pred == gold_norm:
            total_correct += 1

        # binary confusion matrix
        # positive class = "support"
        if gold_norm == "support":
            if pred == "support":
                tp += 1
            else:
                fn += 1
        else:  # gold = not support
            if pred == "support":
                fp += 1
            else:
                tn += 1

    if total_with_gold > 0:
        accuracy = total_correct / total_with_gold

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        outputs["metrics"] = {
            "num_with_gold": total_with_gold,
            "num_correct": total_correct,
            "accuracy": accuracy,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision_support": precision,
            "recall_support": recall,
            "f1_support": f1,
        }

        logger.info(
            f"[{SPLIT_NAME}] "
            f"acc={accuracy:.4f}, P={precision:.4f}, R={recall:.4f}, F1={f1:.4f} "
            f"(N={total_with_gold})"
        )
    else:
        logger.warning("No usable gold labels found in split – metrics cannot be computed.")

    out_key = f"{VERI_PREFIX}deberta_mil_{SPLIT_NAME}.json"
    s3_write_json(out_key, outputs)


if __name__ == "__main__":
    evaluate_split()
