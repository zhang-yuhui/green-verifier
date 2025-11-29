# stage2_infer.py  (pre–fine-tuned DeBERTa baseline)

import os
import io
import json
import boto3
import logging
from typing import List, Dict, Any, Optional

import pdfplumber
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
import torch
import torch.nn.functional as F
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import CFG

# -------------------------
# Logging & Config
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stage2")

AWS_REGION     = CFG["aws"]["region"]
S3_BUCKET      = CFG["aws"]["s3_bucket_raw"]
RESULTS_PREFIX = CFG["aws"]["output_prefix"].rstrip("/") + "/"
SPLIT_PREFIX   = RESULTS_PREFIX + "splits/"
VERI_PREFIX    = RESULTS_PREFIX + "verification/"

# which split to run inference on: "train", "dev", or "test"
# for the baseline accuracy you typically set SPLIT_NAME=test
SPLIT_NAME = os.getenv("SPLIT_NAME", "test")

# Same chunking as Stage 0
CHUNK_SIZE    = 400
CHUNK_OVERLAP = 100
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# Cross-encoder (3-way classification)
MODEL_NAME = os.getenv("CROSS_ENCODER_NAME", "microsoft/deberta-v3-base")
NUM_LABELS = 3
ID2LABEL = {
    0: "support",
    1: "not support",
    2: "don't know",
}
LABEL2ID   = {v: k for k, v in ID2LABEL.items()}

TEXT_CACHE: Dict[str, List[str]] = {}



# -------------------------
# S3 Helpers
# -------------------------
def s3():
    return boto3.client("s3", region_name=AWS_REGION)


def s3_read_json(key: str) -> Dict[str, Any]:
    obj = s3().get_object(Bucket=S3_BUCKET, Key=key)
    return json.loads(obj["Body"].read())


def s3_delete_if_exists(key: str):
    """Delete an S3 object if it already exists (for clean overwrites)."""
    client = s3()
    try:
        client.head_object(Bucket=S3_BUCKET, Key=key)
        client.delete_object(Bucket=S3_BUCKET, Key=key)
        logger.info(f"Deleted existing file: s3://{S3_BUCKET}/{key}")
    except client.exceptions.ClientError:
        # Object does not exist -> nothing to delete
        pass


def s3_write_json(key: str, data: Dict[str, Any]):
    # ensure we overwrite cleanly
    s3_delete_if_exists(key)
    body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    s3().put_object(Bucket=S3_BUCKET, Key=key, Body=body, ContentType="application/json")
    logger.info(f"Saved verification to s3://{S3_BUCKET}/{key}")


def download_s3_bytes(key: str) -> bytes:
    obj = s3().get_object(Bucket=S3_BUCKET, Key=key)
    return obj["Body"].read()


# -------------------------
# Stage 0-compatible text utilities
# -------------------------
def simple_html_to_text(html_bytes: bytes) -> str:
    html = html_bytes.decode("utf-8", errors="ignore")
    import re
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def pdf_to_text(pdf_bytes: bytes) -> str:
    parts = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    return "\n".join(parts)


def tokenize_ws(text: str) -> List[str]:
    return text.split()


def chunk_tokens(tokens: List[str], chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
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
            logger.warning(f"Unsupported format for {report_key}")
            return None
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


# -------------------------
# Cross-encoder Inference
# -------------------------
def load_model():
    tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME)
    model = DebertaV2ForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(DEVICE)
    model.eval()
    return tokenizer, model


def predict_pair_probs(tokenizer, model, claim: str, evidence: str, max_length: int = 512) -> torch.Tensor:
    enc = tokenizer(
        claim,
        evidence,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    ).to(DEVICE)
    with torch.no_grad():
        logits = model(**enc).logits  # [1, 3]
        probs = F.softmax(logits, dim=-1)  # [1, 3]
    return probs.squeeze(0).cpu()  # [3]


def normalize_gold_label(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    t = text.strip().lower().replace("_", " ")

    if t in {"support"}:
        return "support"

    if t in {"not support", "notsupport", "no support"}:
        return "not support"

    if t in {"don't know", "dont know", "unknown"}:
        return "don't know"

    return text  # fallback



# -------------------------
# Aggregation per the paper
# -------------------------
def aggregate_over_candidates(claim_item: Dict[str, Any], tokenizer, model) -> Dict[str, Any]:
    """
    claim_item:
      {
        "claim": str,
        "report_s3_key": str,
        "candidates": [{id, collection, score, source, chunk_id, sentence}, ...],
        "label": "support|contradict|notsupport"   # optional ground-truth label
      }
    """
    claim = claim_item["claim"]
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

        probs = predict_pair_probs(tokenizer, model, claim, evidence_text)  # [3]
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

    # keep ground-truth label if present (for later metric computation)
    if "label" in claim_item:
        out["gold_label"] = claim_item["label"]

    return out


# -------------------------
# Main
# -------------------------
def run_stage2():
    tokenizer, model = load_model()

    split_key = f"{SPLIT_PREFIX}claims_{SPLIT_NAME}.json"
    logger.info(f"Loading split from s3://{S3_BUCKET}/{split_key}")
    data = s3_read_json(split_key)

    items = data.get("items", [])
    logger.info(f"Loaded {len(items)} items for split '{SPLIT_NAME}'.")

    outputs = {
        "split": SPLIT_NAME,
        "source_split_key": split_key,
        "cross_encoder": MODEL_NAME,
        "aggregation": "argmax_y max_i P(y|c,e_i)",
        "results": [],
        "metrics": {},   # will be filled if gold labels exist
    }

    total_with_gold = 0
    total_correct   = 0

    for claim_item in items:
        out = aggregate_over_candidates(claim_item, tokenizer, model)
        outputs["results"].append(out)

        # compute accuracy if gold label is present
        gold = normalize_gold_label(out.get("gold_label"))

        if gold is not None:
            total_with_gold += 1
            if out["final_label"] == gold:
                total_correct += 1


    if total_with_gold > 0:
        accuracy = total_correct / total_with_gold
        outputs["metrics"] = {
            "num_with_gold": total_with_gold,
            "num_correct": total_correct,
            "accuracy": accuracy,
        }
        logger.info(
            f"[{SPLIT_NAME}] Accuracy on {total_with_gold} labeled items: "
            f"{total_correct}/{total_with_gold} = {accuracy:.4f}"
        )
    else:
        logger.warning("No gold labels found in split – accuracy cannot be computed.")

    out_key = f"{VERI_PREFIX}deberta_base_{SPLIT_NAME}.json"
    s3_write_json(out_key, outputs)


if __name__ == "__main__":
    run_stage2()
