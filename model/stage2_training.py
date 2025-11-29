
import os
import io
import json
import logging
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter


import boto3
import pdfplumber
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import (
    DebertaV2Tokenizer,
    DebertaV2ForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from torch.optim import AdamW


import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import CFG

# -------------------------
# Logging & basic config
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stage2_train")

AWS_REGION     = CFG["aws"]["region"]
S3_BUCKET      = CFG["aws"]["s3_bucket_raw"]
RESULTS_PREFIX = CFG["aws"]["output_prefix"].rstrip("/") + "/"
SPLIT_PREFIX   = RESULTS_PREFIX + "splits/"
VERI_PREFIX    = RESULTS_PREFIX + "verification/"

DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME     = os.getenv("CROSS_ENCODER_NAME", "microsoft/deberta-v3-base")

# -------------------------
# Labels (model space)
# -------------------------
ID2LABEL = {
    0: "support",
    1: "not support",
    2: "don't know",
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
NUM_LABELS = len(ID2LABEL)

# Raw labels as they appear in claims_*.json  -> normalized model labels
RAW_TO_NORM_LABEL = {
    "support": "support",
    "notsupport": "not support",
    "not_support": "not support",
    "dontknow": "don't know",
    "dont_know": "don't know",
    "don't know": "don't know",
}

# -------------------------
# Chunking config
# -------------------------
CHUNK_SIZE    = 400
CHUNK_OVERLAP = 100

BATCH_SIZE    = 8
NUM_EPOCHS    = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY  = 0.01
SAVE_DIR      = "./deberta_finetuned"
TOP_K         = 5          # number of candidate evidences per claim to train on

TEXT_CACHE: Dict[str, List[str]] = {}

# -------------------------
# S3 helpers
# -------------------------

JSON_CACHE: Dict[str, Any] = {}
BYTES_CACHE: Dict[str, bytes] = {}
MAX_BYTES_CACHE = 512  # avoid storing too many very large reports in memory

def s3():
    return boto3.client("s3", region_name=AWS_REGION)


def s3_read_json(key: str) -> Dict[str, Any]:
    if key in JSON_CACHE:
        return JSON_CACHE[key]
    obj = s3().get_object(Bucket=S3_BUCKET, Key=key)
    data = json.loads(obj["Body"].read())
    JSON_CACHE[key] = data
    return data

def download_s3_bytes(key: str) -> bytes:
    if key in BYTES_CACHE:
        return BYTES_CACHE[key]
    body = s3().get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
    # simple bounded cache: drop oldest when too big
    if len(BYTES_CACHE) >= MAX_BYTES_CACHE:
        BYTES_CACHE.pop(next(iter(BYTES_CACHE)))
    BYTES_CACHE[key] = body
    return body

def s3_delete_if_exists(key: str):
    client = s3()
    try:
        client.head_object(Bucket=S3_BUCKET, Key=key)
        client.delete_object(Bucket=S3_BUCKET, Key=key)
        logger.info(f"Deleted existing file: s3://{S3_BUCKET}/{key}")
    except client.exceptions.ClientError:
        pass  # not present

def s3_write_json(key: str, data: Dict[str, Any]):
    s3_delete_if_exists(key)
    body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    s3().put_object(Bucket=S3_BUCKET, Key=key, Body=body,
                    ContentType="application/json")
    logger.info(f"Saved s3://{S3_BUCKET}/{key}")

# -------------------------
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
        else:
            # treat everything else as HTML / text (Stage 0 logic)
            text = simple_html_to_text(raw)
        toks = tokenize_ws(text)
        chunks = chunk_tokens(toks)
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
# Dataset
# -------------------------
class ClaimEvidenceDataset(Dataset):
    """
    Dataset holding *pre-tokenized* encodings and labels.
    No tokenization happens inside __getitem__.
    """
    def __init__(self, encodings: Dict[str, torch.Tensor], labels: torch.Tensor):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, idx: int):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


# -------------------------
# Build examples directly from split files (for training)
# -------------------------
def build_examples_for_split(split_name: str,
                             tokenizer: DebertaV2Tokenizer,
                             top_k: int = TOP_K):

    split_key = f"{SPLIT_PREFIX}claims_{split_name}.json"
    logger.info(f"Building examples for split '{split_name}' from s3://{S3_BUCKET}/{split_key}")

    data = s3_read_json(split_key)
    items = data.get("items", [])

    claims_list = []
    evid_list   = []
    label_ids   = []

    missing_label    = 0
    missing_evidence = 0

    for it in items:
        claim      = it.get("claim")
        report_key = it.get("report_s3_key")
        raw_label  = it.get("label")

        if not claim or not report_key or raw_label is None:
            missing_label += 1
            continue

        # normalize label
        norm_key = raw_label.lower()
        norm_label = RAW_TO_NORM_LABEL.get(norm_key)
        if norm_label is None or norm_label not in LABEL2ID:
            missing_label += 1
            continue

        label_id = LABEL2ID[norm_label]

        # take top-k candidates
        candidates = it.get("candidates", [])[:top_k]
        if not candidates:
            missing_evidence += 1
            continue

        for cand in candidates:
            # KPI short sentence
            if cand["collection"] == "reports_kpi" and cand.get("sentence"):
                evidence_text = cand["sentence"]
            else:
                cid = cand.get("chunk_id")
                if cid is None:
                    missing_evidence += 1
                    continue

                evidence_text = get_text_chunk(report_key=cand["source"], chunk_id=cid)
                if not evidence_text:
                    missing_evidence += 1
                    continue

            claims_list.append(claim)
            evid_list.append(evidence_text)
            label_ids.append(label_id)

    logger.info(
        f"Split '{split_name}': built {len(label_ids)} examples "
        f"(missing_label={missing_label}, missing_evidence={missing_evidence})"
    )

    # ---- Tokenize ONCE (big performance win)
    encodings = tokenizer(
        claims_list,
        evid_list,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )

    labels_tensor = torch.tensor(label_ids, dtype=torch.long)

    return encodings, labels_tensor


# -------------------------
# Evaluation on pair level (for dev during training)
# -------------------------
def evaluate_pairwise(model: DebertaV2ForSequenceClassification,
                      dataloader: DataLoader) -> float:
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            labels = batch.pop("labels")
            outputs = model(**batch)
            logits  = outputs.logits
            preds   = torch.argmax(logits, dim=-1)
            total  += labels.size(0)
            correct += (preds == labels).sum().item()
    return correct / total if total > 0 else 0.0

# -------------------------
# Inference helpers (aggregation over candidates)
# -------------------------
def normalize_gold_label(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    key = raw.lower()
    return RAW_TO_NORM_LABEL.get(key)

def predict_pair_probs(tokenizer: DebertaV2Tokenizer,
                       model: DebertaV2ForSequenceClassification,
                       claim: str,
                       evidence: str,
                       max_length: int = 512) -> torch.Tensor:
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
        probs  = torch.softmax(logits, dim=-1)  # [1, 3]
    return probs.squeeze(0).cpu()  # [3]

def aggregate_over_candidates(claim_item: Dict[str, Any],
                              tokenizer: DebertaV2Tokenizer,
                              model: DebertaV2ForSequenceClassification) -> Dict[str, Any]:
    """
    claim_item:
      {
        "claim": str,
        "report_s3_key": str,
        "candidates": [{id, collection, score, source, chunk_id, sentence}, ...],
        "label": raw_label   # optional gold label
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

    # keep gold label (normalized + raw) if present
    if "label" in claim_item:
        raw = claim_item["label"]
        norm = normalize_gold_label(raw)
        out["gold_label_raw"]  = raw
        if norm is not None:
            out["gold_label"] = norm

    return out

def run_aggregated_inference_on_split(model: DebertaV2ForSequenceClassification,
                                      tokenizer: DebertaV2Tokenizer,
                                      split_name: str = "test"):
    """
    Run Stage 2 aggregation inference on claims_{split}.json
    using the (fine-tuned) model and write verification JSON to S3.
    """
    split_key = f"{SPLIT_PREFIX}claims_{split_name}.json"
    logger.info(f"[Inference] Loading split from s3://{S3_BUCKET}/{split_key}")
    data = s3_read_json(split_key)
    items = data.get("items", [])
    logger.info(f"[Inference] Loaded {len(items)} items for split '{split_name}'.")

    outputs = {
        "split": split_name,
        "source_split_key": split_key,
        "cross_encoder": MODEL_NAME + " (fine-tuned)",
        "aggregation": "argmax_y max_i P(y|c,e_i)",
        "results": [],
        "metrics": {},
    }

    total_with_gold = 0
    total_correct   = 0

    for claim_item in items:
        out = aggregate_over_candidates(claim_item, tokenizer, model)
        outputs["results"].append(out)

        gold = out.get("gold_label")  # already normalized string
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
            f"[{split_name}] Aggregated accuracy on {total_with_gold} labeled items: "
            f"{total_correct}/{total_with_gold} = {accuracy:.4f}"
        )
    else:
        logger.warning("[Inference] No gold labels found in split – accuracy cannot be computed.")

    out_key = f"{VERI_PREFIX}deberta_ft_{split_name}.json"
    s3_write_json(out_key, outputs)

# -------------------------
# Main training + inference
# -------------------------
def main():
    logger.info(f"Loading base model: {MODEL_NAME}")
    tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME)
    model = DebertaV2ForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(DEVICE)

    # ---- Build train & dev examples directly from split files
    train_enc, train_labels = build_examples_for_split("train", tokenizer, top_k=TOP_K)
    dev_enc,   dev_labels   = build_examples_for_split("dev", tokenizer, top_k=TOP_K)
    
    print("Train label dist:", Counter(train_labels.tolist()))
    print("Dev label dist:", Counter(dev_labels.tolist()))  

    train_dataset = ClaimEvidenceDataset(train_enc, train_labels)
    dev_dataset   = ClaimEvidenceDataset(dev_enc, dev_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader   = DataLoader(dev_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    # ---- Optimizer & scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)

    num_training_steps = NUM_EPOCHS * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(0, int(0.1 * num_training_steps)),
        num_training_steps=num_training_steps,
    )

    # ---- Training loop
    logger.info(f"Starting training for {NUM_EPOCHS} epochs on {len(train_dataset)} examples")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader):
            
            if step % 10 == 0:
                logger.info(f"Epoch {epoch+1}, step {step}/{len(train_loader)}")
                
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            labels = batch.pop("labels")

            outputs = model(**batch, labels=labels)
            loss = outputs.loss

            loss.backward()
            total_loss += loss.item()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if (step + 1) % 50 == 0:
                avg_loss = total_loss / (step + 1)
                logger.info(f"Epoch {epoch+1} Step {step+1}/{len(train_loader)} - loss={avg_loss:.4f}")

        avg_train_loss = total_loss / max(1, len(train_loader))
        dev_acc = evaluate_pairwise(model, dev_loader) if len(dev_dataset) > 0 else float("nan")
        logger.info(
            f"Epoch {epoch+1} finished. "
            f"train_loss={avg_train_loss:.4f}, dev_pairwise_accuracy={dev_acc:.4f}"
        )

    # ---- Save fine-tuned model
    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    logger.info(f"Saved fine-tuned model to {SAVE_DIR}")
    logger.info("To reuse later, set CROSS_ENCODER_NAME to this folder.")

    # ---- Run Stage 2 aggregated inference on test split with the fine-tuned model
    run_aggregated_inference_on_split(model, tokenizer, split_name="test")

if __name__ == "__main__":
    main()
