import os
import io
import json
import logging
import random
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter

import boto3
import pdfplumber
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch import amp


from transformers import (
    DebertaV2Tokenizer,
    DebertaV2ForSequenceClassification,
    get_linear_schedule_with_warmup,
)

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ingestion.config import CFG

# ----------------------------
# Logging & basic config
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stage2_train_mil")

random.seed(42)

AWS_REGION     = CFG["aws"]["region"]
S3_BUCKET      = CFG["aws"]["s3_bucket_raw"]
RESULTS_PREFIX = CFG["aws"]["output_prefix"].rstrip("/") + "/"
SPLIT_PREFIX   = RESULTS_PREFIX + "splits/"
VERI_PREFIX    = RESULTS_PREFIX + "verification/"

DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME     = os.getenv("CROSS_ENCODER_NAME", "microsoft/deberta-v3-base")

# where to save
SAVE_DIR  = "./deberta_finetuned_mil"

#upload model on S3 under results/
MODEL_S3_PREFIX = RESULTS_PREFIX + "models/deberta_finetuned_mil/"

# ----------
# Labels 
# ---------
ID2LABEL = {
    0: "support",
    1: "not support",
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
NUM_LABELS = len(ID2LABEL)

# Raw labels from splits to normalized to our 2-way space
RAW_TO_NORM_LABEL = {
    "support": "support",
    "notsupport": "not support",
    "not support": "not support",
    "not_support": "not support",
    "no support": "not support",
}

# -------------------------
# Chunking & training config (tuned for lower VRAM)
# ---------------------------
CHUNK_SIZE = 400     
CHUNK_OVERLAP = 100       

MAX_SEQ_LEN = 512     

BATCH_SIZE = 4        # bags per batch 
GRAD_ACCUM_STEPS = 8       # effective batch 

NUM_EPOCHS = 15
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01

TOP_K = 5        # retrieved candidates per claim 
NEG_PER_CLAIM = 2        # random negative chunks per claim 

TEXT_CACHE: Dict[str, List[str]] = {}
JSON_CACHE: Dict[str, Any] = {}
BYTES_CACHE: Dict[str, bytes] = {}
MAX_BYTES_CACHE = 256  # number of docs cached in memory

torch.backends.cudnn.benchmark = True  


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


def download_s3_bytes(key: str) -> bytes:
    if key in BYTES_CACHE:
        return BYTES_CACHE[key]
    body = s3().get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
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
        pass


def s3_write_json(key: str, data: Dict[str, Any]):
    s3_delete_if_exists(key)
    body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    s3().put_object(Bucket=S3_BUCKET, Key=key, Body=body, ContentType="application/json")
    logger.info(f"Saved s3://{S3_BUCKET}/{key}")


def upload_dir_to_s3(local_dir: str, bucket: str, s3_prefix: str):
    """Upload all files in a local directory to S3 under s3_prefix."""
    client = s3()
    for root, _, files in os.walk(local_dir):
        for fname in files:
            local_path = os.path.join(root, fname)
            rel_path = os.path.relpath(local_path, local_dir)  # e.g. config.json
            s3_key = os.path.join(s3_prefix, rel_path).replace("\\", "/")
            client.upload_file(local_path, bucket, s3_key)
            logger.info(f"Uploaded {local_path} -> s3://{bucket}/{s3_key}")


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
        elif report_key.lower().endswith((".html", ".htm")):
            text = simple_html_to_text(raw)
        else:
            # fallback: assume plain text / html
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


# -------------------------
# Dataset (bag-level / MIL)
# -------------------------
class ClaimBagDataset(Dataset):
    """
    Each item is:
      (claim_text, [evidence_text_1, ..., evidence_text_m], label_id)
    We'll pool over evidences inside the batch collate_fn.
    """
    def __init__(self,
                 claims: List[str],
                 bags: List[List[str]],
                 labels: List[int]):
        assert len(claims) == len(bags) == len(labels)
        self.claims = claims
        self.bags = bags
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.claims[idx], self.bags[idx], self.labels[idx]


def mil_collate_fn(batch, tokenizer: DebertaV2Tokenizer):
    """
    batch: list of (claim, [e1, e2, ...], label_id)

    Returns:
      {
        'input_ids': [N_pairs, L],
        'attention_mask': [N_pairs, L],
        'bag_ids': [N_pairs],          # index of bag per pair
        'bag_labels': [num_bags]
      }
    """
    all_claims: List[str] = []
    all_evidences: List[str] = []
    bag_ids: List[int] = []
    bag_labels: List[int] = []

    for bag_idx, (claim, evidences, label_id) in enumerate(batch):
        # ensure at least one evidence
        if len(evidences) == 0:
            continue
        for ev in evidences:
            all_claims.append(claim)
            all_evidences.append(ev)
            bag_ids.append(bag_idx)
        bag_labels.append(label_id)

    encodings = tokenizer(
        all_claims,
        all_evidences,
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LEN,
        return_tensors="pt",
    )

    bag_ids_tensor = torch.tensor(bag_ids, dtype=torch.long)
    bag_labels_tensor = torch.tensor(bag_labels, dtype=torch.long)

    return {
        **encodings,
        "bag_ids": bag_ids_tensor,
        "bag_labels": bag_labels_tensor,
    }


# -------------------------
# Build MIL bags from split
# -------------------------
def build_bags_for_split(split_name: str,
                         tokenizer: DebertaV2Tokenizer,
                         top_k: int = TOP_K,
                         neg_per_claim: int = NEG_PER_CLAIM
                         ) -> Tuple[ClaimBagDataset, Counter]:
    split_key = f"{SPLIT_PREFIX}claims_{split_name}.json"
    logger.info(f"Building MIL bags for split '{split_name}' from s3://{S3_BUCKET}/{split_key}")

    data = s3_read_json(split_key)
    items = data.get("items", [])

    claims_out: List[str] = []
    bags_out: List[List[str]] = []
    labels_out: List[int] = []

    missing_label = 0
    missing_evidence = 0
    label_counter = Counter()

    for it in items:
        claim      = it.get("claim")
        report_key = it.get("report_s3_key")
        raw_label  = it.get("label")

        if not claim or not report_key or raw_label is None:
            missing_label += 1
            continue

        norm_key = raw_label.lower()
        norm_label = RAW_TO_NORM_LABEL.get(norm_key)
        if norm_label is None or norm_label not in LABEL2ID:
            # skip don't know / unknown labels
            missing_label += 1
            continue

        label_id = LABEL2ID[norm_label]

        candidates = it.get("candidates", [])[:top_k]
        evidences: List[str] = []
        used_chunk_ids = set()

        #retrieved candidates
        for cand in candidates:
            ev_text: Optional[str] = None
            if cand["collection"] == "reports_kpi" and cand.get("sentence"):
                ev_text = cand["sentence"]
            else:
                cid = cand.get("chunk_id")
                if cid is None:
                    continue
                ev_text = get_text_chunk(report_key=cand["source"], chunk_id=cid)
                if ev_text:
                    used_chunk_ids.add(cid)

            if ev_text:
                evidences.append(ev_text)

        #random negative chunks from same report
        all_chunks = get_chunks_for_report(report_key)
        if all_chunks and neg_per_claim > 0:
            candidate_ids = [i for i in range(len(all_chunks)) if i not in used_chunk_ids]
            random.shuffle(candidate_ids)
            for cid in candidate_ids[:neg_per_claim]:
                evidences.append(all_chunks[cid])

        if not evidences:
            missing_evidence += 1
            continue

        claims_out.append(claim)
        bags_out.append(evidences)
        labels_out.append(label_id)
        label_counter[label_id] += 1

    logger.info(
        f"Split '{split_name}': built {len(labels_out)} MIL bags "
        f"(missing_label={missing_label}, missing_evidence={missing_evidence})"
    )
    logger.info(f"Label distribution (id:count): {label_counter}")

    dataset = ClaimBagDataset(claims_out, bags_out, labels_out)
    return dataset, label_counter


# -------------------------
# MIL evaluation (bag-level)
# -------------------------
def evaluate_mil(model: DebertaV2ForSequenceClassification,
                 dataloader: DataLoader) -> float:
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in dataloader:
            bag_ids = batch.pop("bag_ids").to(DEVICE)
            bag_labels = batch.pop("bag_labels").to(DEVICE)
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            outputs = model(**batch)
            logits = outputs.logits  # [N_pairs, 2]

            n_bags = bag_labels.size(0)
            bag_logits = torch.full(
                (n_bags, NUM_LABELS),
                float("-inf"),
                device=logits.device,
            )

            for b in range(n_bags):
                idxs = (bag_ids == b)
                if idxs.sum() == 0:
                    continue
                #soft pooling: logsumexp over candidates in bag
                bag_logits[b] = torch.logsumexp(logits[idxs], dim=0)

            preds = torch.argmax(bag_logits, dim=-1)
            total  += n_bags
            correct += (preds == bag_labels).sum().item()

    return correct / total if total > 0 else 0.0


# -------------------------
# Main training
# -------------------------
def train():
    logger.info(f"Loading base model: {MODEL_NAME}")
    tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME)
    model = DebertaV2ForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(DEVICE)
    
    #Freeze most of the encoder,fine-tune top N layers and classifier
    N_UNFROZEN_LAYERS = 4  # tune

    # Freeze everything by default
    for param in model.parameters():
        param.requires_grad = False

    #Unfreeze classifier head
    for param in model.classifier.parameters():
        param.requires_grad = True

    #Unfreeze top N transformer layers
    encoder_layers = model.deberta.encoder.layer
    for layer in encoder_layers[-N_UNFROZEN_LAYERS:]:
        for param in layer.parameters():
            param.requires_grad = True


    # build MIL datasets
    train_dataset, train_label_dist = build_bags_for_split("train", tokenizer)
    dev_dataset,   dev_label_dist   = build_bags_for_split("dev",   tokenizer)

    logger.info(f"Train label dist (id:count): {train_label_dist}")
    logger.info(f"Dev label dist (id:count):   {dev_label_dist}")

    def collate(batch):
        return mil_collate_fn(batch, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    # optimizer and scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
    {
        "params": [
            p for n, p in model.named_parameters()
            if p.requires_grad and not any(nd in n for nd in no_decay)
        ],
        "weight_decay": WEIGHT_DECAY,
    },
    {
        "params": [
            p for n, p in model.named_parameters()
            if p.requires_grad and any(nd in n for nd in no_decay)
        ],
        "weight_decay": 0.0,
    },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)

    num_training_steps = NUM_EPOCHS * max(1, len(train_loader)) // max(1, GRAD_ACCUM_STEPS)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(0, int(0.1 * num_training_steps)),
        num_training_steps=num_training_steps,
    )

    logger.info(
        f"Starting MIL training for {NUM_EPOCHS} epochs "
        f"on {len(train_dataset)} bags "
        f"(batch_size={BATCH_SIZE}, grad_accum={GRAD_ACCUM_STEPS}, max_seq_len={MAX_SEQ_LEN})"
    )
    
    scaler = amp.GradScaler(device="cuda")
    optimizer.zero_grad(set_to_none=True)

    
    # best_dev_acc = -1.0


    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader):
            bag_ids = batch.pop("bag_ids").to(DEVICE)
            bag_labels = batch.pop("bag_labels").to(DEVICE)
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            with amp.autocast(device_type="cuda"):
                outputs = model(**batch)
                logits = outputs.logits  # [N_pairs, 2]

                n_bags = bag_labels.size(0)
                bag_logits = torch.full(
                    (n_bags, NUM_LABELS),
                    float("-inf"),
                    device=logits.device,
                )

                # soft pooling: logsumexp over candidates per bag
                for b in range(n_bags):
                    idxs = (bag_ids == b)
                    if idxs.sum() == 0:
                        continue
                    bag_logits[b] = torch.logsumexp(logits[idxs], dim=0)

                loss = torch.nn.functional.cross_entropy(bag_logits, bag_labels)
                loss = loss / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()
            total_loss += loss.item()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()


            if (step + 1) % 50 == 0:
                avg_loss = total_loss / (step + 1)
                logger.info(
                    f"Epoch {epoch+1} Step {step+1}/{len(train_loader)} "
                    f"- loss={avg_loss:.4f}"
                )
        
        if (step + 1) % GRAD_ACCUM_STEPS != 0:
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        avg_train_loss = total_loss / max(1, len(train_loader))
        dev_acc = evaluate_mil(model, dev_loader) if len(dev_dataset) > 0 else float("nan")
        logger.info(
            f"Epoch {epoch+1} finished. "
            f"train_loss={avg_train_loss:.4f}, dev_bag_accuracy={dev_acc:.4f}"
        )

        # if not torch.isnan(torch.tensor(dev_acc)) and dev_acc > best_dev_acc:
        #     best_dev_acc = dev_acc
        #     os.makedirs(SAVE_DIR, exist_ok=True)
        #     model.save_pretrained(SAVE_DIR)
        #     tokenizer.save_pretrained(SAVE_DIR)
        #     logger.info(f"New best dev acc={dev_acc:.4f}, saved model to {SAVE_DIR}")


    #Save fine-tuned model locally
    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    logger.info(f"Saved fine-tuned MIL model locally to {SAVE_DIR}")

    #Upload model to S3 under results/models/...
    upload_dir_to_s3(SAVE_DIR, S3_BUCKET, MODEL_S3_PREFIX)
    logger.info(f"Uploaded fine-tuned MIL model to s3://{S3_BUCKET}/{MODEL_S3_PREFIX}")


if __name__ == "__main__":
    train()
