import os
import io
import re
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from random import sample

import boto3
import pdfplumber
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import sys
from openai import OpenAI
from ingestion.config import CFG

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llm_claim_gen")

# -------------------------
# Config from ENV + CFG
# -------------------------

# Hugging Face model name
HF_MODEL = os.environ.get("HF_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
AWS_REGION = CFG["aws"]["region"]
S3_BUCKET = CFG["aws"]["s3_bucket_raw"]
PREFIX_EDGAR = CFG["aws"]["prefix_edgar"].rstrip("/") + "/"
PREFIX_SITE = CFG["aws"]["prefix_site"].rstrip("/") + "/"
OUTPUT_PREFIX = CFG["aws"]["output_prefix"].rstrip("/") + "/"  # "results/"

# Safety: truncate very long docs before sending to LLM
MAX_TEXT_CHARS = int(os.environ.get("MAX_TEXT_CHARS", "8000"))

# claims per report (minimum target)
N_CLAIMS = 6
API_KEY = os.environ.get("API_KEY")

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
# LLM initialization with 4-bit quantization
# -------------------------

def load_model() -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model with 4-bit quantization and CUDA support."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, trust_remote_code=True)
    return model, tokenizer

# -------------------------
# LLM prompt + call (Local Model)
# -------------------------

def zero_shot(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, doc: str, claim: str, answer: bool) -> int:
    prompt = f"""this a ESG report, and a claim, verify the claim is support or not support.
Output your answer with only "support" or "not support" other output will be consider as wrong answer
claim: {claim}
doc: {doc}
answer: """
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.3,
        top_p=0.7,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the generated part (after the prompt)
    response = response[len(prompt):].lower().strip()
    
    # Cleanup memory
    torch.cuda.empty_cache()
    
    if "support" in response and "not support" not in response:
        return 1
    elif "not support" in response:
        return 0
    else:
        return -1
    
def few_shot(client: OpenAI, doc: str, claim: str, answer: bool, examples: List[Dict[str, str]], cot: bool) -> int:
    if len(examples) == 0:
        return -1
    
    prompt = """this a ESG report, and a claim, verify the claim is support or not support.
Output your answer with only "support" or "not support" other output will be consider as wrong answer
Here are some examples"""
    
    if cot:
        prompt = """this a ESG report, and a claim, verify the claim is support or not support.
Before output the answer, find the evidence, and then reasons through it.
Finally output your answer in the last line using format of "answer: " and your answer. Your answer should only be either "support" or "not support" other output will be consider as wrong answer
Here are some examples
"""
    
    for example in examples:
        prompt += f"\nclaim: {example['claim']}\nanswer: {example['label']}"
        if example.get("gold_evidence") is not None:
            prompt += f"\nevidence: {example['gold_evidence']}"
    
    prompt += f"\nNow answer if this claim support or not\nclaim: {claim}\ndoc: {doc}\nevidence & answer: "
    
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0.3,
        top_p=0.7,
    ).choices[0].message.content
    print("LLM response:", response)
    
    # Cleanup memory
    #torch.cuda.empty_cache()
    
    
        
    response = next((line.strip() for line in response.splitlines() if "answer" in line), None)
    if response is None:
        return -1
    if "not" in response:
        return 0
    elif "support" in response:
        return 1
    else:
        return -1
# -------------------------
# Main pipeline
# -------------------------

def main():
    logger.info("Using HF model: %s", HF_MODEL)
    logger.info("Loading model with 4-bit quantization...")
    client = OpenAI(api_key=API_KEY)
    logger.info("Model loaded successfully")

    NUM_EXAMPLE = 3
    COT = True

    # 1: read from train and test dataset
    with open("train_data.json", 'r') as file:
        train = json.load(file)
    with open("test_data.json", 'r') as file:
        test = json.load(file)

    print(f"Loaded {len(train)} train items and {len(test)} test items")

    results : List[dict[str, int]] = []
    # global label counters
    outputs = []
    idx = 1
    for data in test[:200]:
        logger.info("Processing file %d / %d", idx, len(test))
        idx += 1
        key = data.get("report_s3_key", None)
        claim = data.get("claim", None)
        label = data.get("label", None)
        evidence = data.get("gold_evidence", None)
        s3_url = f"s3://{S3_BUCKET}/{key}"
        logger.info("Processing %s", s3_url)
        if claim is None or label is None or key is None:
            print("No claim or label found, skipping", key)
            continue
        if COT and evidence is None:
            print("No evidence found for cot, skipping", key)
            continue
        
        doc = extract_text_for_key(key)
        if doc is None:
            print(f"Failed to extract text for {s3_url}, skipping")
            continue
        output = {"s3_key": key, "claim": claim, "label": label}
        if NUM_EXAMPLE > 0:
            examples = sample(train, min(NUM_EXAMPLE, len(train)))
            #print(examples)
            answer  = few_shot(client, doc, claim, label, examples, COT)
            results.append({
                "label": 1 if label == "support" else 0,
                "prediction": answer
            })
            if answer == -1:
                print(f"LLM returned invalid answer for {s3_url}")
                output["prediction"] = "invalid"
            else:
                output["prediction"] = "support" if answer == 1 else "not support"
        else:
            answer  = zero_shot(model, tokenizer, doc, claim, label)
            results.append({
                "label": 1 if label == "support" else 0,
                "prediction": answer
            })
            if answer == -1:
                print(f"LLM returned invalid answer for {s3_url}")
                output["prediction"] = "invalid"
            else:
                output["prediction"] = "support" if answer == 1 else "not support"
        outputs.append(output)

    # save outputs
    with open("baseline_outputs.json", "w") as f:
        json.dump(outputs, f, indent=2)
    with open("baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    # compute accuracy
    total = len(results)
    correct = sum(1 for r in results if r["label"] == r["prediction"])
    print(f"Accuracy: {correct} / {total} = {correct / total:.4f}")

if __name__ == "__main__":
    main()
