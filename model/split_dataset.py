import json
import boto3
import random
import logging
from collections import defaultdict
from typing import Dict, Any, List
import sys
import os
import re

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import CFG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("split")

AWS_REGION = CFG["aws"]["region"]
S3_BUCKET = CFG["aws"]["s3_bucket_raw"]
RESULTS_PREFIX = CFG["aws"]["output_prefix"].rstrip("/") + "/"
RETR_PREFIX = RESULTS_PREFIX + "retrieval/"
SPLIT_PREFIX = RESULTS_PREFIX + "splits/"

random.seed(42)  # reproducible splits


# ---------------------------------------------------
# Explicit mapping: SITE folder name  ->  EDGAR ticker
# (keys are cleaned: only letters, lowercase)
# ---------------------------------------------------
SITE_NAME_TO_TICKER = {
    # direct overlaps EDGAR <-> SITE
    "appleinc": "AAPL",
    "amazoncominc": "AMZN",
    "alphabetinc": "GOOGL",
    "citigroupinc": "C",
    "generalelectriccompany": "GE",
    "ibmcorporation": "IBM",
    "johnsonjohnson": "JNJ",
    "jpmorganchaseco": "JPM",
    "microsoftcorporation": "MSFT",
    "pepsicoinc": "PEP",
    "pfizerinc": "PFE",
    "proctergamblecompany": "PG",
    "thecocacolacompany": "KO",
    # SITE-only companies remain as separate "companies"
    # (AstraZeneca, Danone, H&M, LVMH, Roche, Siemens, TotalEnergies, Toyota ...)
    # will fall back to their cleaned name as company id.
}


# --------- S3 helpers ---------
def s3():
    return boto3.client("s3", region_name=AWS_REGION)


def list_retrieval_files() -> List[str]:
    """List all retrieval JSONs (currently you only have one batch file)."""
    keys: List[str] = []
    paginator = s3().get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=RETR_PREFIX):
        for it in page.get("Contents", []):
            if it["Key"].endswith(".json"):
                keys.append(it["Key"])
    return keys


def s3_read_json(key: str) -> Dict[str, Any]:
    obj = s3().get_object(Bucket=S3_BUCKET, Key=key)
    return json.loads(obj["Body"].read())


def s3_write_json(key: str, data: Dict[str, Any]):
    body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    s3().put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=body,
        ContentType="application/json",
    )
    logger.info(f"Saved s3://{S3_BUCKET}/{key}")


# --------- Company normalization ---------
def _clean_site_name(raw_company: str) -> str:
    """Strip non-letters and lowercase, e.g. 'Apple_Inc.' -> 'appleinc'."""
    return re.sub(r"[^A-Za-z]", "", raw_company).lower()


def extract_company_from_s3_key(report_s3_key: str, known_tickers: List[str]) -> str:
    """
    Normalize company identifiers so EDGAR and SITE reports map to a common key.
        
    Examples:
      'edgar/AAPL/2023-11-03_x.html'        -> 'AAPL'
      'site/Apple_Inc./2012_853c8cdb12.pdf' -> 'AAPL'  (via explicit mapping)
      'site/Roche_Holding_AG/...'          -> 'ROCHEHOLDINGAG' (site-only)
    """
    path = report_s3_key.replace("s3://", "")
    parts = path.split("/")

    # If path starts with bucket name, drop it
    if parts and parts[0] == S3_BUCKET:
        parts = parts[1:]

    if len(parts) < 2:
        return "UNKNOWN"

    source = parts[0].lower()   # 'edgar' or 'site'
    raw_company = parts[1]      # e.g. 'AAPL' or 'Apple_Inc.'

    # EDGAR: company dir is already the ticker
    if source == "edgar":
        return raw_company.upper()

    # SITE: normalize folder name, then map to ticker if we know it
    if source == "site":
        cleaned = _clean_site_name(raw_company)  # 'Apple_Inc.' -> 'appleinc'

        # 1) explicit mapping first (guarantees AAPL <-> Apple_Inc., etc.)
        if cleaned in SITE_NAME_TO_TICKER:
            return SITE_NAME_TO_TICKER[cleaned]

        # 2) fallback heuristic: if a known ticker appears verbatim in folder name
        for t in known_tickers:
            if t.lower() in cleaned:
                return t.upper()

        # 3) final fallback: treat the site company as its own identifier
        return cleaned.upper()

    return "UNKNOWN"


# --------- Main split logic ---------
def run_split():
    retr_keys = list_retrieval_files()
    logger.info(f"Found {len(retr_keys)} retrieval files to split: {retr_keys}")

    all_items: List[Dict[str, Any]] = []

    for rk in retr_keys:
        data = s3_read_json(rk)
        items = data.get("items", [])
        logger.info(f"{rk} contains {len(items)} items.")
        all_items.extend(items)

    if not all_items:
        logger.warning("No items found in retrieval files. Aborting.")
        return

    # 1) Collect known EDGAR tickers from report_s3_key (for heuristic fallback)
    known_tickers = set()
    for item in all_items:
        rk = item.get("report_s3_key", "")
        path = rk.replace("s3://", "")
        parts = path.split("/")
        if parts and parts[0] == S3_BUCKET:
            parts = parts[1:]
        if len(parts) >= 2 and parts[0].lower() == "edgar":
            known_tickers.add(parts[1].upper())

    logger.info(f"Known EDGAR tickers: {sorted(known_tickers)}")

    # 2) Group claim-items by normalized company id
    company_to_items = defaultdict(list)
    for item in all_items:
        report_key = item.get("report_s3_key")
        if not report_key:
            logger.warning(f"Item missing 'report_s3_key': {item.keys()}")
            continue
        company = extract_company_from_s3_key(report_key, list(known_tickers))
        company_to_items[company].append(item)

    # Optional sanity log: see how many items per company
    for c, items in company_to_items.items():
        logger.info(f"Company {c}: {len(items)} items")

    companies = [c for c in company_to_items.keys() if c != "UNKNOWN"]
    random.shuffle(companies)

    n = len(companies)
    n_train = int(0.70 * n)
    n_dev = int(0.15 * n)

    train_companies = companies[:n_train]
    dev_companies = companies[n_train : n_train + n_dev]
    test_companies = companies[n_train + n_dev :]

    logger.info(f"Total companies: {n}")
    logger.info(f"Train companies: {len(train_companies)}")
    logger.info(f"Dev companies:   {len(dev_companies)}")
    logger.info(f"Test companies:  {len(test_companies)}")

    # 3) Build split-wise item lists
    train_items: List[Dict[str, Any]] = []
    dev_items: List[Dict[str, Any]] = []
    test_items: List[Dict[str, Any]] = []

    for company, items in company_to_items.items():
        if company in train_companies:
            train_items.extend(items)
        elif company in dev_companies:
            dev_items.extend(items)
        elif company in test_companies:
            test_items.extend(items)
        else:
            # this should only catch UNKNOWN/edge cases
            logger.debug(f"Company {company} not in any split; defaulting to test.")
            test_items.extend(items)

    logger.info(f"Train items: {len(train_items)}")
    logger.info(f"Dev items:   {len(dev_items)}")
    logger.info(f"Test items:  {len(test_items)}")

    # 4) Ensure splits/ "folder" exists (optional)
    s3().put_object(Bucket=S3_BUCKET, Key=SPLIT_PREFIX, Body=b"")

    # 5) Write split JSONs
    s3_write_json(f"{SPLIT_PREFIX}claims_train.json", {
        "split": "train",
        "items": train_items,
    })
    s3_write_json(f"{SPLIT_PREFIX}claims_dev.json", {
        "split": "dev",
        "items": dev_items,
    })
    s3_write_json(f"{SPLIT_PREFIX}claims_test.json", {
        "split": "test",
        "items": test_items,
    })


if __name__ == "__main__":
    run_split()