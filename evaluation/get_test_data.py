import json
import boto3
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # or adjust path
from ingestion.config import CFG

AWS_REGION = CFG["aws"]["region"]          # e.g. "us-east-1"
BUCKET     = CFG["aws"]["s3_bucket_raw"]   # "greenverifier-raw"
SPLIT_PREFIX = CFG["aws"]["output_prefix"].rstrip("/") + "/splits/"  # "results/splits/"

s3 = boto3.client("s3", region_name=AWS_REGION)

def load_split(split_name: str):
    """
    split_name: "train", "dev", or "test"
    returns the parsed JSON dict for that split
    """
    key = f"{SPLIT_PREFIX}claims_{split_name}.json"
    resp = s3.get_object(Bucket=BUCKET, Key=key)
    data = json.loads(resp["Body"].read())
    return data

# Example: load train + test
train_data = load_split("train").get("items", [])
test_data  = load_split("test").get("items", [])

# Save data to files
with open("train_data.json", "w") as f:
    json.dump(train_data, f, indent=2)

with open("test_data.json", "w") as f:
    json.dump(test_data, f, indent=2)


print(f"Saved {len(train_data)} train items to train_data.json")
print(f"Saved {len(test_data)} test items to test_data.json")