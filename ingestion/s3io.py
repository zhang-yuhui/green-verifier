import hashlib, io, time
import boto3
from botocore.config import Config
from ingestion.config import CFG

_s3 = boto3.client("s3", region_name=CFG["aws"]["region"],
                   config=Config(retries={"max_attempts": 5}))

def hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def put_bytes(bucket: str, key: str, body: bytes, content_type: str):
    _s3.put_object(Bucket=bucket, Key=key, Body=io.BytesIO(body), ContentType=content_type)
    return f"s3://{bucket}/{key}"

def polite_sleep(sec=0.4):
    time.sleep(sec)
