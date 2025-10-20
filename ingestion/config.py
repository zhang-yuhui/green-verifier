import os
from dotenv import load_dotenv

load_dotenv()

CFG = {
    "pg": {
        "host": os.getenv("PG_HOST"),
        "db": os.getenv("PG_DB"),
        "user": os.getenv("PG_USER"),
        "password": os.getenv("PG_PASSWORD"),
        "port": int(os.getenv("PG_PORT", "5432")),
    },
    "aws": {
        "region": os.getenv("AWS_REGION", "eu-central-1"),
        "s3_bucket_raw": os.getenv("S3_BUCKET_RAW"),
    },
    "sec": {
        "user_agent": os.getenv("SEC_USER_AGENT"),
    },
}
