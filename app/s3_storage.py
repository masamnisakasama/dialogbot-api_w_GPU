# app/s3_storage.py
from __future__ import annotations
import os, json, boto3, typing as T
from datetime import datetime
from botocore.config import Config

AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-1")
S3_BUCKET  = os.getenv("S3_BUCKET", "")
S3_PREFIX  = os.getenv("S3_PREFIX", "app/")   # 例: app/
KMS_KEY_ID = os.getenv("KMS_KEY_ID", "")      # 例: arn:aws:kms:ap-northeast-1:xxxx:key/....

_s3 = boto3.client("s3", region_name=AWS_REGION, config=Config(s3={"addressing_style": "virtual"}))

def _path(user_id: str, rel: str) -> str:
    today = datetime.utcnow().strftime("%Y/%m/%d")
    return f"{S3_PREFIX}{user_id}/{today}/{rel}".replace("//", "/")

def put_bytes_user(user_id: str, data: bytes, relkey: str, content_type: str):
    key = _path(user_id, relkey)
    extra = {
        "Bucket": S3_BUCKET, "Key": key, "Body": data, "ContentType": content_type,
        "ServerSideEncryption": "aws:kms" if KMS_KEY_ID else "AES256",
    }
    if KMS_KEY_ID:
        extra["SSEKMSKeyId"] = KMS_KEY_ID
    _s3.put_object(**extra)
    return key

def put_text_user(user_id: str, text: str, relkey: str, content_type="text/plain; charset=utf-8"):
    return put_bytes_user(user_id, text.encode("utf-8"), relkey, content_type)

def put_json_user(user_id: str, obj: T.Dict, relkey: str):
    data = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return put_bytes_user(user_id, data, relkey, "application/json")
