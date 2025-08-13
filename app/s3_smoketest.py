# app/s3_smoketest.py
from __future__ import annotations
import os, sys, json
from uuid import uuid4
from pathlib import Path
from datetime import datetime
from typing import Optional

def main(user_id: Optional[str] = "smoketest"):
    # 1) .env を app/.env から読み込む（ここがミソ）
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("python-dotenv が見つかりません。先に `pip install python-dotenv` を実行してください。")
        sys.exit(1)

    env_path = Path(__file__).resolve().parent / ".env"
    ok = load_dotenv(dotenv_path=env_path)
    print(f"[env] loaded={ok} path={env_path}")

    # 2) 必須環境変数を取得
    AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
    S3_BUCKET  = os.getenv("S3_BUCKET")
    S3_PREFIX  = os.getenv("S3_PREFIX", "app/")
    KMS_KEY_ID = os.getenv("KMS_KEY_ID")  # 任意

    missing = [k for k in ["AWS_ACCESS_KEY_ID","AWS_SECRET_ACCESS_KEY","S3_BUCKET"] if not os.getenv(k)]
    if missing:
        print(f"[error] .env に不足しているキーがあります: {missing}")
        print("       例：AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, S3_BUCKET")
        sys.exit(2)

    # 3) boto3 クライアント
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        print("boto3 が見つかりません。先に `pip install boto3` を実行してください。")
        sys.exit(1)

    s3 = boto3.client("s3", region_name=AWS_REGION)
    print(f"[cfg] region={AWS_REGION} bucket={S3_BUCKET} prefix={S3_PREFIX} user_id={user_id}")

    # 4) バケット存在＆権限チェック
    try:
        resp = s3.head_bucket(Bucket=S3_BUCKET)
        print(f"[ok] head_bucket HTTP {resp['ResponseMetadata']['HTTPStatusCode']}")
    except ClientError as e:
        print(f"[error] head_bucket: {e}")
        sys.exit(3)

    # 5) オブジェクト1件アップロード（暗号化はSSE-KMS優先、なければSSE-S3）
    base = f"{S3_PREFIX.rstrip('/')}/{user_id}/{datetime.utcnow():%Y/%m/%d}/{uuid4().hex}"
    key_txt = f"{base}.txt"
    body = f"hello from s3_smoketest at {datetime.utcnow().isoformat()}Z\n"
    extra = {"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": KMS_KEY_ID} if KMS_KEY_ID else {"ServerSideEncryption": "AES256"}

    try:
        s3.put_object(Bucket=S3_BUCKET, Key=key_txt, Body=body.encode("utf-8"),
                      ContentType="text/plain; charset=utf-8", **extra)
        print(f"[ok] put_object s3://{S3_BUCKET}/{key_txt}")
    except ClientError as e:
        print(f"[error] put_object: {e}")
        sys.exit(4)

    # 6) 一覧で確認（先頭1件）
    try:
        lst = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=f"{S3_PREFIX}{user_id}/", MaxKeys=1)
        print(f"[ok] list_objects_v2 prefix={S3_PREFIX}{user_id}/ KeyCount={lst.get('KeyCount')}")
    except ClientError as e:
        print(f"[error] list_objects_v2: {e}")
        sys.exit(5)

    # 7) 署名付きURL（確認用、15分）
    try:
        url = s3.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": S3_BUCKET, "Key": key_txt},
            ExpiresIn=900,
        )
        print(f"[ok] presigned_url (15min): {url}")
    except Exception as e:
        print(f"[warn] presigned_url: {e}（URL生成は任意項目です）")

    print("[done] S3 スモークテスト完了")

if __name__ == "__main__":
    # `python app/s3_smoketest.py your_user_id` でユーザー別プレフィックスも試せます
    user = sys.argv[1] if len(sys.argv) > 1 else "smoketest"
    main(user)
