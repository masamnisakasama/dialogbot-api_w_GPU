# app/results_router.py
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

import boto3
try:
    from app.s3_storage import s3_client as _s3_client  # あれば利用
except Exception:
    _s3_client = None

S3_BUCKET = os.getenv("S3_BUCKET")
S3_PREFIX = os.getenv("S3_PREFIX", "app/").rstrip("/")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

router = APIRouter(prefix="/results", tags=["results"])


def _client():
    if _s3_client:
        return _s3_client()
    return boto3.client("s3", region_name=AWS_REGION)


def _iter_day_prefixes(base_prefix: str, days: int) -> List[str]:
    tz = timezone.utc
    today = datetime.now(tz).date()
    out: List[str] = []
    for i in range(max(0, int(days)) + 1):
        d = today - timedelta(days=i)
        out.append(f"{base_prefix}/{d:%Y/%m/%d}/")
    return out


def _group_objects(objs: List[Dict]) -> List[Dict]:
    groups: Dict[str, Dict] = {}
    for o in objs:
        key = o["Key"]
        parts = key.split("/")
        if len(parts) < 2:
            continue
        base = "/".join(parts[:-1])
        last = o.get("LastModified")
        g = groups.setdefault(base, {"base": base, "keys": {}, "last_modified": last})
        if last and (not g["last_modified"] or last > g["last_modified"]):
            g["last_modified"] = last

        if key.endswith("/transcript.txt"):
            g["keys"]["transcript"] = key
        elif key.endswith("/result.json"):
            g["keys"]["result"] = key
        elif key.endswith("/audio_metrics.json"):
            g["keys"]["metrics"] = key
        elif key.endswith(".rawreq") or "/rawreq/" in key or key.endswith("/rawreq"):
            g["keys"]["rawreq"] = key
        else:
            pass

    out = sorted(
        groups.values(),
        key=lambda x: x["last_modified"] or datetime(1970, 1, 1, tzinfo=timezone.utc),
        reverse=True,
    )
    return out


@router.get("/list")
def list_results(
    user_id: str = Query(...),
    days: int = Query(7, ge=1, le=31),
    limit: int = Query(50, ge=1, le=200),
    presign_secs: int = Query(900, ge=60, le=86400),
):
    """
    指定 user_id の直近 days 日に生成された結果を、S3 から集約して返す。
    各アイテムには存在するファイルの署名付きURLを付与。
    """
    if not S3_BUCKET:
        return JSONResponse(
            {"items": [], "count": 0, "user_id": user_id, "reason": "S3_BUCKET not set"},
            status_code=200,
        )

    cli = _client()
    base_prefix = f"{S3_PREFIX}/{user_id}".strip("/")

    objs: List[Dict] = []
    for pfx in _iter_day_prefixes(base_prefix, days):
        token: Optional[str] = None
        while True:
            resp = cli.list_objects_v2(
                Bucket=S3_BUCKET, Prefix=pfx, ContinuationToken=token
            ) if token else cli.list_objects_v2(Bucket=S3_BUCKET, Prefix=pfx)
            objs.extend(resp.get("Contents", []))
            if not resp.get("IsTruncated"):
                break
            token = resp.get("NextContinuationToken")

    groups = _group_objects(objs)

    items: List[Dict] = []
    for g in groups[:limit]:
        files: Dict[str, Dict] = {}
        for typ, key in g["keys"].items():
            url = cli.generate_presigned_url(
                ClientMethod="get_object",
                Params={"Bucket": S3_BUCKET, "Key": key},
                ExpiresIn=int(presign_secs),
            )
            files[typ] = {"key": key, "url": url}

        items.append({
            "base": g["base"],
            "updated_at": (
                g["last_modified"].astimezone(timezone.utc).isoformat()
                if g["last_modified"] else None
            ),
            "files": files,
        })

    return {"items": items, "count": len(items), "user_id": user_id, "prefix": f"{base_prefix}/"}
