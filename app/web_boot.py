# app/web_boot.py
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(dotenv_path=(Path(__file__).resolve().parent / ".env"))

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import os, json, re, time
from uuid import uuid4
from typing import Callable, Awaitable, List, Dict

# メインアプリ（profile / logic / stt / mlops などを登録済み）
from app.main import app as _app

# 結果一覧APIを追加登録（メインアプリ読込後に行う）
from app.results_router import router as results_router
_app.include_router(results_router)

# =========================
# S3 ユーティリティ
# =========================
try:
    from app.s3_storage import put_bytes_user, put_text_user, put_json_user
    _S3_AVAILABLE = True
except Exception:
    _S3_AVAILABLE = False
    def put_bytes_user(*a, **k): return None
    def put_text_user(*a, **k): return None
    def put_json_user(*a, **k): return None

# =========================
# CORS 設定
# =========================
def _normalize_origin(o: str) -> str:
    o = o.strip()
    return o[:-1] if o.endswith("/") else o

_raw = (os.getenv("CORS_ORIGINS") or os.getenv("ALLOWED_ORIGINS") or "*").strip()

if _raw == "*":
    allow_origins = ["*"]
    allow_credentials = False  # "*" と併用不可
else:
    allow_origins = [_normalize_origin(o) for o in _raw.split(",") if o.strip()]
    allow_credentials = True

_app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],  # X-User-Id 等も許可
    max_age=86400,
)

_allowed = set(_normalize_origin(o) for o in (allow_origins if allow_origins != ["*"] else []))
_allow_all = (allow_origins == ["*"])

@_app.on_event("startup")
async def _warm_whisper():
    """Whisper モデルのウォームアップ（失敗しても起動は続行）"""
    try:
        import asyncio
        try:
            # 正しい配置（app/stt/whisper_utils.py）
            from app.stt.whisper_utils import get_model
        except Exception:
            # 互換（もし直下にある場合）
            from app.whisper_utils import get_model  # type: ignore
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, get_model)
        print("[startup] whisper model preloaded")
    except Exception as e:
        print("[startup] whisper preload skipped:", e)

# すべてのパスのプリフライト応答
@_app.options("/{rest_of_path:path}", include_in_schema=False)
async def _preflight_any(rest_of_path: str, request: Request):
    origin = request.headers.get("origin", "")
    req_headers = request.headers.get("access-control-request-headers", "content-type,x-user-id")
    headers = {
        "Access-Control-Allow-Methods": "DELETE, GET, HEAD, OPTIONS, PATCH, POST, PUT",
        "Access-Control-Allow-Headers": req_headers,
        "Access-Control-Max-Age": "600",
    }
    if _allow_all:
        headers["Access-Control-Allow-Origin"] = "*"
    else:
        if origin and _normalize_origin(origin) in _allowed:
            headers["Access-Control-Allow-Origin"] = origin
            headers["Vary"] = "Origin"
            if allow_credentials:
                headers["Access-Control-Allow-Credentials"] = "true"
    return Response(status_code=204, headers=headers)

# =========================
# /stt-full I/O を S3 保存するミドルウェア
# =========================
TARGET_PATHS = {"/stt-full", "/stt-full/"}  # どちらにも対応
USER_HEADER = "x-user-id"

class STTS3Middleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        if request.url.path not in TARGET_PATHS:
            return await call_next(request)

        user_id = request.headers.get(USER_HEADER, request.headers.get("x-user", "web-client"))

        body = await request.body()
        try:
            # 下流でも body を読めるようにする（FastAPI内部仕様につき自己責任）
            request._body = body  # type: ignore[attr-defined]
        except Exception:
            pass

        response = await call_next(request)

        # レスポンスを取り出し（StreamingResponse の場合は吸い上げて包み直す）
        resp_bytes = b""
        try:
            if isinstance(response, StreamingResponse) and not isinstance(response, JSONResponse):
                chunks = [chunk async for chunk in response.body_iterator]
                resp_bytes = b"".join(chunks)
                response = Response(
                    content=resp_bytes,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.media_type,
                )
            else:
                resp_bytes = await response.body()
        except Exception:
            pass

        # S3 保存（ベストエフォート）
        if _S3_AVAILABLE:
            try:
                date_prefix = time.strftime("%Y/%m/%d")
                base = f"{user_id}/{date_prefix}/{uuid4().hex}"

                if body:
                    put_bytes_user(user_id, body, f"{base}.rawreq", "application/octet-stream")

                if resp_bytes:
                    try:
                        result_obj = json.loads(resp_bytes.decode("utf-8"))
                    except Exception:
                        put_bytes_user(user_id, resp_bytes, f"{base}.result.bin", "application/octet-stream")
                    else:
                        put_json_user(user_id, result_obj, f"{base}.result.json")
                        text = (
                            result_obj.get("text")
                            or result_obj.get("transcript")
                            or result_obj.get("result", {}).get("text")
                        )
                        if isinstance(text, str) and text.strip():
                            put_text_user(user_id, text, f"{base}.txt")

                        metrics = (
                            result_obj.get("metrics")
                            or result_obj.get("audio_metrics")
                            or result_obj.get("scores")
                        )
                        if isinstance(metrics, dict):
                            put_json_user(user_id, metrics, f"{base}.metrics.json")
            except Exception:
                pass

        return response

_app.add_middleware(STTS3Middleware)

# =========================
# 重要: /analyze-logic の“簡易版”は定義しない
# （app.main で include_router(logic_router) 済みの本来版を使う）
# =========================

# 公開アプリ
app: FastAPI = _app

# 直接起動用（ローカル実行時）
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8015"))
    uvicorn.run("app.web_boot:app", host="127.0.0.1", port=port, reload=True)
