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

from app.main import app as _app

# _appインポート後にルータを登録したあとに追加しないとエラー
from app.results_router import router as results_router
_app.include_router(results_router)

# S3 ユーティリティ 関連
try:
    from app.s3_storage import put_bytes_user, put_text_user, put_json_user
    _S3_AVAILABLE = True
except Exception:
    _S3_AVAILABLE = False
    def put_bytes_user(*a, **k): return None
    def put_text_user(*a, **k): return None
    def put_json_user(*a, **k): return None

# =========================
# CORS 設定 Renewed
# =========================
def _normalize_origin(o: str) -> str:
    o = o.strip()
    return o[:-1] if o.endswith("/") else o

_raw = (os.getenv("CORS_ORIGINS") or os.getenv("ALLOWED_ORIGINS") or "*").strip()

if _raw == "*":
    allow_origins = ["*"]
    allow_credentials = False  # "*" を使う場合は必ず False
else:
    allow_origins = [_normalize_origin(o) for o in _raw.split(",") if o.strip()]
    allow_credentials = True   # 具体オリジンは True

_app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],  # X-User-Id 等のカスタムヘッダも許可
    max_age=86400,
)

# すべての OPTIONS を 204 で即時返す（レスポンスは外側の CORS がヘッダ付与）
class _PreflightMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        if request.method.upper() == "OPTIONS":
            return Response(status_code=204)
        return await call_next(request)

_app.add_middleware(_PreflightMiddleware)

# =========================
# /stt-full の入出力を S3 に保存するミドルウェア
# =========================
TARGET_PATHS = {"/stt-full", "/stt-full/"}
USER_HEADER  = "x-user-id"

class STTS3Middleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        if request.url.path not in TARGET_PATHS:
            return await call_next(request)

        user_id = request.headers.get(USER_HEADER, request.headers.get("x-user", "web-client"))

        body = await request.body()
        try:
            request._body = body  # type: ignore[attr-defined]
        except Exception:
            pass

        response = await call_next(request)

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
# /analyze-logic　既存が無い場合の補助で、ヒューリスティック的なやつ
# =========================
def _split_sentences_ja(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"[。\n!?]+", text) if s.strip()]

def _score_logic(text: str) -> Dict:
    t = (text or "").strip()
    if not t:
        return {"total": 0.0, "scores": {k: 0 for k in ["clarity","consistency","cohesion","density","cta"]},
                "outline": [], "advice": []}

    sents = _split_sentences_ja(t)
    chars = len(t)

    has_intro = bool(re.search(r"(結論|要点|本日|今日は|概要|ポイント)", t[:min(160, len(t))]))
    has_cta = bool(re.search(r"(ご連絡|ご返信|予約|デモ|お申し込み|お願いします|お問い合わせ|クリック|こちら)", t[-min(260, len(t)):]))
    transitions = len(re.findall(r"(まず|次に|一方で|つまり|結果として|ただし|なお)", t))
    numbers = len(re.findall(r"\d+", t))
    headings = len(re.findall(r"^\s*[■#・\-・\*・●]", t, flags=re.M))

    clip = lambda x: max(0.0, min(100.0, float(x)))
    scores = {
        "clarity":     clip(60 + (20 if has_intro else 0) + min(20, headings*5)),
        "consistency": clip(55 + min(25, max(0, numbers - 1) * 5)),
        "cohesion":    clip(50 + min(30, transitions * 6)),
        "density":     clip(50 + min(35, int((numbers + (chars/400)) * 3))),
        "cta":         clip(50 + (30 if has_cta else 0)),
    }
    total = round(sum(scores.values()) / 5, 1)

    paras = [p.strip() for p in re.split(r"\n{2,}", t) if p.strip()]
    outline = []
    for p in paras[:8]:
        first = _split_sentences_ja(p[:200])
        if first:
            outline.append(first[0][:60])

    advice = []
    if scores["clarity"] < 70: advice.append("冒頭に『結論→要点』を置くと明瞭さが上がります。")
    if scores["cohesion"] < 70: advice.append("段落の接続語（まず/次に/つまり/結果として）を増やすと流れが滑らかです。")
    if scores["cta"] < 65:     advice.append("最後に次アクション（ご連絡/予約/返信依頼など）を明示しましょう。")

    return {"total": total, "scores": scores, "outline": outline, "advice": advice}

@_app.post("/analyze-logic")
async def analyze_logic(req: Request):
    data = await req.json()
    text = (data or {}).get("text", "") or ""
    return JSONResponse(_score_logic(text))

# 公開アプリ
app: FastAPI = _app

# 直接起動用
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8015"))
    uvicorn.run("app.web_boot:app", host="127.0.0.1", port=port, reload=True)
