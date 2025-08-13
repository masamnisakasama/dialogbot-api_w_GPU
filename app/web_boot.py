# app/web_boot.py
from __future__ import annotations


# --- 重要：どこから実行しても app パッケージを解決できるようにする ---
import sys
from pathlib import Path

# /path/to/project を sys.path に追加（このファイルの親の親をルートとみなす）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# --- .env 読み込み（app/.env を確実に読む） ---
from dotenv import load_dotenv
load_dotenv(dotenv_path=(Path(__file__).resolve().parent / ".env"))

# --- 既存 FastAPI アプリを読み込む（改造不要） ---
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import os, json, re, time
from uuid import uuid4
from typing import Callable, Awaitable, List, Dict

# 既存の main.py が持つ app を取り込む
from app.main import app as _app

# _appインポート後にルータインポートしないと動かない
from app.results_router import router as results_router
_app.include_router(results_router)

# --- S3 ユーティリティ（既存 s3_storage.py を使用） ---
# もし未配置でも動作が落ちないように NO-OP を用意
try:
    from app.s3_storage import put_bytes_user, put_text_user, put_json_user
    _S3_AVAILABLE = True
except Exception:
    _S3_AVAILABLE = False

    def put_bytes_user(*a, **k): return None
    def put_text_user(*a, **k): return None
    def put_json_user(*a, **k): return None

# =========================
# CORS（フロントから叩けるように）
# =========================
origins_env = os.getenv("CORS_ORIGINS", "*")
origins = [o.strip() for o in origins_env.split(",")] if origins_env else ["*"]
_AppForCors = _app  # alias
_AppForCors.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# /stt-full の入出力を S3 に保存するミドルウェア
# =========================
TARGET_PATHS = {"/stt-full", "/stt-full/"}
USER_HEADER  = "x-user-id"  # フロントが付けられるヘッダ（任意）

class STTS3Middleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        # 対象パス以外はスルー
        if request.url.path not in TARGET_PATHS:
            return await call_next(request)

        # ユーザーID（無ければ web-client）
        user_id = request.headers.get(USER_HEADER, request.headers.get("x-user", "web-client"))

        # リクエスト本文を確保（multipart全体）
        body = await request.body()
        try:
            # 下流でも同じボディが使えるよう内部キャッシュへ
            request._body = body  # type: ignore[attr-defined]
        except Exception:
            pass

        # 下流に処理を渡す
        response = await call_next(request)

        # レスポンス取り出し（StreamingResponse 対応）
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

        # S3 保存（失敗してもレスポンスはそのまま返す）
        if _S3_AVAILABLE:
            try:
                # key 例: app/<user_id>/YYYY/MM/DD/<uuid>.(result.json|txt|metrics.json|rawreq)
                date_prefix = time.strftime("%Y/%m/%d")
                base = f"{user_id}/{date_prefix}/{uuid4().hex}"

                # リクエスト原文（解析失敗時のリカバリ用に保存）
                if body:
                    put_bytes_user(user_id, body, f"{base}.rawreq", "application/octet-stream")

                if resp_bytes:
                    # JSONなら分解保存、ダメならバイナリで保存
                    try:
                        result_obj = json.loads(resp_bytes.decode("utf-8"))
                    except Exception:
                        put_bytes_user(user_id, resp_bytes, f"{base}.result.bin", "application/octet-stream")
                    else:
                        put_json_user(user_id, result_obj, f"{base}.result.json")
                        # テキスト候補（複数キー名に対応）
                        text = (
                            result_obj.get("text")
                            or result_obj.get("transcript")
                            or result_obj.get("result", {}).get("text")
                        )
                        if isinstance(text, str) and text.strip():
                            put_text_user(user_id, text, f"{base}.txt")

                        # メトリクス候補（キー名の揺れに対応）
                        metrics = (
                            result_obj.get("metrics")
                            or result_obj.get("audio_metrics")
                            or result_obj.get("scores")
                        )
                        if isinstance(metrics, dict):
                            put_json_user(user_id, metrics, f"{base}.metrics.json")
            except Exception:
                # ここでは黙って続行（本処理を止めない）
                pass

        return response

# ミドルウェアを差し込む
_app.add_middleware(STTS3Middleware)

# =========================
# /analyze-logic エンドポイント（フロントが呼ぶ）
# 既存に無ければここで提供。あれば既存を使ってもOK。
# =========================
def _split_sentences_ja(text: str) -> List[str]:
    # ざっくり日本語の文分割
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
    clarity     = clip(60 + (20 if has_intro else 0) + min(20, headings*5))
    consistency = clip(55 + min(25, max(0, numbers - 1) * 5))
    cohesion    = clip(50 + min(30, transitions * 6))
    density     = clip(50 + min(35, int((numbers + (chars/400)) * 3)))
    cta         = clip(50 + (30 if has_cta else 0))

    scores = {"clarity": clarity, "consistency": consistency, "cohesion": cohesion, "density": density, "cta": cta}
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

# 最終的に公開するアプリ
app: FastAPI = _app

# --- 直接起動にも対応（python app/web_boot.py） ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8015"))
    uvicorn.run("app.web_boot:app", host="127.0.0.1", port=port, reload=True)
