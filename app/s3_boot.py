# app/s3_boot.py
from __future__ import annotations
import json, time
from uuid import uuid4
from typing import Callable, Awaitable
from starlette.requests import Request
from starlette.responses import Response, JSONResponse, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.datastructures import MutableHeaders
from app.s3_storage import put_bytes_user, put_json_user, put_text_user

# 既存アプリをそのまま読み込む（編集不要）
from app.main import app as _app  # ← 既存 main.py の app をインポート

TARGET_PATHS = {"/stt-full", "/stt-full/"}  # ここへのリクエストだけS3保存
USER_HEADER  = "x-user-id"                  # フロントが付けるユーザー識別ヘッダ

class STTS3Middleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        path = request.url.path
        if path not in TARGET_PATHS:
            return await call_next(request)

        # 1) ユーザーID（なければ匿名に落とす）
        user_id = request.headers.get(USER_HEADER, "anonymous")

        # 2) リクエストボディを先読み（multipartの音声を確保）
        #    下流にも同じbodyを渡すため、request._body を埋める
        body = await request.body()
        try:
            request._body = body  # Starletteの内部キャッシュをセット
        except Exception:
            pass

        # 3) 下流の本処理実行
        response = await call_next(request)

        # 4) 返却JSONを取り出す（取り出せなければそのまま返す）
        resp_bytes = b""
        try:
            if isinstance(response, StreamingResponse) and not isinstance(response, JSONResponse):
                chunks = [chunk async for chunk in response.body_iterator]
                resp_bytes = b"".join(chunks)
                response = Response(content=resp_bytes, status_code=response.status_code,
                                    headers=dict(response.headers), media_type=response.media_type)
            else:
                resp_bytes = await response.body()
        except Exception:
            pass

        # 5) S3 へ保存（失敗してもレスポンスはそのまま返す）
        try:
            base = f"{uuid4().hex}-{int(time.time())}"

            # 5-1) 音声：multipart からバイトを抜く（“file”フィールド想定）
            #       形式不明でも丸ごと保存しておくと後から復元可能
            if body:
                # まずはそのまま原文保存（再解釈用）
                put_bytes_user(user_id, body, f"{base}.rawreq", "application/octet-stream")
                # 可能なら簡易パースしてファイル抽出（best effort）
                if b"filename=" in body:
                    # 非厳密：最初のファイルバイナリっぽい区間だけ抜いて保存
                    # （厳密実装は不要なので省略、rawreqがあれば後から解析できます）
                    pass

            # 5-2) 解析結果（JSON）の保存
            if resp_bytes:
                try:
                    result_obj = json.loads(resp_bytes.decode("utf-8"))
                except Exception:
                    result_obj = {"_raw": True}
                    put_bytes_user(user_id, resp_bytes, f"{base}.result.bin", "application/octet-stream")
                else:
                    put_json_user(user_id, result_obj, f"{base}.result.json")
                    # 文字起こしテキストがあれば抜いて保存（任意キー名を推測）
                    text = (
                        result_obj.get("text")
                        or result_obj.get("transcript")
                        or result_obj.get("stt", {}).get("text")
                    )
                    if isinstance(text, str) and text.strip():
                        put_text_user(user_id, text, f"{base}.txt")

                    # メトリクスっぽい辞書を保存
                    metrics = (
                        result_obj.get("metrics")
                        or result_obj.get("analysis")
                        or result_obj.get("scores")
                    )
                    if isinstance(metrics, dict):
                        put_json_user(user_id, metrics, f"{base}.metrics.json")

        except Exception:
            # 失敗しても落とさない
            pass

        return response

# 既存アプリにミドルウェアを差し込んだ“起動用app”を公開
_app.add_middleware(STTS3Middleware)
app = _app
