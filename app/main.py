# app/main.py
from dotenv import load_dotenv
import os

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(env_path, override=False)
import logging

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

# プロジェクト内モジュール
from app.database import get_db, init_db
from app import schemas, crud, features

# ルーター群
from app.stt.stt_router import router as stt_router           # /analyze/audio など
from app.mlops.retrain_api import router as retrain_router    # /mlops/retrain
from app.mlops.drift_router import router as drift_router     # /drift/rebase, /drift/status
from app.profile_router import router as profile_router
from app.logic_router import router as logic_router

# ===== アプリ生成（ここだけで作る） =====
app = FastAPI(title="Dialog Bot API")
app.include_router(profile_router)
app.include_router(logic_router, prefix="") 

#　web_boot.pyのallow_origins=allow_originsでコケないため
raw = os.getenv("CORS_ORIGINS", "").strip()
if raw in ("*", "wildcard", "WILDCARD"):
    origins = ["*"]
else:
    origins = [o.strip() for o in raw.split(",") if o.strip()]
if not origins:
    # 開発
    origins = ["http://localhost:3000", "http://127.0.0.1:3000","https://roaring-pavlova-76f278.netlify.app"]

# ===== ミドルウェア =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,         
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== ルーター登録 =====
app.include_router(stt_router)
app.include_router(retrain_router)
app.include_router(drift_router)

# 追加：起動時にDB init（テーブルが無ければ作るとうまくいく可能性）
@app.on_event("startup")
def _startup():
    init_db()

class UpsertRequest(BaseModel):
    text: str
    speaker: str | None = None
    top_k: int = 5

class UpsertResponse(BaseModel):
    id: int
    similar: list[dict]


# ===== 単発エンドポイント（必要最小限） =====
class UpsertRequest(BaseModel):
    text: str
    speaker: str | None = None
    top_k: int = 5

class UpsertResponse(BaseModel):
    id: int
    similar: list[dict]  # {id, user, message, similarity}

# 現在使っていないが、MLOps拡張に必要となる可能性(本当？)

@app.post("/conversations/upsert_and_search", response_model=UpsertResponse)
def upsert_and_search(body: UpsertRequest, db: Session = Depends(get_db)):
    """
    OpenAI Embedding を一回だけ実行 → 保存 → 既存データとの類似Top-kを返す。
    """
    # 1) Embedding（OpenAI）
    emb_bytes = features.get_openai_embedding(body.text)

    # 2) 保存
    conv_create = schemas.ConversationCreate(
        user=body.speaker or "user",
        message=body.text
    )
    saved = crud.create_conversation(
        db=db,
        conv=conv_create,
        style=None,
        embedding=emb_bytes,
        sentiment=None,  # モデルに列が無い場合も安全に動くようcrud側でhasattrチェック済み
    )

    # 3) 類似検索（自分は除外）
    sims = crud.topk_similar(
        db,
        query_emb=emb_bytes,
        top_k=body.top_k,
        exclude_id=saved.id
    )
    return UpsertResponse(id=saved.id, similar=sims)

# healthcheck
@app.get("/health")
def health():
    return {"status": "ok"}

# 一応healthzも入れておくわ
@app.get("/healthz", tags=["meta"])
def healthz():
    return {"ok": True}

# ===== ログ設定（任意） =====
logging.basicConfig(level=logging.INFO)

# ===== 起動時デバッグ：ルート一覧を出力（不要なら削除OK） =====
for r in app.routes:
    try:
        print("ROUTE", getattr(r, "methods", None), getattr(r, "path", None))
    except Exception:
        pass

# ===== 直起動用（uvicornから呼ぶなら不要） =====
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8015, reload=True)
