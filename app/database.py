import os
from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# ===== 環境・URL決定（本番は外部DB（PostreSQLのつもり））=====
APP_ENV = os.getenv("APP_ENV", "development").lower()

# Cloud Runでは /tmp が書き込み可。SQLiteを使う場合のデフォルトを /tmp に。
SQLITE_PATH = os.getenv("SQLITE_PATH", "/tmp/app.db")

# DATABASE_URL が無ければ SQLite にフォールバック（これは開発用だけどね）
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{SQLITE_PATH}")
IS_SQLITE = SQLALCHEMY_DATABASE_URL.startswith("sqlite")

# 本番でSQLiteを誤って使わないためのブロック（必要に応じて false で解除可）
BLOCK_SQLITE_IN_PROD = os.getenv("BLOCK_SQLITE_IN_PROD", "true").lower() in ("1", "true", "yes")
if APP_ENV == "production" and IS_SQLITE and BLOCK_SQLITE_IN_PROD:
    raise RuntimeError(
        "SQLite (test.db) は本番で無効です。DATABASE_URL を外部DB(PostgreSQL等)に設定するか、"
        "一時的に BLOCK_SQLITE_IN_PROD=false にしてください。"
    )

# ===== エンジン作成 =====
engine_kwargs = {"pool_pre_ping": True}
if IS_SQLITE:
    # SQLiteはスレッド制約があるので解除
    engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(SQLALCHEMY_DATABASE_URL, **engine_kwargs)

# SQLite利用時は最低限のチューニング（WAL/タイムアウト）
if IS_SQLITE:
    @event.listens_for(engine, "connect")
    def _sqlite_tuning(dbapi_conn, _):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute("PRAGMA busy_timeout=5000;")  # 5秒
        cur.close()

# セッションローカル
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ベースクラス
Base = declarative_base()

# DBセッション依存
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 初期化（既存のまま）
def init_db():
    # 重要：models を import してから create_all しないとテーブルが作られない
    from app import models  # noqa: F401
    Base.metadata.create_all(bind=engine)

# 起動ログ（パスワードは出さない簡易マスク）
def _mask_url(url: str) -> str:
    if "@" in url:
        head, tail = url.rsplit("@", 1)
        return head.split("://")[0] + "://****@" + tail
    return url

print(f"[DB] Using: { _mask_url(SQLALCHEMY_DATABASE_URL) }  (env={APP_ENV})")
