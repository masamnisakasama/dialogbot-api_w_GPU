from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import os, math

# ── LLM 判定（全キーを必ず返す） ─────────────────────────────
from .llm_judger import (
    judge_with_openai,
    STYLE_KEYS, MOOD_KEYS, INTEREST_KEYS,
)

# ── DB があれば使う（なければ None→メモリにフォールバック） ──
try:
    from app import database as _dbmod
    SessionLocal = getattr(_dbmod, "SessionLocal", None)
except Exception:
    SessionLocal = None

try:
    from app import models as _models_mod  # モデル名は不定前提
except Exception:
    _models_mod = None

# ── インメモリ（DB無い場合用） ───────────────────────────────
_MEM: Dict[str, List[Dict[str, Any]]] = {}

# ── ユーティリティ ──────────────────────────────────────────
def _clip01(x: float) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    return 0.0 if v < 0 else 1.0 if v > 1 else v

def _blank_scores() -> Dict[str, Dict[str, float]]:
    return {
        "style":    {k: 0.0 for k in STYLE_KEYS},
        "mood":     {k: 0.0 for k in MOOD_KEYS},
        "interest": {k: 0.0 for k in INTEREST_KEYS},
    }

def _avg(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0

def _reduce_mean(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    out = _blank_scores()
    if not records:
        return out
    for k in STYLE_KEYS:
        out["style"][k] = _avg([float(r["style"].get(k, 0.0)) for r in records])
    for k in MOOD_KEYS:
        out["mood"][k] = _avg([float(r["mood"].get(k, 0.0)) for r in records])
    for k in INTEREST_KEYS:
        out["interest"][k] = _avg([float(r["interest"].get(k, 0.0)) for r in records])
    return out

# ── モデル自動検出（DBスキーマが不定でも動くように） ─────────────
def _pick_model_and_cols(models_mod) -> Tuple[Any, Dict[str, str]]:
    """
    会話テーブルと text/user/time カラム名を推測して返す。
    候補:
      モデル名: Utterance, Conversation, Message, Transcript, Chat, Dialogue
      text列:   text, message, content, transcript
      user列:   user, user_id, speaker, speaker_id, author, uid
      time列:   created_at, timestamp, ts, time, datetime
    """
    candidates = ["Utterance", "Conversation", "Message", "Transcript", "Chat", "Dialogue"]
    text_c = ["text", "message", "content", "transcript"]
    user_c = ["user", "user_id", "speaker", "speaker_id", "author", "uid"]
    time_c = ["created_at", "timestamp", "ts", "time", "datetime"]

    M = None
    for name in candidates:
        if hasattr(models_mod, name):
            M = getattr(models_mod, name)
            break
    if M is None:
        raise RuntimeError("会話モデルが見つかりません（Utterance/Conversation/Message/...）")

    def pick(obj, names):
        for n in names:
            if hasattr(obj, n):
                return n
        return None

    tcol = pick(M, text_c)
    ucol = pick(M, user_c)
    ccol = pick(M, time_c)
    if not (tcol and ucol and ccol):
        raise RuntimeError("text/user/time のカラムが見つかりません")
    return M, {"text": tcol, "user": ucol, "time": ccol}

# ── DB からユーザーの発言を取得 ──────────────────────────────
def _load_rows_from_db(user_id: str, days: int) -> Tuple[List[Any], Any, Dict[str, str]]:
    from sqlalchemy import asc
    if SessionLocal is None or _models_mod is None:
        return [], None, {}
    M, core = _pick_model_and_cols(_models_mod)

    since_dt = datetime.utcnow() - timedelta(days=days)
    db = SessionLocal()
    try:
        q = db.query(M).filter(
            getattr(M, core["user"]) == user_id,
            getattr(M, core["time"]) >= since_dt,
        ).order_by(asc(getattr(M, core["time"])))
        rows = q.all()
        return rows, M, core
    finally:
        db.close()

# ── DB 既存の特徴量列（style_xxx / mood_xxx / interest_xxx）を平均 ─────
def _aggregate_from_db_columns(rows: List[Any], M: Any) -> Dict[str, Dict[str, float]]:
    if not rows:
        return _blank_scores()

    def has_col(name: str) -> bool:
        return hasattr(M, name)

    out = _blank_scores()

    # style_*, mood_*, interest_* を自動集計
    for k in STYLE_KEYS:
        col = f"style_{k}"
        if has_col(col):
            vals = [getattr(r, col) for r in rows if getattr(r, col) is not None]
            if vals:
                out["style"][k] = _clip01(_avg([float(v) for v in vals]))

    mood_aliases = {
        "pos": ["mood_pos", "sentiment_pos", "positive"],
        "neg": ["mood_neg", "sentiment_neg", "negative"],
        "arousal": ["mood_arousal", "arousal"],
        "calm": ["mood_calm", "calm"],
        "excited": ["mood_excited", "excited"],
        "confident": ["mood_confident", "confident"],
        "anxious": ["mood_anxious", "anxious"],
        "frustrated": ["mood_frustrated", "frustrated"],
        "satisfied": ["mood_satisfied", "satisfied"],
        "curious": ["mood_curious", "curious"],
    }
    for k, cands in mood_aliases.items():
        for col in cands:
            if has_col(col):
                vals = [getattr(r, col) for r in rows if getattr(r, col) is not None]
                if vals:
                    out["mood"][k] = _clip01(_avg([float(v) for v in vals]))
                break

    for k in INTEREST_KEYS:
        col = f"interest_{k}"
        if has_col(col):
            vals = [getattr(r, col) for r in rows if getattr(r, col) is not None]
            if vals:
                out["interest"][k] = _clip01(_avg([float(v) for v in vals]))

    return out

# ── DBに列が無い指標は LLM で補完（コスト抑制のためサンプリング） ───────
def _fill_missing_with_llm(rows: List[Any], M: Any, core: Dict[str, str], base: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    texts = [getattr(r, core["text"]) or "" for r in rows] if rows else []
    if not texts:
        return base

    max_n = int(os.getenv("MAX_LLM_EVAL", "40"))
    if len(texts) > max_n:
        step = max(1, len(texts) // max_n)
        texts = texts[::step]

    # LLM で平均
    tmp = _blank_scores()
    cnt = 0
    for t in texts:
        jr = judge_with_openai(t)
        cnt += 1
        for k in STYLE_KEYS:    tmp["style"][k]    += float(jr.style.get(k, 0.0))
        for k in MOOD_KEYS:     tmp["mood"][k]     += float(jr.mood.get(k, 0.0))
        for k in INTEREST_KEYS: tmp["interest"][k] += float(jr.interest.get(k, 0.0))

    if cnt == 0:
        return base

    for k in STYLE_KEYS:    tmp["style"][k]    = _clip01(tmp["style"][k]    / cnt)
    for k in MOOD_KEYS:     tmp["mood"][k]     = _clip01(tmp["mood"][k]     / cnt)
    for k in INTEREST_KEYS: tmp["interest"][k] = _clip01(tmp["interest"][k] / cnt)

    # base の 0 を LLM 値で埋める（DBに列がある指標はDB優先）
    out = _blank_scores()
    for k in STYLE_KEYS:
        out["style"][k] = base["style"][k] if base["style"][k] > 0 else tmp["style"][k]
    for k in MOOD_KEYS:
        out["mood"][k] = base["mood"][k] if base["mood"][k] > 0 else tmp["mood"][k]
    for k in INTEREST_KEYS:
        out["interest"][k] = base["interest"][k] if base["interest"][k] > 0 else tmp["interest"][k]
    return out

# ── 公開API：メモリ書き込み（必要なら使う。DBは変更しない） ─────────
def ingest_message(user_id: str, text: str, ts: Optional[float] = None) -> Dict[str, Any]:
    """
    main.py を変えずに使えるよう、DB書き込みは行わず「任意でメモリに保持」します。
    既存の会話は DB から、直近の未保存分はメモリから集計できます。
    """
    from time import time as _now
    ts = ts or _now()
    jr = judge_with_openai(text)
    rec = {
        "ts": ts,
        "text": text,
        "style": {k: float(jr.style.get(k, 0.0)) for k in STYLE_KEYS},
        "mood":  {k: float(jr.mood.get(k, 0.0))  for k in MOOD_KEYS},
        "interest": {k: float(jr.interest.get(k, 0.0)) for k in INTEREST_KEYS},
    }
    _MEM.setdefault(user_id, []).append(rec)
    return rec

# ── 公開API：スナップショット（DB優先・不足分はLLMで補完・メモリも加味） ─────
def get_snapshot(user_id: str, days: int = 30) -> Dict[str, Any]:
    """
    返り値は {style:{全キー}, mood:{全キー}, interest:{全キー}, count, updated_at}
    - DBがあれば DB の会話履歴から集計（既存の特徴量列を優先）
    - 無い指標は LLM で補完（最大 MAX_LLM_EVAL 件をサンプル）
    - さらにメモリにある未保存レコードも平均に加算
    """
    # 1) DBから読み取り（あれば）
    rows, M, core = _load_rows_from_db(user_id, days)
    if rows:
        from_db = _aggregate_from_db_columns(rows, M)
        filled  = _fill_missing_with_llm(rows, M, core, from_db)
        base = filled
        updated_at = getattr(rows[-1], core["time"])
        if isinstance(updated_at, (int, float)):
            updated_at = datetime.utcfromtimestamp(float(updated_at))
    else:
        base = _blank_scores()
        updated_at = None

    # 2) メモリ（ingest_message で貯めた分）も合算
    #    ※ DBがあるプロダクションならメモリは使わない運用でもOK
    cutoff_ts = (datetime.utcnow() - timedelta(days=days)).timestamp()
    mem_recs = [r for r in _MEM.get(user_id, []) if r["ts"] >= cutoff_ts]
    if mem_recs:
        mem_mean = _reduce_mean(mem_recs)
        # 単純平均（重み付けが必要なら件数で重みを付けてください）
        merged = _blank_scores()
        for k in STYLE_KEYS:
            merged["style"][k] = _avg([base["style"][k], mem_mean["style"][k]])
        for k in MOOD_KEYS:
            merged["mood"][k] = _avg([base["mood"][k], mem_mean["mood"][k]])
        for k in INTEREST_KEYS:
            merged["interest"][k] = _avg([base["interest"][k], mem_mean["interest"][k]])
        base = merged
        if updated_at is None and mem_recs:
            from time import gmtime
            updated_at = datetime.utcfromtimestamp(max(r["ts"] for r in mem_recs))

    return {
        "style": base["style"],
        "mood": base["mood"],
        "interest": base["interest"],
        "count": (len(rows) if rows else 0) + len(mem_recs),
        "updated_at": (updated_at.isoformat() + "Z") if isinstance(updated_at, datetime) else None,
    }
