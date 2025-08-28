# app/mlops/drift_router.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import numpy as np
import pickle, json, os, time, re

from app.database import get_db
from app import models

router = APIRouter()

# 参照統計の保存先（とりまJSONで本番はGCSに切り替えるつもり）
REF_PATH = os.getenv("DRIFT_REF_PATH", "./drift_ref.json")

# PSI（Population Stability Index）とシフトの基準（Severity）
PSI_SEV_LOW = float(os.getenv("DRIFT_PSI_SEV_LOW", "0.10"))
PSI_SEV_HIGH = float(os.getenv("DRIFT_PSI_SEV_HIGH", "0.25"))
SHIFT_SEV_LOW = float(os.getenv("DRIFT_SHIFT_SEV_LOW", "0.03"))
SHIFT_SEV_HIGH = float(os.getenv("DRIFT_SHIFT_SEV_HIGH", "0.07"))
MIN_CHARS = int(os.getenv("MLOPS_MIN_CHARS", "1000")) # 最小１０００文字で学習開始　短い音声避ける


# ----------------- ユーティリティ -----------------

def _load_ref() -> Optional[Dict[str, Any]]:
     # もし本番のGCS環境なら...
     if REF_PATH.startswith("gs://"):
        from google.cloud import storage
        _, _, bucket, *key = REF_PATH.split("/", 3)
        blob = "/".join(key)
        b = storage.Client().bucket(bucket).blob(blob)
        if not b.exists(): return None
        return json.loads(b.download_as_text())
     
     # ローカルなら...
     if os.path.exists(REF_PATH):
        with open(REF_PATH, "r") as f:
            return json.load(f)
        return None

def _save_ref(payload: Dict[str, Any]) -> None:
    # 本番のGCS環境のみreferenceを保存
    if REF_PATH.startswith("gs://"):
        from google.cloud import storage
        _, _, bucket, *key = REF_PATH.split("/", 3)
        blob = "/".join(key)
        storage.Client().bucket(bucket).blob(blob).upload_from_string(
            json.dumps(payload), content_type="application/json"
        )
        return
    with open(REF_PATH, "w") as f: json.dump(payload, f)

# pickled bytes → numpy.float32ベクトルに変換し保存　エラーで何も返さない
def _safe_vec(b: Optional[bytes]) -> Optional[np.ndarray]:
    if not b: return None
    try:
        v = pickle.loads(b)
        a = np.asarray(v, dtype=np.float32).ravel()
        return a if a.size > 0 else None
    except Exception:
        return None
    
# created_at があれば最優先。無ければ timestamp を使う
def _time_col():
    return getattr(models.Conversation, "created_at", None) or getattr(models.Conversation, "timestamp", None)

# timestamp を使えるようヘルパー差し替えて、1000文字未満除外を追加
def _fetch_reference_rows(db: Session, days: int) -> List[models.Conversation]:
    col = _time_col()
    q = db.query(models.Conversation)
    if col is not None:
        since = datetime.utcnow() - timedelta(days=days)
        q = q.filter(col >= since)
    return q.all()

# timestamp を使えるようヘルパー差し替えて、1000文字未満除外を追加
def _fetch_recent_rows(db: Session, hours: int, recent_n: int = 200) -> List[models.Conversation]:
    col = _time_col()
    q = db.query(models.Conversation)
    if col is not None:
        since = datetime.utcnow() - timedelta(hours=hours)
        q = q.filter(col >= since)
        return q.all()
    return q.order_by(models.Conversation.id.desc()).limit(recent_n).all()

# ----------- 特徴抽出（数値） -----------

_LATIN_RE = re.compile(r"[A-Za-z0-9]")  # 日本語は False 英数字のみ True 大体の英語率測定

# messageの長さと、英数字率を配列で返す定義　直感的に一番有用そうなPSIの指標なので導入
# _text_stats / _emb_matrixが1000文字未満弾くように
def _text_stats(rows: List[models.Conversation]) -> Dict[str, np.ndarray]:
    lens, latin_ratio = [], []
    for r in rows:
        t = (r.message or "").strip()
        if not t or len(t) < MIN_CHARS:
            continue
        lens.append(len(t))
        latin = sum(1 for ch in t if _LATIN_RE.match(ch))
        latin_ratio.append(latin / max(1, len(t)))
    return {"len": np.asarray(lens, dtype=np.float32), "latin_ratio": np.asarray(latin_ratio, dtype=np.float32)}

# 結果（rows） の embedding を集めて行列化　意味的なズレをPSIで測定するための整理
def _emb_matrix(rows: List[models.Conversation]) -> np.ndarray:
    emb = []
    for r in rows:
        t = (r.message or "").strip()
        if not t or len(t) < MIN_CHARS:
            continue
        v = _safe_vec(getattr(r, "embedding", None))
        if v is not None:
            emb.append(v)
    return np.asarray(emb, dtype=np.float32)


# ----------- 指標計算 -----------

# PSIの計算　大きいと参照との乖離 感覚的には0.2で結構離れるくらいに調整
def _psi_from_hist(ref_p: np.ndarray, cur_p: np.ndarray) -> float:
    # 安全クリップ + ログPSI
    r = np.clip(ref_p, 1e-6, 1)
    c = np.clip(cur_p, 1e-6, 1)
    return float(np.sum((r - c) * np.log(r / c)))


# 埋め込み中心のコサイン距離（0=同一, 1=直交）
def _cos_center_shift(ref_mat: np.ndarray, cur_mat: np.ndarray) -> float:
    if ref_mat.size == 0 or cur_mat.size == 0:
        return 0.0
    rc = ref_mat.mean(axis=0)
    cc = cur_mat.mean(axis=0)
    na = np.linalg.norm(rc); nb = np.linalg.norm(cc)
    if na == 0 or nb == 0:
        return 0.0
    return float(1 - np.dot(rc, cc) / (na * nb))

# ハードコード回避
def _sev_from_psi(v: float) -> str:
    if v < PSI_SEV_LOW: return "low"
    if v < PSI_SEV_HIGH: return "moderate"
    return "high"
def _sev_from_shift(v: float) -> str:
    if v < SHIFT_SEV_LOW: return "low"
    if v < SHIFT_SEV_HIGH: return "moderate"
    return "high"

# ----------------- API -----------------
# 直近7日もしくは全件のデータを再計算して参照
# ヒストグラムの bin 境界（edges）と参照頻度（ref_p）も保存して、PSIの安定性を担保
# refはたくさんあるが基準の重みづけは生み出したサンプルをベースにして考える

@router.post("/drift/rebase")
def rebase_reference(days: int = 7, db: Session = Depends(get_db)) -> Dict[str, Any]:
    rows = _fetch_reference_rows(db, days=days)
    stats = _text_stats(rows); emb = _emb_matrix(rows)
    len_vals = stats["len"]; latin_vals = stats["latin_ratio"]
    # 参照側で bin を決める感じ
    len_edges = np.histogram_bin_edges(len_vals, bins=10) if len_vals.size else np.array([], dtype=np.float64)
    latin_edges = np.histogram_bin_edges(latin_vals, bins=10) if latin_vals.size else np.array([], dtype=np.float64)


    # 正規化必須でしょう
    def _ref_p(vals, edges):
        if edges.size == 0: return []
        hist, _ = np.histogram(vals, bins=edges)
        s = max(1, hist.sum()); return (hist / s).astype(float).tolist()

    ref_payload = {
        "ts": time.time(),
        "window_days": days,
        "n_rows": len(rows),
        "len_mean": float(len_vals.mean()) if len_vals.size else 0.0,
        "len_std": float(len_vals.std()) if len_vals.size else 1.0,
        "latin_mean": float(latin_vals.mean()) if latin_vals.size else 0.0,
        "latin_std": float(latin_vals.std()) if latin_vals.size else 1.0,
        "emb_center": emb.mean(axis=0).tolist() if emb.size else [],
        "emb_dim": int(emb.shape[1]) if emb.ndim == 2 and emb.shape[1] else 1536,
        "len_edges": len_edges.tolist(),
        "len_ref_p": _ref_p(len_vals, len_edges),
        "latin_edges": latin_edges.tolist(),
        "latin_ref_p": _ref_p(latin_vals, latin_edges),
    }
    _save_ref(ref_payload)
    return {
        "status": "ok",
        "reference": {**ref_payload, "ts_iso": datetime.utcfromtimestamp(ref_payload["ts"]).isoformat() + "Z"}
    }

# 直近との比較でドリフト計算　デフォ７２時間かつ２００件
@router.get("/drift/status")
def drift_status(hours: int = 72, db: Session = Depends(get_db)) -> Dict[str, Any]:
    ref = _load_ref()
    if not ref:
        return {"status": "no_reference", "hint": "POST /drift/rebase を先に実行してください"}

    rows = _fetch_recent_rows(db, hours=hours, recent_n=200)
    MIN_N = 10
    if len(rows) < MIN_N:
        return {"status": "insufficient_data", "n_recent": len(rows)}

    stats_cur = _text_stats(rows); emb_cur = _emb_matrix(rows)

    def psi_with_saved(ref_p_key: str, edges_key: str, cur_vals: np.ndarray) -> float:
        ref_p = np.asarray(ref.get(ref_p_key, []), dtype=np.float64)
        edges = np.asarray(ref.get(edges_key, []), dtype=np.float64)
        if cur_vals.size == 0 or ref_p.size == 0 or edges.size == 0: return 0.0
        cur_hist, _ = np.histogram(cur_vals, bins=edges)
        cur_p = cur_hist / max(1, cur_hist.sum())
        return _psi_from_hist(ref_p, cur_p)
    
    # 参照の ref_p を用いて PSI を計算
    psi_len = psi_with_saved("len_ref_p", "len_edges", stats_cur["len"])
    psi_lat = psi_with_saved("latin_ref_p", "latin_edges", stats_cur["latin_ratio"])

    # 埋め込み中心距離の計算 0.05で高いくらいだと嬉しい
    ref_center = np.asarray(ref.get("emb_center", []), dtype=np.float32)
    center_shift = 0.0 if ref_center.size == 0 else _cos_center_shift(ref_center.reshape(1, -1), emb_cur)

    sev = {
        "psi_len": _sev_from_psi(psi_len),
        "psi_latin": _sev_from_psi(psi_lat),
        "center_shift": _sev_from_shift(center_shift),
    }
    order = {"low": 0, "moderate": 1, "high": 2}
    overall = max(sev.values(), key=lambda k: order[k])
    label = {"low": "stable", "moderate": "watch", "high": "drifting"}[overall]

    return {
        "status": label,
        "window_hours": hours,
        "n_recent": len(rows),
        "metrics": {
            "psi_len": psi_len,
            "psi_latin": psi_lat,
            "center_shift": center_shift,
        },
        "severity": sev,
        "reference": {
            "ts": ref["ts"],
            "ts_iso": datetime.utcfromtimestamp(ref["ts"]).isoformat() + "Z",
            "window_days": ref.get("window_days"),
            "n_rows": ref.get("n_rows"),
            "emb_dim": ref.get("emb_dim"),
        },
    }