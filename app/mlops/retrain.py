# retain.py
# ドリフト基準（参照統計）の再作成　難しいので、最小構成MCPから少しづつ大きくしていく
import os, time, numpy as np
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.mlops.drift_router import _fetch_reference_rows, _text_stats, _emb_matrix, _save_ref

def retrain_model():
    days = int(os.getenv("RETRAIN_REF_DAYS", "30"))
    db: Session = SessionLocal()
    try:
        rows = _fetch_reference_rows(db, days=days)
        stats = _text_stats(rows); emb = _emb_matrix(rows)
        len_vals = stats["len"]; latin_vals = stats["latin_ratio"]
        len_edges = np.histogram_bin_edges(len_vals, bins=10) if len_vals.size else np.array([], np.float64)
        latin_edges = np.linspace(0.0, 1.0, 11, dtype=np.float64)

        def _ref_p(vals, edges):
            if edges.size == 0: return []
            hist, _ = np.histogram(vals, bins=edges)
            s = max(1, hist.sum()); return (hist / s).astype(float).tolist()

        payload = {
            "ts": time.time(), "window_days": days, "n_rows": len(rows),
            "len_mean": float(len_vals.mean()) if len_vals.size else 0.0,
            "len_std": float(len_vals.std()) if len_vals.size else 1.0,
            "latin_mean": float(latin_vals.mean()) if latin_vals.size else 0.0,
            "latin_std": float(latin_vals.std()) if latin_vals.size else 1.0,
            "emb_center": emb.mean(axis=0).tolist() if emb.size else [],
            "emb_dim": int(emb.shape[1]) if emb.ndim == 2 and emb.shape[1] else 1536,
            "len_edges": len_edges.tolist(), "len_ref_p": _ref_p(len_vals, len_edges),
            "latin_edges": latin_edges.tolist(), "latin_ref_p": _ref_p(latin_vals, latin_edges),
        }
        _save_ref(payload)
        return {"rebase": "ok", "n_rows": len(rows), "emb_dim": payload["emb_dim"]}
    finally:
        db.close()