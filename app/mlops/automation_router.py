# app/mlops/automation_router.py
# mlopsのアートメーション部分を担う予定のpyファイル

from fastapi import APIRouter, Depends, BackgroundTasks, Query, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional
import os, traceback, numpy as np

from app.database import get_db, SessionLocal
from app import models
from app.mlops import retrain as retrain_mod
from app.mlops.drift_router import (
    _load_ref, _fetch_recent_rows, _text_stats, _emb_matrix,
    _psi_from_hist, _cos_center_shift, _sev_from_psi, _sev_from_shift
)
from app.mlops.utils import get_running_job, create_job, mark_started, mark_success, mark_failed, list_jobs

router = APIRouter()
_LEVEL_ORDER = {"low":0, "moderate":1, "high":2}

def _calc_drift(db: Session, hours: int = 720) -> Dict[str, Any]:
    ref = _load_ref()
    if not ref:
        return {"status": "no_reference"}
    rows = _fetch_recent_rows(db, hours=hours, recent_n=200)
    if len(rows) < 10:
        return {"status": "insufficient_data", "n_recent": len(rows)}

    stats = _text_stats(rows); emb_cur = _emb_matrix(rows)
    def psi_with_saved(ref_p_key, edges_key, cur_vals):
        import numpy as _np
        ref_p = _np.asarray(ref.get(ref_p_key, []), dtype=_np.float64)
        edges = _np.asarray(ref.get(edges_key, []), dtype=_np.float64)
        if cur_vals.size == 0 or ref_p.size == 0 or edges.size == 0:
            return 0.0
        cur_hist, _ = _np.histogram(cur_vals, bins=edges)
        cur_p = cur_hist / max(1, cur_hist.sum())
        return _psi_from_hist(ref_p, cur_p)

    psi_len = psi_with_saved("len_ref_p", "len_edges", stats["len"])
    psi_lat = psi_with_saved("latin_ref_p", "latin_edges", stats["latin_ratio"])

    ref_center = np.asarray(ref.get("emb_center", []), dtype=np.float32)
    center_shift = 0.0 if ref_center.size == 0 else _cos_center_shift(ref_center.reshape(1, -1), emb_cur)

    sev = {"psi_len": _sev_from_psi(psi_len), "psi_latin": _sev_from_psi(psi_lat), "center_shift": _sev_from_shift(center_shift)}
    overall = max(sev.values(), key=lambda k: _LEVEL_ORDER[k])
    label = {"low":"stable","moderate":"watch","high":"drifting"}[overall]

    return {"status": label, "metrics": {"psi_len": psi_len, "psi_latin": psi_lat, "center_shift": center_shift},
            "severity": sev, "n_recent": len(rows),
            "reference": {"ts": ref.get("ts"), "window_days": ref.get("window_days"), "n_rows": ref.get("n_rows")}}

def _run_retrain_job(job_id: int):
    db = SessionLocal()
    try:
        job = db.query(models.MLOPSJob).get(job_id)
        if not job:
            return
        mark_started(db, job)
        retrain_mod.retrain_model()
        mark_success(db, job)
    except Exception:
        try:
            job = db.query(models.MLOPSJob).get(job_id)
            if job:
                mark_failed(db, job, traceback.format_exc())
        finally:
            pass
    finally:
        db.close()

# ドリフトを評価、しきい値超えなら再学習を行うルータ
@router.post("/drift/check-and-trigger", tags=["mlops"])
def check_and_trigger(background: BackgroundTasks,
    hours: int = Query(default=int(os.getenv("DRIFT_CRON_HOURS", "720"))),
    trigger_level: str = Query(default=os.getenv("DRIFT_TRIGGER_LEVEL", "high")),
    db: Session = Depends(get_db)) -> Dict[str, Any]:

    if trigger_level not in _LEVEL_ORDER:
        raise HTTPException(400, f"invalid trigger_level: {trigger_level}")

    drift = _calc_drift(db, hours=hours)
    if drift.get("status") in ("no_reference", "insufficient_data"):
        return {"status": "skipped", "reason": drift.get("status"), "detail": drift}

    overall = max(drift["severity"].values(), key=lambda k: _LEVEL_ORDER[k])
    if _LEVEL_ORDER[overall] < _LEVEL_ORDER[trigger_level]:
        return {"status": "skipped", "reason": f"below_threshold({trigger_level})", "detail": drift}

    if get_running_job(db):
        return {"status": "skipped", "reason": "already_running", "detail": drift}

    job = create_job(db, triggered_by="auto", reason=f"drift:{drift['status']}", severity=overall, metrics=drift)
    background.add_task(_run_retrain_job, job.id)
    return {"status": "scheduled", "job_id": job.id, "detail": drift}

# 実行中ジョブが無ければ scheduled状態に変更
@router.post("/retrain/trigger", tags=["mlops"])
def manual_trigger(background: BackgroundTasks, reason: Optional[str] = "manual", db: Session = Depends(get_db)) -> Dict[str, Any]:
    if get_running_job(db):
        return {"status": "skipped", "reason": "already_running"}
    job = create_job(db, triggered_by="manual", reason=reason or "manual", severity=None, metrics=None)
    background.add_task(_run_retrain_job, job.id)
    return {"status": "scheduled", "job_id": job.id}

# retrainの履歴を返す
@router.get("/retrain/history", tags=["mlops"])
def job_history(limit: int = 50, db: Session = Depends(get_db)):
    rows = list_jobs(db, limit=limit)
    return [{"id": r.id, "status": r.status, "triggered_by": r.triggered_by, "reason": r.reason,
             "severity": r.severity, "created_at": r.created_at, "started_at": r.started_at,
             "finished_at": r.finished_at,
             "error": (r.error[:200] + "...") if r.error and len(r.error) > 200 else r.error} for r in rows]
