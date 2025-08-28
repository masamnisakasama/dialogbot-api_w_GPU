import os
import pickle
import json
from app import database, models
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional, Dict, Any
from app.models import MLOPSJob

BASELINE_STATS_PATH = "baseline_stats.json"

def get_db_session():
    return database.SessionLocal()

def load_embeddings_from_db():
    db = get_db_session()
    try:
        conversations = db.query(models.Conversation).filter(models.Conversation.embedding != None).all()
        embeddings = [pickle.loads(conv.embedding) for conv in conversations if conv.embedding]
        return embeddings
    finally:
        db.close()

def save_baseline_stats(stats: dict):
    with open(BASELINE_STATS_PATH, "w") as f:
        json.dump(stats, f)

def load_baseline_stats():
    if not os.path.exists(BASELINE_STATS_PATH):
        return None
    with open(BASELINE_STATS_PATH, "r") as f:
        return json.load(f)


RUNNING_STATUSES = {"running"}

def get_running_job(db: Session) -> Optional[MLOPSJob]:
    return db.query(MLOPSJob).filter(MLOPSJob.status.in_(list(RUNNING_STATUSES))).order_by(MLOPSJob.id.desc()).first()

def create_job(db: Session, triggered_by: str, reason: str, severity: Optional[str], metrics: Optional[Dict[str, Any]]) -> MLOPSJob:
    job = MLOPSJob(triggered_by=triggered_by, reason=reason, severity=severity, metrics=metrics or {}, status="scheduled")
    db.add(job); db.commit(); db.refresh(job)
    return job

# バックグラウンド実行開始時
def mark_started(db: Session, job: MLOPSJob):
    job.status = "running"; job.started_at = datetime.utcnow()
    db.commit(); db.refresh(job)

# 学習成功時のメッセージ保存
def mark_success(db: Session, job: MLOPSJob):
    job.status = "success"; job.finished_at = datetime.utcnow()
    db.commit(); db.refresh(job)

# 学習失敗時のメッセージ保存
def mark_failed(db: Session, job: MLOPSJob, error: str):
    job.status = "failed"; job.error = (error or "")[:4000]; job.finished_at = datetime.utcnow()
    db.commit(); db.refresh(job)

def list_jobs(db: Session, limit: int = 50):
    return db.query(MLOPSJob).order_by(MLOPSJob.id.desc()).limit(limit).all()