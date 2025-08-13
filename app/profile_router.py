# app/profile_router.py
from __future__ import annotations
from fastapi import APIRouter, Body, Query
from typing import Dict, Any
from .profile_service import ingest_message, get_snapshot

router = APIRouter(prefix="/profile", tags=["profile"])

@router.post("/ingest")
def ingest(
    user_id: str = Query(...),
    text: str = Body(..., embed=True)
) -> Dict[str, Any]:
    rec = ingest_message(user_id, text)
    return {
        "ok": True,
        "style": rec["style"],
        "mood": rec["mood"],
        "interest": rec["interest"],
        "ts": rec["ts"],
    }

@router.get("/snapshot")
def snapshot(
    user_id: str = Query(...),
    days: int = Query(30)
) -> Dict[str, Any]:
    # App.js は top-level の style/mood/interest を読む
    return get_snapshot(user_id, days)

