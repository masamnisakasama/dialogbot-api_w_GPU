# app/logic_router.py
from fastapi import APIRouter, Body
from app.logic_scorer import analyze_structure, combine_structure_total

router = APIRouter()

@router.post("/analyze-logic")
def analyze_logic(payload: dict = Body(...)):
    """
    入力:  { "text": "<ASRテキスト>" }
    出力:  {
      "scores": {clarity, consistency, cohesion, density, cta},
      "total": float,
      "outline": [...],
      "advice": [...]
    }
    """
    text = (payload or {}).get("text") or ""
    res = analyze_structure(text)
    return {
        "scores": res.scores,
        "total": combine_structure_total(res.scores),
        "outline": res.outline,
        "advice": res.advice,
    }
