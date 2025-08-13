# app/profile_service_fallback.py
from __future__ import annotations
from typing import Dict, Any
import re

POLITE_PATTERNS = [r"です", r"ます", r"でしょうか", r"お願(い|いいたします)", r"恐れ入ります", r"いただけます"]
ABSTRACT_WORDS = ["概念","本質","一般的","抽象","理論","普遍","前提","価値観","哲学","思想","フレーム","認識","モデル"]
CONCRETE_HINTS = [r"\b\d+(\.\d+)?\b", "mm","cm","km","kg","ms","秒","件","個","円","％","%","API","URL","Python","FastAPI","Whisper","GPU","CPU"]
EXPERT_TERMS = ["アルゴリズム","最適化","推論","ベイズ","回帰","埋め込み","クラスタ","トピック","分散","正規化","トークン","API","SDK"]
INTEREST_TECH = ["AI","機械学習","モデル","GPU","API","Python","FastAPI","データ","ベクトル","埋め込み","Git"]
INTEREST_ART  = ["音楽","絵画","デザイン","美術","詩","リズム","メロディ","表現","アート"]
INTEREST_PHIL = ["哲学","倫理","存在","意味","価値","意識","思考","認識","カント","ニーチェ","現象学"]

def _clip(x: float) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    return 0.0 if v < 0 else 1.0 if v > 1 else v

def _re_count(ps, t: str) -> int:
    s = 0
    for p in ps:
        try: s += len(re.findall(p, t))
        except Exception: pass
    return s

def quick_style_mood_interest_from_text(text: str) -> Dict[str, Dict[str, float]]:
    txt = text or ""
    polite = _clip(_re_count(POLITE_PATTERNS, txt) / max(1, len(txt)/20))
    abstract_hits = sum(1 for w in ABSTRACT_WORDS if w in txt)
    concrete_hits = _re_count(CONCRETE_HINTS, txt)
    abstract = _clip( abstract_hits / max(1, abstract_hits + concrete_hits) )
    concise = _clip(1.2 - _clip(len(txt)/200.0))
    expert  = _clip( sum(1 for w in EXPERT_TERMS if w in txt) / 5.0 )

    pos = _clip( sum(1 for w in ["良い","すごい","最高","嬉しい","助かる","ありがたい","素晴らしい"] if w in txt) / 3.0 )
    neg = _clip( sum(1 for w in ["悪い","最悪","困る","厳しい","問題","無理","嫌だ"] if w in txt) / 3.0 )
    arousal = _clip( (txt.count("！")+txt.count("!")+txt.count("？")+txt.count("?"))/6.0 )

    tech  = sum(1 for w in INTEREST_TECH if w in txt)
    art   = sum(1 for w in INTEREST_ART  if w in txt)
    philo = sum(1 for w in INTEREST_PHIL if w in txt)
    s = max(1, tech+art+philo)
    return {
        "style":    {"polite":polite, "abstract":abstract, "concise":concise, "expert":expert},
        "mood":     {"pos":pos, "neg":neg, "arousal":arousal},
        "interest": {"tech": tech/s, "art": art/s, "philo": philo/s},
    }
