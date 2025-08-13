# app/logic_scorer.py
from __future__ import annotations
from typing import Dict, Any, List
import os, json, re
from dataclasses import dataclass
import math

_HEDGE_PAT = re.compile(r"(?:多分|たぶん|おそらく|だいたい|と思い?ます|っぽい|みたい|感じ|一旦|とか|など)")
_NUM_VAGUE_PAT = re.compile(r"[0-9０-９]+(?:\s*[％%分件本日月年])?\s*(?:くらい|程度|前後|とか)")
# 環境変数でON/OFF
_BADNESS_ON = os.getenv("LOGIC_BADNESS_PENALTY", "1") not in ("0","false","False")

# ====== 軽い設定 ======
STRUCT_WEIGHTS: Dict[str, int] = {  # total の重み（env で無効化可）
    "clarity": 30,       # 構成の明瞭さ
    "consistency": 25,   # 論理的一貫性
    "cohesion": 20,      # まとまり/結束性
    "density": 15,       # 要点密度（冗長さの反対）
    "cta": 10,           # CTA の明確さ
}
_WEIGHTED_TOTAL = os.getenv("LOGIC_TOTAL_WEIGHTED", "1") not in ("0", "false", "False")

# ====== LLM ラッパ（あれば使う/無ければヒューリ） ======
try:
    from openai import OpenAI
    _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    _OPENAI = True
except Exception:
    _client = None
    _OPENAI = False

def _chat_json(messages, model: str | None = None, temperature: float = 0.1) -> Dict[str, Any]:
    if not _OPENAI:
        return {}
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    try:
        res = _client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=messages,
            response_format={"type": "json_object"},
        )
        return json.loads(res.choices[0].message.content)
    except Exception:
        return {}

# ====== テキスト統計（ヒューリスティクスの材料） ======
def _basic_stats(text: str) -> Dict[str, float]:
    t = text.strip()
    sents = [s for s in re.split(r"[。．!?！？]\s*", t) if s.strip()]
    avg_sent_len = (sum(len(s) for s in sents) / max(1, len(sents))) if sents else len(t)

    # token 多様度
    tokens = re.findall(r"\w+|[^\s\w]", t, flags=re.UNICODE)
    ttr = len(set(tokens)) / max(1, len(tokens))

    # 目次語（signpost）
    signposts = ["まず","次に","一方で","結論として","まとめると","では","ここで","結果",
                 "デモ","アーキテクチャ","課題","解決","効果","最後に"]
    sp_count = sum(t.count(w) for w in signposts)
    sp_density = sp_count / max(1, len(sents))

    # 数値・具体語（根拠っぽさ）
    digit_ratio = len(re.findall(r"[0-9０-９%％¥￥$＄]", t)) / max(1, len(t))
    proper_nouns = re.findall(r"[A-Za-z][A-Za-z0-9\-_/]+|[ァ-ヴー]{3,}", t)  # 製品名/固有名詞・片仮名長め
    proper_ratio = len(proper_nouns) / max(1, len(sents))

    # 重複（n-gram の被り）
    chars = list(t)
    bigrams = [chars[i] + chars[i+1] for i in range(len(chars)-1)] if len(chars) > 1 else []
    dup_ratio = 0.0
    if bigrams:
        uniq = len(set(bigrams))
        dup_ratio = 1.0 - (uniq / len(bigrams))  # 0=多様 / 1=強い重複

    # 長さ（薄い原稿の上振れ抑制）
    char_len = len(t)
    return {
        "avg_sent_len": avg_sent_len, "ttr": ttr,
        "sp_count": sp_count, "sp_density": sp_density,
        "digit_ratio": digit_ratio, "proper_ratio": proper_ratio,
        "dup_ratio": dup_ratio, "char_len": char_len, "sent_count": len(sents)
    }

def _short_text_cap(x: float, stats: Dict[str, float]) -> float:
    """短文や薄い原稿の時に上限をなだらかに抑える（過大評価対策）"""
    L = stats["char_len"]
    if L >= 1200:   # 十分な長さ
        return x
    # 400〜1200 字の間で徐々に天井を上げる
    ceil = 60.0 if L <= 400 else 60 + (L - 400) * (30 / 800)  # 400→60点, 1200→90点
    return min(x, ceil)

# === 低品質検知（悪さだけ取る） =================================
def _badness_features(text: str, bs: dict) -> dict:
    hedges = len(_HEDGE_PAT.findall(text))
    num_vague = len(_NUM_VAGUE_PAT.findall(text))
    # 根拠の薄さ（低いほど悪い）→ 0〜1 を「不足度」に反転
    evidence_lack = max(0.0, 0.4 - bs.get("digit_ratio",0)) * 100 + max(0.0, 0.10 - bs.get("proper_ratio",0)) * 100
    # 重複（0=良い,1=悪い）
    dup = bs.get("dup_ratio", 0) * 100
    return {"hedges": hedges, "num_vague": num_vague, "evidence_lack": evidence_lack, "dup": dup}

def _one_sided_penalty(raw_score: float, bad: dict) -> float:
    """
    低いスコアにだけ強く効く片側減点。
    raw>80 ではほぼ 0、raw が 70→60→50 と下がるほど効く。
    """
    if not _BADNESS_ON:
        return raw_score
    # 最大減点（特徴量ベース）。上限は安全に抑える。
    hard = min(30.0, 4.0*bad["hedges"] + 6.0*bad["num_vague"] + 0.25*bad["evidence_lack"] + 20.0*(bad["dup"]/100))
    # 片側シグモイド係数：raw が 70 未満で効き始め、60 未満でほぼ最大。
    factor = 1.0 / (1.0 + math.exp(-(70.0 - raw_score)/6.0))
    return max(0.0, raw_score - hard*factor)

# ====== Clarity ======
def score_clarity(text: str) -> float:
    sys = "Rate an IT/keynote presentation. Return JSON int 0-100 for structure,specificity,brevity."
    data = _chat_json([{"role":"system","content":sys},{"role":"user","content":f"Text:\n{text}"}])

    def _clamp(v, d=50):
        try: return max(0, min(100, int(v)))
        except: return d

    # LLM 経路でも cap→減点 を必ず通す
    if data:
        structure   = _clamp(data.get("structure", 50))
        specificity = _clamp(data.get("specificity", 50))
        brevity     = _clamp(data.get("brevity", 50))
        raw = (structure + specificity + brevity) / 3
        bs  = _basic_stats(text)
        bad = _badness_features(text, bs)
        return round(_one_sided_penalty(_short_text_cap(raw, bs), bad), 1)

    # ヒューリスティック経路
    bs = _basic_stats(text)
    structure = min(100, 35 + min(bs["sp_count"], 5) * 9 + min(bs["sp_density"]*10, 10))
    brevity   = max(0, min(100, 100 - (bs["avg_sent_len"] / 3)))
    specificity = max(0, min(100, 25 + bs["ttr"]*120 + bs["digit_ratio"]*160 + bs["proper_ratio"]*40 - bs["dup_ratio"]*60))
    raw = (structure + specificity + brevity) / 3
    bad = _badness_features(text, bs)
    return round(_one_sided_penalty(_short_text_cap(raw, bs), bad), 1)

# ====== CTA ======
def score_cta(text: str) -> float:
    sys = ("Detect calls-to-action (CTA) in IT presentations. "
           "Classify as strong/moderate/implicit/none and return JSON "
           'like {"category":"strong","evidence":"..."}')
    data = _chat_json([{"role":"system","content":sys},{"role":"user","content":f"Text:\n{text}"}])

    cat = str(data.get("category", "none")).lower() if data else "none"
    if cat.startswith("strong"):   return 100.0
    if cat.startswith("moderate"): return 70.0
    if cat.startswith("implicit"): return 40.0
    # 以降ヒューリ
    has_timeframe = bool(re.search(r"(今日|今週|来週|今月|本日|締切|まで|[0-9０-９]{1,2}日|[0-9０-９]{1,2}月)", text))
    strong_pat   = r"(今すぐ|いますぐ|登録|申し込|お申込|ダウンロード|購入|参加(?!しませんか)|導入|契約|お問い合わせ|お問合せ|無料(?:登録|トライアル)|デモを(?:予約|申し込)|PoCに参加|スプリント.*導入|試しましょう)"
    moderate_pat = r"(参加しませんか|試しませんか|いかがでしょうか|どうでしょうか|ご検討ください|ご参加ください|お越しください|始めましょう)"
    implicit_pat = r"(提案します|提案したい|検討したい|検討します)"
    score = 0.0
    if re.search(strong_pat, text):     score = 100.0
    elif re.search(moderate_pat, text): score = 70.0
    elif re.search(implicit_pat, text): score = 40.0
    if has_timeframe and 40.0 <= score < 100.0:
        score = 70.0 if score < 70.0 else 100.0
    if score == 0.0 and re.search(r"(してください|お願いします|使って|導入|登録|参加|試して|お問い合わせ)", text):
        score = 70.0
    return score

# ====== Consistency / Cohesion / Density ======
def _ccd_scores(text: str) -> Dict[str, float]:
    sys = "Score an IT presentation. Return JSON int 0-100 for consistency,cohesion,density."
    data = _chat_json([{"role":"system","content":sys},{"role":"user","content":f"Text:\n{text}"}])

    def _get(k, d=50):
        try: return float(max(0, min(100, int(data.get(k, d))))) if data else d
        except: return float(d)

    # LLM 経路でも cap→減点 を必ず通す
    if data:
        bs  = _basic_stats(text)
        bad = _badness_features(text, bs)
        consistency = _one_sided_penalty(_short_text_cap(_get("consistency", 50), bs), bad)
        cohesion    = _one_sided_penalty(_short_text_cap(_get("cohesion",    50), bs), bad)
        density     = _one_sided_penalty(_short_text_cap(_get("density",     50), bs), bad)
        return {"consistency": round(consistency,1), "cohesion": round(cohesion,1), "density": round(density,1)}

    # ヒューリスティック経路
    bs = _basic_stats(text)
    cohesion    = max(0, min(100, 38 + min(bs["sp_count"], 6) * 6 + min(bs["sp_density"]*10, 8) - bs["dup_ratio"]*40))
    consistency = max(0, min(100, 42 + min(bs["sp_count"], 6) * 5 - bs["dup_ratio"]*35))
    density     = max(0, min(100, 32 + (1 / max(1, bs["avg_sent_len"])) * 140 + bs["ttr"]*70
                               + bs["digit_ratio"]*140 + bs["proper_ratio"]*50 - bs["dup_ratio"]*60))
    # 短文抑制 → 片側減点
    cohesion    = _short_text_cap(cohesion, bs)
    consistency = _short_text_cap(consistency, bs)
    density     = _short_text_cap(density, bs)
    bad = _badness_features(text, bs)
    cohesion    = _one_sided_penalty(cohesion, bad)
    consistency = _one_sided_penalty(consistency, bad)
    density     = _one_sided_penalty(density, bad)
    return {"consistency": round(consistency,1), "cohesion": round(cohesion,1), "density": round(density,1)}

def score_consistency(text: str) -> float: return _ccd_scores(text)["consistency"]
def score_cohesion(text: str)    -> float: return _ccd_scores(text)["cohesion"]
def score_density(text: str)     -> float: return _ccd_scores(text)["density"]

# ====== 合計点 ======
def total_score(scores: Dict[str, float]) -> float:
    if not scores: return 0.0
    if _WEIGHTED_TOTAL:
        wsum = float(sum(STRUCT_WEIGHTS.values()))
        total = sum(float(scores.get(k, 0.0)) * (w/wsum) for k, w in STRUCT_WEIGHTS.items())
        return round(total, 1)
    # 互換モード：単純平均
    keys = ["clarity","consistency","cohesion","density","cta"]
    vals = [scores[k] for k in keys if k in scores]
    return round(sum(vals) / len(vals), 1) if vals else 0.0

# ====== 互換：アウトライン/助言（短く高速） ======
def _make_outline(text: str) -> list:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    bullet_pat = re.compile(r"^(\d+[\).\s]|[-•・]|第[一二三四五六七八九十]|まず|次に|一方で|最後に|まとめ|結論)")
    bullets = [l for l in lines if bullet_pat.search(l)]
    if not bullets:
        sents = re.split(r"[。．?!？！]\s*", text)
        kw = ("課題","背景","目的","方針","解決","設計","デモ","結果","効果","次に","最後に","まとめ","結論")
        bullets = [s.strip() for s in sents if any(k in s for k in kw)] or [s.strip() for s in sents if s.strip()]
    seen, out = set(), []
    for b in bullets:
        if b in seen:
            continue
        seen.add(b)
        out.append(b if len(b) <= 120 else b[:117] + "…")
        if len(out) >= 8:
            break
    return out

def _make_advice(text: str, scores: dict) -> list:
    adv = []
    c = scores.get("clarity", 50)
    cons = scores.get("consistency", 50)
    coh = scores.get("cohesion", 50)
    den = scores.get("density", 50)
    cta = scores.get("cta", 0)
    if c   < 65: adv.append("見出し→要点→根拠の順に統一し、各セクションを1〜2文でご要約ください。")
    if den < 65: adv.append("平均文長を短くし、重複表現を削除して情報密度を高めていただけると幸いです。")
    if coh < 65: adv.append("章間のつなぎに『まず/次に/したがって/一方で/結論として』等の接続語をご追加ください。")
    if cons< 65: adv.append("主張とデータ・事例の対応関係を明確化し、矛盾しうる箇所は言い換えをご検討ください。")
    if cta<= 40: adv.append("締めに『次の一歩』（例: PoC参加/お申込み/期日）を一文でご明示ください。")
    if not adv:
        adv.append("各セクション末尾に1行サマリと次アクションを置くと、結論がより伝わりやすくなります。")
    return adv[:5]

class _AnalyzeResult:
    def __init__(self, scores: dict, text: str = ""):
        self.scores = scores
        self.outline = _make_outline(text)
        self.advice = _make_advice(text, scores)  # フロントが参照
        self.text = text

def analyze_structure(text: str, include_aux: bool = False):
    scores = {
        "clarity":      score_clarity(text),
        "consistency":  score_consistency(text),
        "cohesion":     score_cohesion(text),
        "density":      score_density(text),
        "cta":          score_cta(text),
    }
    # 任意のポストキャリブレーション（環境変数 LOGIC_CALIB_POINTS で有効化）
    scores = _calibrate_scores_if_needed(scores)
    scores["total"] = total_score(scores)
    return _AnalyzeResult(scores, text=text)

def combine_structure_total(scores_or_obj) -> float:
    s = getattr(scores_or_obj, "scores", scores_or_obj)
    return total_score(s)

# ====== （任意）点カーブ調整（既存機能そのまま） ======
def _interp_piecewise_linear(x: float, pts):
    try:
        if not pts: return x
        pts = sorted(pts, key=lambda p: p[0])
        if x <= pts[0][0]: return float(pts[0][1])
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            if x <= x2:
                if x2 == x1: return float(y2)
                t = (x - x1) / (x2 - x1)
                return float(y1 + t * (y2 - y1))
        return float(pts[-1][1])
    except Exception:
        return x

def _load_calib_points():
    try:
        raw = os.getenv("LOGIC_CALIB_POINTS", "")
        data = json.loads(raw) if raw else None
        return data if isinstance(data, dict) else None
    except Exception:
        return None

_CALIB_POINTS = _load_calib_points()

def _calibrate_scores_if_needed(scores: dict) -> dict:
    if not _CALIB_POINTS:
        return scores
    out = dict(scores)
    for k in ("clarity","consistency","cohesion","density","cta"):
        v = scores.get(k)
        if v is None: continue
        pts = _CALIB_POINTS.get(k) or _CALIB_POINTS.get("global")
        if pts:
            out[k] = round(_interp_piecewise_linear(float(v), pts), 2)
    return out
