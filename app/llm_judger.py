from __future__ import annotations
import os, json, time
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

# === App.js と 1:1 のキー ===
STYLE_KEYS: List[str] = [
    "polite","friendly","assertive","empathetic","formal","casual",
    "abstract","concrete","concise","verbose","expert","explanatory",
    "humorous","persuasive",
]
MOOD_KEYS: List[str] = [
    "pos","neg","arousal","calm","excited","confident","anxious",
    "frustrated","satisfied","curious",
]
INTEREST_KEYS: List[str] = [
    "tech","science","art","design","philo","business","finance","history",
    "literature","education","health","sports","entertain","travel","food","gaming",
]

OPENAI_MODEL = os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT_SEC", "20"))

SYSTEM_PROMPT = (
    "あなたは会話テキストから『スタイル/ムード/興味』を0.0〜1.0で採点する評価器です。"
    "複数が同時に高くても良い。文脈と語気を重視して判定し、JSONのみを返してください。"
)
USER_PROMPT = (
    "【Transcript】\n{t}\n\n"
    "次のキーに0.0〜1.0でスコアを付けてJSONで返してください。\n"
    "- style: polite, friendly, assertive, empathetic, formal, casual, abstract, concrete, "
    "concise, verbose, expert, explanatory, humorous, persuasive\n"
    "- mood:  pos, neg, arousal, calm, excited, confident, anxious, frustrated, satisfied, curious\n"
    "- interest: tech, science, art, design, philo, business, finance, history, literature, education, "
    "health, sports, entertain, travel, food, gaming\n"
    "正規化は不要です。"
)

@dataclass
class JudgeResult:
    style: Dict[str, float]
    mood: Dict[str, float]
    interest: Dict[str, float]

def _clip01(x: float) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    return 0.0 if v < 0 else 1.0 if v > 1 else v

def _num_props(keys: List[str]) -> Dict[str, Any]:
    return {k: {"type": "number", "minimum": 0, "maximum": 1} for k in keys}

def _json_schema() -> Dict[str, Any]:
    return {
        "name": "ConversationPreferenceScores",
        "schema": {
            "type": "object",
            "properties": {
                "style": {"type": "object", "properties": _num_props(STYLE_KEYS)},
                "mood": {"type": "object", "properties": _num_props(MOOD_KEYS)},
                "interest": {"type": "object", "properties": _num_props(INTEREST_KEYS)},
            },
            "required": ["style","mood","interest"],
            "additionalProperties": False,
            "strict": True,
        },
    }

def _parse_json_loose(s: str) -> Optional[Dict[str, Any]]:
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        s2 = s.strip().strip("`")
        L, R = s2.find("{"), s2.rfind("}")
        if L >= 0 and R > L:
            try:
                return json.loads(s2[L:R+1])
            except Exception:
                return None
        return None

def _safe_result(d: Dict[str, Any]) -> JudgeResult:
    s = d.get("style", {}) or {}
    m = d.get("mood", {}) or {}
    i = d.get("interest", {}) or {}
    return JudgeResult(
        style={k: _clip01(s.get(k, 0)) for k in STYLE_KEYS},
        mood={k: _clip01(m.get(k, 0)) for k in MOOD_KEYS},
        interest={k: _clip01(i.get(k, 0)) for k in INTEREST_KEYS},
    )

def _call_openai(messages: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
    if not OPENAI_API_KEY:
        return None
    # 新SDK（openai>=1.x）
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            response_format={"type": "json_schema", "json_schema": _json_schema()},
            temperature=0,
            timeout=OPENAI_TIMEOUT,
        )
        return _parse_json_loose(resp.choices[0].message.content or "")
    except Exception:
        pass
    # 旧SDK（openai<=0.x）
    try:
        import openai  # type: ignore
        openai.api_key = OPENAI_API_KEY
        resp = openai.ChatCompletion.create(
            model=OPENAI_MODEL, messages=messages, temperature=0, timeout=OPENAI_TIMEOUT
        )
        return _parse_json_loose(resp["choices"][0]["message"]["content"])
    except Exception:
        return None

def _heuristics(text: str) -> JudgeResult:
    t = (text or "").lower()
    def has(*ws): return any(w.lower() in t for w in ws)

    style = {k:0.0 for k in STYLE_KEYS}
    mood  = {k:0.0 for k in MOOD_KEYS}
    intr  = {k:0.0 for k in INTEREST_KEYS}

    style["polite"] += 0.8 if has("お願いします","です","ます") else 0.2
    style["friendly"] += 0.6 if has("！","〜","ね","よ","w","笑") else 0.2
    style["assertive"] += 0.6 if has("べき","必ず","断言") else 0.2
    style["empathetic"] += 0.6 if has("大変","お気持ち","わかります","つらい") else 0.2
    style["formal"] += 0.6 if has("致します","存じます") else 0.2
    style["casual"] += 0.6 if has("だよ","かな","ねぇ","w","笑") else 0.2
    style["abstract"] += 0.6 if has("理念","概念","本質","目的") else 0.2
    style["concrete"] += 0.6 if has("例えば","具体","数値","手順") else 0.2
    style["concise"] += 0.7 if len(t) < 80 else 0.2
    style["verbose"] += 0.7 if len(t) > 240 else 0.2
    style["expert"] += 0.7 if has("アルゴリズム","api","推論","埋め込み","ベクトル") else 0.2
    style["explanatory"] += 0.7 if has("つまり","要するに","理由は") else 0.2
    style["humorous"] += 0.6 if has("w","笑","😂","😆") else 0.1
    style["persuasive"] += 0.6 if has("おすすめ","ぜひ","ご検討") else 0.2

    mood["pos"] += 0.7 if has("ありがとう","助かる","良い","最高") else 0.3
    mood["neg"] += 0.7 if has("遅い","困る","最悪","バグ","エラー") else 0.3
    exclam = text.count("！") + text.count("!")
    mood["arousal"] += min(1.0, 0.2 + 0.15*exclam)
    mood["calm"] += 0.6 if has("落ち着いて","ゆっくり","丁寧に") else 0.3
    mood["excited"] += 0.6 if exclam >= 2 else 0.2
    mood["confident"] += 0.6 if has("できます","確実","問題ありません") else 0.3
    mood["anxious"] += 0.6 if has("不安","心配","焦る") else 0.2
    mood["frustrated"] += 0.6 if has("苛立ち","イライラ","もうやだ") else 0.2
    mood["satisfied"] += 0.6 if has("満足","助かった","解決") else 0.2
    mood["curious"] += 0.6 if has("なぜ","どうして","気になる") else 0.2

    def bump(keys, s):
        for k in keys: intr[k] += s
    if has("api","モデル","デプロイ","ベクトル","推論","python","swift","react"): bump(["tech","science"], 0.6)
    if has("美術","色彩","構図","作品","表現"): bump(["art","design"], 0.6)
    if has("哲学","倫理","本質","意味"): bump(["philo"], 0.7)
    if has("事業","市場","売上","利益","kpi"): bump(["business","finance"], 0.6)
    if has("歴史","戦争","時代","文化"): bump(["history","literature"], 0.6)
    if has("教育","学習","授業","教材"): bump(["education"], 0.6)
    if has("健康","睡眠","運動","食事"): bump(["health","food","sports"], 0.6)
    if has("映画","ドラマ","音楽","ゲーム","旅行"): bump(["entertain","gaming","travel"], 0.6)

    return JudgeResult(
        style={k:_clip01(v) for k,v in style.items()},
        mood={k:_clip01(v) for k,v in mood.items()},
        interest={k:_clip01(v) for k,v in intr.items()},
    )

def judge_with_openai(text: str) -> JudgeResult:
    text = (text or "").strip()
    if not text:
        return _heuristics("")
    messages = [
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":USER_PROMPT.format(t=text[:8000])},
    ]
    for attempt in range(3):
        data = _call_openai(messages)
        if data:
            try:
                return _safe_result(data)
            except Exception:
                pass
        time.sleep(0.4 * (attempt + 1))
    return _heuristics(text)

def judge_text_dict(text: str) -> Dict[str, Dict[str, float]]:
    r = judge_with_openai(text)
    return {"style": r.style, "mood": r.mood, "interest": r.interest}
