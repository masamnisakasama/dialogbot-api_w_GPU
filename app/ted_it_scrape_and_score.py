#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TED から IT系トークのスクリプトを収集し、/analyze-logic で採点して保存。

APIスクレイピングしてみたけど日本語ないし仕様変わりがちからダメそう　Youtubeの方が良いね

- 取得モード:
  - sitemap: TEDのサイトマップ(XML/ XML.GZ)から /talks/<slug> を抽出（推奨）
  - search : 検索ページ（簡易。取りこぼしあり得る）
  - slugs  : 引数で与えた slug 群を直接処理（動作確認用）

注意: 利用規約/robots を尊重。研究/社内評価目的を想定。
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# --------------------
# 定数
# --------------------
HEADERS = {
    "User-Agent": "DialogBot-Research/1.0 (+https://example.com; contact: research@example.com)"
}
SEARCH_URL = "https://www.ted.com/talks"
TRANSCRIPT_URL = "https://www.ted.com/talks/{slug}/transcript"
TALK_URL = "https://www.ted.com/talks/{slug}"
SITEMAP_INDEX = "https://www.ted.com/sitemap.xml"

IT_KEYWORDS = [
    "tech", "technology", "software", "developer", "program", "programming",
    "computer", "computing", "ai", "machine learning", "data", "cloud",
    "security", "internet", "web", "hardware", "robot", "cyber", "devops",
    "platform", "saas", "api"
]

# --------------------
# 便利関数
# --------------------
def sleep_polite(sec: float = 1.0) -> None:
    time.sleep(sec)

def get_bytes(url: str, params: Dict | None = None, retries: int = 3, sleep: float = 1.2) -> Optional[bytes]:
    for i in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=25)
            if r.status_code == 200:
                return r.content
            if r.status_code in (429, 500, 502, 503, 504):
                sleep_polite(sleep * (i + 1))
                continue
            return None
        except requests.RequestException:
            sleep_polite(sleep * (i + 1))
    return None

def get_text(url: str, params: Dict | None = None, retries: int = 3, sleep: float = 1.2) -> Optional[str]:
    data = get_bytes(url, params=params, retries=retries, sleep=sleep)
    if data is None:
        return None
    # .gz 対応
    if url.endswith(".gz"):
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(data)) as gz:
                return gz.read().decode("utf-8", errors="replace")
        except OSError:
            # サーバが Content-Encoding:gzip で返してくれて解凍済みの可能性
            return data.decode("utf-8", errors="replace")
    return data.decode("utf-8", errors="replace")

def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def looks_it_related(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in IT_KEYWORDS)

# --------------------
# データ構造
# --------------------
@dataclass
class Talk:
    title: str
    speaker: str
    url: str
    slug: str
    language: str
    tags: List[str]
    transcript: str

@dataclass
class ScoredTalk(Talk):
    scores: Dict[str, float]
    total: float
    outline: List[str]
    advice: List[str]
    transcript_chars: int

# --------------------
# スラグ取得
# --------------------
def load_slugs_from_sitemap(max_sitemaps: int = 4, debug: bool = False) -> List[Tuple[str, str]]:
    """
    サイトマップ index → talks系サブサイトマップ(.xml / .xml.gz) → /talks/<slug> URL抽出
    """
    slugs: Set[Tuple[str, str]] = set()

    # 1) index
    idx = get_text(SITEMAP_INDEX)
    if not idx:
        if debug: print("[warn] sitemap index fetch failed")
        return []
    root = ET.fromstring(idx)
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    locs = [loc.text for loc in root.findall(".//sm:loc", ns) if loc is not None]

    # talks 系だけ
    talk_maps = [u for u in locs if "/talks" in u]
    if debug: print(f"[debug] talks sitemaps in index: {len(talk_maps)}")
    if not talk_maps:
        return []

    # 2) サブサイトマップを順に
    for sm_url in talk_maps[:max_sitemaps]:
        xml_text = get_text(sm_url)
        if not xml_text:
            if debug: print(f"[warn] skip (fetch failed): {sm_url}")
            continue
        try:
            root2 = ET.fromstring(xml_text)
        except ET.ParseError as e:
            if debug: print(f"[warn] parse error on {sm_url}: {e}")
            continue

        for loc in root2.findall(".//sm:loc", ns):
            url = (loc.text or "").strip()
            m = re.match(r"^https?://www\.ted\.com/talks/([a-zA-Z0-9_\-]+)$", url)
            if m:
                slug = m.group(1)
                slugs.add((slug, url))
        if debug: print(f"[debug] {sm_url} -> cum slugs: {len(slugs)}")
        sleep_polite(0.8)

    return list(slugs)

def search_talk_slugs(query: str, pages: int = 1, debug: bool = False) -> List[Tuple[str, str]]:
    """
    検索ページから <a href="/talks/<slug>"> を拾う（動的要素で取りこぼしあり得る）
    """
    found: Set[Tuple[str, str]] = set()
    for page in range(1, pages + 1):
        html = get_text(SEARCH_URL, params={"q": query, "page": page})
        if not html:
            if debug: print(f"[warn] search fetch failed: q={query} page={page}")
            continue
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            m = re.match(r"^/talks/([a-zA-Z0-9_\-]+)$", href)
            if m:
                slug = m.group(1)
                found.add((slug, f"https://www.ted.com{href}"))
        if debug: print(f"[debug] search '{query}' page={page}: cum slugs={len(found)}")
        sleep_polite(0.6)
    return list(found)

# --------------------
# ページ取得
# --------------------
def fetch_transcript(slug: str, lang: str, debug: bool = False) -> Optional[str]:
    html = get_text(TRANSCRIPT_URL.format(slug=slug), params={"language": lang})
    if not html:
        if debug: print(f"[warn] transcript fetch failed: slug={slug}, lang={lang}")
        return None
    soup = BeautifulSoup(html, "html.parser")

    # 新レイアウト: data-qa
    paras = soup.find_all(attrs={"data-qa": "talk-transcript__para"})
    # フォールバック（旧）
    if not paras:
        paras = soup.select("div.Grid__cell p, div.talk-transcript__para p")

    text = "\n".join(
        norm_ws(p.get_text(" ", strip=True))
        for p in paras
        if p.get_text(strip=True)
    )
    return text or None

def fetch_title_speaker_tags(slug: str, debug: bool = False) -> Tuple[str, str, List[str]]:
    html = get_text(TALK_URL.format(slug=slug))
    if not html:
        if debug: print(f"[warn] talk page fetch failed: slug={slug}")
        return "", "", []
    soup = BeautifulSoup(html, "html.parser")
    # タイトル
    title = ""
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        title = og["content"]
    else:
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(strip=True)
    # スピーカー
    speaker = ""
    sp = soup.select_one('[data-qa="talk-speaker__name"]') or soup.select_one('a[href^="/speakers/"]')
    if sp: speaker = sp.get_text(strip=True)
    # タグ
    tags: List[str] = []
    for a in soup.select('a[href^="/topics/"]'):
        lab = a.get_text(strip=True)
        if lab and lab not in tags:
            tags.append(lab)
    return norm_ws(title), norm_ws(speaker), tags

# --------------------
# 採点
# --------------------
def score_text_via_api(text: str, api_base: str) -> Optional[Dict]:
    try:
        r = requests.post(
            f"{api_base.rstrip('/')}/analyze-logic",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"text": text[:8000]}),
            timeout=40,
        )
        if r.status_code != 200:
            return None
        return r.json()
    except requests.RequestException:
        return None

def fallback_score(text: str) -> Dict:
    sents = [t for t in re.split(r"[。.!?]\s*", text) if t.strip()]
    has_intro = any(re.search(r"(まず|本日|目的|概要)", s) for s in sents[:3])
    has_claim = any(re.search(r"(結論|要点|主張|ポイント)", s) for s in sents)
    has_reason = any(re.search(r"(理由|根拠|だから|そのため)", s) for s in sents)
    has_sum = any(re.search(r"(まとめ|以上|最後に)", s) for s in sents[-3:])
    has_cta = any(re.search(r"(お願いします|ご検討|お申込み|次に|お問い合わせ|行動|参加|購入)", s) for s in sents[-4:])
    clarity = (has_intro + has_claim + has_reason + has_sum) / 4 * 100
    consistency = (80 if (has_claim and has_reason) else 40) + (20 if re.search(r"(だから|そのため)", text) else 0)
    cohesion = 60 + min(text.count("そして")+text.count("しかし"), 8) * 5
    density = 60 + min(text.count("ポイント")+text.count("要点"), 6) * 6
    cta = 100 if has_cta else (30 if has_sum else 0)
    scores = {
        "clarity": round(min(max(clarity, 0), 100), 1),
        "consistency": round(min(max(consistency, 0), 100), 1),
        "cohesion": round(min(max(cohesion, 0), 100), 1),
        "density": round(min(max(density, 0), 100), 1),
        "cta": round(min(max(cta, 0), 100), 1),
    }
    w = {"clarity":30,"consistency":25,"cohesion":20,"density":15,"cta":10}
    total = sum(scores[k]*w[k] for k in w)/sum(w.values())
    return {"scores":scores,"total":round(total,1),"outline":[],"advice":[]}

# --------------------
# メイン
# --------------------
def main():
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--api-base", default="http://127.0.0.1:8015")
    ap.add_argument("--mode", choices=["sitemap","search"], default="sitemap")
    ap.add_argument("--sitemap-count", type=int, default=4)
    ap.add_argument("--queries", nargs="+", default=["technology","software","AI","data"])
    ap.add_argument("--pages", type=int, default=1)
    ap.add_argument("--max-talks", type=int, default=60)
    ap.add_argument("--lang", nargs="+", default=["en","ja"])
    ap.add_argument("--out-prefix", default="ted_it")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--slugs", nargs="*", default=None)
    ap.add_argument("--no-it-filter", action="store_true")
    args = ap.parse_args()

    # 1) スラグ一覧
    slugs: List[Tuple[str, str]] = []
    if args.slugs:
        for s in args.slugs:
            slugs.append((s, TALK_URL.format(slug=s)))
    elif args.mode == "sitemap":
        slugs = load_slugs_from_sitemap(max_sitemaps=args.sitemap_count, debug=args.debug)
    else:  # search
        seen: Set[Tuple[str, str]] = set()
        for q in tqdm(args.queries, desc="search queries"):
            pairs = search_talk_slugs(q, pages=args.pages, debug=args.debug)
            seen.update(pairs)
            sleep_polite(0.6)
        slugs = list(seen)

    print(f"Found {len(slugs)} candidate talks (before filters)")

    if args.max_talks and len(slugs) > args.max_talks:
        slugs = slugs[:args.max_talks]

    # 2) 取得 + ITフィルタ + 採点
    results: List[ScoredTalk] = []
    for slug, url in tqdm(slugs, desc="fetch+score"):
        title, speaker, tags = fetch_title_speaker_tags(slug, debug=args.debug)
        meta_text = " ".join([title, speaker, " ".join(tags)]).lower()

        if not args.no_it_filter and not looks_it_related(meta_text):
            continue

        transcript = None
        lang_used = None
        for lg in args.lang:
            transcript = fetch_transcript(slug, lg, debug=args.debug)
            if transcript:
                lang_used = lg
                break
        if not transcript:
            if args.debug: print(f"[skip] no transcript: {slug}")
            continue

        scored = score_text_via_api(transcript, args.api_base) or fallback_score(transcript)

        results.append(
            ScoredTalk(
                title=title or "",
                speaker=speaker or "",
                url=url,
                slug=slug,
                language=lang_used or (args.lang[0] if args.lang else "en"),
                tags=tags,
                transcript=transcript,
                scores=scored["scores"],
                total=scored["total"],
                outline=scored.get("outline", []),
                advice=scored.get("advice", []),
                transcript_chars=len(transcript),
            )
        )
        sleep_polite(0.9)

    # 3) 保存
    rows = []
    for r in results:
        rows.append({
            "title": r.title,
            "speaker": r.speaker,
            "url": r.url,
            "slug": r.slug,
            "language": r.language,
            "tags": ", ".join(r.tags),
            "transcript_chars": r.transcript_chars,
            "clarity": r.scores.get("clarity"),
            "consistency": r.scores.get("consistency"),
            "cohesion": r.scores.get("cohesion"),
            "density": r.scores.get("density"),
            "cta": r.scores.get("cta"),
            "total": r.total,
        })

    csv_path = f"{args.out_prefix}.scored.csv"
    jsonl_path = f"{args.out_prefix}.scored.jsonl"

    df = pd.DataFrame(rows)
    if len(df) == 0:
        print("No talks collected after filtering. Tips: add --no-it-filter, increase --sitemap-count, or test with --slugs <slug>.")
        df.to_csv(csv_path, index=False, encoding="utf-8")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            pass
        return

    df = df.sort_values("total", ascending=False)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    print(f"\nSaved: {csv_path} ({len(df)} rows)")
    print(f"Saved: {jsonl_path}")
    print("\nTop 5 (by total):")
    print(df[["title","speaker","total","clarity","consistency","cohesion","density","cta"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
