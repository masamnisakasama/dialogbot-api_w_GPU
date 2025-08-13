#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YouTube 日本語プレゼンの字幕 -> /analyze-logic 採点 -> CSV/JSONL 保存

- 入力: urls.txt（1行1URL or 動画ID）
- 字幕取得: yt-dlp（手動字幕→自動字幕の順、日本語優先。必要なら英語も可）
- 整形: 日本語字幕のノイズ除去＆句読点補完で LLM 採点の安定性向上
- 出力: <out-prefix>.scored.csv / <out-prefix>.scored.jsonl

使い方（まず1本で動作確認）:
  printf "https://www.youtube.com/watch?v=WTjR15oL8iY\n" > urls.txt
  python -m app.ja_collect_and_score --api-base http://127.0.0.1:8015 \
    --infile urls.txt --out-prefix ja_one --min_chars 150 --allow_en --debug
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Optional, Tuple, List, Dict
from urllib.parse import urlparse, parse_qs

import requests
import pandas as pd
from tqdm import tqdm


# -----------------------
# URL/ID ユーティリティ
# -----------------------
def extract_video_id(url_or_id: str) -> Optional[str]:
    s = url_or_id.strip()
    # 素のID（11文字想定だが短いものもあるためゆるく）
    if re.fullmatch(r"[A-Za-z0-9_-]{6,}", s):
        return s
    try:
        u = urlparse(s)
        if u.hostname in ("youtu.be",):
            return u.path.lstrip("/")
        if u.hostname and "youtube.com" in u.hostname:
            qs = parse_qs(u.query)
            if "v" in qs and qs["v"]:
                return qs["v"][0]
            m = re.search(r"/(shorts|live)/([A-Za-z0-9_-]{6,})", u.path)
            if m:
                return m.group(2)
    except Exception:
        pass
    return None


# -----------------------
# 字幕（VTT/SRV3）→ テキスト
# -----------------------
def vtt_to_text(vtt: str) -> str:
    lines: List[str] = []
    for line in vtt.splitlines():
        if not line:
            continue
        if line.startswith("WEBVTT"):
            continue
        if re.match(r"^\d+$", line.strip()):
            continue
        if re.match(r"^\d{2}:\d{2}:\d{2}\.\d{3}\s+-->\s+\d{2}:\d{2}:\d{2}\.\d{3}", line):
            continue
        # <c>タグなどの簡易除去
        line = re.sub(r"<[^>]+>", "", line)
        lines.append(line.strip())
    text = re.sub(r"\s*\n\s*", "\n", "\n".join([l for l in lines if l]))
    return text


def srv3_to_text(xml: str) -> str:
    # YouTubeのsrv3はXML (<text>…)。タグ除去＆結合
    xml = re.sub(r"<br\s*/?>", "\n", xml)
    texts = re.findall(r"<text[^>]*>(.*?)</text>", xml, flags=re.S)
    import html
    texts = [html.unescape(re.sub(r"<[^>]+>", "", t)).strip() for t in texts]
    joined = "\n".join([t for t in texts if t])
    return joined


def clean_caption_jp(text: str) -> str:
    """日本語字幕のノイズを除去して、文として読みやすく整える（LLM採点の安定化）"""
    # よくあるノイズ
    text = re.sub(r"\[(拍手|音楽|笑い|Music|Applause)\]", "", text)
    text = re.sub(r"（(拍手|音楽|笑い)）", "", text)
    text = text.replace("♪", "").replace("■", "")
    # 行番号取りこぼし
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

    # 行末に句点が無ければ「。」を補う（日本語っぽい行のみ）
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    fixed: List[str] = []
    for l in lines:
        if re.search(r"[。！？!?]$", l) or len(l) < 8:
            fixed.append(l)
        else:
            if re.search(r"[\u3040-\u30FF\u4E00-\u9FFF]", l):  # ひらがな/カタカナ/漢字
                fixed.append(l + "。")
            else:
                fixed.append(l)
    out = "".join(fixed)

    # 句読点/空白整形
    out = re.sub(r"([。！？!?])\s*", r"\1", out)
    out = re.sub(r"(。){3,}", "。。", out)
    return out.strip()


# -----------------------
# yt-dlp で字幕を取得（日本語優先 → 英語、最長トラックを採用）
# -----------------------
def get_transcript_any(video_id: str, allow_en: bool, debug: bool = False):
    import yt_dlp

    prefer_ja = ["ja", "ja-orig", "ja-Hans", "ja-Hant", "ja-JP"]
    prefer = prefer_ja + (["en"] if allow_en else [])

    def fetch_both_parsers(url: str) -> tuple[str, int]:
        try:
            r = requests.get(url, timeout=25)
            raw = r.text
        except Exception:
            return "", 0
        # 1) VTTとして
        txt_vtt = vtt_to_text(raw) if raw.lstrip().startswith("WEBVTT") else ""
        # 2) SRV3として（XMLっぽければ）
        txt_srv = srv3_to_text(raw) if raw.lstrip().startswith("<") else ""
        # 3) どちらでもなさそうでも両方試す
        if not txt_vtt:
            txt_vtt = vtt_to_text(raw)
        if not txt_srv and raw.lstrip().startswith("<"):
            txt_srv = srv3_to_text(raw)
        # 長い方を採用
        cand = max((txt_vtt, len(txt_vtt)), (txt_srv, len(txt_srv)), key=lambda x: x[1])
        text = clean_caption_jp(cand[0])
        return text, len(text)

    def best_text_from_tracks(tracks: list[dict]) -> tuple[str, int]:
        best_txt, best_len = "", 0
        for e in tracks or []:
            url = e.get("url")
            if not url:
                continue
            txt, L = fetch_both_parsers(url)
            if L > best_len:
                best_txt, best_len = txt, L
        return best_txt, best_len

    ydl_opts = {"quiet": True, "skip_download": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)

    subs = info.get("subtitles") or {}
    caps = info.get("automatic_captions") or {}
    meta = {"provider": "yt-dlp", "available": sorted(set(list(subs.keys()) + list(caps.keys())))}

    # 1) 手動字幕：希望言語順で最長を選ぶ
    for code in prefer:
        best_txt, best_len = best_text_from_tracks(subs.get(code))
        if best_len > 0:
            if debug: print(f"[ytdlp] manual {code}: {best_len} chars")
            return best_txt, code, meta

    # 2) 自動字幕：希望言語順で最長を選ぶ
    for code in prefer:
        best_txt, best_len = best_text_from_tracks(caps.get(code))
        if best_len > 0:
            if debug: print(f"[ytdlp] auto {code}: {best_len} chars")
            return best_txt, code, meta

    if debug: print(f"[ytdlp] no captions usable for {video_id}")
    return None, None, meta


    def fetch_caption_text(entry) -> tuple[str, int]:
        """URLからテキストを取得して (text, length) を返す。拡張子でパーサ切替。"""
        url = entry.get("url")
        if not url:
            return "", 0
        try:
            resp = requests.get(url, timeout=25)
            raw = resp.text
            # extヒント（yt-dlpエントリに "ext" が入っていることが多い）
            ext = (entry.get("ext") or "").lower()
            if ext == "vtt" or raw.lstrip().startswith("WEBVTT"):
                txt = vtt_to_text(raw)
            elif ext == "srv3" or raw.lstrip().startswith("<"):
                txt = srv3_to_text(raw)
            else:
                # ダメ元でVTTとして処理
                txt = vtt_to_text(raw)
            txt = clean_caption_jp(txt)
            return txt, len(txt)
        except Exception:
            return "", 0

    ydl_opts = {"quiet": True, "skip_download": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)

    subs = info.get("subtitles") or {}
    caps = info.get("automatic_captions") or {}
    meta = {"provider": "yt-dlp", "available": sorted(set(list(subs.keys()) + list(caps.keys())))}

    # 1) 手動字幕から、希望言語ごとに「最長」を選ぶ
    for code in prefer:
        entries = subs.get(code) or []
        best_txt, best_len = "", 0
        for e in entries:
            txt, L = fetch_caption_text(e)
            if L > best_len:
                best_txt, best_len = txt, L
        if best_len > 0:
            if debug: print(f"[ytdlp] manual {code}: {best_len} chars")
            return best_txt, code, meta

    # 2) 自動字幕から、希望言語ごとに「最長」を選ぶ
    for code in prefer:
        entries = caps.get(code) or []
        best_txt, best_len = "", 0
        for e in entries:
            txt, L = fetch_caption_text(e)
            if L > best_len:
                best_txt, best_len = txt, L
        if best_len > 0:
            if debug: print(f"[ytdlp] auto {code}: {best_len} chars")
            return best_txt, code, meta

    if debug: print(f"[ytdlp] no captions usable for {video_id}")
    return None, None, meta


# -----------------------
# 採点API
# -----------------------
def score_text_via_api(text: str, api_base: str) -> Optional[dict]:
    try:
        r = requests.post(
            f"{api_base.rstrip('/')}/analyze-logic",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"text": text[:12000]}),  # 念のため制限
            timeout=60,
        )
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


# -----------------------
# メイン
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-base", default="http://127.0.0.1:8015")
    ap.add_argument("--infile", default="urls.txt", help="YouTube URL/ID を1行ずつ")
    ap.add_argument("--out-prefix", default="ja_talks")
    ap.add_argument("--min_chars", type=int, default=200, help="短すぎる字幕を除外する閾値（文字数）")
    ap.add_argument("--allow_en", action="store_true", help="日本語が無い場合は英語字幕でも採点")
    ap.add_argument("--prime", default="以下は日本語のプレゼン書き起こしです。導入→主張→根拠→まとめ→CTAの観点で評価してください。\n---\n",
                    help="採点前に本文の先頭へ付与するプロンプト（空文字で無効化）")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    # 入力読み込み
    with open(args.infile, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]

    rows: List[Dict] = []
    details: List[Dict] = []

    for url in tqdm(urls, desc="fetch+score"):
        vid = extract_video_id(url)
        if not vid:
            if args.debug:
                print(f"[skip] not a YouTube URL/ID: {url}")
            continue

        # 字幕取得
        text, lang, meta = get_transcript_any(vid, allow_en=args.allow_en, debug=args.debug)
        if args.debug:
            print(f"[debug] {vid}: lang={lang}, len(text)={len(text) if text else 0}, meta={meta}")

        if not text or len(text) < args.min_chars:
            if args.debug:
                print(f"[skip] no/short transcript: {vid} chars={len(text) if text else 0}")
            continue

        # 採点（プライム文を先頭に付与）
        payload = (args.prime or "") + text
        if args.debug:
            print(f"[debug] scoring {vid}: {len(payload)} chars -> {args.api_base}/analyze-logic")
        res = score_text_via_api(payload, args.api_base)
        if not res:
            if args.debug:
                print(f"[skip] scoring failed: {vid}")
            continue

        # タイトル（任意）
        title = f"video:{vid}"
        try:
            import yt_dlp
            with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True}) as ydl:
                info = ydl.extract_info(f"https://www.youtube.com/watch?v={vid}", download=False)
                title = info.get("title") or title
        except Exception:
            pass

        rows.append({
            "title": title,
            "url": f"https://www.youtube.com/watch?v={vid}",
            "lang": lang,
            "transcript_chars": len(text),
            "clarity": res.get("scores", {}).get("clarity"),
            "consistency": res.get("scores", {}).get("consistency"),
            "cohesion": res.get("scores", {}).get("cohesion"),
            "density": res.get("scores", {}).get("density"),
            "cta": res.get("scores", {}).get("cta"),
            "total": res.get("total"),
        })
        details.append({
            "title": title,
            "url": f"https://www.youtube.com/watch?v={vid}",
            "lang": lang,
            "text": text,
            **res
        })

    if not rows:
        print("No talks collected. ヒント: --allow_en を付ける / --min_chars を下げる / --debug で原因を確認")
        return

    # 保存
    df = pd.DataFrame(rows).sort_values("total", ascending=False)
    csv_path = f"{args.out_prefix}.scored.csv"
    jsonl_path = f"{args.out_prefix}.scored.jsonl"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"Saved: {csv_path} ({len(df)} rows)")
    print(f"Saved: {jsonl_path}")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
