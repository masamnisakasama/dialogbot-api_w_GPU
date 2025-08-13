#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YouTube -> 音声DL(yt-dlp/m4a) -> ffmpeg で冒頭スキップ/長さ制限 -> /stt-full でSTT
（失敗時 /analyze/audio にフォールバック）-> /analyze-logic で採点
出力: <out-prefix>.stt.scored.csv / .jsonl

依存: pip install yt-dlp pandas tqdm requests
ffmpeg 必須（mac: brew install ffmpeg）
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from typing import Optional, List, Dict
from urllib.parse import urlparse, parse_qs

import pandas as pd
import requests
from tqdm import tqdm


# -----------------------
# URL/ID ユーティリティ
# -----------------------
def extract_video_id(url_or_id: str) -> Optional[str]:
    s = url_or_id.strip()
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
# 日本語: 字幕/書き起こしの整形（軽量）
# -----------------------
def polish_transcript_jp(text: str) -> str:
    """雑音除去＋短文連結＋句読点補完で、LLMの構造検出を安定化"""
    text = re.sub(r"\[(拍手|音楽|笑い|Music|Applause)\]", "", text)
    text = re.sub(r"（(拍手|音楽|笑い)）", "", text)
    text = text.replace("♪", "").replace("■", "")

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    joined: List[str] = []
    for l in lines:
        if joined and re.match(r"^(そして|まず|次に|さらに|一方で|ただ|しかし|なので|だから|そのため|結論として)", l):
            joined[-1] += "、" + l
        else:
            joined.append(l)

    fixed: List[str] = []
    for l in joined:
        if re.search(r"[。！？!?]$", l) or len(l) < 10:
            fixed.append(l)
        elif re.search(r"[\u3040-\u30FF\u4E00-\u9FFF]", l):
            fixed.append(l + "。")
        else:
            fixed.append(l)

    out = "".join(fixed)
    out = re.sub(r"([。！？!?])\s*", r"\1", out)
    out = re.sub(r"(。){3,}", "。。", out)
    return out.strip()


# -----------------------
# YouTube 音声 DL（m4a優先で小さく）
# -----------------------
def download_audio(url_or_id: str, out_dir: str) -> Optional[str]:
    import yt_dlp
    vid = extract_video_id(url_or_id) or url_or_id
    ydl_opts = {
        "quiet": True,
        "format": "bestaudio/best",
        "outtmpl": os.path.join(out_dir, "%(id)s.%(ext)s"),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "m4a"}],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={vid}", download=True)
        base = os.path.join(out_dir, info["id"])
        for ext in (".m4a", ".mp3", ".webm", ".opus"):
            fp = base + ext
            if os.path.exists(fp):
                return fp
    return None


# -----------------------
# ffmpeg トリミング（冒頭スキップ＋最大分数）
# -----------------------
def trim_audio(src: str, dst: str, start_sec: int = 0, max_min: int = 0) -> str:
    """-ss で冒頭を飛ばし、-t で指定分だけに切る（再エンコードなし）"""
    cmd = ["ffmpeg", "-y"]
    if start_sec > 0:
        cmd += ["-ss", str(start_sec)]
    cmd += ["-i", src]
    if max_min > 0:
        cmd += ["-t", str(max_min * 60)]
    cmd += ["-c", "copy", dst]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return dst
    except Exception:
        return src


# -----------------------
# STT API（/stt-full → /analyze/audio）
# -----------------------
def stt_via_api(audio_path: str, api_base: str, user: str = "yt-import") -> Optional[dict]:
    stt_json = None
    for endpoint in ("/stt-full/?detail=true", "/analyze/audio?detail=true"):
        try:
            with open(audio_path, "rb") as f:
                files = {"file": (os.path.basename(audio_path), f, "audio/m4a")}
                data = {"user": user}
                r = requests.post(api_base.rstrip("/") + endpoint, files=files, data=data, timeout=180)
            if r.ok:
                stt_json = r.json()
                break
            else:
                print(f"[debug] {endpoint} -> HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:
            print(f"[debug] {endpoint} error: {e}")
    return stt_json


# -----------------------
# 採点 API
# -----------------------
def analyze_logic(text: str, api_base: str, prime: str = "") -> Optional[dict]:
    payload = {"text": (prime + text)[:12000]}
    try:
        r = requests.post(api_base.rstrip("/") + "/analyze-logic",
                          headers={"Content-Type": "application/json"},
                          data=json.dumps(payload), timeout=60)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return None


# -----------------------
# メイン
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-base", default="http://127.0.0.1:8015")
    ap.add_argument("--infile", default="urls.txt")
    ap.add_argument("--out-prefix", default="ja_talks")
    ap.add_argument("--min_chars", type=int, default=800, help="短すぎる書き起こしを除外する閾値（文字数）")
    ap.add_argument("--prime", default="",
                    help="採点前に本文の先頭へ付与するプロンプト（空文字なら言語別の既定文を挿入）")
    ap.add_argument("--skip-head-sec", type=int, default=0, help="音声の冒頭をこの秒数スキップ")
    ap.add_argument("--max-duration-min", type=int, default=0, help="最大この分数に切り詰めてSTT（0=無効）")
    ap.add_argument("--workdir", default=None, help="DL用一時ディレクトリ（指定しなければ自動生成）")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    with open(args.infile, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]

    tmp_dir = args.workdir or tempfile.mkdtemp(prefix="ytstt_")
    if args.debug:
        print(f"[debug] workdir: {tmp_dir}")

    rows: List[Dict] = []
    details: List[Dict] = []

    try:
        for url in tqdm(urls, desc="dl+stt+score"):
            try:
                # 1) 音声DL（m4a）
                audio = download_audio(url, tmp_dir)
                if not audio:
                    if args.debug:
                        print(f"[skip] audio DL failed: {url}")
                    continue

                # 1.5) 冒頭スキップ & 長さ制限（必要時）
                if args.skip_head_sec > 0 or args.max_duration_min > 0:
                    trimmed = os.path.join(tmp_dir, "trimmed.m4a")
                    audio = trim_audio(audio, trimmed, args.skip_head_sec, args.max_duration_min) or audio
                    if args.debug:
                        print(f"[debug] trimmed audio -> {audio}")

                # 2) STT
                stt = stt_via_api(audio, args.api_base, user="yt-import")
                text = (stt.get("text") or stt.get("transcript") or stt.get("result", {}).get("text") or "").strip() if stt else ""
                if args.debug:
                    print(f"[debug] text chars: {len(text)}")
                if not text or len(text) < args.min_chars:
                    if args.debug:
                        print(f"[skip] short transcript: {len(text)} < {args.min_chars}")
                    continue

                # 3) 整形（日本語のみ軽く）
                raw_text = text
                if (stt.get("language") or "").lower().startswith("ja"):
                    text = polish_transcript_jp(text)
                    if args.debug:
                        print(f"[debug] cleaned chars: {len(text)} (raw {len(raw_text)})")

                # 4) タイトル/長さ
                import yt_dlp
                title = "video"
                vid = None
                duration_min = None
                with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True}) as ydl:
                    info = ydl.extract_info(url, download=False)
                    title = info.get("title") or title
                    vid = info.get("id")
                    duration_sec = info.get("duration") or 0
                    if duration_sec:
                        duration_min = round(duration_sec / 60)
                pretty_url = f"https://www.youtube.com/watch?v={vid}" if vid else url

                # 5) 採点（言語でプライム自動出し分け）
                lang = (stt.get("language") or "").lower() if stt else ""
                prime = args.prime
                if prime == "":
                    if lang.startswith("en"):
                        approx = f" (about {duration_min} min)" if duration_min else ""
                        prime = (
                            f"This is a full transcript of a presentation{approx}. "
                            "Evaluate it on: Introduction → Main claim → Evidence → Summary → a concrete Call-To-Action (CTA). "
                            "Penalize redundancy.\n---\n"
                        )
                    else:
                        approx = f"（全体約{duration_min}分）" if duration_min else ""
                        prime = (
                            f"以下はプレゼンの書き起こしです{approx}。"
                            "導入→結論→根拠→まとめ→具体的な次アクション（CTA）を抽出し、冗長表現は不利に評価してください。\n---\n"
                        )

                res = analyze_logic(text, args.api_base, prime=prime)
                if not res:
                    if args.debug:
                        print(f"[skip] analyze-logic failed")
                    continue

                # 6) 保存用
                rows.append({
                    "title": title,
                    "url": pretty_url,
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
                    "url": pretty_url,
                    "text": text,
                    **res
                })

            except Exception as e:
                print(f"[warn] {url}: {e}")

        if not rows:
            print("No talks collected. ヒント: 動画を差し替える / --min_chars を下げる / --debug でログ確認")
            return

        df = pd.DataFrame(rows).sort_values("total", ascending=False)
        csv_path = f"{args.out_prefix}.stt.scored.csv"
        jsonl_path = f"{args.out_prefix}.stt.scored.jsonl"
        df.to_csv(csv_path, index=False, encoding="utf-8")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for d in details:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

        print(f"Saved: {csv_path} ({len(df)} rows)")
        print(f"Saved: {jsonl_path}")
        print(df.head(5).to_string(index=False))

    finally:
        if not args.workdir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
