import argparse, csv, json, sys, time, re, os, glob
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
from urllib.parse import urlparse
import requests

# ====== URLテキスト抽出（必要時のみ trafilatura/bs4 を使う） ======
def _extract_with_trafilatura(url: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        import trafilatura
        downloaded = trafilatura.fetch_url(url, timeout=20)
        if not downloaded:
            return None, None
        data = trafilatura.extract(
            downloaded,
            output="json",
            favor_precision=True,
            include_comments=False,
            include_tables=False,
        )
        if not data:
            return None, None
        j = json.loads(data)
        return (j.get("text") or None, j.get("title") or None)
    except Exception:
        return None, None

def _extract_with_bs4(url: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        from bs4 import BeautifulSoup
        r = requests.get(url, timeout=20, headers={"User-Agent": "logic-scorer/1.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        candidates = soup.select("article, main, .article, .post, #content") or [soup.body]
        text = "\n".join(c.get_text(" ", strip=True) for c in candidates if c)
        title = (soup.title.string or "").strip() if soup.title else ""
        text = re.sub(r"\s+", " ", text)
        return (text or None, title or None)
    except Exception:
        return None, None

def extract_text_from_url(url: str) -> Tuple[Optional[str], Optional[str]]:
    text, title = _extract_with_trafilatura(url)
    if not text:
        text, title = _extract_with_bs4(url)
    return text, title

# ====== ローカルTXT抽出 ======
def extract_text_from_file(path: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
            txt = f.read()
    txt = txt.strip()
    if not txt:
        return None, None
    # 先頭の非空行をタイトル相当に（長すぎる場合は短縮）
    first_line = next((l.strip() for l in txt.splitlines() if l.strip()), "")
    title = first_line if len(first_line) <= 120 else first_line[:117] + "…"
    return txt, (title or os.path.basename(path))

# ====== チャンク＆集計 ======
def chunk_text(t: str, max_chars: int = 2200) -> List[str]:
    t = t.strip()
    if len(t) <= max_chars:
        return [t]
    # 文区切りでなるべく境界を保つ
    sents = re.split(r"(?<=[。．!?！？])\s*", t)
    chunks, cur = [], ""
    for s in sents:
        if not s:
            continue
        if len(cur) + len(s) <= max_chars:
            cur += s
        else:
            if cur:
                chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)
    return chunks

def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        if x != x:  # NaN
            return default
        # だいたい 0〜100 に収める（外れ値が来たときの保険）
        return max(0.0, min(100.0, x))
    except Exception:
        return default

def _normalize_scores(d: Dict) -> Dict[str, float]:
    """バックエンドの返却を吸収。{scores:{...}} or {...} の両対応に正規化"""
    s = d.get("scores") if isinstance(d.get("scores"), dict) else d
    out = {k: _to_float(s.get(k, 0.0)) for k in ("clarity","consistency","cohesion","density","cta")}
    # total は top-level も尊重
    total = d.get("total", s.get("total") if isinstance(s, dict) else None)
    if total is not None:
        out["total"] = _to_float(total)
    else:
        # 無ければ平均
        vals = [out[k] for k in ("clarity","consistency","cohesion","density","cta") if k in out]
        out["total"] = round(sum(vals)/len(vals), 2) if vals else 0.0
    return out

def aggregate(scores_list: List[Tuple[Dict[str, float], int]]) -> Dict[str, float]:
    """(scores, weight=文字数) を長さ重み平均。無いキーはスキップ。"""
    keys = ["clarity", "consistency", "cohesion", "density", "cta", "total"]
    out: Dict[str, float] = {}
    W = sum(w for _, w in scores_list) or 1
    for k in keys:
        num = sum((s.get(k, 0.0) * w) for s, w in scores_list if k in s)
        den = sum(w for s, w in scores_list if k in s) or W
        out[k] = round(num / den, 2)
    return out

# ====== API ======
def call_api(api: str, text: str, timeout: int = 60) -> Dict:
    r = requests.post(api, json={"text": text}, timeout=timeout)
    r.raise_for_status()
    return r.json()

# ====== メイン ======
@dataclass
class Row:
    id: str       # URL or file path
    title: str
    site: str     # netloc or 'local'
    chars: int
    chunks: int
    clarity: float
    consistency: float
    cohesion: float
    density: float
    cta: float
    total: float

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", help="1行1URLのリストファイル（省略可）")
    ap.add_argument("--files", nargs="*", help="評価するローカルtxt（ワイルドカード可）")
    ap.add_argument("--api", default="http://127.0.0.1:8015/analyze-logic")
    ap.add_argument("--out", default="results.csv", help="集計CSV（各指標＋total）")
    ap.add_argument("--jsonl", default="results.jsonl", help="集計JSONL（チャンク明細を含む）")
    ap.add_argument("--per-chunk-csv", default=None, help="チャンクごとの素点を書き出すCSV（任意）")
    ap.add_argument("--sleep", type=float, default=0.8)
    ap.add_argument("--max-chars", type=int, default=2200)
    args = ap.parse_args()

    if not args.urls and not args.files:
        print("usage: --files corpora/test.txt [複数可]  または  --urls urls.txt", file=sys.stderr)
        sys.exit(2)

    # 収集対象を列挙
    targets: List[Tuple[str, str]] = []  # (kind, id)  kind in {"file","url"}
    if args.files:
        expanded: List[str] = []
        for pat in args.files:
            expanded.extend(glob.glob(pat))
        for p in sorted(set(expanded)):
            targets.append(("file", p))
    if args.urls:
        with open(args.urls, "r", encoding="utf-8") as f:
            for line in f:
                u = line.strip()
                if u and not u.startswith("#"):
                    targets.append(("url", u))

    rows: List[Row] = []
    per_chunk_writer = None
    per_chunk_file = None
    if args.per_chunk_csv:
        per_chunk_file = open(args.per_chunk_csv, "w", newline="", encoding="utf-8")
        per_chunk_writer = csv.writer(per_chunk_file)
        per_chunk_writer.writerow(["id","title","site","chunk_idx","chunk_chars",
                                   "clarity","consistency","cohesion","density","cta","total"])

    with open(args.jsonl, "w", encoding="utf-8") as jsonl:
        for i, (kind, ident) in enumerate(targets, 1):
            try:
                if kind == "file":
                    text, title = extract_text_from_file(ident)
                    site = "local"
                    display_id = ident
                else:
                    text, title = extract_text_from_url(ident)
                    site = urlparse(ident).netloc
                    display_id = ident

                if not text or len(text) < 50:
                    print(f"[{i}/{len(targets)}] SKIP (no/short text) {display_id}")
                    continue

                parts = chunk_text(text, max_chars=args.max_chars)
                scored: List[Tuple[Dict[str, float], int]] = []
                chunk_details = []

                for j, part in enumerate(parts, 1):
                    data = call_api(args.api, part)
                    scores = _normalize_scores(data)
                    weight = len(part)
                    scored.append((scores, weight))
                    chunk_details.append({
                        "chunk_index": j,
                        "chars": weight,
                        "scores": scores
                    })
                    if per_chunk_writer:
                        per_chunk_writer.writerow([
                            display_id, (title or ""), site, j, weight,
                            scores.get("clarity",0.0), scores.get("consistency",0.0),
                            scores.get("cohesion",0.0), scores.get("density",0.0),
                            scores.get("cta",0.0), scores.get("total",0.0)
                        ])
                    time.sleep(args.sleep)

                agg = aggregate(scored)
                row = Row(
                    id=display_id, title=title or "", site=site,
                    chars=sum(len(p) for p in parts), chunks=len(parts),
                    clarity=agg.get("clarity", 0.0),
                    consistency=agg.get("consistency", 0.0),
                    cohesion=agg.get("cohesion", 0.0),
                    density=agg.get("density", 0.0),
                    cta=agg.get("cta", 0.0),
                    total=agg.get("total", 0.0),
                )
                rows.append(row)

                # JSONL: 集計＋チャンク明細
                json.dump({
                    "source": kind, "id": display_id, "title": row.title, "site": row.site,
                    "agg_scores": agg, "chars": row.chars, "chunks": row.chunks,
                    "chunk_details": chunk_details
                }, jsonl, ensure_ascii=False)
                jsonl.write("\n")

                print(f"[{i}/{len(targets)}] OK {display_id}  "
                      f"clarity={row.clarity:.2f} consistency={row.consistency:.2f} "
                      f"cohesion={row.cohesion:.2f} density={row.density:.2f} "
                      f"cta={row.cta:.2f} total={row.total:.2f}")

            except requests.HTTPError as e:
                print(f"[{i}/{len(targets)}] HTTP {e.response.status_code} {ident}", file=sys.stderr)
            except Exception as e:
                print(f"[{i}/{len(targets)}] ERR {ident}  {e}", file=sys.stderr)

    if per_chunk_file:
        per_chunk_file.close()

    # CSV 保存（集計）
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id","title","site","chars","chunks",
                    "clarity","consistency","cohesion","density","cta","total"])
        for r in rows:
            w.writerow([r.id, r.title, r.site, r.chars, r.chunks,
                        r.clarity, r.consistency, r.cohesion, r.density, r.cta, r.total])

    print(f"done: {len(rows)} rows -> {args.out} / {args.jsonl}"
          + (f" (+ {args.per_chunk_csv})" if args.per_chunk_csv else ""))
    

if __name__ == "__main__":
    main()
