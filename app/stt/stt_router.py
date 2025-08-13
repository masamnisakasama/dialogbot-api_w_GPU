from __future__ import annotations

import os
import re
import json
import uuid
import shutil
import tempfile
import wave
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import boto3
from fastapi import APIRouter, UploadFile, File, Query, Form, HTTPException
from fastapi.responses import JSONResponse

# ===== Whisper / STT ユーティリティ =====
# transcribe_with_segments / get_model_name は既存を利用
try:
    from .whisper_utils import (
        transcribe_with_segments,
        get_model_name,
    )
except Exception:
    from app.stt.whisper_utils import (
        transcribe_with_segments,
        get_model_name,
    )

MIN_BYTES = 1024          # 1KB 未満はエラー
DEFAULT_LANG = "ja"       # 自動検出にしたいなら None
ENABLE_WAV_FALLBACK = True

router = APIRouter(tags=["stt"])


# ============= 補助（長さ・メトリクス） =============

def _duration_from_wav(path: str) -> float:
    """標準ライブラリ wave で WAV の長さを出す（WAV 以外は 0 を返す）"""
    try:
        if not str(path).lower().endswith((".wav", ".wave")):
            return 0.0
        with wave.open(path, "rb") as w:
            frames = w.getnframes()
            rate = w.getframerate()
            if rate > 0:
                return frames / float(rate)
    except Exception:
        pass
    return 0.0


def _duration_from_segments(segments: List[Dict[str, Any]]) -> float:
    """Whisper のセグメントから全体長を推定"""
    try:
        if not segments:
            return 0.0
        starts = [float(s.get("start", 0.0)) for s in segments]
        ends   = [float(s.get("end",   0.0)) for s in segments]
        if starts and ends:
            return max(0.0, max(ends) - min(starts))
    except Exception:
        pass
    return 0.0


def _strip_spaces_len(text: str) -> int:
    """空白類を除いた長さ（CPSの参考値用）"""
    # ★ raw string を使う（\s の警告回避）
    return len(re.sub(r"\s+", "", text or ""))


def _metrics_from_segments(segments: List[Dict[str, Any]],
                           duration_sec: float,
                           text: str) -> Dict[str, Any]:
    """最小限のメトリクスを計算（既存があれば置き換え可）"""
    try:
        n = len(segments)
        starts = [float(s.get("start", 0.0)) for s in segments] if segments else []
        ends   = [float(s.get("end",   0.0)) for s in segments] if segments else []
        seg_durs = [(e - st) for st, e in zip(starts, ends) if e > st] if (starts and ends) else []
        voiced_time = sum(seg_durs) if seg_durs else 0.0
        avg_seg = (voiced_time / n) if n > 0 else 0.0

        # 文字/秒（日本語ではこちらをメイン指標に）
        cps = (_strip_spaces_len(text) / duration_sec) if duration_sec > 0 else 0.0
        # 語/分（英語向けの参考）
        words = len((text or "").split())
        wpm = (words * 60.0 / duration_sec) if duration_sec > 0 else 0.0

        silence = max(0.0, duration_sec - voiced_time)
        pause_ratio = (silence / duration_sec) if duration_sec > 0 else 0.0

        return {
            "speech_rate_cps": cps,
            "speech_rate_wpm": wpm,
            "pause_ratio": pause_ratio,
            "num_pauses": max(0, n - 1),
            "avg_pause_sec": (silence / max(1, n - 1)) if n > 1 else 0.0,
            "median_pause_sec": pause_ratio,  # 簡略
            "voiced_time_sec": voiced_time,
            "utterance_density": (voiced_time / duration_sec) if duration_sec > 0 else 0.0,
            "avg_segment_sec": avg_seg,
            "num_segments": n,
        }
    except Exception:
        return {}


def _make_speaking_advice(metrics: Dict[str, Any]) -> List[str]:
    tips: List[str] = []
    cps = float(metrics.get("speech_rate_cps") or 0.0)
    seg = float(metrics.get("avg_segment_sec") or 0.0)

    # 日本語の目安：CPS 3.0〜5.0
    if cps > 5.0:
        tips.append("話速がやや速めです。キーワード前後に 0.2〜0.4 秒の間を意識すると聞き取りやすくなります。")
    elif 0 < cps < 3.0:
        tips.append("話速がややゆっくりです。文末の無音を少し短くし、接続詞でテンポを作ると自然になります。")

    if seg > 5.0:
        tips.append("1 セグメントがやや長い傾向です。3〜5 秒程度で区切るとより明瞭です。")
    elif 0 < seg < 2.8:
        tips.append("1 セグメントが短い傾向です。1文をもう少し長くすると自然に聞こえます。")

    return tips


# ============= S3 保存（AES256 を常時付与：バケツのポリシー準拠） =============

def _s3_put_triplet(*, bucket: str, base: str, text: str,
                    result_obj: Dict[str, Any],
                    audio_metrics: Optional[Dict[str, Any]] = None) -> None:
    """
    バケットポリシーが 「s3:x-amz-server-side-encryption == AES256」を強制しているため、
    常に SSE-S3(AES256) を付けて PutObject する。
    """
    s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1"))
    extra = {"ServerSideEncryption": "AES256"}

    s3.put_object(
        Bucket=bucket, Key=f"{base}/transcript.txt",
        Body=(text or "").encode("utf-8"),
        ContentType="text/plain; charset=utf-8",
        **extra,
    )
    s3.put_object(
        Bucket=bucket, Key=f"{base}/result.json",
        Body=json.dumps(result_obj, ensure_ascii=False).encode("utf-8"),
        ContentType="application/json; charset=utf-8",
        **extra,
    )
    if audio_metrics:
        s3.put_object(
            Bucket=bucket, Key=f"{base}/audio_metrics.json",
            Body=json.dumps(audio_metrics, ensure_ascii=False).encode("utf-8"),
            ContentType="application/json; charset=utf-8",
            **extra,
        )


# ============= ルーター =============

@router.get("/stt/model")
def get_stt_model_name():
    try:
        return {"model": get_model_name()}
    except Exception:
        return {"model": "loading_on_first_use"}


@router.post("/stt-full/")
async def stt_full(
    file: UploadFile = File(...),
    detail: bool = Query(False),
    user: str = Form("web-client"),
):
    suffix = Path(file.filename).suffix or ".webm"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="stt_"); os.close(fd)

    try:
        # 一時ファイルに保存
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # サイズガード
        if os.path.getsize(tmp_path) < MIN_BYTES:
            raise HTTPException(status_code=400, detail="Audio too small (<1KB). Please record at least 2 seconds.")

        # STT 実行（segments 取得）
        lang = DEFAULT_LANG
        stt = transcribe_with_segments(tmp_path, language=lang)
        text = (stt.get("text") or "").strip()
        segments = stt.get("segments") or []

        # 長さの算出（WAV → それ以外は segments から）
        dur = 0.0
        if ENABLE_WAV_FALLBACK:
            dur = _duration_from_wav(tmp_path)
        if dur <= 0.0:
            dur = _duration_from_segments(segments)

        # メトリクス
        audio_metrics = _metrics_from_segments(segments, dur, text)
        advice = _make_speaking_advice(audio_metrics)

        # レスポンス
        resp: Dict[str, Any] = {
            "text": text,
            "model": get_model_name(),
            "language": lang,
            "duration_sec": dur,
            "audio_metrics": audio_metrics,
            "advice": advice,
        }
        if detail:
            resp["segments"] = segments

        # S3 保存（AES256）
        try:
            bucket = os.getenv("S3_BUCKET")
            if not bucket:
                raise RuntimeError("S3_BUCKET is not set")
            prefix = os.getenv("S3_PREFIX", "app/").rstrip("/")
            now = datetime.utcnow()
            base = f"{prefix}/{user}/{now:%Y/%m/%d}/{uuid.uuid4().hex[:12]}"

            _s3_put_triplet(
                bucket=bucket, base=base, text=text,
                result_obj=resp, audio_metrics=audio_metrics
            )
            print(f"[stt_full] S3 saved: s3://{bucket}/{base}")
        except Exception:
            import traceback
            print("[stt_full] S3 save skipped:\n", traceback.format_exc())

        return JSONResponse(resp)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT failed: {e}")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
