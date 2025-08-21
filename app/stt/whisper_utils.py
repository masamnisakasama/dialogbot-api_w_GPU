# app/whisper_utils.py
from __future__ import annotations

import io
import os
import tempfile
import threading
from pathlib import Path
from typing import Optional, Union, List

import shutil
import whisper 
# バグると怖いのでローカルだと既存の OpenAI Whisper を温存
# MacOSでGPU動かせないので、本番は「頼む通ってくれ」と願うことしかできない

# ───────────────────────────────────────────────────────────────────
# 使い方
#  - ローカルではFASTER_WHISPER=0でwhisper/本番はFASTER_WHISPER=1でfaster-whisper
#  - 悲しいことにMacのGPUは使えないのでローカルはCPU
#  - ローカル(CPU): USE_GPU=0, FASTER_WHISPER=0, WHISPER_MODEL=small, COMPUTE_TYPE=int8
#  - GKE(GPU): USE_GPU=1, FASTER_WHISPER=1, WHISPER_MODEL=medium, COMPUTE_TYPE=float16
#   ※ FASTER_WHISPER を設定しない場合は USE_GPU=1 なら自動で faster-whisper 
# ───────────────────────────────────────────────────────────────────

def _env_true(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")

USE_GPU = _env_true("USE_GPU", False)
# デフォルト: GPUなら faster-whisper、CPUなら  Whisper
USE_FASTER = _env_true("FASTER_WHISPER", USE_GPU)

# モデル名
ENV_MODEL = os.getenv("WHISPER_MODEL", "").strip()
DEFAULT_CANDIDATES: List[str] = ["small", "base", "tiny"]
# GPU ならturboを既定、CPUはmediumを既定に
DEFAULT_MODEL = "turbo" if USE_GPU else "medium"
MODEL = ENV_MODEL or DEFAULT_MODEL

# CPU/GPU 共通の制限
MAX_UPLOAD_MB = float(os.getenv("WHISPER_MAX_UPLOAD_MB", "50"))

# OpenAI Whisper 用の推論設定　既存を置いておく
TRANSCRIBE_KW = dict(
    temperature=0.0,
    beam_size=1,
    best_of=1,
    condition_on_previous_text=False,
    word_timestamps=False,
    fp16=False,  # CPUなので fp16 は無効（OpenAI Whisper 経路でのみ使用）
)

# faster-whisper 用（GPU で有効化推奨）
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16" if USE_GPU else "int8")
CPU_THREADS = int(os.getenv("CPU_THREADS", "4"))
NUM_WORKERS = int(os.getenv("WHISPER_WORKERS", "1"))

# 実装切替フラグ
_USE_FASTER_EFFECTIVE = False
try:
    if USE_FASTER:
        from faster_whisper import WhisperModel as FWModel  # type: ignore
        _USE_FASTER_EFFECTIVE = True
except Exception as _e:
    # faster-whisperが未インストールとか読み込み失敗 → 既存WhisperでFB
    _USE_FASTER_EFFECTIVE = False


def _ensure_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install ffmpeg or use a Docker image that contains it."
        )

# ===== OpenAI Whisper (従来) =====
_DEVICE_OPENAI: str = "cpu"
_model_openai = None
_model_name_openai: Optional[str] = None
_model_lock_openai = threading.Lock()

def _load_model_with_fallback_openai():
    global _model_openai, _model_name_openai
    if _model_openai is not None:
        return
    candidates: List[str] = (
        [ENV_MODEL] + [m for m in DEFAULT_CANDIDATES if m != ENV_MODEL]
        if ENV_MODEL else DEFAULT_CANDIDATES
    )
    last_err: Optional[Exception] = None
    for name in candidates:
        try:
            local = whisper.load_model(name, device=_DEVICE_OPENAI)
            _model_openai = local
            _model_name_openai = name
            print(f"[whisper_utils] Loaded OpenAI-Whisper '{name}' on cpu.")
            return
        except Exception as e:
            print(f"[whisper_utils] Failed to load '{name}' on cpu: {e}")
            last_err = e
            continue
    raise RuntimeError(f"Failed to load Whisper model. Tried: {candidates}. Last error: {last_err}")

def _get_openai_model():
    global _model_openai
    if _model_openai is not None:
        return _model_openai
    with _model_lock_openai:
        if _model_openai is not None:
            return _model_openai
        _load_model_with_fallback_openai()
        return _model_openai

# ===== faster-whisper (GPU/CPU) =====
_model_fw = None
_model_lock_fw = threading.Lock()

def _get_fw_model():
    global _model_fw
    if _model_fw is not None:
        return _model_fw
    with _model_lock_fw:
        if _model_fw is not None:
            return _model_fw
        # 初回ロード
        _model_fw = FWModel(
            MODEL,
            device=("cuda" if USE_GPU else "cpu"),
            compute_type=COMPUTE_TYPE,
            cpu_threads=CPU_THREADS,
            num_workers=NUM_WORKERS,
        )
        print(f"[whisper_utils] Loaded faster-whisper '{MODEL}' on "
              f"{'cuda' if USE_GPU else 'cpu'} ({COMPUTE_TYPE}).")
        return _model_fw

# 互換 expose（古いコードが whisper_utils.model を参照しても壊れにくく）
model = None  # get_model() 呼び出し後に実体をセットする

def get_model_name() -> Optional[str]:
    # どちらの実装でも MODEL 名を返す
    return MODEL

def _filesize_mb(path: Union[str, Path]) -> float:
    p = Path(path)
    return p.stat().st_size / (1024 * 1024)

def _to_temp_audio_file(
    data: Union[bytes, io.BufferedReader, io.BytesIO, str, Path],
    suffix: str = ".wav",
) -> str:
    """file-like/bytes/パス いずれも受け取り、一時ファイルのパスを返す。"""
    if isinstance(data, (str, Path)):
        p = Path(data)
        if not p.exists():
            raise FileNotFoundError(f"Audio file not found: {p}")
        if _filesize_mb(p) > MAX_UPLOAD_MB:
            raise ValueError(f"Audio file too large (> {MAX_UPLOAD_MB} MB).")
        return str(p)

    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="whisper_")
    os.close(fd)
    try:
        with open(tmp_path, "wb") as f:
            if isinstance(data, (io.BufferedReader, io.BytesIO)):
                f.write(data.read())
            elif isinstance(data, (bytes, bytearray)):
                f.write(data)
            else:
                raise TypeError(f"Unsupported audio data type: {type(data)}")
    except Exception:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise

    if _filesize_mb(tmp_path) > MAX_UPLOAD_MB:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise ValueError(f"Audio file too large (> {MAX_UPLOAD_MB} MB).")

    return tmp_path

# ===== 共通 API: get_model / transcribe_* =====

def get_model():
    """
    互換API。どちらの実装でも model を返す。
    """
    global model
    if _USE_FASTER_EFFECTIVE:
        mdl = _get_fw_model()
    else:
        mdl = _get_openai_model()
    model = mdl
    return mdl

def transcribe_audio(
    file_like_or_path: Union[bytes, io.BufferedReader, io.BytesIO, str, Path],
    *,
    language: Optional[str] = "ja",
    initial_prompt: Optional[str] = None,
) -> str:
    mdl = get_model()
    tmp = _to_temp_audio_file(file_like_or_path, suffix=".wav")
    try:
        if _USE_FASTER_EFFECTIVE:
            segs, info = mdl.transcribe(
                tmp, language=language, vad_filter=True, initial_prompt=initial_prompt
            )
            return "".join(s.text for s in segs).strip()
        else:
            _ensure_ffmpeg()
            result = mdl.transcribe(
                tmp,
                language=language,
                initial_prompt=initial_prompt or "",
                **TRANSCRIBE_KW,
            )
            return (result or {}).get("text", "").strip()
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass

def transcribe_with_segments(
    file_like_or_path: Union[bytes, io.BufferedReader, io.BytesIO, str, Path],
    *,
    language: Optional[str] = "ja",
    initial_prompt: Optional[str] = None,
) -> dict:
    """
    Whisper の結果を { text, segments, duration, language, model } 形式で返す。
    既存の stt_router.py が期待する形に合わせている。
    """
    mdl = get_model()
    tmp = _to_temp_audio_file(file_like_or_path, suffix=".wav")
    try:
        if _USE_FASTER_EFFECTIVE:
            segs, info = mdl.transcribe(
                tmp, language=language, vad_filter=True, initial_prompt=initial_prompt
            )
            segments = [{"start": s.start, "end": s.end, "text": s.text} for s in segs]
            text = "".join(s["text"] for s in segments).strip()
            return {
                "text": text,
                "segments": segments,
                "duration": getattr(info, "duration", None),
                "language": getattr(info, "language", language),
                "model": MODEL,
            }
        else:
            _ensure_ffmpeg()
            result = mdl.transcribe(
                tmp,
                language=language,
                initial_prompt=initial_prompt or "",
                **TRANSCRIBE_KW,
            ) or {}
            # OpenAI Whisper の result は { "text": ..., "segments": [...] } の形
            return result
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass
