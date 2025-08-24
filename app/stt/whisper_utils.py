from __future__ import annotations
import io, os, tempfile, threading, shutil
from pathlib import Path
from typing import Optional, Union, List
import whisper, torch

print(f"[whisper_utils HOTFIX] file={__file__} torch={torch.__version__} cuda_avail={torch.cuda.is_available()}")

def _env_true(name: str, default: bool=False)->bool:
    v=os.getenv(name); 
    return default if v is None else v.strip().lower() in ("1","true","yes","on")

USE_GPU=_env_true("USE_GPU", True)
ENV_MODEL=(os.getenv("WHISPER_MODEL","turbo") or "turbo").lower()
DEFAULT_CANDIDATES=["small","base","tiny"]
MODEL=ENV_MODEL
MAX_UPLOAD_MB=float(os.getenv("WHISPER_MAX_UPLOAD_MB","200"))

# GPU専用：CUDAなければ即失敗して気づく
if USE_GPU and not torch.cuda.is_available():
    raise RuntimeError("GPU_ONLY: USE_GPU=1 だが CUDA を利用できません（GPUなし or CPU版PyTorch）")

_DEVICE="cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu"
TRANSCRIBE_KW=dict(
  temperature=0.0, beam_size=1, best_of=1,
  condition_on_previous_text=False, word_timestamps=False,
  fp16=(_DEVICE=="cuda"),
)

_model=None; _lock=threading.Lock()

def _filesize_mb(p: Union[str,Path])->float:
    p=Path(p); return p.stat().st_size/(1024*1024)

def _to_temp_audio_file(data: Union[bytes, io.BufferedReader, io.BytesIO, str, Path], suffix=".wav")->str:
    if isinstance(data,(str,Path)):
        p=Path(data); 
        if not p.exists(): raise FileNotFoundError(p)
        if _filesize_mb(p)>MAX_UPLOAD_MB: raise ValueError(f"Audio too large (> {MAX_UPLOAD_MB} MB)")
        return str(p)
    fd,tmp=tempfile.mkstemp(suffix=suffix,prefix="whisper_"); os.close(fd)
    try:
        with open(tmp,"wb") as f:
            if isinstance(data,(io.BufferedReader,io.BytesIO)): f.write(data.read())
            elif isinstance(data,(bytes,bytearray)): f.write(data)
            else: raise TypeError(type(data))
    except Exception:
        try: os.remove(tmp)
        except Exception: pass
        raise
    if _filesize_mb(tmp)>MAX_UPLOAD_MB:
        try: os.remove(tmp)
        except Exception: pass
        raise ValueError(f"Audio too large (> {MAX_UPLOAD_MB} MB)")
    return tmp

def _ensure_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")

def _load_model():
    global _model
    if _model is not None: return _model
    with _lock:
        if _model is not None: return _model
        candidates=[MODEL]+[m for m in DEFAULT_CANDIDATES if m!=MODEL]
        last=None
        for name in candidates:
            try:
                m=whisper.load_model(name, device=_DEVICE)
                try: param_dev=next(m.parameters()).device.type
                except Exception: param_dev=_DEVICE
                print(f"[whisper_utils HOTFIX] Loaded '{name}' on {_DEVICE} (param_device={param_dev})")
                if _DEVICE=="cuda" and param_dev!="cuda":
                    raise RuntimeError(f"Expected CUDA but got {param_dev}")
                _model=m; return m
            except Exception as e:
                print(f"[whisper_utils HOTFIX] Failed to load '{name}': {e}"); last=e
        raise RuntimeError(f"Failed to load model. Tried {candidates}. Last: {last}")

def get_model_name()->Optional[str]:
    return MODEL

def get_model():
    return _load_model()

def transcribe_audio(file_like_or_path, *, language: Optional[str]="ja", initial_prompt: Optional[str]=None)->str:
    mdl=_load_model(); tmp=_to_temp_audio_file(file_like_or_path,".wav")
    try:
        _ensure_ffmpeg()
        r=mdl.transcribe(tmp, language=language, initial_prompt=initial_prompt or "", **TRANSCRIBE_KW) or {}
        return (r or {}).get("text","").strip()
    finally:
        try: os.remove(tmp)
        except Exception: pass

def transcribe_with_segments(file_like_or_path, *, language: Optional[str]="ja", initial_prompt: Optional[str]=None)->dict:
    mdl=_load_model(); tmp=_to_temp_audio_file(file_like_or_path,".wav")
    try:
        _ensure_ffmpeg()
        r=mdl.transcribe(tmp, language=language, initial_prompt=initial_prompt or "", **TRANSCRIBE_KW) or {}
        return r
    finally:
        try: os.remove(tmp)
        except Exception: pass
