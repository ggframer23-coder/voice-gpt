from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    base_dir: Path
    db_path: Path
    index_path: Path
    model_name: str
    offline: bool
    whisper_bin: Path | None
    vad_bin: Path | None
    vad_model: Path | None
    parakeet_model: str
    parakeet_dir: Path | None
    parakeet_quant: str | None


def load_settings() -> Settings:
    base_dir = Path(os.environ.get("STT_HOME", Path.home() / ".stt")).expanduser()
    db_path = Path(os.environ.get("STT_DB", base_dir / "journal.sqlite")).expanduser()
    index_path = Path(os.environ.get("STT_INDEX", base_dir / "faiss.index")).expanduser()
    model_name = os.environ.get("STT_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    whisper_bin_raw = os.environ.get("STT_WHISPER_BIN")
    whisper_bin = Path(whisper_bin_raw).expanduser() if whisper_bin_raw else None
    vad_bin_raw = os.environ.get("STT_VAD_BIN")
    vad_bin = Path(vad_bin_raw).expanduser() if vad_bin_raw else None
    vad_model_raw = os.environ.get("STT_VAD_MODEL")
    vad_model = Path(vad_model_raw).expanduser() if vad_model_raw else None
    parakeet_model = os.environ.get("STT_PARAKEET_MODEL", "nemo-parakeet-tdt-0.6b-v3")
    parakeet_dir_raw = os.environ.get("STT_PARAKEET_DIR")
    parakeet_dir = Path(parakeet_dir_raw).expanduser() if parakeet_dir_raw else None
    parakeet_quant_raw = os.environ.get("STT_PARAKEET_QUANT")
    if parakeet_quant_raw is None:
        parakeet_quant = "int8"
    else:
        parakeet_quant = None if parakeet_quant_raw.strip().lower() in {"", "none"} else parakeet_quant_raw

    offline_raw = os.environ.get("STT_OFFLINE")
    if offline_raw is None:
        offline = True
    else:
        offline = offline_raw.strip().lower() in {"1", "true", "yes", "on"}
    return Settings(
        base_dir=base_dir,
        db_path=db_path,
        index_path=index_path,
        model_name=model_name,
        offline=offline,
        whisper_bin=whisper_bin,
        vad_bin=vad_bin,
        vad_model=vad_model,
        parakeet_model=parakeet_model,
        parakeet_dir=parakeet_dir,
        parakeet_quant=parakeet_quant,
    )
