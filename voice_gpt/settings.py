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
    whisper_bin: Path | None


def load_settings() -> Settings:
    base_dir = Path(os.environ.get("VOICE_GPT_HOME", Path.home() / ".voice-gpt")).expanduser()
    db_path = Path(os.environ.get("VOICE_GPT_DB", base_dir / "journal.sqlite")).expanduser()
    index_path = Path(os.environ.get("VOICE_GPT_INDEX", base_dir / "faiss.index")).expanduser()
    model_name = os.environ.get("VOICE_GPT_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    whisper_bin_raw = os.environ.get("VOICE_GPT_WHISPER_BIN")
    whisper_bin = Path(whisper_bin_raw).expanduser() if whisper_bin_raw else None
    return Settings(
        base_dir=base_dir,
        db_path=db_path,
        index_path=index_path,
        model_name=model_name,
        whisper_bin=whisper_bin,
    )
