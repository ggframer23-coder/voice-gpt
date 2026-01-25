from __future__ import annotations

import builtins
import json
import shutil
import sqlite3
import hashlib
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import typer
from rich import print

from .settings import load_settings

if TYPE_CHECKING:
    from .journal import add_entry, has_audio, init_store, search
    from .transcribe import (
        TranscriptionError,
        _collect_word_timestamps,
        load_faster_whisper_model,
        load_parakeet_model,
        transcribe_audio,
        transcribe_audio_faster_whisper,
        transcribe_audio_parakeet,
        transcribe_audio_whisperx,
    )
    from .vad import create_vad_clips

def _show_help_on_no_args(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        print(ctx.get_help())
        raise typer.Exit()


TMP_DIR = Path("tmp")

app = typer.Typer(add_completion=False, invoke_without_command=True, callback=_show_help_on_no_args)


def _default_vad_dir(audio: Path) -> Path:
    return audio.parent / f"{audio.stem}.vad"


def _vad_metadata_path(vad_dir: Path) -> Path:
    return vad_dir / "segments.json"


def _write_vad_metadata(path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8")


def _archive_week_dir(base_dir: Path, audio_path: Path) -> Path:
    recorded_at = datetime.fromtimestamp(audio_path.stat().st_mtime).astimezone()
    week = recorded_at.isocalendar().week
    return base_dir / str(recorded_at.year) / f"{week:02d}"


def _audio_duration_seconds(audio_path: Path) -> Optional[float]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return None
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def _resolve_vad_paths(settings, vad_bin: Optional[Path], vad_model: Optional[Path]) -> tuple[Optional[Path], Optional[Path]]:
    return vad_bin or settings.vad_bin, vad_model or settings.vad_model


def _resolve_parakeet_settings(
    settings,
    parakeet_model: Optional[str],
    parakeet_dir: Optional[Path],
    parakeet_quant: Optional[str],
    model_fallback: Optional[str] = None,
) -> tuple[str, Optional[Path], Optional[str]]:
    model = parakeet_model or model_fallback or settings.parakeet_model
    model_dir = parakeet_dir or settings.parakeet_dir
    quant = parakeet_quant if parakeet_quant is not None else settings.parakeet_quant
    return model, model_dir, quant


def _build_audio_index(audio_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in audio_dir.rglob("*"):
        if path.is_file():
            index.setdefault(path.name, path)
    return index


def _iter_audio_files(directory: Path, recursive: bool, extensions: str) -> list[Path]:
    ext_set = {ext.strip().lower().lstrip(".") for ext in extensions.split(",") if ext.strip()}
    files: list[Path] = []
    if recursive:
        for path in directory.rglob("*"):
            if any(part.endswith(".vad") for part in path.parts):
                continue
            if path.is_file() and path.suffix.lower().lstrip(".") in ext_set:
                files.append(path)
    else:
        for path in directory.iterdir():
            if path.is_file() and path.suffix.lower().lstrip(".") in ext_set:
                files.append(path)
    files.sort()
    return files


def _format_timestamp_ms(ms: int) -> str:
    seconds = ms / 1000.0
    minutes = int(seconds // 60)
    seconds -= minutes * 60
    return f"{minutes:02d}:{seconds:06.3f}"


def _format_elapsed_seconds(seconds: float) -> str:
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _inject_elapsed_markers(words: list[dict], interval_seconds: int = 300) -> str:
    if not words:
        return ""
    parts: list[str] = []
    next_marker = interval_seconds
    for word in words:
        start = float(word.get("start", 0.0))
        while start >= next_marker:
            parts.append(f" [{_format_elapsed_seconds(next_marker)}]")
            next_marker += interval_seconds
        parts.append(str(word.get("word", "")))
    return "".join(parts).strip()


def _inject_elapsed_lines(words: list[dict], interval_minutes: int = 5) -> str:
    if not words:
        return ""
    parts: list[str] = []
    interval_seconds = max(1, interval_minutes) * 60
    next_marker = interval_seconds
    for word in words:
        start = float(word.get("start", 0.0))
        while start >= next_marker:
            parts.append(f"\nElapsed: {int(next_marker // 60)}min\n")
            next_marker += interval_seconds
        parts.append(str(word.get("word", "")))
    return "".join(parts).strip()


def _format_duration(duration_seconds: Optional[float]) -> str:
    if duration_seconds is None:
        return "n/a"
    return _format_elapsed_seconds(duration_seconds)


def _print_speed_ratio(metadata: dict, audio_path: Path, duration_seconds: Optional[float] = None) -> None:
    if duration_seconds is None and audio_path.exists():
        duration_seconds = _audio_duration_seconds(audio_path)
    transcription_meta = metadata.get("transcription") or {}
    elapsed_seconds = transcription_meta.get("elapsed_seconds")
    if isinstance(elapsed_seconds, (int, float)) and duration_seconds:
        ratio = elapsed_seconds / duration_seconds
        builtins.print(f"speed_ratio: {ratio:.3f}")


def _record_timing(metadata: dict, key: str, duration_seconds: float) -> None:
    timings = metadata.setdefault("timings", {})
    steps = timings.setdefault("steps", {})
    steps[key] = duration_seconds


AUTO_ENGINE_THRESHOLD_SECONDS = 300


def _resolve_engine_for_audio(engine: str, audio_path: Path) -> tuple[str, Optional[float]]:
    if engine != "auto":
        return engine, None
    duration_seconds = _audio_duration_seconds(audio_path) if audio_path.exists() else None
    if duration_seconds is not None and duration_seconds < AUTO_ENGINE_THRESHOLD_SECONDS:
        return "faster-whisper", duration_seconds
    return "whisperx", duration_seconds


def _build_transcription_metadata(
    settings,
    engine: str,
    model: str,
    parakeet_model: Optional[str],
    parakeet_dir: Optional[Path],
    parakeet_quant: Optional[str],
    whisperx_model: str,
    whisperx_device: str,
    whisperx_diarize: bool,
    whisperx_compute_type: Optional[str],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"engine": engine}
    if engine == "faster-whisper":
        metadata["model"] = model
        metadata["compute_type"] = "int8"
    elif engine == "parakeet":
        metadata["model"] = parakeet_model or model
        if parakeet_dir is not None:
            metadata["model_dir"] = str(parakeet_dir)
        if parakeet_quant is not None:
            metadata["quantization"] = parakeet_quant
    elif engine == "whisperx":
        metadata["model"] = whisperx_model or model
        metadata["device"] = whisperx_device
        metadata["diarize"] = whisperx_diarize
        if whisperx_compute_type is not None:
            metadata["compute_type"] = whisperx_compute_type
    else:
        metadata["model"] = model
        if settings.whisper_bin:
            metadata["whisper_bin"] = str(settings.whisper_bin)
    return metadata


def _select_metadata_payload(metadata: dict, full: bool) -> dict:
    if full:
        return metadata
    return metadata.get("transcription") or {}


def _format_clip_line(clip: dict, text: str) -> str:
    start = _format_timestamp_ms(int(clip["start_ms"]))
    end = _format_timestamp_ms(int(clip["end_ms"]))
    return f"[{start}-{end}] {text}".strip()


def _transcribe_to_text(
    settings,
    audio: Path,
    model: str,
    convert: bool,
    vad: bool,
    vad_dir: Optional[Path],
    vad_threshold: float,
    vad_min_speech_ms: int,
    vad_min_silence_ms: int,
    vad_speech_pad_ms: int,
    vad_max_speech_s: Optional[float],
    vad_samples_overlap: float,
    vad_bin: Optional[Path],
    vad_model: Optional[Path],
    vad_timestamps: bool,
    engine: str,
    parakeet_model: Optional[str],
    parakeet_dir: Optional[Path],
    parakeet_quant: Optional[str],
    whisperx_model: str,
    whisperx_device: str,
    whisperx_diarize: bool,
    whisperx_diarize_model: Optional[str],
    whisperx_compute_type: Optional[str] = None,
    language: str = "en",
) -> tuple[str, dict, list[dict], Optional[Path]]:
    from .transcribe import (
        TranscriptionError,
        transcribe_audio,
        transcribe_audio_faster_whisper,
        transcribe_audio_parakeet,
        transcribe_audio_whisperx,
    )
    from .vad import create_vad_clips

    metadata: dict = {}
    word_timestamps: list[dict] = []
    vad_metadata_path: Optional[Path] = None
    resolved_parakeet = _resolve_parakeet_settings(
        settings,
        parakeet_model,
        parakeet_dir,
        parakeet_quant,
        model_fallback=model,
    )
    resolved_engine, _duration_seconds = _resolve_engine_for_audio(engine, audio)
    transcription_metadata = _build_transcription_metadata(
        settings,
        resolved_engine,
        model,
        parakeet_model,
        parakeet_dir,
        parakeet_quant,
        whisperx_model,
        whisperx_device,
        whisperx_diarize,
        whisperx_compute_type,
    )
    metadata: dict = {"transcription": transcription_metadata}
    if resolved_engine == "whisperx" and vad:
        raise TranscriptionError("WhisperX does not support VAD splitting.")

    if vad:
        output_dir = vad_dir or _default_vad_dir(audio)
        resolved_bin, resolved_model = _resolve_vad_paths(settings, vad_bin, vad_model)
        metadata = create_vad_clips(
            audio_path=audio,
            output_dir=output_dir,
            threshold=vad_threshold,
            min_speech_ms=vad_min_speech_ms,
            min_silence_ms=vad_min_silence_ms,
            speech_pad_ms=vad_speech_pad_ms,
            max_speech_s=vad_max_speech_s,
            samples_overlap=vad_samples_overlap,
            vad_bin=resolved_bin,
            vad_model=resolved_model,
        )
        metadata["transcription"] = transcription_metadata
        clips = metadata["segments"]
        if not clips:
            raise TranscriptionError("No speech segments detected.")
        if resolved_engine == "whisperx":
            raise TranscriptionError("WhisperX does not support VAD splitting.")
        if resolved_engine == "faster-whisper":
            start_time = time.perf_counter()
            text, word_timestamps = _transcribe_clips_faster_whisper(
                clips,
                model,
                language,
                vad_timestamps,
                settings.offline,
            )
            elapsed = time.perf_counter() - start_time
            transcription_metadata["elapsed_seconds"] = elapsed
            _record_timing(metadata, "transcribe_faster_whisper", elapsed)
        elif resolved_engine == "parakeet":
            parakeet_model_name, parakeet_model_dir, parakeet_quantization = resolved_parakeet
            start_time = time.perf_counter()
            text = _transcribe_clips_parakeet(
                clips,
                parakeet_model_name,
                parakeet_model_dir,
                parakeet_quantization,
                vad_timestamps,
            )
            elapsed = time.perf_counter() - start_time
            transcription_metadata["elapsed_seconds"] = elapsed
            _record_timing(metadata, "transcribe_parakeet", elapsed)
        else:
            start_time = time.perf_counter()
            text = _transcribe_clips_whispercpp(settings, clips, Path(model), language, vad_timestamps)
            elapsed = time.perf_counter() - start_time
            transcription_metadata["elapsed_seconds"] = elapsed
            _record_timing(metadata, "transcribe_whispercpp", elapsed)
        vad_metadata_path = _vad_metadata_path(output_dir)
        return text, metadata, word_timestamps, vad_metadata_path

    if resolved_engine == "faster-whisper":
        start_time = time.perf_counter()
        text, word_timestamps = transcribe_audio_faster_whisper(
            audio_path=audio,
            model_name_or_path=model,
            language=language,
            convert=convert,
            offline=settings.offline,
        )
        elapsed = time.perf_counter() - start_time
        transcription_metadata["elapsed_seconds"] = elapsed
        _record_timing(metadata, "transcribe_faster_whisper", elapsed)
    elif resolved_engine == "parakeet":
        parakeet_model_name, parakeet_model_dir, parakeet_quantization = resolved_parakeet
        start_time = time.perf_counter()
        text = transcribe_audio_parakeet(
            audio_path=audio,
            model_name=parakeet_model_name,
            model_dir=parakeet_model_dir,
            quantization=parakeet_quantization,
            convert=convert,
        )
        elapsed = time.perf_counter() - start_time
        transcription_metadata["elapsed_seconds"] = elapsed
        _record_timing(metadata, "transcribe_parakeet", elapsed)
    elif resolved_engine == "whisperx":
        duration_seconds = _audio_duration_seconds(audio)
        auto_diarize = bool(duration_seconds and duration_seconds > 300)
        start_time = time.perf_counter()
        text, word_timestamps, wx_metadata = transcribe_audio_whisperx(
            audio_path=audio,
            model_name=whisperx_model or model,
            device=whisperx_device,
            language=language,
            convert=convert,
            diarize=whisperx_diarize or auto_diarize,
            diarize_model=whisperx_diarize_model,
            offline=settings.offline,
            compute_type=whisperx_compute_type,
        )
        elapsed = time.perf_counter() - start_time
        transcription_metadata["elapsed_seconds"] = elapsed
        _record_timing(metadata, "transcribe_whisperx", elapsed)
        metadata.update(wx_metadata)
    else:
        start_time = time.perf_counter()
        text = transcribe_audio(
            settings,
            audio_path=audio,
            model_path=Path(model),
            language=language,
            convert=convert,
        )
        elapsed = time.perf_counter() - start_time
        transcription_metadata["elapsed_seconds"] = elapsed
        _record_timing(metadata, "transcribe_whispercpp", elapsed)
    return text, metadata, word_timestamps, vad_metadata_path


def _transcribe_clips_whispercpp(
    settings, clips: list[dict], model_path: Path, language: str, include_timestamps: bool
) -> str:
    from .transcribe import transcribe_audio

    texts = []
    for clip in clips:
        clip_path = Path(clip["clip_path"])
        text = transcribe_audio(
            settings,
            audio_path=clip_path,
            model_path=model_path,
            language=language,
            convert=False,
        )
        clip["text"] = text
        if text:
            if include_timestamps:
                texts.append(_format_clip_line(clip, text))
            else:
                texts.append(text)
    return "\n".join(texts).strip()


def _transcribe_clips_faster_whisper(
    clips: list[dict],
    model_name_or_path: str,
    language: str,
    include_timestamps: bool,
    offline: bool,
) -> tuple[str, list[dict]]:
    from .transcribe import _collect_word_timestamps, load_faster_whisper_model

    model = load_faster_whisper_model(model_name_or_path, device="cpu", compute_type="int8", offline=offline)
    texts = []
    words: list[dict] = []
    for clip in clips:
        clip_path = Path(clip["clip_path"])
        segments, _info = model.transcribe(str(clip_path), language=language, word_timestamps=True)
        segment_list = list(segments)
        lines = [seg.text.strip() for seg in segment_list if seg.text]
        text = "\n".join(lines).strip()
        clip["text"] = text
        if text:
            if include_timestamps:
                texts.append(_format_clip_line(clip, text))
            else:
                texts.append(text)
        clip_offset = float(clip.get("start_ms", 0.0)) / 1000.0
        words.extend(_collect_word_timestamps(segment_list, offset_seconds=clip_offset))
    return "\n".join(texts).strip(), words


def _transcribe_clips_parakeet(
    clips: list[dict],
    model_name: str,
    model_dir: Optional[Path],
    quantization: Optional[str],
    include_timestamps: bool,
    offline: bool,
) -> str:
    from .transcribe import load_parakeet_model

    model = load_parakeet_model(model_name, model_dir=model_dir, quantization=quantization, offline=offline)
    texts = []
    for clip in clips:
        clip_path = Path(clip["clip_path"])
        text = model.recognize(str(clip_path)).strip()
        clip["text"] = text
        if text:
            if include_timestamps:
                texts.append(_format_clip_line(clip, text))
            else:
                texts.append(text)
    return "\n".join(texts).strip()


@app.command()
def help(ctx: typer.Context) -> None:
    """Show help for the CLI."""
    print(ctx.get_help())


@app.command()
def init() -> None:
    """Initialize local storage and vector index."""
    from .journal import init_store

    settings = load_settings()
    init_store(settings)
    print(f"Initialized store at {settings.base_dir}")


@app.command()
def add_text(
    text: str = typer.Argument(..., help="Journal text to store."),
    source: Optional[str] = typer.Option(None, help="Source label."),
) -> None:
    """Add a journal entry from raw text."""
    from .journal import add_entry

    settings = load_settings()
    entry_id = add_entry(settings, text=text, source=source)
    print(f"Added entry {entry_id}")


@app.command()
def dedupe() -> None:
    """Remove duplicate audio entries (keeps latest by recorded_at)."""
    settings = load_settings()
    db_path = settings.db_path
    if not db_path.exists():
        builtins.print("No journal database found.")
        raise typer.Exit()
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            "DELETE FROM entries "
            "WHERE audio_path IS NOT NULL AND audio_path != '' "
            "AND EXISTS ("
            "  SELECT 1 FROM entries e2 "
            "  WHERE e2.audio_path = entries.audio_path "
            "    AND (e2.recorded_at > entries.recorded_at "
            "         OR (e2.recorded_at = entries.recorded_at AND e2.id > entries.id))"
            ")"
        )
        conn.commit()
        builtins.print(f"Removed {cur.rowcount} duplicate entries")


@app.command()
def vad(
    audio: Path = typer.Argument(..., help="Path to audio file."),
    output_dir: Optional[Path] = typer.Option(None, help="Directory to store VAD clips + metadata."),
    vad_threshold: float = typer.Option(0.5, help="VAD speech threshold (0.0-1.0)."),
    vad_min_speech_ms: int = typer.Option(250, help="VAD minimum speech duration (ms)."),
    vad_min_silence_ms: int = typer.Option(100, help="VAD minimum silence duration (ms)."),
    vad_speech_pad_ms: int = typer.Option(30, help="VAD padding around speech (ms)."),
    vad_max_speech_s: Optional[float] = typer.Option(None, help="VAD max speech duration (s)."),
    vad_samples_overlap: float = typer.Option(0.1, help="VAD samples overlap (seconds)."),
    vad_bin: Optional[Path] = typer.Option(None, help="Path to whisper.cpp vad-speech-segments binary."),
    vad_model: Optional[Path] = typer.Option(None, help="Path to whisper.cpp VAD model (GGML)."),
) -> None:
    """Detect speech segments and write clips + metadata."""
    from .transcribe import TranscriptionError
    from .vad import create_vad_clips

    settings = load_settings()
    try:
        target_dir = output_dir or _default_vad_dir(audio)
        resolved_bin, resolved_model = _resolve_vad_paths(settings, vad_bin, vad_model)
        metadata = create_vad_clips(
            audio_path=audio,
            output_dir=target_dir,
            threshold=vad_threshold,
            min_speech_ms=vad_min_speech_ms,
            min_silence_ms=vad_min_silence_ms,
            speech_pad_ms=vad_speech_pad_ms,
            max_speech_s=vad_max_speech_s,
            samples_overlap=vad_samples_overlap,
            vad_bin=resolved_bin,
            vad_model=resolved_model,
        )
        _write_vad_metadata(_vad_metadata_path(target_dir), metadata)
    except TranscriptionError as exc:
        raise typer.Exit(str(exc))

    print(
        f"VAD segments: {len(metadata['segments'])} (metadata: {_vad_metadata_path(target_dir)})"
    )


@app.command("audio")
def audio(
    audio: Optional[Path] = typer.Argument(
        None, help="Path to audio file or directory (defaults to ./audio)."
    ),
    model: str = typer.Argument(
        "base.en",
        help="GGUF path for whisper.cpp, model name/path for faster-whisper, or Parakeet model name.",
    ),
    save: Optional[Path] = typer.Option(None, help="Save transcript to file."),
    ingest: bool = typer.Option(True, help="Store transcript in journal."),
    source: Optional[str] = typer.Option("whisper.cpp", help="Source label."),
    convert: bool = typer.Option(True, help="Convert to 16kHz mono WAV with ffmpeg."),
    vad: bool = typer.Option(
        False,
        help="Use whisper.cpp VAD to split audio into speech clips first (forces 16kHz mono conversion).",
    ),
    vad_dir: Optional[Path] = typer.Option(None, help="Directory to store VAD clips + metadata."),
    vad_threshold: float = typer.Option(0.5, help="VAD speech threshold (0.0-1.0)."),
    vad_min_speech_ms: int = typer.Option(250, help="VAD minimum speech duration (ms)."),
    vad_min_silence_ms: int = typer.Option(100, help="VAD minimum silence duration (ms)."),
    vad_speech_pad_ms: int = typer.Option(30, help="VAD padding around speech (ms)."),
    vad_max_speech_s: Optional[float] = typer.Option(None, help="VAD max speech duration (s)."),
    vad_samples_overlap: float = typer.Option(0.1, help="VAD samples overlap (seconds)."),
    vad_bin: Optional[Path] = typer.Option(None, help="Path to whisper.cpp vad-speech-segments binary."),
    vad_model: Optional[Path] = typer.Option(None, help="Path to whisper.cpp VAD model (GGML)."),
    vad_timestamps: bool = typer.Option(
        True,
        help="Include clip timestamps in transcript when VAD is enabled.",
    ),
    engine: str = typer.Option(
        "auto",
        help="Transcription engine: auto (faster-whisper <5 min, whisperx >=5 min + diarize), faster-whisper, whispercpp, parakeet, or whisperx.",
    ),
    parakeet_model: Optional[str] = typer.Option(
        None,
        help="Parakeet model name for onnx-asr (e.g., nemo-parakeet-tdt-0.6b-v3).",
    ),
    parakeet_dir: Optional[Path] = typer.Option(
        None,
        help="Directory containing Parakeet model files (optional).",
    ),
    parakeet_quant: Optional[str] = typer.Option(
        None,
        help="Parakeet quantization suffix (e.g., int8).",
    ),
    whisperx_model: str = typer.Option(
        "medium",
        help="WhisperX model to use when engine=whisperx.",
    ),
    whisperx_device: str = typer.Option(
        "cpu",
        help="Device to run WhisperX on (cpu/cuda).",
    ),
    whisperx_diarize: bool = typer.Option(
        False,
        help="Run WhisperX diarization (speaker labeling).",
    ),
    whisperx_diarize_model: Optional[str] = typer.Option(
        None,
        help="Pyannote diarization model for WhisperX (default: pyannote/speaker-diarization).",
    ),
    language: str = typer.Option("en", help="Language code (e.g., en)."),
) -> None:
    """Transcribe audio offline with whisper.cpp or faster-whisper."""
    from .journal import add_entry
    from .transcribe import TranscriptionError

    default_audio_dir = False
    if audio is None:
        audio = Path("audio")
        default_audio_dir = True

    if audio.is_dir():
        # Directory ingest path mirrors `ingest-dir` behavior.
        ingest_dir(
            directory=audio,
            model=model,
            engine=engine,
            convert=convert,
            vad=vad,
            vad_dir=vad_dir,
            vad_threshold=vad_threshold,
            vad_min_speech_ms=vad_min_speech_ms,
            vad_min_silence_ms=vad_min_silence_ms,
            vad_speech_pad_ms=vad_speech_pad_ms,
            vad_max_speech_s=vad_max_speech_s,
            vad_samples_overlap=vad_samples_overlap,
            vad_bin=vad_bin,
            vad_model=vad_model,
            vad_timestamps=vad_timestamps,
            source=source,
            parakeet_model=parakeet_model,
            parakeet_dir=parakeet_dir,
            parakeet_quant=parakeet_quant,
            archive_dir=None,
            recursive=default_audio_dir or audio.name == "audio",
            extensions="wav,mp3,m4a,flac,ogg,opus,webm",
            whisperx_model=whisperx_model,
            whisperx_device=whisperx_device,
            whisperx_diarize=whisperx_diarize,
            whisperx_diarize_model=whisperx_diarize_model,
            language=language,
        )
        return
    settings = load_settings()
    try:
        text, metadata, word_timestamps, vad_metadata_path = _transcribe_to_text(
            settings,
            audio,
            model,
            convert,
            vad,
            vad_dir,
            vad_threshold,
            vad_min_speech_ms,
            vad_min_silence_ms,
            vad_speech_pad_ms,
            vad_max_speech_s,
            vad_samples_overlap,
            vad_bin,
            vad_model,
            vad_timestamps,
            engine,
            parakeet_model,
            parakeet_dir,
            parakeet_quant,
            whisperx_model,
            whisperx_device,
            whisperx_diarize,
            whisperx_diarize_model,
            language=language,
        )
    except TranscriptionError as exc:
        raise typer.Exit(str(exc))

    metadata["word_timestamps"] = word_timestamps
    if vad_metadata_path is not None:
        _write_vad_metadata(vad_metadata_path, metadata)
    raw_text = text
    display_text = text
    if word_timestamps and not vad:
        display_text = _inject_elapsed_markers(word_timestamps)

    if save:
        save.parent.mkdir(parents=True, exist_ok=True)
        save.write_text(display_text, encoding="utf-8")

    if ingest:
        entry_id = add_entry(
            settings,
            text=raw_text,
            source=source,
            audio_path=str(audio),
            metadata=metadata,
        )
        print(f"Added entry {entry_id}")
        _print_speed_ratio(metadata, audio)
        return entry_id
    else:
        print(display_text)
        _print_speed_ratio(metadata, audio)
        return None


@app.command("list")
def list_entries(
    audio_dir: Optional[Path] = typer.Option(
        None,
        help="Base directory to resolve moved audio files (searched recursively).",
    ),
    dedupe: bool = typer.Option(
        True,
        help="Collapse duplicate audio_path entries (keeps latest by recorded_at).",
    ),
    debug: bool = typer.Option(False, help="Print per-entry resolution details."),
    db_only: bool = typer.Option(False, help="Use only stored DB metadata (no ffprobe)."),
) -> None:
    """List already transcribed audio files with size, minutes, word count, and diarization."""
    settings = load_settings()
    if not settings.db_path.exists():
        builtins.print("No journal database found.")
        raise typer.Exit()

    resolved_audio_dir = audio_dir
    if resolved_audio_dir is None:
        default_audio_dir = Path("audio")
        if default_audio_dir.exists():
            resolved_audio_dir = default_audio_dir

    audio_index: dict[str, Path] = {}
    if resolved_audio_dir and resolved_audio_dir.exists():
        audio_index = _build_audio_index(resolved_audio_dir)

    with sqlite3.connect(settings.db_path) as conn:
        conn.row_factory = sqlite3.Row
        from .journal import ensure_audio_columns
        ensure_audio_columns(conn)
        _migrate_drop_duration_minutes(conn)
        rows = conn.execute(
            "SELECT id, recorded_at, text, audio_path, audio_size_bytes, audio_duration_seconds, metadata FROM entries "
            "WHERE audio_path IS NOT NULL AND audio_path != '' "
            "ORDER BY recorded_at"
        ).fetchall()

    if not rows:
        builtins.print("No audio entries found.")
        raise typer.Exit()

    if dedupe:
        deduped: dict[str, sqlite3.Row] = {}
        for row in rows:
            deduped[row["audio_path"]] = row
        rows = list(deduped.values())

    items = []
    seen_display: set[str] = set()
    updates: list[tuple[Optional[int], Optional[float], int]] = []
    for row in rows:
        audio_path = row["audio_path"]
        text = row["text"] or ""
        word_count = len(text.split())
        resolved_path = Path(audio_path)
        if not resolved_path.exists():
            resolved_path = audio_index.get(resolved_path.name, resolved_path)

        size_bytes = row["audio_size_bytes"]
        if size_bytes is None and resolved_path.exists() and not db_only:
            size_bytes = resolved_path.stat().st_size
        size_mb = f"{(size_bytes / (1024 * 1024)):.2f}" if size_bytes is not None else "n/a"

        duration_seconds = row["audio_duration_seconds"]
        if duration_seconds is None and resolved_path.exists() and not db_only:
            duration_seconds = _audio_duration_seconds(resolved_path)
        duration_minutes = (duration_seconds / 60) if duration_seconds is not None else None
        minutes_display = f"{duration_minutes:.1f}" if duration_minutes is not None else "n/a"

        display_path = str(resolved_path)
        if not resolved_path.exists():
            display_path = f"{display_path} (missing)"
        if dedupe:
            normalized = display_path.strip().lower()
            if normalized in seen_display:
                continue
            seen_display.add(normalized)
        metadata = json.loads(row["metadata"] or "{}")
        transcription_meta = metadata.get("transcription") or {}
        engine_display = transcription_meta.get("engine") or "n/a"
        model_display = transcription_meta.get("model") or "n/a"
        diarize_flag = transcription_meta.get("diarize")
        diarization_segments = (metadata.get("whisperx") or {}).get("diarization")
        diarized = diarize_flag is True or bool(diarization_segments)
        diarize_display = "yes" if diarized else ("no" if diarize_flag is False else "n/a")
        option_parts = []
        for key in ("device", "compute_type", "quantization", "model_dir", "whisper_bin"):
            value = transcription_meta.get(key)
            if value:
                option_parts.append(f"{key}={value}")
        if diarize_flag is True:
            option_parts.append("diarize")
        options_display = ",".join(option_parts) if option_parts else "n/a"
        elapsed_seconds = transcription_meta.get("elapsed_seconds")
        ratio = None
        if isinstance(elapsed_seconds, (int, float)) and duration_seconds:
            ratio = elapsed_seconds / duration_seconds
        transcription_minutes = None
        if isinstance(elapsed_seconds, (int, float)):
            transcription_minutes = elapsed_seconds / 60
        if transcription_minutes is not None and duration_minutes:
            xtran = transcription_minutes / duration_minutes
        else:
            xtran = None
        xtran_display = f"{xtran:.2f}" if xtran is not None else "n/a"
        ratio_display = f"{ratio:.2f}" if ratio is not None else "n/a"
        items.append(
            (
                display_path,
                size_mb,
                minutes_display,
                xtran_display,
                word_count,
                diarize_display,
                ratio_display,
                engine_display,
                model_display,
                options_display,
            )
        )
        if debug:
            builtins.print(
                f"debug: path={audio_path} resolved={resolved_path} "
                f"size={size_bytes} duration={duration_seconds}"
            )
        if not db_only and (row["audio_size_bytes"] is None or row["audio_duration_seconds"] is None):
            updates.append((size_bytes, duration_seconds, row["id"]))
    if updates:
        with sqlite3.connect(settings.db_path) as conn:
            conn.executemany(
                "UPDATE entries SET audio_size_bytes = ?, audio_duration_seconds = ? WHERE id = ?",
                updates,
            )
            conn.commit()

    path_width = max([len("audio_path")] + [len(item[0]) for item in items]) + 5
    builtins.print(
        f"{'audio_path':<{path_width}}size_mb min xtran words diarize rt engine model options"
    )
    for (
        display_path,
        size_mb,
        minutes_display,
        xtran_display,
        word_count,
        diarize_display,
        ratio_display,
        engine_display,
        model_display,
        options_display,
    ) in items:
        builtins.print(
            f"{display_path:<{path_width}}{size_mb:>7} {minutes_display:>7} {xtran_display:>7} "
            f"{word_count:>5} {diarize_display:>7} {ratio_display:>5} {engine_display:>7} "
            f"{model_display:>7} {options_display:>12}"
        )


@app.command()
def dump(
    output_dir: Path = typer.Option(TMP_DIR, help="Directory for per-recording transcript files."),
    audio_dir: Optional[Path] = typer.Option(
        None,
        help="Base directory to resolve moved audio files (searched recursively).",
    ),
    interval_minutes: int = typer.Option(5, help="Elapsed marker interval in minutes."),
) -> None:
    """Write one transcript file per recording."""
    settings = load_settings()
    if not settings.db_path.exists():
        builtins.print("No journal database found.")
        raise typer.Exit()

    resolved_audio_dir = audio_dir
    if resolved_audio_dir is None:
        default_audio_dir = Path("audio")
        if default_audio_dir.exists():
            resolved_audio_dir = default_audio_dir

    audio_index: dict[str, Path] = {}
    if resolved_audio_dir and resolved_audio_dir.exists():
        audio_index = _build_audio_index(resolved_audio_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    interval_seconds = max(1, interval_minutes) * 60

    with sqlite3.connect(settings.db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, recorded_at, text, audio_path, metadata FROM entries "
            "WHERE audio_path IS NOT NULL AND audio_path != '' "
            "ORDER BY recorded_at"
        ).fetchall()

    if not rows:
        builtins.print("No audio entries found.")
        raise typer.Exit()

    for row in rows:
        audio_path = row["audio_path"]
        resolved_path = Path(audio_path)
        if not resolved_path.exists():
            resolved_path = audio_index.get(resolved_path.name, resolved_path)

        duration_seconds = _audio_duration_seconds(resolved_path) if resolved_path.exists() else None
        metadata = json.loads(row["metadata"] or "{}")
        word_timestamps = metadata.get("word_timestamps") or []
        if word_timestamps:
            transcript_text = _inject_elapsed_lines(word_timestamps, interval_minutes=interval_minutes)
            marker_line = f"Elapsed markers: every {interval_minutes} min"
        else:
            transcript_text = (row["text"] or "").strip()
            marker_line = "Elapsed markers: unavailable (no word timestamps)"

        file_base = resolved_path.stem if resolved_path.name else f"entry_{row['id']}"
        out_path = output_dir / f"{file_base}_{row['id']}.txt"
        header_lines = [
            f"Filename: {audio_path}",
            f"Duration: {_format_duration(duration_seconds)}",
            marker_line,
            "",
        ]
        out_path.write_text("\n".join(header_lines) + transcript_text + "\n", encoding="utf-8")

    builtins.print(f"Wrote {len(rows)} transcript files to {output_dir}")


@app.command()
def metadata(
    entry_id: Optional[int] = typer.Option(None, "--id", "-i", help="Entry id to inspect."),
    audio: Optional[Path] = typer.Option(
        None,
        "--audio",
        "-a",
        help="Audio path to match (exact match in journal).",
    ),
    full: bool = typer.Option(False, "--full", help="Show full metadata (default: transcription only)."),
    all_entries: bool = typer.Option(
        False,
        "--all",
        help="Show all matching entries (default: most recent match).",
    ),
) -> None:
    """Show stored transcription metadata."""
    settings = load_settings()
    if not settings.db_path.exists():
        builtins.print("No journal database found.")
        raise typer.Exit()
    if entry_id is None and audio is None and not all_entries:
        raise typer.BadParameter("Provide --id or --audio (or use --all).")
    if entry_id is not None and audio is not None:
        raise typer.BadParameter("Use --id or --audio, not both.")

    query = "SELECT id, recorded_at, audio_path, metadata FROM entries "
    params: list[object] = []
    if entry_id is not None:
        query += "WHERE id = ?"
        params.append(entry_id)
    elif audio is not None:
        query += "WHERE audio_path = ?"
        params.append(str(audio))
        if not all_entries:
            query += " ORDER BY recorded_at DESC"
    if not all_entries:
        query += " LIMIT 1"

    with sqlite3.connect(settings.db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, params).fetchall()

    if not rows:
        builtins.print("No matching entries found.")
        raise typer.Exit()

    payloads = []
    for row in rows:
        metadata = json.loads(row["metadata"] or "{}")
        payload = _select_metadata_payload(metadata, full)
        if all_entries:
            payloads.append(
                {
                    "id": row["id"],
                    "recorded_at": row["recorded_at"],
                    "audio_path": row["audio_path"],
                    "metadata": payload,
                }
            )
        else:
            payloads.append(payload)

    output = payloads if all_entries else payloads[0]
    builtins.print(json.dumps(output, indent=2, ensure_ascii=True))


@app.command()
def export(
    output_dir: Path = typer.Argument(TMP_DIR, help="Directory for per-recording transcript files."),
    audio_dir: Optional[Path] = typer.Option(
        None,
        help="Base directory to resolve moved audio files (searched recursively).",
    ),
    interval_minutes: int = typer.Option(5, help="Elapsed marker interval in minutes."),
) -> None:
    """Export transcripts with one file per recording."""
    dump(output_dir=output_dir, audio_dir=audio_dir, interval_minutes=interval_minutes)


@app.command()
def ingest_dir(
    directory: Path = typer.Argument(..., help="Directory with audio files."),
    model: str = typer.Argument(
        "base.en",
        help="GGUF path for whisper.cpp, model name/path for faster-whisper, or Parakeet model name.",
    ),
    engine: str = typer.Option(
        "auto",
        help="Transcription engine: auto (faster-whisper <5 min, whisperx >=5 min + diarize), faster-whisper, whispercpp, or parakeet.",
    ),
    convert: bool = typer.Option(True, help="Convert to 16kHz mono WAV with ffmpeg."),
    vad: bool = typer.Option(
        False,
        help="Use whisper.cpp VAD to split audio into speech clips first (forces 16kHz mono conversion).",
    ),
    vad_dir: Optional[Path] = typer.Option(None, help="Directory to store VAD clips + metadata."),
    vad_threshold: float = typer.Option(0.5, help="VAD speech threshold (0.0-1.0)."),
    vad_min_speech_ms: int = typer.Option(250, help="VAD minimum speech duration (ms)."),
    vad_min_silence_ms: int = typer.Option(100, help="VAD minimum silence duration (ms)."),
    vad_speech_pad_ms: int = typer.Option(30, help="VAD padding around speech (ms)."),
    vad_max_speech_s: Optional[float] = typer.Option(None, help="VAD max speech duration (s)."),
    vad_samples_overlap: float = typer.Option(0.1, help="VAD samples overlap (seconds)."),
    vad_bin: Optional[Path] = typer.Option(None, help="Path to whisper.cpp vad-speech-segments binary."),
    vad_model: Optional[Path] = typer.Option(None, help="Path to whisper.cpp VAD model (GGML)."),
    vad_timestamps: bool = typer.Option(
        True,
        help="Include clip timestamps in transcript when VAD is enabled.",
    ),
    source: Optional[str] = typer.Option("whisper.cpp", help="Source label."),
    parakeet_model: Optional[str] = typer.Option(
        None,
        help="Parakeet model name for onnx-asr (e.g., nemo-parakeet-tdt-0.6b-v3).",
    ),
    parakeet_dir: Optional[Path] = typer.Option(
        None,
        help="Directory containing Parakeet model files (optional).",
    ),
    parakeet_quant: Optional[str] = typer.Option(
        None,
        help="Parakeet quantization suffix (e.g., int8).",
    ),
    archive_dir: Optional[Path] = typer.Option(None, help="Move processed files here."),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Recurse into subdirectories."),
    extensions: str = typer.Option(
        "wav,mp3,m4a,flac,ogg,opus,webm",
        help="Comma-separated list of extensions to ingest.",
    ),
    whisperx_model: str = typer.Option(
        "medium",
        help="Override model name when engine=whisperx.",
    ),
    whisperx_device: str = typer.Option(
        "cpu",
        help="Device for WhisperX (e.g., cpu, cuda).",
    ),
    whisperx_diarize: bool = typer.Option(
        False,
        help="Run WhisperX diarization (speaker labels only).",
    ),
    whisperx_diarize_model: Optional[str] = typer.Option(
        None,
        help="Pyannote diarization model identifier (WhisperX only).",
    ),
    language: str = typer.Option("en", help="Language code (e.g., en)."),
) -> None:
    """Ingest and transcribe all audio files in a directory."""
    from .journal import add_entry, has_audio
    from .transcribe import (
        TranscriptionError,
        transcribe_audio,
        transcribe_audio_faster_whisper,
        transcribe_audio_parakeet,
        transcribe_audio_whisperx,
    )
    from .vad import create_vad_clips

    settings = load_settings()
    resolved_parakeet = _resolve_parakeet_settings(
        settings,
        parakeet_model,
        parakeet_dir,
        parakeet_quant,
        model_fallback=model,
    )
    if not directory.exists():
        raise typer.Exit(f"Directory not found: {directory}")

    files = _iter_audio_files(directory, recursive=recursive, extensions=extensions)
    if not files:
        print("No audio files found.")
        raise typer.Exit()

    if archive_dir:
        archive_dir.mkdir(parents=True, exist_ok=True)

    for audio in files:
        archive_target = archive_dir
        if archive_target is None and directory.name == "audio":
            archive_target = _archive_week_dir(directory, audio)

        if has_audio(settings, str(audio)):
            print(f"Skipping already ingested: {audio}")
            if archive_target:
                archive_target.mkdir(parents=True, exist_ok=True)
                target = archive_target / audio.name
                audio.replace(target)
                default_vad_dir = _default_vad_dir(audio)
                if default_vad_dir.exists():
                    default_vad_dir.replace(archive_target / default_vad_dir.name)
            continue
        try:
            metadata: dict = {}
            word_timestamps: list[dict] = []
            vad_metadata_path: Optional[Path] = None
            resolved_engine, duration_seconds = _resolve_engine_for_audio(engine, audio)
            transcription_metadata = _build_transcription_metadata(
                settings,
                resolved_engine,
                model,
                parakeet_model,
                parakeet_dir,
                parakeet_quant,
                whisperx_model,
                whisperx_device,
                whisperx_diarize,
                None,
            )
            start_time = time.perf_counter()
            if vad:
                output_dir = (vad_dir / audio.stem) if vad_dir else _default_vad_dir(audio)
                resolved_bin, resolved_model = _resolve_vad_paths(settings, vad_bin, vad_model)
                metadata = create_vad_clips(
                    audio_path=audio,
                    output_dir=output_dir,
                    threshold=vad_threshold,
                    min_speech_ms=vad_min_speech_ms,
                    min_silence_ms=vad_min_silence_ms,
                    speech_pad_ms=vad_speech_pad_ms,
                    max_speech_s=vad_max_speech_s,
                    samples_overlap=vad_samples_overlap,
                    vad_bin=resolved_bin,
                    vad_model=resolved_model,
                )
                metadata["transcription"] = transcription_metadata
                clips = metadata["segments"]
                if not clips:
                    raise TranscriptionError("No speech segments detected.")
                if resolved_engine == "faster-whisper":
                    start_time = time.perf_counter()
                    text, word_timestamps = _transcribe_clips_faster_whisper(
                        clips,
                        model,
                        language,
                        vad_timestamps,
                        settings.offline,
                    )
                    elapsed = time.perf_counter() - start_time
                    transcription_metadata["elapsed_seconds"] = elapsed
                    _record_timing(metadata, "transcribe_faster_whisper", elapsed)
                elif resolved_engine == "parakeet":
                    parakeet_model_name, parakeet_model_dir, parakeet_quantization = resolved_parakeet
                    start_time = time.perf_counter()
                    text = _transcribe_clips_parakeet(
                        clips,
                        parakeet_model_name,
                        parakeet_model_dir,
                        parakeet_quantization,
                        vad_timestamps,
                    )
                    elapsed = time.perf_counter() - start_time
                    transcription_metadata["elapsed_seconds"] = elapsed
                    _record_timing(metadata, "transcribe_parakeet", elapsed)
                else:
                    start_time = time.perf_counter()
                    text = _transcribe_clips_whispercpp(
                        settings,
                        clips,
                        Path(model),
                        language,
                        vad_timestamps,
                    )
                    elapsed = time.perf_counter() - start_time
                    transcription_metadata["elapsed_seconds"] = elapsed
                    _record_timing(metadata, "transcribe_whispercpp", elapsed)
                vad_metadata_path = _vad_metadata_path(output_dir)
            else:
                metadata["transcription"] = transcription_metadata
                if resolved_engine == "faster-whisper":
                    start_time = time.perf_counter()
                    text, word_timestamps = transcribe_audio_faster_whisper(
                        audio_path=audio,
                        model_name_or_path=model,
                        language=language,
                        convert=convert,
                        offline=settings.offline,
                    )
                    elapsed = time.perf_counter() - start_time
                    transcription_metadata["elapsed_seconds"] = elapsed
                    _record_timing(metadata, "transcribe_faster_whisper", elapsed)
                elif resolved_engine == "parakeet":
                    parakeet_model_name, parakeet_model_dir, parakeet_quantization = resolved_parakeet
                    start_time = time.perf_counter()
                    text = transcribe_audio_parakeet(
                        audio_path=audio,
                        model_name=parakeet_model_name,
                        model_dir=parakeet_model_dir,
                        quantization=parakeet_quantization,
                        convert=convert,
                        )
                    elapsed = time.perf_counter() - start_time
                    transcription_metadata["elapsed_seconds"] = elapsed
                    _record_timing(metadata, "transcribe_parakeet", elapsed)
                elif resolved_engine == "whisperx":
                    auto_diarize = bool(duration_seconds and duration_seconds > 300)
                    start_time = time.perf_counter()
                    text, word_timestamps, wx_metadata = transcribe_audio_whisperx(
                        audio_path=audio,
                        model_name=whisperx_model or model,
                        device=whisperx_device,
                        language=language,
                        convert=convert,
                        diarize=whisperx_diarize or auto_diarize,
                        diarize_model=whisperx_diarize_model,
                        offline=settings.offline,
                    )
                    elapsed = time.perf_counter() - start_time
                    transcription_metadata["elapsed_seconds"] = elapsed
                    _record_timing(metadata, "transcribe_whisperx", elapsed)
                    metadata.update(wx_metadata)
                else:
                    start_time = time.perf_counter()
                    text = transcribe_audio(
                        settings,
                        audio_path=audio,
                        model_path=Path(model),
                        language=language,
                        convert=convert,
                    )
                    elapsed = time.perf_counter() - start_time
                    transcription_metadata["elapsed_seconds"] = elapsed
                    _record_timing(metadata, "transcribe_whispercpp", elapsed)
            transcription_metadata["elapsed_seconds"] = time.perf_counter() - start_time
        except TranscriptionError as exc:
            print(f"Failed: {audio} ({exc})")
            continue

        metadata["word_timestamps"] = word_timestamps
        if vad_metadata_path is not None:
            _write_vad_metadata(vad_metadata_path, metadata)
        entry_id = add_entry(settings, text=text, source=source, audio_path=str(audio), metadata=metadata)
        print(f"Added entry {entry_id} from {audio}")
        _print_speed_ratio(metadata, audio, duration_seconds)

        if archive_target:
            archive_target.mkdir(parents=True, exist_ok=True)
            target = archive_target / audio.name
            audio.replace(target)
            if vad and vad_dir is None:
                default_vad_dir = _default_vad_dir(audio)
                if default_vad_dir.exists():
                    default_vad_dir.replace(archive_target / default_vad_dir.name)


@app.command()
def reingest(
    audio: Path = typer.Argument(..., help="Audio file or directory."),
    model: str = typer.Argument(
        "base.en",
        help="GGUF path for whisper.cpp, model name/path for faster-whisper, or Parakeet model name.",
    ),
    engine: str = typer.Option(
        "auto",
        help="Transcription engine: auto (faster-whisper <5 min, whisperx >=5 min + diarize), faster-whisper, whispercpp, or parakeet.",
    ),
    convert: bool = typer.Option(True, help="Convert to 16kHz mono WAV with ffmpeg."),
    vad: bool = typer.Option(
        False,
        help="Use whisper.cpp VAD to split audio into speech clips first (forces 16kHz mono conversion).",
    ),
    vad_dir: Optional[Path] = typer.Option(None, help="Directory to store VAD clips + metadata."),
    vad_threshold: float = typer.Option(0.5, help="VAD speech threshold (0.0-1.0)."),
    vad_min_speech_ms: int = typer.Option(250, help="VAD minimum speech duration (ms)."),
    vad_min_silence_ms: int = typer.Option(100, help="VAD minimum silence duration (ms)."),
    vad_speech_pad_ms: int = typer.Option(30, help="VAD padding around speech (ms)."),
    vad_max_speech_s: Optional[float] = typer.Option(None, help="VAD max speech duration (s)."),
    vad_samples_overlap: float = typer.Option(0.1, help="VAD samples overlap (seconds)."),
    vad_bin: Optional[Path] = typer.Option(None, help="Path to whisper.cpp vad-speech-segments binary."),
    vad_model: Optional[Path] = typer.Option(None, help="Path to whisper.cpp VAD model (GGML)."),
    vad_timestamps: bool = typer.Option(
        True,
        help="Include clip timestamps in transcript when VAD is enabled.",
    ),
    source: Optional[str] = typer.Option("whisper.cpp", help="Source label."),
    parakeet_model: Optional[str] = typer.Option(
        None,
        help="Parakeet model name for onnx-asr (e.g., nemo-parakeet-tdt-0.6b-v3).",
    ),
    parakeet_dir: Optional[Path] = typer.Option(
        None,
        help="Directory containing Parakeet model files (optional).",
    ),
    parakeet_quant: Optional[str] = typer.Option(
        None,
        help="Parakeet quantization suffix (e.g., int8).",
    ),
    whisperx_model: str = typer.Option(
        "medium",
        help="WhisperX model to use when engine=whisperx.",
    ),
    whisperx_device: str = typer.Option(
        "cpu",
        help="Device to run WhisperX on (cpu/cuda).",
    ),
    whisperx_diarize: bool = typer.Option(
        False,
        help="Run WhisperX diarization (speaker labels only).",
    ),
    whisperx_diarize_model: Optional[str] = typer.Option(
        None,
        help="Pyannote diarization model identifier (WhisperX only).",
    ),
    language: str = typer.Option("en", help="Language code (e.g., en)."),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Recurse into subdirectories."),
    extensions: str = typer.Option(
        "wav,mp3,m4a,flac,ogg,opus,webm",
        help="Comma-separated list of extensions to ingest.",
    ),
) -> None:
    """Force transcribe even if already ingested."""
    from .journal import add_entry
    from .transcribe import TranscriptionError

    settings = load_settings()
    if audio.is_dir():
        files = _iter_audio_files(audio, recursive=recursive, extensions=extensions)
        if not files:
            print("No audio files found.")
            raise typer.Exit()
        for path in files:
            try:
                text, metadata, word_timestamps, vad_metadata_path = _transcribe_to_text(
                    settings,
                    path,
                    model,
                    convert,
                    vad,
                    vad_dir,
                    vad_threshold,
                    vad_min_speech_ms,
                    vad_min_silence_ms,
                    vad_speech_pad_ms,
                    vad_max_speech_s,
                    vad_samples_overlap,
                    vad_bin,
                    vad_model,
                    vad_timestamps,
                    engine,
                    parakeet_model,
                    parakeet_dir,
                    parakeet_quant,
                    whisperx_model,
                    whisperx_device,
                    whisperx_diarize,
                    whisperx_diarize_model,
                    language=language,
                )
            except TranscriptionError as exc:
                print(f"Failed: {path} ({exc})")
                continue

            metadata["word_timestamps"] = word_timestamps
            if vad_metadata_path is not None:
                _write_vad_metadata(vad_metadata_path, metadata)

            entry_id = add_entry(settings, text=text, source=source, audio_path=str(path), metadata=metadata)
            duration = _audio_duration_seconds(path) if path.exists() else None
            print(f"Added entry {entry_id}: {path} {_format_duration(duration)}")
            _print_speed_ratio(metadata, path, duration)
        return

    try:
        text, metadata, word_timestamps, vad_metadata_path = _transcribe_to_text(
            settings,
            audio,
            model,
            convert,
            vad,
            vad_dir,
            vad_threshold,
            vad_min_speech_ms,
            vad_min_silence_ms,
            vad_speech_pad_ms,
            vad_max_speech_s,
            vad_samples_overlap,
            vad_bin,
            vad_model,
            vad_timestamps,
            engine,
            parakeet_model,
            parakeet_dir,
            parakeet_quant,
            whisperx_model,
            whisperx_device,
            whisperx_diarize,
            whisperx_diarize_model,
            language=language,
        )
    except TranscriptionError as exc:
        raise typer.Exit(str(exc))

    metadata["word_timestamps"] = word_timestamps
    if vad_metadata_path is not None:
        _write_vad_metadata(vad_metadata_path, metadata)

    entry_id = add_entry(settings, text=text, source=source, audio_path=str(audio), metadata=metadata)
    duration = _audio_duration_seconds(audio) if audio.exists() else None
    print(f"Added entry {entry_id}: {audio} {_format_duration(duration)}")
    _print_speed_ratio(metadata, audio, duration)


def _print_query_results(results: list[dict]) -> None:
    if not results:
        print("No results")
        return
    for idx, item in enumerate(results, start=1):
        print(f"[{idx}] score={item['score']:.3f} entry={item['entry_id']} chunk={item['chunk_id']}")
        if item.get("recorded_at"):
            print(f"recorded_at={item['recorded_at']}")
        print(item["chunk_text"])
        print()


def _sanitize_label(value: str) -> str:
    safe = []
    for ch in value:
        if ch.isalnum() or ch in {"-", "_", "."}:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "default"


def _build_llm_prompt(rows: list[sqlite3.Row]) -> str:
    parts = [
        "You are given voice transcript entries. Produce:",
        "1) Summary (5-10 bullets)",
        "2) Task list (deduped, prioritized)",
        "3) Plan (next steps in order)",
        "4) Ideas (top 10)",
        "5) Open questions",
        "",
        "Transcript entries:",
        "",
    ]
    for row in rows:
        parts.append(f"File: {row['audio_path']}")
        if row["recorded_at"]:
            parts.append(f"Recorded: {row['recorded_at']}")
        parts.append("Text:")
        parts.append(row["text"] or "")
        parts.append("---")
    return "\n".join(parts).strip() + "\n"


def _load_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        return {}
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_cache(cache_path: Path, cache: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=True), encoding="utf-8")


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _run_llm_command(
    command: list[str],
    prompt: str,
    output_dir: Path,
    label: str,
    debug: bool,
) -> tuple[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = output_dir / f"prompt_{label}.txt"
    response_path = output_dir / f"response_{label}.txt"
    debug_path = output_dir / f"debug_{label}.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    start_time = time.perf_counter()
    result = subprocess.run(command, input=prompt, text=True, capture_output=True)
    elapsed = time.perf_counter() - start_time
    response_path.write_text(result.stdout, encoding="utf-8")
    debug_sections = [
        f"command: {shlex.join(command)}",
        f"exit_code: {result.returncode}",
        f"duration_seconds: {elapsed:.3f}",
        "stderr:",
        result.stderr.strip(),
        "",
    ]
    if debug:
        debug_sections.extend(
            [
                "prompt:",
                prompt,
                "",
                "response:",
                result.stdout,
                "",
            ]
        )
        print(prompt)
        print(result.stdout)
    debug_path.write_text("\n".join(debug_sections), encoding="utf-8")
    if result.returncode != 0:
        raise typer.Exit(f"LLM command failed (see {debug_path})")
    return result.stdout, elapsed


def _ensure_llm_runs_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS llm_runs ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "created_at TEXT NOT NULL,"
        "backend TEXT NOT NULL,"
        "model TEXT,"
        "label TEXT NOT NULL,"
        "prompt_hash TEXT NOT NULL,"
        "prompt TEXT NOT NULL,"
        "response TEXT NOT NULL,"
        "recorded_from TEXT,"
        "recorded_to TEXT,"
        "limit_count INTEGER,"
        "duration_seconds REAL"
        ")"
    )
    conn.commit()
    cols = [row[1] for row in conn.execute("PRAGMA table_info(llm_runs)").fetchall()]
    if "duration_seconds" not in cols:
        conn.execute("ALTER TABLE llm_runs ADD COLUMN duration_seconds REAL")
        conn.commit()


def _migrate_drop_duration_minutes(conn: sqlite3.Connection) -> None:
    cols = [row[1] for row in conn.execute("PRAGMA table_info(entries)").fetchall()]
    if "audio_duration_minutes" not in cols:
        return
    conn.execute("BEGIN")
    conn.execute(
        "CREATE TABLE entries_new ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "created_at TEXT NOT NULL,"
        "recorded_at TEXT,"
        "text TEXT NOT NULL,"
        "source TEXT,"
        "audio_path TEXT,"
        "audio_size_bytes INTEGER,"
        "audio_duration_seconds REAL,"
        "metadata TEXT"
        ")"
    )
    conn.execute(
        "INSERT INTO entries_new "
        "(id, created_at, recorded_at, text, source, audio_path, "
        "audio_size_bytes, audio_duration_seconds, metadata) "
        "SELECT id, created_at, recorded_at, text, source, audio_path, "
        "audio_size_bytes, audio_duration_seconds, metadata FROM entries"
    )
    conn.execute("DROP TABLE entries")
    conn.execute("ALTER TABLE entries_new RENAME TO entries")
    conn.execute("COMMIT")


def _store_llm_run(
    db_path: Path,
    backend: str,
    model: Optional[str],
    label: str,
    prompt_hash: str,
    prompt: str,
    response: str,
    recorded_from: Optional[str],
    recorded_to: Optional[str],
    limit: Optional[int],
    duration_seconds: Optional[float],
) -> None:
    with sqlite3.connect(db_path) as conn:
        _ensure_llm_runs_table(conn)
        conn.execute(
            "INSERT INTO llm_runs "
            "(created_at, backend, model, label, prompt_hash, prompt, response, recorded_from, recorded_to, "
            "limit_count, duration_seconds) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(),
                backend,
                model,
                label,
                prompt_hash,
                prompt,
                response,
                recorded_from,
                recorded_to,
                limit,
                duration_seconds,
            ),
        )
        conn.commit()


@app.command()
def summarize_db(
    backend: str = typer.Option(
        "both",
        help="Which CLI to run: codex, claude, or both.",
    ),
    codex_models: Optional[str] = typer.Option(
        None,
        help="Comma-separated Codex model list (omit to use default).",
    ),
    claude_models: Optional[str] = typer.Option(
        None,
        help="Comma-separated Claude model list (omit to use default).",
    ),
    output_dir: Path = typer.Option(TMP_DIR, help="Directory for prompts/responses/debug."),
    debug: bool = typer.Option(False, help="Print prompt/response and include them in debug files."),
    dump_only: bool = typer.Option(
        True,
        help="Print last stored LLM response without calling a model.",
    ),
    recorded_from: Optional[str] = typer.Option(None, help="Filter by recorded_at >= (ISO 8601)."),
    recorded_to: Optional[str] = typer.Option(None, help="Filter by recorded_at <= (ISO 8601)."),
    limit: Optional[int] = typer.Option(None, help="Limit number of entries included."),
) -> None:
    """Summarize the journal using local LLM CLIs."""
    settings = load_settings()
    if not settings.db_path.exists():
        builtins.print("No journal database found.")
        raise typer.Exit()

    if dump_only:
        with sqlite3.connect(settings.db_path) as conn:
            conn.row_factory = sqlite3.Row
            _ensure_llm_runs_table(conn)
            backend = backend.strip().lower()
            params: list[object] = []
            where = []
            if backend != "both":
                where.append("backend = ?")
                params.append(backend)
            where_sql = f"WHERE {' AND '.join(where)}" if where else ""
            rows = conn.execute(
                "SELECT backend, model, label, response, created_at "
                f"FROM llm_runs {where_sql} ORDER BY created_at DESC",
                params,
            ).fetchall()
        if not rows:
            builtins.print("No LLM responses stored.")
            raise typer.Exit()
        if backend == "both":
            grouped: dict[str, sqlite3.Row] = {}
            for row in rows:
                if row["label"] not in grouped:
                    grouped[row["label"]] = row
            for label, row in grouped.items():
                builtins.print(f"[{label}] {row['created_at']}")
                builtins.print(row["response"])
                builtins.print()
        else:
            row = rows[0]
            builtins.print(f"[{row['label']}] {row['created_at']}")
            builtins.print(row["response"])
        return

    params: list[object] = []
    where = ["audio_path IS NOT NULL", "audio_path != ''"]
    if recorded_from:
        where.append("recorded_at >= ?")
        params.append(recorded_from)
    if recorded_to:
        where.append("recorded_at <= ?")
        params.append(recorded_to)
    where_sql = " AND ".join(where)
    limit_sql = f" LIMIT {int(limit)}" if limit else ""

    with sqlite3.connect(settings.db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT recorded_at, text, audio_path FROM entries "
            f"WHERE {where_sql} ORDER BY recorded_at{limit_sql}",
            params,
        ).fetchall()

    if not rows:
        builtins.print("No audio entries found.")
        raise typer.Exit()

    prompt = _build_llm_prompt(rows)
    prompt_digest = _prompt_hash(prompt)
    cache_path = output_dir / "llm_cache.json"
    cache = _load_cache(cache_path)
    if debug:
        builtins.print(
            f"Loaded {len(rows)} entries; prompt hash {prompt_digest}; cache keys {len(cache)}"
        )
    backend = backend.strip().lower()
    if backend not in {"codex", "claude", "both"}:
        raise typer.Exit("backend must be codex, claude, or both")

    if backend in {"codex", "both"}:
        models = [m.strip() for m in (codex_models or "").split(",") if m.strip()] or [""]
        for model in models:
            label = _sanitize_label(f"codex_{model}" if model else "codex_default")
            command = ["codex", "exec"]
            if model:
                command += ["-m", model]
            command.append("-")
            cache_key = f"{label}"
            if cache.get(cache_key) == prompt_digest:
                builtins.print(f"Skipped {label} (no changes)")
                continue
            if debug:
                builtins.print(f"Running {label}")
            response, duration_seconds = _run_llm_command(command, prompt, output_dir, label, debug)
            _store_llm_run(
                settings.db_path,
                "codex",
                model or None,
                label,
                prompt_digest,
                prompt,
                response,
                recorded_from,
                recorded_to,
                limit,
                duration_seconds,
            )
            cache[cache_key] = prompt_digest

    if backend in {"claude", "both"}:
        models = [m.strip() for m in (claude_models or "").split(",") if m.strip()] or [""]
        for model in models:
            label = _sanitize_label(f"claude_{model}" if model else "claude_default")
            command = ["claude", "-p"]
            if model:
                command += ["--model", model]
            cache_key = f"{label}"
            if cache.get(cache_key) == prompt_digest:
                builtins.print(f"Skipped {label} (no changes)")
                continue
            if debug:
                builtins.print(f"Running {label}")
            response, duration_seconds = _run_llm_command(command, prompt, output_dir, label, debug)
            _store_llm_run(
                settings.db_path,
                "claude",
                model or None,
                label,
                prompt_digest,
                prompt,
                response,
                recorded_from,
                recorded_to,
                limit,
                duration_seconds,
            )
            cache[cache_key] = prompt_digest

    _save_cache(cache_path, cache)
    builtins.print(f"Wrote prompts/responses to {output_dir}")


@app.command()
def query(
    q: Optional[str] = typer.Argument(None, help="Query text."),
    k: int = typer.Option(5, help="Top results to return."),
    recorded_from: Optional[str] = typer.Option(None, help="Filter by recorded_at >= (ISO 8601)."),
    recorded_to: Optional[str] = typer.Option(None, help="Filter by recorded_at <= (ISO 8601)."),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Prompt for queries in a loop."),
) -> None:
    """Search the journal using the configured index backend."""
    from .journal import search

    settings = load_settings()

    def run_query(text: str) -> None:
        results = search(settings, query=text, k=k, recorded_from=recorded_from, recorded_to=recorded_to)
        _print_query_results(results)

    if interactive:
        print("Interactive mode. Enter a query (type 'quit' to exit).")
        while True:
            try:
                text = input("query> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not text:
                continue
            if text.lower() in {"quit", "exit", "q"}:
                break
            run_query(text)
        return

    if not q:
        raise typer.BadParameter("Missing query text (or use --interactive).")

    run_query(q)
