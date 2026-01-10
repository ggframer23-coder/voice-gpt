from __future__ import annotations

import builtins
import json
import shutil
import sqlite3
import hashlib
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from rich import print

from .journal import add_entry, has_audio, init_store, search
from .settings import load_settings
from .transcribe import (
    TranscriptionError,
    _collect_word_timestamps,
    load_faster_whisper_model,
    load_parakeet_model,
    transcribe_audio,
    transcribe_audio_faster_whisper,
    transcribe_audio_parakeet,
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
            if ".vad" in path.parts:
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
) -> tuple[str, dict, list[dict], Optional[Path]]:
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
        clips = metadata["segments"]
        if not clips:
            raise TranscriptionError("No speech segments detected.")
        if engine == "faster-whisper":
            text, word_timestamps = _transcribe_clips_faster_whisper(
                clips,
                model,
                vad_timestamps,
                settings.offline,
            )
        elif engine == "parakeet":
            parakeet_model_name, parakeet_model_dir, parakeet_quantization = resolved_parakeet
            text = _transcribe_clips_parakeet(
                clips,
                parakeet_model_name,
                parakeet_model_dir,
                parakeet_quantization,
                vad_timestamps,
            )
        else:
            text = _transcribe_clips_whispercpp(settings, clips, Path(model), vad_timestamps)
        vad_metadata_path = _vad_metadata_path(output_dir)
        return text, metadata, word_timestamps, vad_metadata_path

    if engine == "faster-whisper":
        text, word_timestamps = transcribe_audio_faster_whisper(
            audio_path=audio,
            model_name_or_path=model,
            convert=convert,
            offline=settings.offline,
        )
    elif engine == "parakeet":
        parakeet_model_name, parakeet_model_dir, parakeet_quantization = resolved_parakeet
        text = transcribe_audio_parakeet(
            audio_path=audio,
            model_name=parakeet_model_name,
            model_dir=parakeet_model_dir,
            quantization=parakeet_quantization,
            convert=convert,
        )
    else:
        text = transcribe_audio(
            settings,
            audio_path=audio,
            model_path=Path(model),
            convert=convert,
        )
    return text, metadata, word_timestamps, vad_metadata_path


def _transcribe_clips_whispercpp(
    settings, clips: list[dict], model_path: Path, include_timestamps: bool
) -> str:
    texts = []
    for clip in clips:
        clip_path = Path(clip["clip_path"])
        text = transcribe_audio(settings, audio_path=clip_path, model_path=model_path, convert=False)
        clip["text"] = text
        if text:
            if include_timestamps:
                texts.append(_format_clip_line(clip, text))
            else:
                texts.append(text)
    return "\n".join(texts).strip()


def _transcribe_clips_faster_whisper(
    clips: list[dict], model_name_or_path: str, include_timestamps: bool, offline: bool
) -> tuple[str, list[dict]]:
    model = load_faster_whisper_model(model_name_or_path, device="cpu", compute_type="int8", offline=offline)
    texts = []
    words: list[dict] = []
    for clip in clips:
        clip_path = Path(clip["clip_path"])
        segments, _info = model.transcribe(str(clip_path), language="en", word_timestamps=True)
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
    """Initialize local storage and FAISS index."""
    settings = load_settings()
    init_store(settings)
    print(f"Initialized store at {settings.base_dir}")


@app.command()
def add_text(
    text: str = typer.Argument(..., help="Journal text to store."),
    source: Optional[str] = typer.Option(None, help="Source label."),
) -> None:
    """Add a journal entry from raw text."""
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


@app.command()
def transcribe(
    audio: Path = typer.Argument(..., help="Path to audio file."),
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
    engine: str = typer.Option("faster-whisper", help="Transcription engine: faster-whisper, whispercpp, or parakeet."),
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
) -> None:
    """Transcribe audio offline with whisper.cpp or faster-whisper."""
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
        return entry_id
    else:
        print(display_text)
        return None


@app.command()
def summary(
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
    """List already transcribed audio files with size, minutes, and word count."""
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
            "SELECT id, recorded_at, text, audio_path, audio_size_bytes, audio_duration_seconds FROM entries "
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
        minutes = f"{(duration_seconds / 60):.1f}" if duration_seconds is not None else "n/a"

        display_path = str(resolved_path)
        if not resolved_path.exists():
            display_path = f"{display_path} (missing)"
        if dedupe:
            normalized = display_path.strip().lower()
            if normalized in seen_display:
                continue
            seen_display.add(normalized)
        items.append((display_path, size_mb, minutes, word_count))
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
    builtins.print(f"{'audio_path':<{path_width}}size_mb minutes words")
    for display_path, size_mb, minutes, word_count in items:
        builtins.print(f"{display_path:<{path_width}}{size_mb:>7} {minutes:>7} {word_count:>5}")


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
    engine: str = typer.Option("faster-whisper", help="Transcription engine: faster-whisper, whispercpp, or parakeet."),
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
    extensions: str = typer.Option(
        "wav,mp3,m4a,flac,ogg,opus,webm",
        help="Comma-separated list of extensions to ingest.",
    ),
) -> None:
    """Ingest and transcribe all audio files in a directory."""
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

    files = _iter_audio_files(directory, recursive=False, extensions=extensions)
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
                clips = metadata["segments"]
                if not clips:
                    raise TranscriptionError("No speech segments detected.")
                if engine == "faster-whisper":
                    text, word_timestamps = _transcribe_clips_faster_whisper(
                        clips,
                        model,
                        vad_timestamps,
                        settings.offline,
                    )
                elif engine == "parakeet":
                    parakeet_model_name, parakeet_model_dir, parakeet_quantization = resolved_parakeet
                    text = _transcribe_clips_parakeet(
                        clips,
                        parakeet_model_name,
                        parakeet_model_dir,
                        parakeet_quantization,
                        vad_timestamps,
                    )
                else:
                    text = _transcribe_clips_whispercpp(settings, clips, Path(model), vad_timestamps)
                vad_metadata_path = _vad_metadata_path(output_dir)
            else:
                if engine == "faster-whisper":
                    text, word_timestamps = transcribe_audio_faster_whisper(
                        audio_path=audio,
                        model_name_or_path=model,
                        convert=convert,
                        offline=settings.offline,
                    )
                elif engine == "parakeet":
                    parakeet_model_name, parakeet_model_dir, parakeet_quantization = resolved_parakeet
                    text = transcribe_audio_parakeet(
                        audio_path=audio,
                        model_name=parakeet_model_name,
                        model_dir=parakeet_model_dir,
                        quantization=parakeet_quantization,
                        convert=convert,
                    )
                else:
                    text = transcribe_audio(
                        settings,
                        audio_path=audio,
                        model_path=Path(model),
                        convert=convert,
                    )
        except TranscriptionError as exc:
            print(f"Failed: {audio} ({exc})")
            continue

        metadata["word_timestamps"] = word_timestamps
        if vad_metadata_path is not None:
            _write_vad_metadata(vad_metadata_path, metadata)
        entry_id = add_entry(settings, text=text, source=source, audio_path=str(audio), metadata=metadata)
        print(f"Added entry {entry_id} from {audio}")

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
    engine: str = typer.Option("faster-whisper", help="Transcription engine: faster-whisper, whispercpp, or parakeet."),
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
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Recurse into subdirectories."),
    extensions: str = typer.Option(
        "wav,mp3,m4a,flac,ogg,opus,webm",
        help="Comma-separated list of extensions to ingest.",
    ),
) -> None:
    """Force transcribe even if already ingested."""
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
        )
    except TranscriptionError as exc:
        raise typer.Exit(str(exc))

    metadata["word_timestamps"] = word_timestamps
    if vad_metadata_path is not None:
        _write_vad_metadata(vad_metadata_path, metadata)

    entry_id = add_entry(settings, text=text, source=source, audio_path=str(audio), metadata=metadata)
    duration = _audio_duration_seconds(audio) if audio.exists() else None
    print(f"Added entry {entry_id}: {audio} {_format_duration(duration)}")


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
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = output_dir / f"prompt_{label}.txt"
    response_path = output_dir / f"response_{label}.txt"
    debug_path = output_dir / f"debug_{label}.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    result = subprocess.run(command, input=prompt, text=True, capture_output=True)
    response_path.write_text(result.stdout, encoding="utf-8")
    debug_sections = [
        f"command: {shlex.join(command)}",
        f"exit_code: {result.returncode}",
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
    return result.stdout


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
        "limit_count INTEGER"
        ")"
    )
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
) -> None:
    with sqlite3.connect(db_path) as conn:
        _ensure_llm_runs_table(conn)
        conn.execute(
            "INSERT INTO llm_runs "
            "(created_at, backend, model, label, prompt_hash, prompt, response, recorded_from, recorded_to, limit_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
            response = _run_llm_command(command, prompt, output_dir, label, debug)
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
            response = _run_llm_command(command, prompt, output_dir, label, debug)
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
    """Search the journal using FAISS."""
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
