from __future__ import annotations

import builtins
import json
import shutil
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich import print

from .journal import add_entry, has_audio, init_store, search
from .settings import load_settings
from .transcribe import (
    TranscriptionError,
    load_parakeet_model,
    transcribe_audio,
    transcribe_audio_faster_whisper,
    transcribe_audio_parakeet,
)
from .vad import create_vad_clips

app = typer.Typer(add_completion=False)


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


def _format_timestamp_ms(ms: int) -> str:
    seconds = ms / 1000.0
    minutes = int(seconds // 60)
    seconds -= minutes * 60
    return f"{minutes:02d}:{seconds:06.3f}"


def _format_clip_line(clip: dict, text: str) -> str:
    start = _format_timestamp_ms(int(clip["start_ms"]))
    end = _format_timestamp_ms(int(clip["end_ms"]))
    return f"[{start}-{end}] {text}".strip()


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
    clips: list[dict], model_name_or_path: str, include_timestamps: bool
) -> str:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise TranscriptionError("faster-whisper is not installed; uv pip install -e '.[faster]'") from exc

    model = WhisperModel(model_name_or_path, device="cpu", compute_type="int8")
    texts = []
    for clip in clips:
        clip_path = Path(clip["clip_path"])
        segments, _info = model.transcribe(str(clip_path), language="en")
        lines = [seg.text.strip() for seg in segments if seg.text]
        text = "\n".join(lines).strip()
        clip["text"] = text
        if text:
            if include_timestamps:
                texts.append(_format_clip_line(clip, text))
            else:
                texts.append(text)
    return "\n".join(texts).strip()


def _transcribe_clips_parakeet(
    clips: list[dict],
    model_name: str,
    model_dir: Optional[Path],
    quantization: Optional[str],
    include_timestamps: bool,
) -> str:
    model = load_parakeet_model(model_name, model_dir=model_dir, quantization=quantization)
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
        ...,
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
    metadata: Optional[dict] = None
    resolved_parakeet = _resolve_parakeet_settings(
        settings,
        parakeet_model,
        parakeet_dir,
        parakeet_quant,
        model_fallback=model,
    )
    try:
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
                raise typer.Exit("No speech segments detected.")
            if engine == "faster-whisper":
                text = _transcribe_clips_faster_whisper(clips, model, vad_timestamps)
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
            _write_vad_metadata(_vad_metadata_path(output_dir), metadata)
        else:
            if engine == "faster-whisper":
                text = transcribe_audio_faster_whisper(
                    audio_path=audio,
                    model_name_or_path=model,
                    convert=convert,
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
        raise typer.Exit(str(exc))

    if save:
        save.parent.mkdir(parents=True, exist_ok=True)
        save.write_text(text, encoding="utf-8")

    if ingest:
        entry_id = add_entry(settings, text=text, source=source, audio_path=str(audio), metadata=metadata)
        print(f"Added entry {entry_id}")
    else:
        print(text)


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
        rows = conn.execute(
            "SELECT recorded_at, text, audio_path FROM entries "
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
    for row in rows:
        audio_path = row["audio_path"]
        text = row["text"] or ""
        word_count = len(text.split())
        resolved_path = Path(audio_path)
        if not resolved_path.exists():
            resolved_path = audio_index.get(resolved_path.name, resolved_path)

        size_bytes = resolved_path.stat().st_size if resolved_path.exists() else None
        size_mb = f"{(size_bytes / (1024 * 1024)):.2f}" if size_bytes is not None else "n/a"

        duration_seconds = _audio_duration_seconds(resolved_path) if resolved_path.exists() else None
        minutes = f"{(duration_seconds / 60):.1f}" if duration_seconds is not None else "n/a"

        display_path = str(resolved_path)
        if not resolved_path.exists():
            display_path = f"{display_path} (missing)"
        items.append((display_path, size_mb, minutes, word_count))

    path_width = max([len("audio_path")] + [len(item[0]) for item in items]) + 5
    builtins.print(f"{'audio_path':<{path_width}}size_mb minutes words")
    for display_path, size_mb, minutes, word_count in items:
        builtins.print(f"{display_path:<{path_width}}{size_mb:>7} {minutes:>7} {word_count:>5}")


@app.command()
def ingest_dir(
    directory: Path = typer.Argument(..., help="Directory with audio files."),
    model: str = typer.Argument(
        ...,
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

    ext_set = {ext.strip().lower().lstrip(".") for ext in extensions.split(",") if ext.strip()}
    files = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower().lstrip(".") in ext_set]
    files.sort()
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
            metadata: Optional[dict] = None
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
                    text = _transcribe_clips_faster_whisper(clips, model, vad_timestamps)
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
                _write_vad_metadata(_vad_metadata_path(output_dir), metadata)
            else:
                if engine == "faster-whisper":
                    text = transcribe_audio_faster_whisper(
                        audio_path=audio,
                        model_name_or_path=model,
                        convert=convert,
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
