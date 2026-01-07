from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich import print

from .journal import add_entry, has_audio, init_store, search
from .settings import load_settings
from .transcribe import TranscriptionError, transcribe_audio, transcribe_audio_faster_whisper
from .vad import create_vad_clips

app = typer.Typer(add_completion=False)


def _default_vad_dir(audio: Path) -> Path:
    return audio.parent / f"{audio.stem}.vad"


def _vad_metadata_path(vad_dir: Path) -> Path:
    return vad_dir / "segments.json"


def _write_vad_metadata(path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8")


def _resolve_vad_paths(settings, vad_bin: Optional[Path], vad_model: Optional[Path]) -> tuple[Optional[Path], Optional[Path]]:
    return vad_bin or settings.vad_bin, vad_model or settings.vad_model


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
    model: str = typer.Argument(..., help="GGUF path for whisper.cpp or model name/path for faster-whisper."),
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
    engine: str = typer.Option("whispercpp", help="Transcription engine: whispercpp or faster-whisper."),
) -> None:
    """Transcribe audio offline with whisper.cpp or faster-whisper."""
    settings = load_settings()
    metadata: Optional[dict] = None
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
def ingest_dir(
    directory: Path = typer.Argument(..., help="Directory with audio files."),
    model: str = typer.Argument(..., help="GGUF path for whisper.cpp or model name/path for faster-whisper."),
    engine: str = typer.Option("whispercpp", help="Transcription engine: whispercpp or faster-whisper."),
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
    archive_dir: Optional[Path] = typer.Option(None, help="Move processed files here."),
    extensions: str = typer.Option(
        "wav,mp3,m4a,flac,ogg,opus,webm",
        help="Comma-separated list of extensions to ingest.",
    ),
) -> None:
    """Ingest and transcribe all audio files in a directory."""
    settings = load_settings()
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
        if has_audio(settings, str(audio)):
            print(f"Skipping already ingested: {audio}")
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

        if archive_dir:
            target = archive_dir / audio.name
            audio.replace(target)


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
