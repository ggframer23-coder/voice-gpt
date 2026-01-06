from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich import print

from .journal import add_entry, has_audio, init_store, search
from .settings import load_settings
from .transcribe import TranscriptionError, transcribe_audio, transcribe_audio_faster_whisper

app = typer.Typer(add_completion=False)


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
def transcribe(
    audio: Path = typer.Argument(..., help="Path to audio file."),
    model: str = typer.Argument(..., help="GGUF path for whisper.cpp or model name/path for faster-whisper."),
    save: Optional[Path] = typer.Option(None, help="Save transcript to file."),
    ingest: bool = typer.Option(True, help="Store transcript in journal."),
    source: Optional[str] = typer.Option("whisper.cpp", help="Source label."),
    convert: bool = typer.Option(True, help="Convert to 16kHz mono WAV with ffmpeg."),
    vad: bool = typer.Option(
        False,
        help="Trim silence with ffmpeg before transcription (forces 16kHz mono WAV conversion).",
    ),
    engine: str = typer.Option("whispercpp", help="Transcription engine: whispercpp or faster-whisper."),
) -> None:
    """Transcribe audio offline with whisper.cpp or faster-whisper."""
    settings = load_settings()
    try:
        if engine == "faster-whisper":
            text = transcribe_audio_faster_whisper(
                audio_path=audio,
                model_name_or_path=model,
                convert=convert,
                vad=vad,
            )
        else:
            text = transcribe_audio(
                settings,
                audio_path=audio,
                model_path=Path(model),
                convert=convert,
                vad=vad,
            )
    except TranscriptionError as exc:
        raise typer.Exit(str(exc))

    if save:
        save.parent.mkdir(parents=True, exist_ok=True)
        save.write_text(text, encoding="utf-8")

    if ingest:
        entry_id = add_entry(settings, text=text, source=source, audio_path=str(audio))
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
        help="Trim silence with ffmpeg before transcription (forces 16kHz mono WAV conversion).",
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
            if engine == "faster-whisper":
                text = transcribe_audio_faster_whisper(
                    audio_path=audio,
                    model_name_or_path=model,
                    convert=convert,
                    vad=vad,
                )
            else:
                text = transcribe_audio(
                    settings,
                    audio_path=audio,
                    model_path=Path(model),
                    convert=convert,
                    vad=vad,
                )
        except TranscriptionError as exc:
            print(f"Failed: {audio} ({exc})")
            continue

        entry_id = add_entry(settings, text=text, source=source, audio_path=str(audio))
        print(f"Added entry {entry_id} from {audio}")

        if archive_dir:
            target = archive_dir / audio.name
            audio.replace(target)


@app.command()
def query(
    q: str = typer.Argument(..., help="Query text."),
    k: int = typer.Option(5, help="Top results to return."),
    recorded_from: Optional[str] = typer.Option(None, help="Filter by recorded_at >= (ISO 8601)."),
    recorded_to: Optional[str] = typer.Option(None, help="Filter by recorded_at <= (ISO 8601)."),
) -> None:
    """Search the journal using FAISS."""
    settings = load_settings()
    results = search(settings, query=q, k=k, recorded_from=recorded_from, recorded_to=recorded_to)
    if not results:
        print("No results")
        raise typer.Exit()
    for idx, item in enumerate(results, start=1):
        print(f"[{idx}] score={item['score']:.3f} entry={item['entry_id']} chunk={item['chunk_id']}")
        if item.get("recorded_at"):
            print(f"recorded_at={item['recorded_at']}")
        print(item["chunk_text"])
        print()
