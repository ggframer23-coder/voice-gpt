# voice-gpt

Offline voice transcription, journaling, and memory retrieval with FAISS.

## Install (app) with uv

```bash
uv venv
. .venv/bin/activate
uv pip install -e .[cli]
```

If you want `faster-whisper` too:

```bash
uv pip install -e .[cli,faster]
```

## Install (library) with uv

```bash
uv pip install -e .
```

## Requirements

- Python 3.10+
- `whisper.cpp` binary (offline transcription)
- A GGUF model (e.g., `tiny.en` or `base.en`)
- `ffmpeg` (for audio conversion)

Set env vars:

```bash
export VOICE_GPT_WHISPER_BIN=/path/to/whisper.cpp/main
export VOICE_GPT_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

## Run (quickstart)

```bash
# 1) Initialize storage
voice-gpt init

# 2) Transcribe an audio file and store the entry
voice-gpt transcribe /path/to/audio.wav /path/to/gguf-model
```

If you already have 16kHz mono WAV and want to skip conversion:

```bash
voice-gpt transcribe /path/to/audio.wav /path/to/gguf-model --no-convert
```

To use faster-whisper (CPU-only):

```bash
voice-gpt transcribe /path/to/audio.wav base.en --engine faster-whisper
```

To save text to a file without storing:

```bash
voice-gpt transcribe /path/to/audio.wav /path/to/gguf-model --save /tmp/out.txt --no-ingest
```

## Ingest a directory

Process every audio file in a folder and store entries:

```bash
voice-gpt ingest-dir /path/to/audio /path/to/gguf-model
```

Archive files after ingest:

```bash
voice-gpt ingest-dir /path/to/audio /path/to/gguf-model --archive-dir /path/to/processed
```

## Search memories

```bash
voice-gpt query "memory pipeline" -k 5
```

Filter by recorded time (local ISO 8601):

```bash
voice-gpt query "memory pipeline" --recorded-from "2026-01-01T00:00:00-08:00" --recorded-to "2026-01-31T23:59:59-08:00"
```

## Add text directly

```bash
voice-gpt add-text "Today I worked on the memory pipeline."
```

## Using as a library

```python
from voice_gpt.settings import load_settings
from voice_gpt.journal import add_entry, search

settings = load_settings()
add_entry(settings, text="My first note.")
results = search(settings, query="first note")
```

## Notes

- FAISS index + SQLite live in `~/.voice-gpt` by default.
- For a different location, set `VOICE_GPT_HOME` or `VOICE_GPT_DB`/`VOICE_GPT_INDEX`.
- Embedding model downloads on first use; cache it to keep offline after that.
- `recorded_at` is stored in local time for each entry; for audio files it uses the file modified time.
- `recorded_at` is included in embedding inputs to make time queries more discoverable.
