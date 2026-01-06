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

## CPU-only install (avoid NVIDIA/CUDA)

`sentence-transformers` pulls in `torch`, which may default to CUDA wheels on some systems. To force CPU-only wheels:

```bash
uv pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1+cpu
uv pip install -e .[cli] --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu --constraint constraints-cpu.txt
```

## Requirements

- Python 3.10+
- `whisper.cpp` binary (offline transcription)
- A ggml model file (e.g., `ggml-base.en.bin`)
- `ffmpeg` (for audio conversion)

Set env vars:

```bash
export VOICE_GPT_WHISPER_BIN=/path/to/whisper.cpp/build/bin/whisper-cli
export VOICE_GPT_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

## Current build environment

These versions are from the current dev machine:

- Ubuntu 24.04.3 LTS (WSL2, kernel 6.6.87.2-microsoft-standard-WSL2)
- Python 3.12.3
- uv 0.6.2
- cmake 3.28.3
- ffmpeg 6.1.1-3ubuntu5
- torch 2.9.1+cpu

## Available models

`whisper.cpp` supports the Whisper family of models in GGUF format. Common model names:

- `tiny`, `tiny.en`
- `base`, `base.en`
- `small`, `small.en`
- `medium`, `medium.en`
- `large-v2`, `large-v3`

For `--engine faster-whisper`, you pass the model name (for example `base.en`), and it will download the model on first use.

## Run (quickstart)

```bash
# 1) Initialize storage
voice-gpt init

# 2) Transcribe an audio file and store the entry
voice-gpt transcribe /path/to/audio.wav /path/to/gguf-model
```

## Makefile defaults

The `Makefile` wraps common commands and sets defaults you can override per call:

- `AUDIO=audio`
- `MODEL_NAME=base.en`
- `MODEL=third_party/whisper.cpp/models/ggml-base.en.bin`
- `K=5`
- `VENV=.venv`
- `VOICE_GPT=.venv/bin/voice-gpt`
- `WHISPER_DIR=third_party/whisper.cpp`
- `WHISPER_BIN=third_party/whisper.cpp/build/bin/whisper-cli`

Example overrides:

```bash
make install-whisper download-model MODEL_NAME=base.en
make transcribe AUDIO=/path/to/audio.wav MODEL=/path/to/model.gguf
make query QUERY="memory pipeline" K=10
```

If you already have 16kHz mono WAV and want to skip conversion:

```bash
voice-gpt transcribe /path/to/audio.wav /path/to/gguf-model --no-convert
voice-gpt transcribe /path/to/audio.wav /path/to/gguf-model --vad
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
voice-gpt ingest-dir /path/to/audio /path/to/gguf-model --vad
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
