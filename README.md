# voice-gpt

Offline voice transcription, journaling, and memory retrieval with FAISS.

## Purpose

- Transcribe audio locally with multiple engines.
- Store transcripts in a local journal (SQLite + FAISS index).
- Query your transcript history by text and recorded time.

## Directory structure

```text
audio/
  2026/
    01/
    02/
```

- `audio/` is the default input folder for batch work.
- Transcribed files are moved into `audio/<year>/<week>/` based on file modified time (local ISO week).
- If VAD is enabled with the default output, the `.vad` folder is moved alongside the audio file.

## Workflow

Quickstart:

```bash
voice-gpt init
voice-gpt transcribe /path/to/audio.wav base.en
```

Batch folder workflow (top-level only):

```bash
make transcribe AUDIO=audio MODEL=base.en
```

Already-ingested files are still moved into their year/week folder.

## Commands

- `voice-gpt init` - initialize storage and index.
- `voice-gpt transcribe AUDIO MODEL` - transcribe one file.
- `voice-gpt ingest-dir DIR MODEL` - transcribe all supported files in a directory.
- `voice-gpt summary` - list ingested audio with size, minutes, and word count.
- `voice-gpt vad AUDIO` - generate VAD clips and metadata only.
- `voice-gpt query QUERY` - search stored transcripts.
- `voice-gpt-query QUERY` - query-only CLI (no transcription commands).
- `voice-gpt add-text TEXT` - store raw text without audio.

Common options:

- `--engine faster-whisper|whispercpp|parakeet` (transcribe, ingest-dir)
- `--no-convert` (transcribe, ingest-dir) to skip ffmpeg conversion
- `--vad` (transcribe, ingest-dir) to run VAD first
- `--vad-dir DIR` to control VAD output location
- `--no-vad-timestamps` to omit timestamps from VAD transcripts
- `--archive-dir DIR` (ingest-dir) to move processed files
- `--save PATH` (transcribe) to save transcript to a file
- `--no-ingest` (transcribe) to skip writing to the journal
- `--recorded-from ISO` / `--recorded-to ISO` (query) to filter by recorded time

Run `voice-gpt --help` or `voice-gpt COMMAND --help` for the full list.

## Install

App (CLI):

```bash
uv venv
. .venv/bin/activate
uv pip install -e .[cli]
```

Add `faster-whisper`:

```bash
uv pip install -e .[cli,faster]
```

Add `parakeet` (onnx-asr):

```bash
uv pip install -e .[cli,parakeet]
```

Library only:

```bash
uv pip install -e .
```

CPU-only install (avoid CUDA):

```bash
uv pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1+cpu
uv pip install -e .[cli] --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu --constraint constraints-cpu.txt
```

## Requirements

- Python 3.10+
- `ffmpeg` (for audio conversion)

## Engines and models

Supported engines:

- `faster-whisper` (default): CPU-only CTranslate2 backend; models by name (e.g., `base.en`).
- `whisper.cpp`: GGUF models + local `whisper-cli` binary (set `VOICE_GPT_WHISPER_BIN`).
- `parakeet` (onnx-asr): ONNX models by name (e.g., `nemo-parakeet-tdt-0.6b-v3`).

Common Whisper model names:

- `tiny`, `tiny.en`
- `base`, `base.en`
- `small`, `small.en`
- `medium`, `medium.en`
- `large-v2`, `large-v3`

Optional VAD:

- `whisper.cpp` VAD can segment audio before transcription (requires `vad-speech-segments` and a VAD model).

Environment variables:

```bash
export VOICE_GPT_WHISPER_BIN=/path/to/whisper.cpp/build/bin/whisper-cli
export VOICE_GPT_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
export VOICE_GPT_PARAKEET_MODEL=nemo-parakeet-tdt-0.6b-v3
export VOICE_GPT_PARAKEET_DIR=/path/to/parakeet-model
export VOICE_GPT_PARAKEET_QUANT=int8
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

## Storage location

The journal database lives at `~/.voice-gpt/journal.sqlite` by default. You can override:

- `VOICE_GPT_HOME` to change the base directory (defaults to `~/.voice-gpt`)
- `VOICE_GPT_DB` to point directly at a specific SQLite file

The FAISS index lives alongside the database as `~/.voice-gpt/faiss.index` unless you set
`VOICE_GPT_INDEX`.

## Examples

Skip conversion if you already have 16kHz mono WAV:

```bash
voice-gpt transcribe /path/to/audio.wav /path/to/gguf-model --no-convert
```

Split speech into VAD clips and include timestamps:

```bash
voice-gpt transcribe /path/to/audio.wav /path/to/gguf-model --vad
```

Only segment audio (no transcription):

```bash
voice-gpt vad /path/to/audio.wav
```

Query interactively:

```bash
voice-gpt query --interactive
```

Filter by recorded time:

```bash
voice-gpt query "memory pipeline" --recorded-from "2026-01-01T00:00:00-08:00" --recorded-to "2026-01-31T23:59:59-08:00"
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
