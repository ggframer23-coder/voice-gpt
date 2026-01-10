# stt

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
stt init
stt transcribe /path/to/audio.wav base.en
```

Batch folder workflow (top-level only):

```bash
make transcribe AUDIO=audio MODEL=base.en
```

Already-ingested files are still moved into their year/week folder.

## Commands

- `stt init` - initialize storage and index.
- `stt transcribe AUDIO MODEL` - transcribe one file.
- `stt reingest AUDIO MODEL` - force transcribe even if already ingested.
- `stt ingest-dir DIR MODEL` - transcribe all supported files in a directory.
- `stt summary` - list ingested audio with size, minutes, and word count.
- `stt export DIR` - export one transcript file per recording into DIR.
- `stt dump` - write one transcript file per recording.
- `stt vad AUDIO` - generate VAD clips and metadata only.
- `stt query QUERY` - search stored transcripts.
- `stt-query QUERY` - query-only CLI (no transcription commands).
- `stt add-text TEXT` - store raw text without audio.
- `stt dedupe` - remove duplicate audio entries (keeps latest by recorded_at).
- `stt summarize-db` - summarize transcripts via Codex/Claude CLI.
  - Use `--dump-only` to print the latest stored LLM response (no model call).

Example (Codex only):

```bash
stt summarize-db --backend codex --codex-models o3
```

Example (Codex only, one day):

```bash
stt summarize-db --backend codex --codex-models o3 --recorded-from "2026-01-05T00:00:00-08:00" --recorded-to "2026-01-05T23:59:59-08:00"
```

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

Run `stt --help` or `stt COMMAND --help` for the full list.

## Install

App (CLI):

```bash
make venv install-cli
```

Add `faster-whisper`:

```bash
make venv install-faster
```

Library only:

```bash
make venv install
```

CPU-only install (avoid CUDA):

```bash
make venv install-cli-cpu
```

## Requirements

- Python 3.10+
- `ffmpeg` (for audio conversion)

## Engines and models

Supported engines:

- `faster-whisper` (default): CPU-only CTranslate2 backend; models by name (e.g., `base.en`).
- `whisper.cpp`: GGUF models + local `whisper-cli` binary (set `STT_WHISPER_BIN`).
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
export STT_WHISPER_BIN=/path/to/whisper.cpp/build/bin/whisper-cli
export STT_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
export STT_OFFLINE=1
export STT_PARAKEET_MODEL=nemo-parakeet-tdt-0.6b-v3
export STT_PARAKEET_DIR=/path/to/parakeet-model
export STT_PARAKEET_QUANT=int8
```

## Makefile defaults

The `Makefile` wraps common commands and sets defaults you can override per call:

- `AUDIO=audio`
- `MODEL_NAME=base.en`
- `MODEL=third_party/whisper.cpp/models/ggml-base.en.bin`
- `K=5`
- `VENV=.venv`
- `STT=.venv/bin/stt`
- `WHISPER_DIR=third_party/whisper.cpp`
- `WHISPER_BIN=third_party/whisper.cpp/build/bin/whisper-cli`

Example overrides:

```bash
make install-whisper download-model MODEL_NAME=base.en
make transcribe AUDIO=/path/to/audio.wav MODEL=/path/to/model.gguf
make query QUERY="memory pipeline" K=10
```

## Storage location

The journal database lives at `~/.stt/journal.sqlite` by default. You can override:

- `STT_HOME` to change the base directory (defaults to `~/.stt`)
- `STT_DB` to point directly at a specific SQLite file

The FAISS index lives alongside the database as `~/.stt/faiss.index` unless you set
`STT_INDEX`.

## Examples

Skip conversion if you already have 16kHz mono WAV:

```bash
stt transcribe /path/to/audio.wav /path/to/gguf-model --no-convert
```

Split speech into VAD clips and include timestamps:

```bash
stt transcribe /path/to/audio.wav /path/to/gguf-model --vad
```

Only segment audio (no transcription):

```bash
stt vad /path/to/audio.wav
```

Query interactively:

```bash
stt query --interactive
```

Filter by recorded time:

```bash
stt query "memory pipeline" --recorded-from "2026-01-01T00:00:00-08:00" --recorded-to "2026-01-31T23:59:59-08:00"
```

## Using as a library

```python
from stt.settings import load_settings
from stt.journal import add_entry, search

settings = load_settings()
add_entry(settings, text="My first note.")
results = search(settings, query="first note")
```

## Notes

- FAISS index + SQLite live in `~/.stt` by default.
- For a different location, set `STT_HOME` or `STT_DB`/`STT_INDEX`.
- Offline mode is enabled by default; set `STT_OFFLINE=0` to allow downloads.
- Ensure embedding and transcription models are cached or referenced by local paths when offline.
- `recorded_at` is stored in local time for each entry; for audio files it uses the file modified time.
- `recorded_at` is included in embedding inputs to make time queries more discoverable.
- Faster-whisper captures word timings by default; they are stored in metadata and printed/saved transcripts include elapsed markers every 5 minutes.
