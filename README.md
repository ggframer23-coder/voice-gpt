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

If you want `parakeet` (onnx-asr) too:

```bash
uv pip install -e .[cli,parakeet]
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
- `ffmpeg` (for audio conversion)

## Transcription toolchain

This project supports three local engines:

- `faster-whisper` (default): CPU-only CTranslate2 backend; models are downloaded by name (e.g., `base.en`).
- `whisper.cpp`: GGUF models + local `whisper-cli` binary (set `VOICE_GPT_WHISPER_BIN`).
- `parakeet` (onnx-asr): ONNX models by name (e.g., `nemo-parakeet-tdt-0.6b-v3`), optional local model directory.

Optional VAD:

- `whisper.cpp` VAD can segment audio before transcription (requires `vad-speech-segments` and a VAD model).

Set env vars:

```bash
export VOICE_GPT_WHISPER_BIN=/path/to/whisper.cpp/build/bin/whisper-cli
export VOICE_GPT_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
export VOICE_GPT_PARAKEET_MODEL=nemo-parakeet-tdt-0.6b-v3
# Optional: local model directory and quantization suffix
export VOICE_GPT_PARAKEET_DIR=/path/to/parakeet-model
export VOICE_GPT_PARAKEET_QUANT=int8
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

The default engine is `faster-whisper`. You pass the model name (for example `base.en`), and it will download the model on first use.

## Run (quickstart)

```bash
# 1) Initialize storage
voice-gpt init

# 2) Transcribe an audio file and store the entry (default engine: faster-whisper)
voice-gpt transcribe /path/to/audio.wav base.en
```

## Commands and options

List commands:

- `voice-gpt init` - initialize storage and index.
- `voice-gpt transcribe AUDIO MODEL` - transcribe one file.
- `voice-gpt ingest-dir DIR MODEL` - transcribe all supported files in a directory.
- `voice-gpt summary` - list ingested audio with size, minutes, and word count.
- `voice-gpt vad AUDIO` - generate VAD clips and metadata only.
- `voice-gpt query QUERY` - search stored transcripts.
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

## Audio directory workflow

When `AUDIO=audio` is used (for example `make transcribe AUDIO=audio ...`), the CLI only scans the
top-level files in `audio/` and then moves each file into a year/week folder under `audio/` after
transcription. The year/week is derived from the file modified time (local ISO week).

Example target layout:

```text
audio/
  2026/
    01/
    02/
```

Notes:

- Already-ingested files are still moved into their year/week folder.
- If VAD is enabled with the default output, the `.vad` folder is moved alongside the audio file.

If you already have 16kHz mono WAV and want to skip conversion:

```bash
voice-gpt transcribe /path/to/audio.wav /path/to/gguf-model --no-convert
```

To split speech into clips with whisper.cpp VAD (timestamps + clips saved under `audio.vad/segments.json`):

```bash
voice-gpt transcribe /path/to/audio.wav /path/to/gguf-model --vad
```

The metadata JSON includes `start_ms`, `end_ms`, `clip_path`, and per-clip `text`.
Use `--vad-dir /path/to/output` to change where clips + metadata are stored.
Disable timestamp prefixes in the transcript with `--no-vad-timestamps`.
Use `--vad-model`/`--vad-bin` (or env `VOICE_GPT_VAD_MODEL`/`VOICE_GPT_VAD_BIN`) to point at the
whisper.cpp VAD model and `vad-speech-segments` binary.
The default looks for `third_party/whisper.cpp/models/for-tests-silero-v6.2.0-ggml.bin`.

To only segment audio (no transcription):

```bash
voice-gpt vad /path/to/audio.wav
```

To use faster-whisper (CPU-only):

```bash
voice-gpt transcribe /path/to/audio.wav base.en --engine faster-whisper
```

To use Parakeet (CPU-only, onnx-asr):

```bash
voice-gpt transcribe /path/to/audio.wav nemo-parakeet-tdt-0.6b-v3 --engine parakeet
```

To use a local Parakeet model directory (for example, `parakeet-tdt-0.6b-v3-int8`):

```bash
voice-gpt transcribe /path/to/audio.wav nemo-parakeet-tdt-0.6b-v3 --engine parakeet \
  --parakeet-dir /path/to/parakeet-tdt-0.6b-v3-int8 --parakeet-quant int8
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

Use whisper.cpp VAD when ingesting a directory:

```bash
voice-gpt ingest-dir /path/to/audio /path/to/gguf-model --vad
```

## Search memories

```bash
voice-gpt query "memory pipeline" -k 5
```

Interactive query mode:

```bash
voice-gpt query --interactive
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
