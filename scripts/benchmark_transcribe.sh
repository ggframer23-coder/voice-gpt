#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VOICE_GPT="${VOICE_GPT:-$ROOT_DIR/.venv/bin/voice-gpt}"
WHISPER_DIR="${WHISPER_DIR:-$ROOT_DIR/third_party/whisper.cpp}"
WHISPER_BIN="${WHISPER_BIN:-$WHISPER_DIR/build/bin/whisper-cli}"
MODEL_NAME="${MODEL_NAME:-base.en}"
WHISPER_MODEL="${WHISPER_MODEL:-$WHISPER_DIR/models/ggml-$MODEL_NAME.bin}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/voice-gpt-bench}"
TIME_BIN="${TIME_BIN:-/usr/bin/time}"
AUDIO="${AUDIO:-}"
NO_CONVERT="${NO_CONVERT:-0}"

if [ -z "$AUDIO" ]; then
  echo "AUDIO is required."
  exit 1
fi

if [ ! -f "$AUDIO" ]; then
  echo "Audio file not found: $AUDIO"
  exit 1
fi

if [ ! -x "$VOICE_GPT" ]; then
  echo "voice-gpt not found at $VOICE_GPT"
  exit 1
fi

if [ ! -x "$WHISPER_BIN" ]; then
  echo "whisper.cpp binary not found at $WHISPER_BIN"
  exit 1
fi

if [ ! -f "$WHISPER_MODEL" ]; then
  echo "Whisper model not found at $WHISPER_MODEL"
  exit 1
fi

if [ ! -x "$TIME_BIN" ]; then
  echo "time binary not found at $TIME_BIN"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

CONVERT_FLAG=()
if [ "$NO_CONVERT" = "1" ]; then
  CONVERT_FLAG+=(--no-convert)
fi

run_timed() {
  local label=$1
  shift
  echo "== $label =="
  "$TIME_BIN" -f "$label real %e s | user %U s | sys %S s" "$@"
  echo ""
}

run_timed "whisper.cpp" \
  env VOICE_GPT_WHISPER_BIN="$WHISPER_BIN" \
  "$VOICE_GPT" transcribe "$AUDIO" "$WHISPER_MODEL" \
  --engine whispercpp --no-ingest --save "$OUTPUT_DIR/whispercpp.txt" \
  "${CONVERT_FLAG[@]}"

if ! run_timed "faster-whisper" \
  "$VOICE_GPT" transcribe "$AUDIO" "$MODEL_NAME" \
  --engine faster-whisper --no-ingest --save "$OUTPUT_DIR/faster-whisper.txt" \
  "${CONVERT_FLAG[@]}"; then
  echo "faster-whisper failed. Install with: make install-faster-cpu"
  exit 1
fi

echo "Saved transcripts:"
echo "  $OUTPUT_DIR/whispercpp.txt"
echo "  $OUTPUT_DIR/faster-whisper.txt"
