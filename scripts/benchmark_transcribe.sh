#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STT="${STT:-$ROOT_DIR/.venv/bin/stt}"
WHISPER_DIR="${WHISPER_DIR:-$ROOT_DIR/third_party/whisper.cpp}"
WHISPER_BIN="${WHISPER_BIN:-$WHISPER_DIR/build/bin/whisper-cli}"
MODEL_NAME="${MODEL_NAME:-base.en}"
WHISPER_MODEL="${WHISPER_MODEL:-$WHISPER_DIR/models/ggml-$MODEL_NAME.bin}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/stt-bench}"
TIME_BIN="${TIME_BIN:-/usr/bin/time}"
AUDIO="${AUDIO:-}"
NO_CONVERT="${NO_CONVERT:-0}"
PARAKEET_MODEL="${PARAKEET_MODEL:-nemo-parakeet-tdt-0.6b-v3}"
PARAKEET_DIR="${PARAKEET_DIR:-}"
PARAKEET_QUANT="${PARAKEET_QUANT:-int8}"
RUN_FASTER="${RUN_FASTER:-1}"
RUN_PARAKEET="${RUN_PARAKEET:-1}"
RUN_WHISPERX="${RUN_WHISPERX:-1}"
RUN_WHISPERX_DIARIZE="${RUN_WHISPERX_DIARIZE:-0}"
WHISPERX_MODEL="${WHISPERX_MODEL:-medium}"
WHISPERX_DEVICE="${WHISPERX_DEVICE:-cpu}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

if [ -z "$AUDIO" ]; then
  echo "AUDIO is required."
  exit 1
fi

if [ ! -f "$AUDIO" ]; then
  echo "Audio file not found: $AUDIO"
  exit 1
fi

if [ ! -x "$STT" ]; then
  echo "stt not found at $STT"
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

export HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE

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
  env STT_WHISPER_BIN="$WHISPER_BIN" \
  "$STT" transcribe "$AUDIO" "$WHISPER_MODEL" \
  --engine whispercpp --no-ingest --save "$OUTPUT_DIR/whispercpp.txt" \
  "${CONVERT_FLAG[@]}"

if [ "$RUN_FASTER" = "1" ]; then
  if ! run_timed "faster-whisper" \
    "$STT" transcribe "$AUDIO" "$MODEL_NAME" \
    --engine faster-whisper --no-ingest --save "$OUTPUT_DIR/faster-whisper.txt" \
    "${CONVERT_FLAG[@]}"; then
    echo "faster-whisper failed. Install with: make install-faster-cpu"
  fi
fi

if [ "$RUN_PARAKEET" = "1" ]; then
  PARAKEET_FLAGS=(--engine parakeet)
  if [ -n "$PARAKEET_DIR" ]; then
    PARAKEET_FLAGS+=(--parakeet-dir "$PARAKEET_DIR")
  fi
  if [ -n "$PARAKEET_QUANT" ]; then
    PARAKEET_FLAGS+=(--parakeet-quant "$PARAKEET_QUANT")
  fi
  if ! run_timed "parakeet" \
    "$STT" transcribe "$AUDIO" "$PARAKEET_MODEL" \
    --no-ingest --save "$OUTPUT_DIR/parakeet.txt" \
    "${PARAKEET_FLAGS[@]}" \
    "${CONVERT_FLAG[@]}"; then
    echo "parakeet failed. Install with: uv pip install -e '.[cli,parakeet]'"
  fi
fi

run_whisperx_compute() {
  local label=$1
  local compute_type=$2
  local output=$3
  run_timed "$label" \
    env WHISPERX_COMPUTE_TYPE="$compute_type" \
    "$STT" transcribe "$AUDIO" "$WHISPERX_MODEL" \
    --engine whisperx --whisperx-model "$WHISPERX_MODEL" \
    --whisperx-device "$WHISPERX_DEVICE" --no-ingest \
    --save "$output" "${CONVERT_FLAG[@]}"
}

if [ "$RUN_WHISPERX" = "1" ]; then
  if ! run_whisperx_compute "whisperx-fp16" "float16" "$OUTPUT_DIR/whisperx-fp16.txt"; then
    echo "whisperx (float16) failed. Install with: make install-whisperx"
  fi
  if ! run_whisperx_compute "whisperx-fp32" "float32" "$OUTPUT_DIR/whisperx-fp32.txt"; then
    echo "whisperx (float32) failed. Install with: make install-whisperx"
  fi
fi

if [ "$RUN_WHISPERX_DIARIZE" = "1" ]; then
  if ! run_timed "whisperx-diarize" \
    "$STT" transcribe "$AUDIO" "$WHISPERX_MODEL" \
    --engine whisperx --whisperx-model "$WHISPERX_MODEL" \
    --whisperx-device "$WHISPERX_DEVICE" --whisperx-diarize --no-ingest \
    --save "$OUTPUT_DIR/whisperx-diarize.txt" "${CONVERT_FLAG[@]}"; then
    echo "whisperx diarize failed. Install with: make install-whisperx"
  fi
fi

echo "Saved transcripts:"
echo "  $OUTPUT_DIR/whispercpp.txt"
if [ "$RUN_FASTER" = "1" ]; then
  echo "  $OUTPUT_DIR/faster-whisper.txt"
fi
if [ "$RUN_PARAKEET" = "1" ]; then
  echo "  $OUTPUT_DIR/parakeet.txt"
fi
if [ "$RUN_WHISPERX" = "1" ]; then
  echo "  $OUTPUT_DIR/whisperx-fp16.txt"
  echo "  $OUTPUT_DIR/whisperx-fp32.txt"
fi
if [ "$RUN_WHISPERX_DIARIZE" = "1" ]; then
  echo "  $OUTPUT_DIR/whisperx-diarize.txt"
fi
