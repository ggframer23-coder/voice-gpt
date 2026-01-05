#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHISPER_DIR="${WHISPER_DIR:-$ROOT_DIR/third_party/whisper.cpp}"

if [ ! -d "$WHISPER_DIR" ]; then
  echo "Cloning whisper.cpp into $WHISPER_DIR"
  git clone https://github.com/ggerganov/whisper.cpp "$WHISPER_DIR"
fi

if [ ! -f "$WHISPER_DIR/Makefile" ]; then
  echo "Missing Makefile in $WHISPER_DIR; expected a whisper.cpp checkout."
  exit 1
fi

echo "Building whisper.cpp..."
(
  cd "$WHISPER_DIR"
  if ! command -v cmake >/dev/null 2>&1; then
    echo "cmake is required but not installed."
    echo "Install it (e.g., your system package manager) and rerun: make install-whisper"
    exit 1
  fi
  cmake -B build
  cmake --build build -j
)

if [ ! -x "$WHISPER_DIR/build/bin/whisper-cli" ]; then
  echo "Build complete but whisper-cli binary not found at $WHISPER_DIR/build/bin/whisper-cli"
  exit 1
fi

echo "Built whisper.cpp at $WHISPER_DIR/build/bin/whisper-cli"
echo "Set: export VOICE_GPT_WHISPER_BIN=$WHISPER_DIR/build/bin/whisper-cli"
