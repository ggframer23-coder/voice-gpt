.DEFAULT_GOAL := help
.PHONY: help clean venv install install-cli install-faster install-cpu install-cli-cpu install-faster-cpu install-whisper download-model benchmark benchmark-parakeet init transcribe reingest ingest-dir query add-text dedupe guard-stt guard-whisper-bin guard-whisper-src guard-model-file

SHELL := /usr/bin/env bash

UV ?= uv
VENV ?= .venv
UV_ENV ?= UV_PROJECT_ENVIRONMENT=$(VENV)
UV_CMD = $(UV_ENV) $(UV)
STT ?= $(VENV)/bin/stt
TORCH_CPU_INDEX ?= https://download.pytorch.org/whl/cpu
PYPI_INDEX ?= https://pypi.org/simple
TORCH_CPU_SPEC ?= torch==2.9.1+cpu
CPU_CONSTRAINTS ?= constraints-cpu.txt
WHISPER_DIR ?= third_party/whisper.cpp
WHISPER_BIN ?= $(WHISPER_DIR)/build/bin/whisper-cli
WHISPER_MODELS_DIR ?= $(WHISPER_DIR)/models
MODEL_NAME ?= base.en
MODEL ?= $(if $(filter faster-whisper,$(ENGINE)),$(MODEL_NAME),$(WHISPER_MODELS_DIR)/ggml-$(MODEL_NAME).bin)
EXTENSIONS ?= wav,mp3,m4a,flac,ogg,opus,webm
BENCH_OUTPUT_DIR ?= /tmp/stt-bench
PARAKEET_MODEL ?= nemo-parakeet-tdt-0.6b-v3
PARAKEET_DIR ?=
PARAKEET_QUANT ?= int8
RUN_FASTER ?= 1
RUN_PARAKEET ?= 1

AUDIO ?= audio
INPUT_DIR ?=
QUERY ?=
TEXT ?=
K ?= 5
ARCHIVE_DIR ?=
RECORDED_FROM ?=
RECORDED_TO ?=
SAVE ?=
ENGINE ?= faster-whisper
NO_CONVERT ?= 0
NO_INGEST ?= 0
REINGEST_RECURSIVE ?= 0
ifneq (,$(findstring r,$(MAKEFLAGS)))
REINGEST_RECURSIVE := 1
endif

TRANSCRIBE_FLAGS :=
ifeq ($(NO_CONVERT),1)
TRANSCRIBE_FLAGS += --no-convert
endif
ifeq ($(NO_INGEST),1)
TRANSCRIBE_FLAGS += --no-ingest
endif
ifneq ($(SAVE),)
TRANSCRIBE_FLAGS += --save $(SAVE)
endif
ifneq ($(ENGINE),)
TRANSCRIBE_FLAGS += --engine $(ENGINE)
endif

INGEST_FLAGS :=
ifneq ($(ARCHIVE_DIR),)
INGEST_FLAGS += --archive-dir $(ARCHIVE_DIR)
endif

QUERY_FLAGS :=
ifneq ($(RECORDED_FROM),)
QUERY_FLAGS += --recorded-from "$(RECORDED_FROM)"
endif
ifneq ($(RECORDED_TO),)
QUERY_FLAGS += --recorded-to "$(RECORDED_TO)"
endif

guard-%:
	@ if [ -z "$($*)" ]; then echo "Missing $*"; exit 1; fi

help:
	@ echo "Targets:"
	@ echo "  clean               Remove venv/build artifacts"
	@ echo "  venv                Create venv with uv"
	@ echo "  install             Install library (editable)"
	@ echo "  install-cli         Install CLI extra (editable)"
	@ echo "  install-faster      Install faster-whisper extra (editable)"
	@ echo "  install-whisperx    Install WhisperX extra (diarization support)"
	@ echo "  install-whisper     Build whisper.cpp (sets WHISPER_BIN path)"
	@ echo "  download-model      Download a whisper.cpp ggml model"
	@ echo "  benchmark           Compare whisper.cpp vs faster-whisper/parakeet timing"
	@ echo "  benchmark-parakeet  Compare whisper.cpp vs parakeet timing"
	@ echo "  install-cpu         Install CPU-only torch, then library"
	@ echo "  install-cli-cpu     Install CPU-only torch, then CLI extra"
	@ echo "  install-faster-cpu  Install CPU-only torch, then CLI+faster"
	@ echo "  init                Initialize storage (~/.stt by default)"
	@ echo "  transcribe          Transcribe AUDIO with MODEL"
	@ echo "  reingest            Force transcribe even if already ingested"
	@ echo "  ingest-dir          Ingest INPUT_DIR with MODEL"
	@ echo "  query               Search memories"
	@ echo "  add-text            Add text entry"
	@ echo "  dedupe              Remove duplicate audio entries (keeps latest by recorded_at)"
	@ echo ""
	@ echo "Examples:"
	@ echo "  make venv install-cli"
	@ echo "  make transcribe AUDIO=/path/a.wav MODEL=/path/model.gguf"
	@ echo "  make transcribe AUDIO=/path/a.wav MODEL=base.en ENGINE=faster-whisper"
	@ echo "  make install-whisper download-model MODEL_NAME=base.en"
	@ echo "  make ingest-dir INPUT_DIR=/path/audio MODEL=/path/model.gguf ARCHIVE_DIR=/path/processed"
	@ echo "  make reingest AUDIO=/path/audio MODEL=/path/model.gguf"
	@ echo "  make reingest -r AUDIO=/path/audio MODEL=/path/model.gguf"
	@ echo "  make benchmark AUDIO=/path/audio.wav MODEL_NAME=base.en"
	@ echo "  make query QUERY='memory pipeline' K=5"
	@ echo "  make add-text TEXT='Today I worked on the memory pipeline.'"
	@ echo "  make dedupe"

venv:
	$(UV_CMD) venv $(VENV)

clean:
	rm -rf $(VENV) build dist *.egg-info .pytest_cache

install:
	$(UV_CMD) pip install -e .

install-cli:
	$(UV_CMD) pip install -e ".[cli]"

install-faster:
	$(UV_CMD) pip install -e ".[cli,faster]"

install-whisperx:
	$(UV_CMD) pip install -e ".[whisperx]"

install-whisper:
	WHISPER_DIR=$(WHISPER_DIR) bash scripts/install_whisper_cpp.sh

download-model: guard-whisper-src
	mkdir -p $(WHISPER_MODELS_DIR)
	sh $(WHISPER_DIR)/models/download-ggml-model.sh $(MODEL_NAME) $(WHISPER_MODELS_DIR)

benchmark: ENGINE=whispercpp
benchmark: guard-stt guard-whisper-bin guard-AUDIO
	AUDIO=$(AUDIO) STT=$(STT) WHISPER_BIN=$(WHISPER_BIN) WHISPER_DIR=$(WHISPER_DIR) MODEL_NAME=$(MODEL_NAME) OUTPUT_DIR=$(BENCH_OUTPUT_DIR) NO_CONVERT=$(NO_CONVERT) PARAKEET_MODEL=$(PARAKEET_MODEL) PARAKEET_DIR=$(PARAKEET_DIR) PARAKEET_QUANT=$(PARAKEET_QUANT) RUN_FASTER=$(RUN_FASTER) RUN_PARAKEET=$(RUN_PARAKEET) bash scripts/benchmark_transcribe.sh

benchmark-parakeet: ENGINE=whispercpp
benchmark-parakeet: guard-stt guard-whisper-bin guard-AUDIO
	AUDIO=$(AUDIO) STT=$(STT) WHISPER_BIN=$(WHISPER_BIN) WHISPER_DIR=$(WHISPER_DIR) MODEL_NAME=$(MODEL_NAME) OUTPUT_DIR=$(BENCH_OUTPUT_DIR) NO_CONVERT=$(NO_CONVERT) PARAKEET_MODEL=$(PARAKEET_MODEL) PARAKEET_DIR=$(PARAKEET_DIR) PARAKEET_QUANT=$(PARAKEET_QUANT) RUN_FASTER=0 RUN_PARAKEET=1 bash scripts/benchmark_transcribe.sh

install-cpu:
	$(UV_CMD) pip install --index-url $(TORCH_CPU_INDEX) $(TORCH_CPU_SPEC)
	$(UV_CMD) pip install -e . --index-url $(PYPI_INDEX) --extra-index-url $(TORCH_CPU_INDEX) --constraint $(CPU_CONSTRAINTS)

install-cli-cpu:
	$(UV_CMD) pip install --index-url $(TORCH_CPU_INDEX) $(TORCH_CPU_SPEC)
	$(UV_CMD) pip install -e ".[cli]" --index-url $(PYPI_INDEX) --extra-index-url $(TORCH_CPU_INDEX) --constraint $(CPU_CONSTRAINTS)

install-faster-cpu:
	$(UV_CMD) pip install --index-url $(TORCH_CPU_INDEX) $(TORCH_CPU_SPEC)
	$(UV_CMD) pip install -e ".[cli,faster]" --index-url $(PYPI_INDEX) --extra-index-url $(TORCH_CPU_INDEX) --constraint $(CPU_CONSTRAINTS)

guard-stt:
	@ if [ ! -x "$(STT)" ]; then echo "Missing $(STT). Run 'make venv install-cli-cpu' first."; exit 1; fi

guard-whisper-src:
	@ if [ ! -f "$(WHISPER_DIR)/models/download-ggml-model.sh" ]; then echo "Missing whisper.cpp checkout at $(WHISPER_DIR). Run 'make install-whisper' first."; exit 1; fi

guard-whisper-bin:
ifeq ($(ENGINE),whispercpp)
	@ if [ ! -x "$(WHISPER_BIN)" ]; then echo "Missing $(WHISPER_BIN). Run 'make install-whisper' or set WHISPER_BIN=/path/to/whisper.cpp/build/bin/whisper-cli"; exit 1; fi
endif

guard-model-file:
ifeq ($(ENGINE),faster-whisper)
	@ :
else ifeq ($(ENGINE),parakeet)
	@ :
else
	@ if [ ! -f "$(MODEL)" ]; then echo "Missing model file $(MODEL). Run 'make download-model MODEL_NAME=base.en' or set MODEL=/path/to/model.bin"; exit 1; fi
endif

init: guard-stt
	$(STT) init

transcribe: guard-stt guard-whisper-bin guard-model-file guard-AUDIO guard-MODEL
	@ if [ -d "$(AUDIO)" ]; then \
		STT_WHISPER_BIN=$(WHISPER_BIN) $(STT) ingest-dir "$(AUDIO)" $(MODEL) $(INGEST_FLAGS); \
	else \
		STT_WHISPER_BIN=$(WHISPER_BIN) $(STT) transcribe "$(AUDIO)" $(MODEL) $(TRANSCRIBE_FLAGS); \
	fi

reingest: guard-stt guard-whisper-bin guard-model-file guard-AUDIO guard-MODEL
	$(STT) reingest $(if $(REINGEST_RECURSIVE),-r,) "$(AUDIO)" $(MODEL) $(TRANSCRIBE_FLAGS)

ingest-dir: guard-stt guard-whisper-bin guard-model-file guard-INPUT_DIR guard-MODEL
	STT_WHISPER_BIN=$(WHISPER_BIN) $(STT) ingest-dir $(INPUT_DIR) $(MODEL) $(INGEST_FLAGS)

query: guard-stt guard-QUERY
	$(STT) query "$(QUERY)" -k $(K) $(QUERY_FLAGS)

add-text: guard-stt guard-TEXT
	$(STT) add-text "$(TEXT)"

dedupe:
	python3 scripts/dedupe_journal.py
