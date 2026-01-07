.DEFAULT_GOAL := help
.PHONY: help clean venv install install-cli install-faster install-cpu install-cli-cpu install-faster-cpu install-whisper download-model benchmark benchmark-parakeet init transcribe reingest ingest-dir query add-text dedupe guard-voice-gpt guard-whisper-bin guard-whisper-src guard-model-file

SHELL := /usr/bin/env bash

UV ?= uv
VENV ?= .venv
UV_ENV ?= UV_PROJECT_ENVIRONMENT=$(VENV)
UV_CMD = $(UV_ENV) $(UV)
VOICE_GPT ?= $(VENV)/bin/voice-gpt
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
BENCH_OUTPUT_DIR ?= /tmp/voice-gpt-bench
PARAKEET_MODEL ?= nemo-parakeet-tdt-0.6b-v3
PARAKEET_DIR ?=
PARAKEET_QUANT ?= int8
RUN_FASTER ?= 1
RUN_PARAKEET ?= 1

AUDIO ?= audio
MODEL ?= base.en
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
	@ echo "  install-whisper     Build whisper.cpp (sets WHISPER_BIN path)"
	@ echo "  download-model      Download a whisper.cpp ggml model"
	@ echo "  benchmark           Compare whisper.cpp vs faster-whisper/parakeet timing"
	@ echo "  benchmark-parakeet  Compare whisper.cpp vs parakeet timing"
	@ echo "  install-cpu         Install CPU-only torch, then library"
	@ echo "  install-cli-cpu     Install CPU-only torch, then CLI extra"
	@ echo "  install-faster-cpu  Install CPU-only torch, then CLI+faster"
	@ echo "  init                Initialize storage (~/.voice-gpt by default)"
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

install-whisper:
	WHISPER_DIR=$(WHISPER_DIR) bash scripts/install_whisper_cpp.sh

download-model: guard-whisper-src
	mkdir -p $(WHISPER_MODELS_DIR)
	sh $(WHISPER_DIR)/models/download-ggml-model.sh $(MODEL_NAME) $(WHISPER_MODELS_DIR)

benchmark: ENGINE=whispercpp
benchmark: guard-voice-gpt guard-whisper-bin guard-AUDIO
	AUDIO=$(AUDIO) VOICE_GPT=$(VOICE_GPT) WHISPER_BIN=$(WHISPER_BIN) WHISPER_DIR=$(WHISPER_DIR) MODEL_NAME=$(MODEL_NAME) OUTPUT_DIR=$(BENCH_OUTPUT_DIR) NO_CONVERT=$(NO_CONVERT) PARAKEET_MODEL=$(PARAKEET_MODEL) PARAKEET_DIR=$(PARAKEET_DIR) PARAKEET_QUANT=$(PARAKEET_QUANT) RUN_FASTER=$(RUN_FASTER) RUN_PARAKEET=$(RUN_PARAKEET) bash scripts/benchmark_transcribe.sh

benchmark-parakeet: ENGINE=whispercpp
benchmark-parakeet: guard-voice-gpt guard-whisper-bin guard-AUDIO
	AUDIO=$(AUDIO) VOICE_GPT=$(VOICE_GPT) WHISPER_BIN=$(WHISPER_BIN) WHISPER_DIR=$(WHISPER_DIR) MODEL_NAME=$(MODEL_NAME) OUTPUT_DIR=$(BENCH_OUTPUT_DIR) NO_CONVERT=$(NO_CONVERT) PARAKEET_MODEL=$(PARAKEET_MODEL) PARAKEET_DIR=$(PARAKEET_DIR) PARAKEET_QUANT=$(PARAKEET_QUANT) RUN_FASTER=0 RUN_PARAKEET=1 bash scripts/benchmark_transcribe.sh

install-cpu:
	$(UV_CMD) pip install --index-url $(TORCH_CPU_INDEX) $(TORCH_CPU_SPEC)
	$(UV_CMD) pip install -e . --index-url $(PYPI_INDEX) --extra-index-url $(TORCH_CPU_INDEX) --constraint $(CPU_CONSTRAINTS)

install-cli-cpu:
	$(UV_CMD) pip install --index-url $(TORCH_CPU_INDEX) $(TORCH_CPU_SPEC)
	$(UV_CMD) pip install -e ".[cli]" --index-url $(PYPI_INDEX) --extra-index-url $(TORCH_CPU_INDEX) --constraint $(CPU_CONSTRAINTS)

install-faster-cpu:
	$(UV_CMD) pip install --index-url $(TORCH_CPU_INDEX) $(TORCH_CPU_SPEC)
	$(UV_CMD) pip install -e ".[cli,faster]" --index-url $(PYPI_INDEX) --extra-index-url $(TORCH_CPU_INDEX) --constraint $(CPU_CONSTRAINTS)

guard-voice-gpt:
	@ if [ ! -x "$(VOICE_GPT)" ]; then echo "Missing $(VOICE_GPT). Run 'make venv install-cli-cpu' first."; exit 1; fi

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

init: guard-voice-gpt
	$(VOICE_GPT) init

transcribe: guard-voice-gpt guard-whisper-bin guard-model-file guard-AUDIO guard-MODEL
	@ if [ -d "$(AUDIO)" ]; then \
		VOICE_GPT_WHISPER_BIN=$(WHISPER_BIN) $(VOICE_GPT) ingest-dir "$(AUDIO)" $(MODEL) $(INGEST_FLAGS); \
	else \
		VOICE_GPT_WHISPER_BIN=$(WHISPER_BIN) $(VOICE_GPT) transcribe "$(AUDIO)" $(MODEL) $(TRANSCRIBE_FLAGS); \
	fi

reingest: guard-voice-gpt guard-whisper-bin guard-model-file guard-AUDIO guard-MODEL
	@ if [ -d "$(AUDIO)" ]; then \
		shopt -s nullglob; \
		IFS=, read -ra exts <<< "$(EXTENSIONS)"; \
		files=(); \
		for ext in "$${exts[@]}"; do \
			ext="$${ext#.}"; \
			files+=( "$(AUDIO)"/*."$$ext" ); \
		done; \
		if [ "$${#files[@]}" -eq 0 ]; then \
			echo "No audio files found."; \
			exit 1; \
		fi; \
		for f in "$${files[@]}"; do \
			[ -f "$$f" ] || continue; \
			VOICE_GPT_WHISPER_BIN=$(WHISPER_BIN) $(VOICE_GPT) transcribe "$$f" $(MODEL) $(TRANSCRIBE_FLAGS); \
		done; \
	else \
		VOICE_GPT_WHISPER_BIN=$(WHISPER_BIN) $(VOICE_GPT) transcribe "$(AUDIO)" $(MODEL) $(TRANSCRIBE_FLAGS); \
	fi

ingest-dir: guard-voice-gpt guard-whisper-bin guard-model-file guard-INPUT_DIR guard-MODEL
	VOICE_GPT_WHISPER_BIN=$(WHISPER_BIN) $(VOICE_GPT) ingest-dir $(INPUT_DIR) $(MODEL) $(INGEST_FLAGS)

query: guard-voice-gpt guard-QUERY
	$(VOICE_GPT) query "$(QUERY)" -k $(K) $(QUERY_FLAGS)

add-text: guard-voice-gpt guard-TEXT
	$(VOICE_GPT) add-text "$(TEXT)"

dedupe:
	python3 scripts/dedupe_journal.py
