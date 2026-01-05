.DEFAULT_GOAL := help
.PHONY: help clean venv install install-cli install-faster install-cpu install-cli-cpu install-faster-cpu init transcribe ingest-dir query add-text guard-voice-gpt

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
ENGINE ?=
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
	@ echo "  install-cpu         Install CPU-only torch, then library"
	@ echo "  install-cli-cpu     Install CPU-only torch, then CLI extra"
	@ echo "  install-faster-cpu  Install CPU-only torch, then CLI+faster"
	@ echo "  init                Initialize storage (~/.voice-gpt by default)"
	@ echo "  transcribe          Transcribe AUDIO with MODEL"
	@ echo "  ingest-dir          Ingest INPUT_DIR with MODEL"
	@ echo "  query               Search memories"
	@ echo "  add-text            Add text entry"
	@ echo ""
	@ echo "Examples:"
	@ echo "  make venv install-cli"
	@ echo "  make transcribe AUDIO=/path/a.wav MODEL=/path/model.gguf"
	@ echo "  make transcribe AUDIO=/path/a.wav MODEL=base.en ENGINE=faster-whisper"
	@ echo "  make ingest-dir INPUT_DIR=/path/audio MODEL=/path/model.gguf ARCHIVE_DIR=/path/processed"
	@ echo "  make query QUERY='memory pipeline' K=5"
	@ echo "  make add-text TEXT='Today I worked on the memory pipeline.'"

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

init: guard-voice-gpt
	$(VOICE_GPT) init

transcribe: guard-voice-gpt guard-AUDIO guard-MODEL
	$(VOICE_GPT) transcribe $(AUDIO) $(MODEL) $(TRANSCRIBE_FLAGS)

ingest-dir: guard-voice-gpt guard-INPUT_DIR guard-MODEL
	$(VOICE_GPT) ingest-dir $(INPUT_DIR) $(MODEL) $(INGEST_FLAGS)

query: guard-voice-gpt guard-QUERY
	$(VOICE_GPT) query "$(QUERY)" -k $(K) $(QUERY_FLAGS)

add-text: guard-voice-gpt guard-TEXT
	$(VOICE_GPT) add-text "$(TEXT)"
