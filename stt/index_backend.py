from __future__ import annotations

import importlib
from pathlib import Path
from typing import Iterable, Protocol, Tuple


class IndexBackend(Protocol):
    def load_or_create(self, index_path: Path, dim: int):
        ...

    def save(self, index, index_path: Path) -> None:
        ...

    def add_vectors(self, index, ids: Iterable[int], vectors: Iterable[Iterable[float]]) -> None:
        ...

    def search(self, index, query_vec: Iterable[float], k: int) -> Tuple[object, object]:
        ...


def _normalize_backend(name: str) -> str:
    return name.strip().lower()


def _module_name(backend: str) -> str:
    name = _normalize_backend(backend)
    if name in {"faiss", "faiss-cpu"}:
        return "stt.faiss_index"
    if name in {"chroma", "chromadb"}:
        return "stt.chroma_index"
    raise ValueError(f"Unknown index backend: {backend}")


def _load_backend(backend: str) -> IndexBackend:
    module_name = _module_name(backend)
    return importlib.import_module(module_name)


def _wrap_missing_dep(backend: str, exc: ModuleNotFoundError) -> RuntimeError:
    name = _normalize_backend(backend)
    if name in {"chroma", "chromadb"}:
        message = "ChromaDB backend requires the chromadb package. Install with `pip install .[chroma]`."
    elif name in {"faiss", "faiss-cpu"}:
        message = "FAISS backend requires the faiss-cpu package. Install dependencies and try again."
    else:
        message = f"Backend {backend} requires additional dependencies."
    return RuntimeError(message)


def load_or_create(backend: str, index_path: Path, dim: int):
    module = _load_backend(backend)
    try:
        return module.load_or_create(index_path, dim)
    except ModuleNotFoundError as exc:
        raise _wrap_missing_dep(backend, exc) from exc


def save(backend: str, index, index_path: Path) -> None:
    module = _load_backend(backend)
    try:
        module.save(index, index_path)
    except ModuleNotFoundError as exc:
        raise _wrap_missing_dep(backend, exc) from exc


def add_vectors(backend: str, index, ids: Iterable[int], vectors: Iterable[Iterable[float]]) -> None:
    module = _load_backend(backend)
    try:
        module.add_vectors(index, ids, vectors)
    except ModuleNotFoundError as exc:
        raise _wrap_missing_dep(backend, exc) from exc


def search(backend: str, index, query_vec: Iterable[float], k: int):
    module = _load_backend(backend)
    try:
        return module.search(index, query_vec, k)
    except ModuleNotFoundError as exc:
        raise _wrap_missing_dep(backend, exc) from exc
