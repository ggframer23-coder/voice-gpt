from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import faiss
import numpy as np


def load_or_create(index_path: Path, dim: int) -> faiss.IndexIDMap:
    if index_path.exists():
        index = faiss.read_index(str(index_path))
        if index.d != dim:
            raise ValueError(f"FAISS index dim {index.d} != embedding dim {dim}")
        return index
    index = faiss.IndexFlatIP(dim)
    id_index = faiss.IndexIDMap(index)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(id_index, str(index_path))
    return id_index


def save(index: faiss.IndexIDMap, index_path: Path) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))


def add_vectors(index: faiss.IndexIDMap, ids: Iterable[int], vectors: Iterable[Iterable[float]]) -> None:
    vecs = np.array(list(vectors), dtype="float32")
    id_arr = np.array(list(ids), dtype="int64")
    index.add_with_ids(vecs, id_arr)


def search(index: faiss.IndexIDMap, query_vec: Iterable[float], k: int) -> Tuple[np.ndarray, np.ndarray]:
    vec = np.array([list(query_vec)], dtype="float32")
    scores, ids = index.search(vec, k)
    return scores[0], ids[0]
