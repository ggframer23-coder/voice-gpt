from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Tuple

if TYPE_CHECKING:
    import chromadb
    import numpy as np


def load_or_create(index_path: Path, dim: int) -> "chromadb.Collection":
    import chromadb

    index_path.parent.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(index_path))
    collection = client.get_or_create_collection(
        name="chunks",
        metadata={"hnsw:space": "cosine", "dim": dim},
    )
    return collection


def save(collection: "chromadb.Collection", index_path: Path) -> None:
    # ChromaDB auto-persists with PersistentClient, no-op
    pass


def add_vectors(collection: "chromadb.Collection", ids: Iterable[int], vectors: Iterable[Iterable[float]]) -> None:
    id_list = [str(i) for i in ids]
    vec_list = [list(v) for v in vectors]
    collection.add(ids=id_list, embeddings=vec_list)


def search(collection: "chromadb.Collection", query_vec: Iterable[float], k: int) -> Tuple["np.ndarray", "np.ndarray"]:
    import numpy as np

    vec = [list(query_vec)]
    results = collection.query(query_embeddings=vec, n_results=k)

    distances = results.get("distances", [[]])[0]
    ids_str = results.get("ids", [[]])[0]

    # Convert cosine distance to similarity: score = 1 - distance
    scores = [1.0 - d for d in distances]
    ids = [int(i) for i in ids_str]

    # Pad with -1 if fewer than k results
    while len(ids) < k:
        scores.append(0.0)
        ids.append(-1)

    return np.array(scores, dtype="float32"), np.array(ids, dtype="int64")
