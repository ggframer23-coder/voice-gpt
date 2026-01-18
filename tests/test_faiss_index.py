import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import stt.faiss_index as faiss_index


class FakeIndex:
    def __init__(self, dim: int) -> None:
        self.d = dim
        self.added = None
        self.search_args = None

    def add_with_ids(self, vectors, ids) -> None:
        self.added = (vectors, ids)

    def search(self, vec, k):
        self.search_args = (vec, k)
        return np.array([[0.2, 0.1]]), np.array([[2, 1]])


def test_load_or_create_existing(tmp_path, monkeypatch) -> None:
    index_path = tmp_path / "index.bin"
    index_path.write_text("x", encoding="utf-8")
    fake = FakeIndex(dim=3)

    fake_faiss = SimpleNamespace(read_index=lambda _p: fake)
    monkeypatch.setitem(sys.modules, "faiss", fake_faiss)

    loaded = faiss_index.load_or_create(index_path, dim=3)

    assert loaded is fake


def test_load_or_create_new(tmp_path, monkeypatch) -> None:
    index_path = tmp_path / "index.bin"
    writes = []

    def fake_flat(dim):
        return FakeIndex(dim)

    def fake_id_map(index):
        return index

    def fake_write(index, path):
        writes.append((index, path))

    fake_faiss = SimpleNamespace(IndexFlatIP=fake_flat, IndexIDMap=fake_id_map, write_index=fake_write)
    monkeypatch.setitem(sys.modules, "faiss", fake_faiss)

    loaded = faiss_index.load_or_create(index_path, dim=4)

    assert isinstance(loaded, FakeIndex)
    assert writes
    assert Path(writes[0][1]) == index_path


def test_add_vectors() -> None:
    index = FakeIndex(dim=2)
    faiss_index.add_vectors(index, [1, 2], [[0.1, 0.2], [0.3, 0.4]])
    vecs, ids = index.added
    assert ids.tolist() == [1, 2]
    assert vecs.shape == (2, 2)


def test_search() -> None:
    index = FakeIndex(dim=2)
    scores, ids = faiss_index.search(index, [0.1, 0.2], k=2)
    assert scores.tolist() == [0.2, 0.1]
    assert ids.tolist() == [2, 1]
