from pathlib import Path

import numpy as np

import stt.chroma_index as chroma_index


def test_load_or_create_new(tmp_path) -> None:
    index_path = tmp_path / "chroma"
    collection = chroma_index.load_or_create(index_path, dim=3)

    assert collection is not None
    assert collection.name == "chunks"
    assert index_path.exists()


def test_load_or_create_existing(tmp_path) -> None:
    index_path = tmp_path / "chroma"

    collection1 = chroma_index.load_or_create(index_path, dim=3)
    chroma_index.add_vectors(collection1, [1], [[0.1, 0.2, 0.3]])

    collection2 = chroma_index.load_or_create(index_path, dim=3)
    assert collection2.count() == 1


def test_add_vectors(tmp_path) -> None:
    index_path = tmp_path / "chroma"
    collection = chroma_index.load_or_create(index_path, dim=2)

    chroma_index.add_vectors(collection, [1, 2], [[0.1, 0.2], [0.3, 0.4]])

    assert collection.count() == 2


def test_search(tmp_path) -> None:
    index_path = tmp_path / "chroma"
    collection = chroma_index.load_or_create(index_path, dim=3)

    chroma_index.add_vectors(collection, [10, 20], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    scores, ids = chroma_index.search(collection, [1.0, 0.0, 0.0], k=2)

    assert len(scores) == 2
    assert len(ids) == 2
    assert ids[0] == 10  # Closest match
    assert scores[0] > scores[1]


def test_search_empty_collection(tmp_path) -> None:
    index_path = tmp_path / "chroma"
    collection = chroma_index.load_or_create(index_path, dim=3)

    scores, ids = chroma_index.search(collection, [0.1, 0.2, 0.3], k=3)

    assert len(scores) == 3
    assert len(ids) == 3
    # Should be padded with -1
    assert all(i == -1 for i in ids)
