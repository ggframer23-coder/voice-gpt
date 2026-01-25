import json
import sqlite3
from pathlib import Path

import numpy as np

import stt.journal as journal
from stt.settings import Settings


class FakeModel:
    def __init__(self, dim: int = 3) -> None:
        self._dim = dim

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim


class FakeIndex:
    def __init__(self) -> None:
        self.added = []

    def add_with_ids(self, vectors, ids) -> None:
        self.added.append((vectors, ids))


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        base_dir=tmp_path,
        db_path=tmp_path / "journal.sqlite",
        index_path=tmp_path / "faiss.index",
        index_backend="faiss",
        model_name="fake-model",
        offline=True,
        whisper_bin=None,
        vad_bin=None,
        vad_model=None,
        parakeet_model="parakeet",
        parakeet_dir=None,
        parakeet_quant=None,
    )


def test_chunk_text_overlap() -> None:
    chunks = journal.chunk_text("one two three four five", max_words=3, overlap=1)
    assert chunks == ["one two three", "three four five"]


def test_chunk_text_empty() -> None:
    assert journal.chunk_text("") == []


def test_ensure_columns(tmp_path) -> None:
    db_path = tmp_path / "db.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE entries (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, text TEXT NOT NULL)"
        )
        journal.ensure_recorded_at_column(conn)
        journal.ensure_audio_columns(conn)
        cols = [row[1] for row in conn.execute("PRAGMA table_info(entries)").fetchall()]
        assert "recorded_at" in cols
        assert "audio_size_bytes" in cols
        assert "audio_duration_seconds" in cols


def test_audio_duration_missing_ffprobe(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(journal.shutil, "which", lambda *_args: None)
    duration = journal._audio_duration_seconds(tmp_path / "missing.wav")
    assert duration is None


def test_add_entry_writes_db(tmp_path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    fake_index = FakeIndex()

    monkeypatch.setattr(journal, "load_model", lambda *_args, **_kwargs: FakeModel())
    monkeypatch.setattr(journal, "load_or_create", lambda *_args, **_kwargs: fake_index)
    monkeypatch.setattr(journal, "save", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(journal, "add_vectors", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(journal, "embed_texts", lambda *_args, **_kwargs: [[0.1, 0.2, 0.3]])
    monkeypatch.setattr(journal, "_audio_duration_seconds", lambda *_args, **_kwargs: 1.25)

    audio_path = tmp_path / "sample.wav"
    audio_path.write_text("x", encoding="utf-8")
    entry_id = journal.add_entry(
        settings,
        text="hello world",
        source="test",
        audio_path=str(audio_path),
        metadata={"note": "ok"},
        recorded_at="2024-01-01T00:00:00+00:00",
    )

    with sqlite3.connect(settings.db_path) as conn:
        row = conn.execute("SELECT * FROM entries WHERE id = ?", (entry_id,)).fetchone()
        assert row is not None
        stored_meta = json.loads(row[8])
        assert stored_meta["note"] == "ok"
        chunk_rows = conn.execute("SELECT * FROM chunks WHERE entry_id = ?", (entry_id,)).fetchall()
        assert len(chunk_rows) == 1

    assert journal.has_audio(settings, str(audio_path)) is True


def test_search_returns_results(tmp_path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    fake_index = FakeIndex()

    monkeypatch.setattr(journal, "load_model", lambda *_args, **_kwargs: FakeModel())
    monkeypatch.setattr(journal, "load_or_create", lambda *_args, **_kwargs: fake_index)
    monkeypatch.setattr(journal, "save", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(journal, "add_vectors", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(journal, "embed_texts", lambda *_args, **_kwargs: [[0.1, 0.2, 0.3]])

    entry_id = journal.add_entry(
        settings,
        text="hello world",
        recorded_at="2024-01-02T00:00:00+00:00",
    )

    with sqlite3.connect(settings.db_path) as conn:
        chunk_id = conn.execute("SELECT id FROM chunks WHERE entry_id = ?", (entry_id,)).fetchone()[0]

    monkeypatch.setattr(
        journal,
        "index_search",
        lambda *_args, **_kwargs: (np.array([0.9]), np.array([chunk_id])),
    )

    results = journal.search(settings, query="hello", recorded_from="2024-01-01T00:00:00+00:00")

    assert len(results) == 1
    assert results[0]["entry_id"] == entry_id
