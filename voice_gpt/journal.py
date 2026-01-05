from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

from .embeddings import embed_texts, load_model
from .faiss_index import add_vectors, load_or_create, save, search as faiss_search
from .settings import Settings


SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  recorded_at TEXT,
  text TEXT NOT NULL,
  source TEXT,
  audio_path TEXT,
  metadata TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  entry_id INTEGER NOT NULL,
  chunk_index INTEGER NOT NULL,
  text TEXT NOT NULL,
  FOREIGN KEY(entry_id) REFERENCES entries(id)
);
"""


def init_store(settings: Settings) -> None:
    settings.base_dir.mkdir(parents=True, exist_ok=True)
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(settings.db_path) as conn:
        conn.executescript(SCHEMA)
        conn.commit()
        ensure_recorded_at_column(conn)

    model = load_model(settings.model_name)
    dim = model.get_sentence_embedding_dimension()
    index = load_or_create(settings.index_path, dim)
    save(index, settings.index_path)


def ensure_recorded_at_column(conn: sqlite3.Connection) -> None:
    cols = [row[1] for row in conn.execute("PRAGMA table_info(entries)").fetchall()]
    if "recorded_at" not in cols:
        conn.execute("ALTER TABLE entries ADD COLUMN recorded_at TEXT")
        conn.commit()


def _local_iso_from_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).astimezone().isoformat()


def _local_iso_now() -> str:
    return datetime.now().astimezone().isoformat()


def chunk_text(text: str, max_words: int = 200, overlap: int = 40) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = max(0, end - overlap)
    return chunks


def add_entry(
    settings: Settings,
    text: str,
    source: Optional[str] = None,
    audio_path: Optional[str] = None,
    metadata: Optional[dict] = None,
    recorded_at: Optional[str] = None,
    max_words: int = 200,
    overlap: int = 40,
) -> int:
    init_store(settings)
    created_at = datetime.now(timezone.utc).isoformat()
    if not recorded_at:
        if audio_path:
            audio_file = Path(audio_path)
            if audio_file.exists():
                recorded_at = _local_iso_from_timestamp(audio_file.stat().st_mtime)
    if not recorded_at:
        recorded_at = _local_iso_now()
    meta_json = json.dumps(metadata or {}, ensure_ascii=True)

    with sqlite3.connect(settings.db_path) as conn:
        ensure_recorded_at_column(conn)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO entries (created_at, recorded_at, text, source, audio_path, metadata) VALUES (?, ?, ?, ?, ?, ?)",
            (created_at, recorded_at, text, source, audio_path, meta_json),
        )
        entry_id = cur.lastrowid

        chunks = chunk_text(text, max_words=max_words, overlap=overlap)
        chunk_ids = []
        for idx, chunk in enumerate(chunks):
            cur.execute(
                "INSERT INTO chunks (entry_id, chunk_index, text) VALUES (?, ?, ?)",
                (entry_id, idx, chunk),
            )
            chunk_ids.append(cur.lastrowid)
        conn.commit()

    if chunk_ids:
        embed_inputs = [f"Recorded at: {recorded_at}\n{chunk}" for chunk in chunks]
        vectors = embed_texts(settings.model_name, embed_inputs)
        model = load_model(settings.model_name)
        dim = model.get_sentence_embedding_dimension()
        index = load_or_create(settings.index_path, dim)
        add_vectors(index, chunk_ids, vectors)
        save(index, settings.index_path)

    return entry_id


def search(
    settings: Settings,
    query: str,
    k: int = 5,
    recorded_from: Optional[str] = None,
    recorded_to: Optional[str] = None,
) -> List[dict]:
    init_store(settings)
    model = load_model(settings.model_name)
    dim = model.get_sentence_embedding_dimension()
    index = load_or_create(settings.index_path, dim)

    query_vec = embed_texts(settings.model_name, [query])[0]
    scores, ids = faiss_search(index, query_vec, k)

    results: List[dict] = []
    with sqlite3.connect(settings.db_path) as conn:
        conn.row_factory = sqlite3.Row
        for score, chunk_id in zip(scores, ids):
            if chunk_id == -1:
                continue
            params = [int(chunk_id)]
            query_sql = (
                "SELECT chunks.id AS chunk_id, chunks.text AS chunk_text, entries.id AS entry_id, "
                "entries.created_at, entries.recorded_at, entries.text AS entry_text, entries.source, "
                "entries.audio_path, entries.metadata "
                "FROM chunks JOIN entries ON chunks.entry_id = entries.id WHERE chunks.id = ?"
            )
            if recorded_from:
                query_sql += " AND entries.recorded_at >= ?"
                params.append(recorded_from)
            if recorded_to:
                query_sql += " AND entries.recorded_at <= ?"
                params.append(recorded_to)
            row = conn.execute(query_sql, params).fetchone()
            if not row:
                continue
            results.append(
                {
                    "score": float(score),
                    "chunk_id": row["chunk_id"],
                    "chunk_text": row["chunk_text"],
                    "entry_id": row["entry_id"],
                    "created_at": row["created_at"],
                    "recorded_at": row["recorded_at"],
                    "entry_text": row["entry_text"],
                    "source": row["source"],
                    "audio_path": row["audio_path"],
                    "metadata": json.loads(row["metadata"] or "{}"),
                }
            )
    return results


def has_audio(settings: Settings, audio_path: str) -> bool:
    init_store(settings)
    with sqlite3.connect(settings.db_path) as conn:
        row = conn.execute(
            "SELECT 1 FROM entries WHERE audio_path = ? LIMIT 1",
            (audio_path,),
        ).fetchone()
        return row is not None
