#!/usr/bin/env python3
"""Migrate data from FAISS (~/.stt) to ChromaDB (~/.stt-chroma)."""

import argparse
import shutil
import sqlite3
from pathlib import Path

from stt.chroma_index import add_vectors, load_or_create
from stt.embeddings import embed_texts, load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate FAISS data to ChromaDB")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    args = parser.parse_args()

    source_dir = Path.home() / ".stt"
    target_dir = Path.home() / ".stt-chroma"
    source_db = source_dir / "journal.sqlite"
    target_db = target_dir / "journal.sqlite"
    chroma_path = target_dir / "chroma"

    if not source_db.exists():
        print(f"Source database not found: {source_db}")
        return

    # Count chunks in source
    with sqlite3.connect(source_db) as conn:
        chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        entry_count = conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]

    print(f"Source: {source_db}")
    print(f"  Entries: {entry_count}")
    print(f"  Chunks: {chunk_count}")
    print()
    print(f"Target: {target_dir}")
    print(f"  Database: {target_db}")
    print(f"  ChromaDB: {chroma_path}")

    if args.dry_run:
        print()
        print("[DRY RUN] Would perform the following:")
        print(f"  1. Create directory: {target_dir}")
        print(f"  2. Copy {source_db} -> {target_db}")
        print(f"  3. Re-embed {chunk_count} chunks")
        print(f"  4. Add vectors to ChromaDB at {chroma_path}")
        return

    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)

    # Copy SQLite database
    print()
    print(f"Copying database to {target_db}...")
    shutil.copy2(source_db, target_db)

    # Load embedding model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Loading embedding model: {model_name}...")
    model = load_model(model_name, offline=True)
    dim = model.get_sentence_embedding_dimension()

    # Create ChromaDB collection
    print(f"Creating ChromaDB collection at {chroma_path}...")
    collection = load_or_create(chroma_path, dim)

    # Read all chunks with their recorded_at from entries
    print("Reading chunks from database...")
    with sqlite3.connect(target_db) as conn:
        rows = conn.execute(
            """
            SELECT chunks.id, chunks.text, entries.recorded_at
            FROM chunks
            JOIN entries ON chunks.entry_id = entries.id
            ORDER BY chunks.id
            """
        ).fetchall()

    if not rows:
        print("No chunks to migrate.")
        return

    # Process in batches
    batch_size = 100
    total = len(rows)
    print(f"Migrating {total} chunks...")

    for i in range(0, total, batch_size):
        batch = rows[i : i + batch_size]
        chunk_ids = [row[0] for row in batch]
        texts = [f"Recorded at: {row[2]}\n{row[1]}" for row in batch]

        vectors = embed_texts(model_name, texts, offline=True)
        add_vectors(collection, chunk_ids, vectors)

        progress = min(i + batch_size, total)
        print(f"  Processed {progress}/{total} chunks")

    print()
    print("Migration complete!")
    print(f"  ChromaDB collection count: {collection.count()}")


if __name__ == "__main__":
    main()
