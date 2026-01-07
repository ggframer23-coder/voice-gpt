#!/usr/bin/env python3
from __future__ import annotations

import os
import sqlite3


def _journal_path() -> str:
    base_dir = os.environ.get("VOICE_GPT_HOME", os.path.expanduser("~/.voice-gpt"))
    return os.environ.get("VOICE_GPT_DB", os.path.join(base_dir, "journal.sqlite"))


def main() -> None:
    db_path = _journal_path()
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(
            "DELETE FROM entries "
            "WHERE audio_path IS NOT NULL AND audio_path != '' "
            "AND EXISTS ("
            "  SELECT 1 FROM entries e2 "
            "  WHERE e2.audio_path = entries.audio_path "
            "    AND (e2.recorded_at > entries.recorded_at "
            "         OR (e2.recorded_at = entries.recorded_at AND e2.id > entries.id))"
            ")"
        )
        conn.commit()
        print(f"Removed {cur.rowcount} duplicate entries")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
