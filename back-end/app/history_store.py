import json
import sqlite3
from pathlib import Path
from threading import Lock
from typing import Any


class ChatHistoryStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = str(Path(db_path).resolve())
        self._lock = Lock()
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        source TEXT NOT NULL,
                        query TEXT NOT NULL,
                        answer TEXT NOT NULL,
                        source_documents TEXT NOT NULL
                    )
                    """
                )
                conn.commit()

    def add_entry(self, source: str, query: str, answer: str, source_documents: list[dict[str, Any]]) -> int:
        payload = json.dumps(source_documents, ensure_ascii=False)
        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO chat_history (source, query, answer, source_documents)
                    VALUES (?, ?, ?, ?)
                    """,
                    (source, query, answer, payload),
                )
                conn.commit()
                return int(cursor.lastrowid)

    def list_entries(self, limit: int = 50) -> list[dict[str, Any]]:
        safe_limit = max(1, min(limit, 500))
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT id, created_at, source, query, answer, source_documents
                    FROM chat_history
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (safe_limit,),
                ).fetchall()

        data: list[dict[str, Any]] = []
        for row in rows:
            data.append(
                {
                    "id": row["id"],
                    "created_at": row["created_at"],
                    "source": row["source"],
                    "query": row["query"],
                    "answer": row["answer"],
                    "source_documents": json.loads(row["source_documents"]),
                }
            )
        return data

    def get_entry(self, entry_id: int) -> dict[str, Any] | None:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT id, created_at, source, query, answer, source_documents
                    FROM chat_history
                    WHERE id = ?
                    """,
                    (entry_id,),
                ).fetchone()

        if row is None:
            return None

        return {
            "id": row["id"],
            "created_at": row["created_at"],
            "source": row["source"],
            "query": row["query"],
            "answer": row["answer"],
            "source_documents": json.loads(row["source_documents"]),
        }
