"""SQLite storage and full-text retrieval for repository chunks."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable


SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    repository_path TEXT NOT NULL,
    file_path TEXT NOT NULL,
    language TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    content TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    file_path,
    language,
    content,
    content='documents',
    content_rowid='id'
);
"""


class Index:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.executescript(SCHEMA)
        return connection

    def replace_documents(self, documents: Iterable[dict[str, object]]) -> int:
        rows = list(documents)
        with self.connect() as connection:
            connection.execute("DELETE FROM documents")
            connection.execute("DELETE FROM documents_fts")
            connection.executemany(
                """
                INSERT INTO documents
                (repository_path, file_path, language, start_line, end_line, content)
                VALUES (:repository_path, :file_path, :language, :start_line, :end_line, :content)
                """,
                rows,
            )
            connection.execute(
                """
                INSERT INTO documents_fts(rowid, file_path, language, content)
                SELECT id, file_path, language, content FROM documents
                """
            )
        return len(rows)

    def search(self, query: str, limit: int = 8) -> list[sqlite3.Row]:
        with self.connect() as connection:
            return connection.execute(
                """
                SELECT documents.*, bm25(documents_fts) AS score
                FROM documents_fts
                JOIN documents ON documents.id = documents_fts.rowid
                WHERE documents_fts MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (query, limit),
            ).fetchall()
