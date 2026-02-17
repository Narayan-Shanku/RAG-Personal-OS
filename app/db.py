import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Any, Dict, List, Tuple


SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    mode TEXT NOT NULL,
    path TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    mode TEXT NOT NULL,
    chunk_hash TEXT NOT NULL,
    text TEXT NOT NULL,
    heading TEXT,
    page INTEGER,
    start_char INTEGER,
    end_char INTEGER,
    FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
);

CREATE INDEX IF NOT EXISTS idx_chunks_mode ON chunks(mode);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
"""


class DB:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def init(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            conn.commit()

    def upsert_document(self, doc_id: str, mode: str, path: str, file_hash: str, updated_at: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO documents(doc_id, mode, path, file_hash, updated_at)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                  mode=excluded.mode,
                  path=excluded.path,
                  file_hash=excluded.file_hash,
                  updated_at=excluded.updated_at
                """,
                (doc_id, mode, path, file_hash, updated_at),
            )
            conn.commit()

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
            return dict(row) if row else None

    def delete_document_and_chunks(self, doc_id: str) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            conn.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
            conn.commit()

    def list_documents_by_mode(self, mode: str) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM documents WHERE mode=? ORDER BY path", (mode,)).fetchall()
            return [dict(r) for r in rows]

    def list_all_documents(self) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM documents ORDER BY mode, path").fetchall()
            return [dict(r) for r in rows]

    def replace_chunks_for_doc(self, doc_id: str, chunks: List[Tuple]) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            conn.executemany(
                """
                INSERT INTO chunks(chunk_id, doc_id, mode, chunk_hash, text, heading, page, start_char, end_char)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                chunks,
            )
            conn.commit()

    def list_chunks_by_mode(self, mode: str) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT c.*, d.path AS source_path
                FROM chunks c
                JOIN documents d ON d.doc_id = c.doc_id
                WHERE c.mode=?
                ORDER BY d.path, c.page, c.start_char
                """,
                (mode,),
            ).fetchall()
            return [dict(r) for r in rows]

    def list_chunks_by_ids(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        if not chunk_ids:
            return []
        placeholders = ",".join(["?"] * len(chunk_ids))
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT c.*, d.path AS source_path
                FROM chunks c
                JOIN documents d ON d.doc_id = c.doc_id
                WHERE c.chunk_id IN ({placeholders})
                """,
                tuple(chunk_ids),
            ).fetchall()
            by_id = {r["chunk_id"]: dict(r) for r in rows}
            return [by_id[cid] for cid in chunk_ids if cid in by_id]