import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.db import DB
from app.ingest.loaders import load_any
from app.ingest.chunker import chunk_loaded_text, Chunk


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def stable_doc_id(mode: str, path: Path) -> str:
    return sha256_text(f"{mode}::{path.as_posix()}")


def stable_chunk_id(doc_id: str, chunk_index: int, chunk_hash: str) -> str:
    return sha256_text(f"{doc_id}::{chunk_index}::{chunk_hash}")


@dataclass
class IndexBuildStats:
    mode: str
    scanned_files: int
    indexed_files: int
    deleted_files: int
    total_chunks: int


class POSIndexer:
    def __init__(self, db: DB):
        self.db = db
        self.model = SentenceTransformer(settings.embedding_model_name)

    def ensure_dirs(self) -> None:
        settings.sources_dir.mkdir(parents=True, exist_ok=True)
        settings.index_dir.mkdir(parents=True, exist_ok=True)
        settings.sqlite_dir.mkdir(parents=True, exist_ok=True)
        for m in settings.modes:
            (settings.sources_dir / m).mkdir(parents=True, exist_ok=True)
            (settings.index_dir / "faiss" / m).mkdir(parents=True, exist_ok=True)

    def list_source_files(self, mode: str) -> List[Path]:
        root = settings.sources_dir / mode
        allowed = {".txt", ".md", ".markdown", ".pdf", ".docx"}
        files = []
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in allowed:
                files.append(p)
        files.sort()
        return files

    def compute_file_hash(self, path: Path) -> str:
        return sha256_bytes(path.read_bytes())

    def index_mode(self, mode: str) -> IndexBuildStats:
        self.ensure_dirs()
        now = datetime.utcnow().isoformat()

        files = self.list_source_files(mode)
        scanned = len(files)

        known_docs = self.db.list_documents_by_mode(mode)
        known_by_path = {d["path"]: d for d in known_docs}

        current_paths = {p.as_posix() for p in files}
        deleted = 0
        for old_path, doc in list(known_by_path.items()):
            if old_path not in current_paths:
                self.db.delete_document_and_chunks(doc["doc_id"])
                deleted += 1

        indexed = 0
        total_chunks = 0

        for path in files:
            pstr = path.as_posix()
            doc_id = stable_doc_id(mode, path)
            file_hash = self.compute_file_hash(path)
            existing = self.db.get_document(doc_id)

            if existing and existing["file_hash"] == file_hash:
                continue

            pages = load_any(path)
            is_md = path.suffix.lower() in [".md", ".markdown"]
            all_chunks: List[Chunk] = []
            for lp in pages:
                all_chunks.extend(
                    chunk_loaded_text(
                        text=lp.text,
                        is_markdown=is_md,
                        page=lp.page,
                        chunk_size=settings.chunk_size_chars,
                        overlap=settings.chunk_overlap_chars,
                    )
                )

            db_rows: List[Tuple] = []
            for idx, ch in enumerate(all_chunks):
                ch_hash = sha256_text(ch.text)
                chunk_id = stable_chunk_id(doc_id, idx, ch_hash)
                db_rows.append(
                    (
                        chunk_id,
                        doc_id,
                        mode,
                        ch_hash,
                        ch.text,
                        ch.heading,
                        ch.page,
                        ch.start_char,
                        ch.end_char,
                    )
                )

            self.db.upsert_document(doc_id=doc_id, mode=mode, path=pstr, file_hash=file_hash, updated_at=now)
            self.db.replace_chunks_for_doc(doc_id, db_rows)

            indexed += 1
            total_chunks += len(db_rows)

        self._rebuild_faiss_index(mode)

        mode_chunks = self.db.list_chunks_by_mode(mode)
        total_chunks_mode = len(mode_chunks)

        return IndexBuildStats(
            mode=mode,
            scanned_files=scanned,
            indexed_files=indexed,
            deleted_files=deleted,
            total_chunks=total_chunks_mode,
        )

    def _rebuild_faiss_index(self, mode: str) -> None:
        chunks = self.db.list_chunks_by_mode(mode)
        faiss_dir = settings.index_dir / "faiss" / mode
        faiss_path = faiss_dir / "index.faiss"
        ids_path = faiss_dir / "chunk_ids.json"

        if not chunks:
            if faiss_path.exists():
                faiss_path.unlink()
            if ids_path.exists():
                ids_path.unlink()
            return

        texts = [c["text"] for c in chunks]
        chunk_ids = [c["chunk_id"] for c in chunks]

        emb = self.model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
        emb = np.asarray(emb, dtype="float32")

        dim = emb.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(emb)

        faiss.write_index(index, str(faiss_path))
        ids_path.write_text(json.dumps(chunk_ids, indent=2), encoding="utf-8")