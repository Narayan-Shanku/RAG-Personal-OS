import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from app.config import settings


@dataclass
class Retrieved:
    chunk_id: str
    score: float


class ModeVectorStore:
    def __init__(self, mode: str):
        self.mode = mode
        self.model = SentenceTransformer(settings.embedding_model_name)
        self.faiss_dir = settings.index_dir / "faiss" / mode
        self.faiss_path = self.faiss_dir / "index.faiss"
        self.ids_path = self.faiss_dir / "chunk_ids.json"

        self._index = None
        self._chunk_ids: List[str] = []

    def load(self) -> bool:
        if not self.faiss_path.exists() or not self.ids_path.exists():
            return False
        self._index = faiss.read_index(str(self.faiss_path))
        self._chunk_ids = json.loads(self.ids_path.read_text(encoding="utf-8"))
        return True

    def search(self, query: str, top_k: int) -> List[Retrieved]:
        if self._index is None:
            ok = self.load()
            if not ok:
                return []

        q = self.model.encode([query], normalize_embeddings=True)
        q = np.asarray(q, dtype="float32")
        scores, idxs = self._index.search(q, top_k)

        out: List[Retrieved] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            if idx >= len(self._chunk_ids):
                continue
            out.append(Retrieved(chunk_id=self._chunk_ids[idx], score=float(score)))
        return out