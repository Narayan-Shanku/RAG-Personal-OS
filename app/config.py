from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    project_root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = project_root / "data"
    sources_dir: Path = data_dir / "sources"
    index_dir: Path = data_dir / "index"
    sqlite_dir: Path = data_dir / "sqlite"

    db_path: Path = sqlite_dir / "pos_rag.sqlite3"

    modes: tuple = ("study", "build", "career", "life", "health")

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size_chars: int = 1400
    chunk_overlap_chars: int = 250

    retrieve_k: int = 8
    candidate_k: int = 40

    min_top_score: float = 0.15
    min_mean_score: float = 0.12


settings = Settings()