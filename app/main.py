from typing import Optional, List
from fastapi import FastAPI
from pydantic import BaseModel

from app.config import settings
from app.db import DB
from app.ingest.indexer import POSIndexer
from app.retrieval.rag import query_pos


app = FastAPI(title="Personal Operating System RAG", version="1.0.0")

db = DB(settings.db_path)
db.init()
indexer = POSIndexer(db=db)
indexer.ensure_dirs()


class QueryRequest(BaseModel):
    mode: str
    question: str
    strict: bool = True
    retrieve_k: Optional[int] = None
    candidate_k: Optional[int] = None
    debug: bool = False


class ReindexRequest(BaseModel):
    modes: Optional[List[str]] = None


@app.get("/status")
def status():
    docs = db.list_all_documents()
    out = {m: 0 for m in settings.modes}
    for d in docs:
        out[d["mode"]] += 1
    return {
        "ok": True,
        "modes": list(settings.modes),
        "documents_per_mode": out,
        "sources_dir": str(settings.sources_dir),
        "db_path": str(settings.db_path),
    }


@app.post("/reindex")
def reindex(req: ReindexRequest):
    modes = req.modes or list(settings.modes)
    modes = [m for m in modes if m in settings.modes]
    stats = []
    for m in modes:
        st = indexer.index_mode(m)
        stats.append(st.__dict__)
    return {"ok": True, "stats": stats}


@app.post("/query")
async def query(req: QueryRequest):
    result = await query_pos(
        db=db,
        mode=req.mode,
        question=req.question,
        strict=req.strict,
        retrieve_k=req.retrieve_k,
        candidate_k=req.candidate_k,
        debug=req.debug,
    )
    return result