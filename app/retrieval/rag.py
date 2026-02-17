from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.config import settings
from app.db import DB
from app.retrieval.vector_store import ModeVectorStore


@dataclass
class Citation:
    chunk_id: str
    source_path: str
    heading: Optional[str]
    page: Optional[int]
    score: float
    snippet: str


def _make_snippet(text: str, limit: int = 350) -> str:
    t = " ".join(text.split())
    if len(t) <= limit:
        return t
    return t[:limit].rstrip() + "…"


def _compose_grounded_answer(question: str, citations: List[Citation]) -> str:
    if not citations:
        return "I don’t have enough information in your sources to answer that."

    parts = []
    parts.append(f"Question: {question}")
    parts.append("")
    parts.append("Grounded answer based on your sources:")
    parts.append("")

    used = 0
    for c in citations:
        used += 1
        head = f" | {c.heading}" if c.heading else ""
        pg = f" | page {c.page}" if c.page else ""
        parts.append(f"{used}) {c.source_path}{head}{pg}")
        parts.append(c.snippet)
        parts.append("")

        if used >= settings.retrieve_k:
            break

    parts.append("Summary:")
    parts.append("The passages above are the most relevant evidence I found. If you want a cleaner final answer, add an LLM on top of this retrieval step (the citations stay the same).")
    return "\n".join(parts)


async def query_pos(
    db: DB,
    mode: str,
    question: str,
    strict: bool = True,
    retrieve_k: Optional[int] = None,
    candidate_k: Optional[int] = None,
    debug: bool = False,
) -> Dict[str, Any]:
    if mode not in settings.modes:
        return {"ok": False, "error": f"Unknown mode: {mode}"}

    rk = retrieve_k or settings.retrieve_k
    ck = candidate_k or settings.candidate_k

    store = ModeVectorStore(mode=mode)
    retrieved = store.search(question, top_k=ck)

    if not retrieved:
        return {
            "ok": True,
            "mode": mode,
            "refused": True,
            "answer": "I don’t have enough information in your sources to answer that.",
            "citations": [],
            "debug": {"reason": "no_index_or_no_results"} if debug else None,
        }

    top = retrieved[:rk]
    top_scores = [r.score for r in top]
    top_score = max(top_scores) if top_scores else 0.0
    mean_score = sum(top_scores) / len(top_scores) if top_scores else 0.0

    should_refuse = (top_score < settings.min_top_score) or (mean_score < settings.min_mean_score)

    chunk_ids = [r.chunk_id for r in top]
    chunk_rows = db.list_chunks_by_ids(chunk_ids)
    by_id = {c["chunk_id"]: c for c in chunk_rows}

    citations: List[Citation] = []
    for r in top:
        row = by_id.get(r.chunk_id)
        if not row:
            continue
        citations.append(
            Citation(
                chunk_id=r.chunk_id,
                source_path=row["source_path"],
                heading=row.get("heading"),
                page=row.get("page"),
                score=r.score,
                snippet=_make_snippet(row["text"]),
            )
        )

    if should_refuse:
        answer = "I don’t have enough high-confidence evidence in your sources to answer that. Try rephrasing, selecting a different mode, or reindexing your documents."
    else:
        print("LLM CALLED ✅")
        from app.llm.ollama_client import OllamaClient
        prompt = build_llm_prompt(question, citations)
        client = OllamaClient()
        answer = await client.generate(prompt)
    dbg = None
    if debug:
        dbg = {
            "top_score": top_score,
            "mean_score": mean_score,
            "thresholds": {"min_top_score": settings.min_top_score, "min_mean_score": settings.min_mean_score},
            "retrieved": [{"chunk_id": r.chunk_id, "score": r.score} for r in retrieved[:min(len(retrieved), 20)]],
        }

    return {
        "ok": True,
        "mode": mode,
        "refused": should_refuse,
        "answer": answer,
        "citations": [c.__dict__ for c in citations],
        "debug": dbg,
    }
def build_llm_prompt(question: str, citations: List[Citation]) -> str:
    context_blocks = []
    for i, c in enumerate(citations, start=1):
        head = f" | heading: {c.heading}" if c.heading else ""
        pg = f" | page: {c.page}" if c.page else ""
        context_blocks.append(
            f"[SOURCE {i}] file: {c.source_path}{head}{pg}\n{c.snippet}\n"
        )

    context = "\n".join(context_blocks)

    return f"""
You are a personal assistant. Answer the user's question using ONLY the sources provided.
If the sources do not contain enough information, say: "I don't have enough information in your sources to answer that."

Rules:
- Do not invent facts.
- Keep it concise and clear.
- After your answer, include a section titled "Citations" listing which SOURCE numbers you used.

User question:
{question}

Sources:
{context}
""".strip()