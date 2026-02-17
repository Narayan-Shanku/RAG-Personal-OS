from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from pypdf import PdfReader
from docx import Document


@dataclass
class LoadedPage:
    text: str
    page: Optional[int] = None
    heading: Optional[str] = None


def load_txt_or_md(path: Path) -> List[LoadedPage]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [LoadedPage(text=text, page=None, heading=None)]


def load_docx(path: Path) -> List[LoadedPage]:
    doc = Document(str(path))
    parts = []
    for p in doc.paragraphs:
        if p.text and p.text.strip():
            parts.append(p.text.strip())
    return [LoadedPage(text="\n".join(parts), page=None, heading=None)]


def load_pdf(path: Path) -> List[LoadedPage]:
    reader = PdfReader(str(path))
    pages: List[LoadedPage] = []
    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        txt = txt.strip()
        if txt:
            pages.append(LoadedPage(text=txt, page=i + 1, heading=None))
    return pages


def load_any(path: Path) -> List[LoadedPage]:
    suffix = path.suffix.lower()
    if suffix in [".txt", ".md", ".markdown"]:
        return load_txt_or_md(path)
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix == ".docx":
        return load_docx(path)
    return load_txt_or_md(path)
