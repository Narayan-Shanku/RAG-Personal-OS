import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Chunk:
    text: str
    heading: Optional[str]
    page: Optional[int]
    start_char: int
    end_char: int


_heading_re = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def split_markdown_by_headings(text: str) -> List[Tuple[Optional[str], str]]:
    matches = list(_heading_re.finditer(text))
    if not matches:
        return [(None, text)]

    blocks: List[Tuple[Optional[str], str]] = []
    for idx, m in enumerate(matches):
        heading = m.group(2).strip()
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            blocks.append((heading, body))
    return blocks if blocks else [(None, text)]


def chunk_text(text: str, heading: Optional[str], page: Optional[int], chunk_size: int, overlap: int) -> List[Chunk]:
    cleaned = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not cleaned:
        return []

    chunks: List[Chunk] = []
    start = 0
    n = len(cleaned)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = cleaned[start:end].strip()

        if end < n:
            back = chunk.rfind("\n\n")
            if back > int(chunk_size * 0.6):
                end = start + back
                chunk = cleaned[start:end].strip()

        if chunk:
            chunks.append(
                Chunk(
                    text=chunk,
                    heading=heading,
                    page=page,
                    start_char=start,
                    end_char=end,
                )
            )

        if end >= n:
            break
        start = max(0, end - overlap)

    return chunks


def chunk_loaded_text(text: str, is_markdown: bool, page: Optional[int], chunk_size: int, overlap: int) -> List[Chunk]:
    if is_markdown:
        blocks = split_markdown_by_headings(text)
        out: List[Chunk] = []
        for heading, body in blocks:
            out.extend(chunk_text(body, heading=heading, page=page, chunk_size=chunk_size, overlap=overlap))
        return out
    return chunk_text(text, heading=None, page=page, chunk_size=chunk_size, overlap=overlap)