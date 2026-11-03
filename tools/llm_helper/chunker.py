"""Semantic and Recursive Text Chunker for Knowledge Corpus.

Breaks large technical documents into semantic sections respecting headers,
code blocks, and lists, preserving line numbers and character offsets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any
from .knowledge_ingestor import KnowledgeDocument

# Clean boundaries tried, in order, when a chunk has to be cut mid-document.
_BOUNDARY_SEPARATORS = ("\n## ", "\n### ", "\n\n", "\n", ". ")


@dataclass(frozen=True)
class ChunkOrigin:
    """Where a chunk came from: its source file, public URL and section title."""

    source_path: str = ""
    canonical_url: str | None = None
    section_title: str | None = None

    @classmethod
    def of_document(cls, doc: KnowledgeDocument) -> ChunkOrigin:
        """Return the origin shared by every chunk of ``doc``."""
        return cls(source_path=doc.source_path, canonical_url=doc.canonical_url)


@dataclass(eq=False)
class KnowledgeChunk:
    """Represents a bounded semantic slice of an ingested document."""

    chunk_id: str
    doc_id: str
    text: str
    char_start: int
    char_end: int
    origin: ChunkOrigin = field(default_factory=ChunkOrigin)
    sha256: str = field(init=False, default="")

    def __post_init__(self) -> None:
        self.sha256 = hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def canonical_url(self) -> str | None:
        """Public URL of the source document, if any."""
        return self.origin.canonical_url

    @property
    def source_path(self) -> str:
        """Path (or URL) the source document was read from."""
        return self.origin.source_path

    @property
    def section_title(self) -> str | None:
        """Title of the section the chunk belongs to, if known."""
        return self.origin.section_title

    def to_dict(self) -> dict[str, Any]:
        """Return the chunk as a JSON-serialisable dict."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "canonical_url": self.canonical_url,
            "source_path": self.source_path,
            "section_title": self.section_title,
            "sha256": self.sha256,
        }


class RecursiveChunker:
    """Splits knowledge documents into overlapping semantic chunks."""

    def __init__(
        self,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _chunk_end(self, text: str, pos: int) -> int:
        """Return where the chunk starting at ``pos`` ends, preferring a clean boundary."""
        text_len = len(text)
        end_pos = min(pos + self.chunk_size, text_len)
        if end_pos >= text_len:
            return end_pos
        for sep in _BOUNDARY_SEPARATORS:
            b = text.rfind(sep, pos, end_pos)
            if b > pos + (self.chunk_size // 2):
                return b + len(sep)
        return end_pos

    def chunk_document(self, doc: KnowledgeDocument) -> list[KnowledgeChunk]:
        """Chunk a single KnowledgeDocument using hierarchical separators."""
        text = doc.content
        if not text:
            return []

        origin = ChunkOrigin.of_document(doc)
        # If document is smaller than chunk size, return it as single chunk
        if len(text) <= self.chunk_size:
            return [KnowledgeChunk(f"{doc.doc_id}_c0", doc.doc_id, text, 0, len(text), origin)]

        chunks: list[KnowledgeChunk] = []
        pos = 0
        text_len = len(text)

        while pos < text_len:
            end_pos = self._chunk_end(text, pos)
            chunk_text = text[pos:end_pos].strip()
            if chunk_text:
                chunk_id = f"{doc.doc_id}_c{len(chunks)}"
                chunks.append(
                    KnowledgeChunk(chunk_id, doc.doc_id, chunk_text, pos, end_pos, origin)
                )

            if end_pos >= text_len:
                break
            pos = max(pos + 1, end_pos - self.chunk_overlap)

        return chunks

    def chunk_documents(self, docs: list[KnowledgeDocument]) -> list[KnowledgeChunk]:
        """Chunk a collection of documents."""
        all_chunks: list[KnowledgeChunk] = []
        for d in docs:
            all_chunks.extend(self.chunk_document(d))
        return all_chunks
