"""Semantic and Recursive Text Chunker for Knowledge Corpus.

Breaks large technical documents into semantic sections respecting headers,
code blocks, and lists, preserving line numbers and character offsets.
"""

from __future__ import annotations

import hashlib
from typing import Any
from .knowledge_ingestor import KnowledgeDocument


class KnowledgeChunk:
    """Represents a bounded semantic slice of an ingested document."""

    def __init__(
        self,
        chunk_id: str,
        doc_id: str,
        text: str,
        char_start: int,
        char_end: int,
        canonical_url: str | None = None,
        source_path: str = "",
        section_title: str | None = None,
    ) -> None:
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.text = text
        self.char_start = char_start
        self.char_end = char_end
        self.canonical_url = canonical_url
        self.source_path = source_path
        self.section_title = section_title
        self.sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()

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

    def chunk_document(self, doc: KnowledgeDocument) -> list[KnowledgeChunk]:
        """Chunk a single KnowledgeDocument using hierarchical separators."""
        text = doc.content
        if not text:
            return []

        # If document is smaller than chunk size, return it as single chunk
        if len(text) <= self.chunk_size:
            chunk_id = f"{doc.doc_id}_c0"
            return [
                KnowledgeChunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    text=text,
                    char_start=0,
                    char_end=len(text),
                    canonical_url=doc.canonical_url,
                    source_path=doc.source_path,
                )
            ]

        chunks: list[KnowledgeChunk] = []
        pos = 0
        chunk_idx = 0
        text_len = len(text)

        while pos < text_len:
            end_pos = min(pos + self.chunk_size, text_len)
            if end_pos < text_len:
                # Look for clean boundary: \n## , \n\n, \n, .
                boundary = -1
                for sep in ["\n## ", "\n### ", "\n\n", "\n", ". "]:
                    b = text.rfind(sep, pos, end_pos)
                    if b > pos + (self.chunk_size // 2):
                        boundary = b + len(sep)
                        break
                if boundary != -1:
                    end_pos = boundary

            chunk_text = text[pos:end_pos].strip()
            if chunk_text:
                chunk_id = f"{doc.doc_id}_c{chunk_idx}"
                chunks.append(
                    KnowledgeChunk(
                        chunk_id=chunk_id,
                        doc_id=doc.doc_id,
                        text=chunk_text,
                        char_start=pos,
                        char_end=end_pos,
                        canonical_url=doc.canonical_url,
                        source_path=doc.source_path,
                    )
                )
                chunk_idx += 1

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
