"""Multi-format Knowledge Ingestor for LLM Datasets.

Ingests offline files (.md, .txt, .json, .jsonl, .yaml, .yml, .csv) and remote web URLs
with deterministic hashing, line preservation, and offline caching.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from pathlib import Path
import re
from typing import Any
import urllib.parse
import urllib.request
import yaml

logger = logging.getLogger("lemdata.llm.ingestor")


class KnowledgeDocument:
    """Represents a normalized ingested document unit."""

    def __init__(
        self,
        doc_id: str,
        content: str,
        source_type: str,
        source_path: str,
        canonical_url: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.doc_id = doc_id
        self.content = content
        self.source_type = source_type
        self.source_path = source_path
        self.canonical_url = canonical_url
        self.metadata = metadata or {}
        self.content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "source_type": self.source_type,
            "source_path": self.source_path,
            "canonical_url": self.canonical_url,
            "content_hash": self.content_hash,
            "content_length": len(self.content),
            "metadata": self.metadata,
        }


class KnowledgeIngestor:
    """Universal multi-format knowledge loader."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or (Path.home() / ".cache" / "lemgendary_datasets" / "knowledge")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def ingest_path(
        self,
        target_path: str | Path,
        glob_pattern: str | None = None,
        url_prefix: str | None = None,
    ) -> list[KnowledgeDocument]:
        """Ingest local file or directory recursively."""
        p = Path(target_path)
        if not p.exists():
            logger.warning("Knowledge target path does not exist: %s", target_path)
            return []

        if p.is_file():
            doc = self._ingest_single_file(p, url_prefix=url_prefix)
            return [doc] if doc is not None else []

        docs: list[KnowledgeDocument] = []
        pattern = glob_pattern or "*.*"
        for item in sorted(p.glob(pattern)):
            if item.is_file():
                d = self._ingest_single_file(item, url_prefix=url_prefix)
                if d is not None:
                    docs.append(d)
        return docs

    def _ingest_single_file(
        self,
        file_path: Path,
        url_prefix: str | None = None,
    ) -> KnowledgeDocument | None:
        ext = file_path.suffix.lower()
        doc_id = file_path.name
        canonical_url = f"{url_prefix.rstrip('/')}/{file_path.name}" if url_prefix else None

        try:
            if ext in (".md", ".txt"):
                content = file_path.read_text(encoding="utf-8", errors="replace")
                return KnowledgeDocument(
                    doc_id=doc_id,
                    content=content,
                    source_type="markdown" if ext == ".md" else "text",
                    source_path=str(file_path),
                    canonical_url=canonical_url,
                )
            elif ext in (".json", ".jsonl"):
                if ext == ".json":
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    content = json.dumps(data, indent=2)
                else:
                    lines = [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                    content = "\n".join(lines)
                return KnowledgeDocument(
                    doc_id=doc_id,
                    content=content,
                    source_type="json",
                    source_path=str(file_path),
                    canonical_url=canonical_url,
                )
            elif ext in (".yaml", ".yml"):
                with open(file_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                content = yaml.safe_dump(data, sort_keys=False)
                return KnowledgeDocument(
                    doc_id=doc_id,
                    content=content,
                    source_type="yaml",
                    source_path=str(file_path),
                    canonical_url=canonical_url,
                )
            elif ext == ".csv":
                with open(file_path, "r", encoding="utf-8", newline="") as f:
                    reader = csv.reader(f)
                    rows = [", ".join(row) for row in reader]
                content = "\n".join(rows)
                return KnowledgeDocument(
                    doc_id=doc_id,
                    content=content,
                    source_type="csv",
                    source_path=str(file_path),
                    canonical_url=canonical_url,
                )
            else:
                logger.debug("Unsupported extension: %s", ext)
                return None
        except Exception as exc:
            logger.error("Failed reading file %s: %s", file_path, exc)
            return None

    def ingest_url(
        self,
        url: str,
        tag: str | None = None,
        ttl_hours: int = 168,
    ) -> KnowledgeDocument | None:
        """Fetch remote URL with caching and clean text normalization."""
        url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        cache_file = self.cache_dir / f"{url_hash}.txt"

        content = ""
        if cache_file.exists():
            content = cache_file.read_text(encoding="utf-8")
        else:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "LemGendary-Knowledge-Ingestor/1.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    raw_bytes = resp.read()
                    html_text = raw_bytes.decode("utf-8", errors="replace")
                    # Basic tag stripping
                    clean_text = re.sub(r"<[^>]+>", " ", html_text)
                    clean_text = re.sub(r"\s+", " ", clean_text).strip()
                    content = clean_text
                    cache_file.write_text(content, encoding="utf-8")
            except Exception as exc:
                logger.warning("Remote URL fetch failed for %s (%s). Falling back.", url, exc)
                return None

        parsed = urllib.parse.urlparse(url)
        doc_id = Path(parsed.path).name or f"url_{url_hash}"
        return KnowledgeDocument(
            doc_id=doc_id,
            content=content,
            source_type="remote_url",
            source_path=url,
            canonical_url=url,
            metadata={"tag": tag or "web_resource"},
        )

