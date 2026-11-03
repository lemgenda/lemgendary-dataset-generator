"""Multi-format Knowledge Ingestor for LLM Datasets.

Ingests offline files (.md, .txt, .json, .jsonl, .yaml, .yml, .csv) and remote web URLs
with deterministic hashing, line preservation, and offline caching.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("lemdata.llm.ingestor")


@dataclass(eq=False)
class KnowledgeDocument:
    """Represents a normalized ingested document unit."""

    doc_id: str
    content: str
    source_type: str
    source_path: str
    canonical_url: str | None = None
    metadata: dict[str, Any] | None = None
    content_hash: str = field(init=False, default="")

    def __post_init__(self) -> None:
        self.metadata = self.metadata or {}
        self.content_hash = hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Return the document as a JSON-serialisable dict."""
        return {
            "doc_id": self.doc_id,
            "source_type": self.source_type,
            "source_path": self.source_path,
            "canonical_url": self.canonical_url,
            "content_hash": self.content_hash,
            "content_length": len(self.content),
            "metadata": self.metadata,
        }


def _read_text(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8", errors="replace")


def _read_json(file_path: Path) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return json.dumps(data, indent=2)


def _read_jsonl(file_path: Path) -> str:
    lines = [
        line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    return "\n".join(lines)


def _read_yaml(file_path: Path) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return yaml.safe_dump(data, sort_keys=False)


def _read_csv(file_path: Path) -> str:
    with open(file_path, "r", encoding="utf-8", newline="") as f:
        rows = [", ".join(row) for row in csv.reader(f)]
    return "\n".join(rows)


# Extension -> (reader returning normalised text, source_type label).
_FILE_READERS: dict[str, tuple[Callable[[Path], str], str]] = {
    ".md": (_read_text, "markdown"),
    ".txt": (_read_text, "text"),
    ".json": (_read_json, "json"),
    ".jsonl": (_read_jsonl, "json"),
    ".yaml": (_read_yaml, "yaml"),
    ".yml": (_read_yaml, "yaml"),
    ".csv": (_read_csv, "csv"),
}


def _strip_html(html_text: str) -> str:
    """Basic tag stripping and whitespace collapse for fetched web pages."""
    clean_text = re.sub(r"<[^>]+>", " ", html_text)
    return re.sub(r"\s+", " ", clean_text).strip()


def _fetch_url_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "LemGendary-Knowledge-Ingestor/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return _strip_html(resp.read().decode("utf-8", errors="replace"))


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
        reader = _FILE_READERS.get(ext)
        if reader is None:
            logger.debug("Unsupported extension: %s", ext)
            return None
        read, source_type = reader
        try:
            content = read(file_path)
        except (OSError, UnicodeDecodeError, ValueError, yaml.YAMLError, csv.Error) as exc:
            logger.error("Failed reading file %s: %s", file_path, exc)
            return None
        return KnowledgeDocument(
            doc_id=file_path.name,
            content=content,
            source_type=source_type,
            source_path=str(file_path),
            canonical_url=f"{url_prefix.rstrip('/')}/{file_path.name}" if url_prefix else None,
        )

    def ingest_url(
        self,
        url: str,
        tag: str | None = None,
        ttl_hours: int = 168,
    ) -> KnowledgeDocument | None:
        """Fetch remote URL with caching and clean text normalization.

        A cached copy younger than ``ttl_hours`` is reused; an older one is
        refreshed from the network, falling back to the stale copy only when
        the refresh fails.
        """
        url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        content = self._cached_url_text(url, self.cache_dir / f"{url_hash}.txt", ttl_hours)
        if content is None:
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

    @staticmethod
    def _cached_url_text(url: str, cache_file: Path, ttl_hours: int) -> str | None:
        """Return fresh cache, else refetch (and cache), else stale cache, else None."""
        cache_exists = cache_file.exists()
        if cache_exists and (time.time() - cache_file.stat().st_mtime) < ttl_hours * 3600:
            return cache_file.read_text(encoding="utf-8")
        try:
            content = _fetch_url_text(url)
            cache_file.write_text(content, encoding="utf-8")
        except (OSError, ValueError) as exc:
            if not cache_exists:
                logger.warning("Remote URL fetch failed for %s (%s). Falling back.", url, exc)
                return None
            logger.warning(
                "Remote URL refresh failed for %s (%s). Using stale cache %s.",
                url,
                exc,
                cache_file,
            )
            return cache_file.read_text(encoding="utf-8")
        return content
