"""Token counting and sequence length alignment bounds for LLM datasets.

Provides the Single Source of Truth (SSOT) for token estimation (``tiktoken``
with deterministic ``len(text) / 3.5`` fallback) and QLoRA sequence ceilings
so ``max_seq_length: 2048`` never truncates assistant completions during
ChatML prompt masking.
"""

from __future__ import annotations

import functools
import importlib
import importlib.util
import logging
import math
from typing import Any

logger = logging.getLogger("lemdata.llm.token_budget")

MAX_TOTAL_TOKENS: int = 1536
SYNTHESIS_MAX_TOTAL_TOKENS: int = 1450
ASSISTANT_TARGET_MIN_TOKENS: int = 150
ASSISTANT_TARGET_MAX_TOKENS: int = 350
ASSISTANT_HARD_MAX_TOKENS: int = 420
FALLBACK_CHARS_PER_TOKEN: float = 3.5


@functools.lru_cache(maxsize=1)
def _load_tiktoken_encoding() -> Any | None:
    """Dynamically load ``tiktoken`` cl100k_base encoding when installed."""
    if importlib.util.find_spec("tiktoken") is None:
        return None
    try:
        tiktoken_mod = importlib.import_module("tiktoken")
        return tiktoken_mod.get_encoding("cl100k_base")
    except (ImportError, KeyError, ValueError, OSError, AttributeError) as exc:
        logger.debug("tiktoken unavailable, using character fallback: %s", exc)
        return None


def count_tokens(text: str) -> int:
    """Count tokens using ``tiktoken`` if available, else ``ceil(len(text) / 3.5)``."""
    if not text:
        return 0
    encoding = _load_tiktoken_encoding()
    if encoding is not None:
        try:
            return len(encoding.encode(text, disallowed_special=()))
        except (ValueError, TypeError) as exc:
            logger.debug("tiktoken encode failed, falling back to char ratio: %s", exc)
    return math.ceil(len(text) / FALLBACK_CHARS_PER_TOKEN)


def count_messages_tokens(messages: list[Any]) -> tuple[int, int]:
    """Return ``(total_tokens, assistant_tokens)`` across ChatML message turns."""
    total_tokens = 0
    assistant_tokens = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content", "")
        if not isinstance(content, str) or not content:
            continue
        turn_tokens = count_tokens(content)
        total_tokens += turn_tokens
        if message.get("role") == "assistant":
            assistant_tokens += turn_tokens
    return total_tokens, assistant_tokens


def count_record_tokens(record: dict[str, Any]) -> tuple[int, int]:
    """Return ``(total_tokens, assistant_tokens)`` for a ChatML record dictionary."""
    messages = record.get("messages")
    if not isinstance(messages, list):
        return 0, 0
    return count_messages_tokens(messages)


def truncate_to_token_limit(text: str, max_tokens: int) -> str:
    """Trim ``text`` cleanly on word or sentence boundaries to stay within ``max_tokens``."""
    stripped = text.strip()
    if not stripped or max_tokens <= 0:
        return ""
    if count_tokens(stripped) <= max_tokens:
        return stripped

    words = stripped.split()
    if not words:
        return ""

    low = 0
    high = len(words)
    best = ""
    while low <= high:
        mid = (low + high) // 2
        candidate = " ".join(words[:mid]).strip()
        if count_tokens(candidate) <= max_tokens:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1

    if not best:
        char_cap = max(1, int(max_tokens * FALLBACK_CHARS_PER_TOKEN) - 3)
        best = stripped[:char_cap].strip()
        while best and count_tokens(best) > max_tokens:
            best = best[:-1].strip()
        return best

    last_period = max(best.rfind(". "), best.rfind("! "), best.rfind("? "))
    if last_period >= int(len(best) * 0.6):
        return best[: last_period + 1].strip()
    return best.rstrip(",;:")
