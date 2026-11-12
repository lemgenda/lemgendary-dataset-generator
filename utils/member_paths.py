"""Archive and remote file paths: wrapper folders, ``--only`` selections and safe local paths.

One place for the rules every partial download shares (byte-range
extraction from a remote zip, Kaggle per-file download):

- a *wrapper* is the one top folder every path shares (for example
  ``LemGendizedRetinaFaceMobileNetLarge/``); paths are taken relative to it,
  so ``labels/`` means the same thing whether or not the upload had one.
  A manifold layout folder (``shards``, ``labels``, ...) is never a wrapper;
- a selection prefix ending in ``/`` (or naming a folder) selects that folder,
  any other prefix selects the one file of that name;
- a relative path becomes a local path only when every part is a plain name
  (no ``..``, drive, backslash or empty part).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

__all__ = [
    "STRUCTURAL_DIRS",
    "UnsafeMemberPath",
    "common_wrapper",
    "local_path",
    "matches_prefixes",
    "relative_to_wrapper",
]

# Manifold layout folders: their children never belong one level up.
STRUCTURAL_DIRS = frozenset({"shards", "data", "images", "targets", "labels", "masks", "mds"})


class UnsafeMemberPath(ValueError):
    """A remote path that would land outside the destination folder."""


def common_wrapper(names: Iterable[str]) -> str:
    """The top folder every file path in ``names`` shares, else ``""``.

    Folder entries (ending in ``/``) are ignored; a structural folder is never
    a wrapper.
    """
    tops: set[str] = set()
    for name in names:
        if name.endswith("/"):
            continue
        head, sep, _rest = name.partition("/")
        if not sep:
            return ""
        tops.add(head)
        if len(tops) > 1:
            return ""
    if len(tops) != 1:
        return ""
    top = tops.pop()
    return "" if top.lower() in STRUCTURAL_DIRS else top


def relative_to_wrapper(name: str, wrapper: str) -> str:
    """``name`` without the ``wrapper/`` folder (unchanged when there is none)."""
    return name[len(wrapper) + 1:] if wrapper and name.startswith(f"{wrapper}/") else name


def matches_prefixes(rel: str, prefixes: tuple[str, ...]) -> bool:
    """True when ``rel`` is selected by one of ``prefixes`` (no prefixes select everything)."""
    if not prefixes:
        return True
    for prefix in prefixes:
        clean = prefix.replace("\\", "/").lstrip("/")
        folder = clean if clean.endswith("/") else f"{clean}/"
        if rel == clean or rel.startswith(folder):
            return True
    return False


def local_path(dest: Path, rel: str) -> Path:
    """``dest / rel`` for a plain relative path; raises :class:`UnsafeMemberPath` otherwise."""
    parts = rel.split("/")
    if any(part in ("", ".", "..") or ":" in part or "\\" in part for part in parts):
        raise UnsafeMemberPath(f"unsafe path in remote listing: {rel!r}")
    return dest.joinpath(*parts)
