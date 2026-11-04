"""
Builders that turn a local ``dataset-metadata.json`` into Kaggle DatasetSettings.

Pure translation, no network: title/subtitle clipping to Kaggle limits, the
single canonical license, provenance sources, keyword de-duplication, the
header image choice, and per-file column descriptors mapped onto the paths
Kaggle actually stores.
"""

import os
from pathlib import Path
from typing import Any

from kagglesdk.datasets.types.dataset_types import (
    DatasetSettings,
    DatasetSettingsFile,
    DatasetSettingsFileColumn,
    SettingsLicense,
)

_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")

# Ordered (substring, canonical Kaggle license name) pairs; first match wins.
_LICENSE_ALIASES = (
    ("cc-by-nc", "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"),
    ("cc-by-sa", "Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)"),
    ("cc-by", "Attribution 4.0 International (CC BY 4.0)"),
    ("cc0", "CC0-1.0"),
    ("public", "CC0-1.0"),
)


def clip_title(raw_title: str) -> str:
    """Fit a title into Kaggle's 6..50 character window (word-boundary trim, space pad)."""
    if len(raw_title) > 50:
        raw_title = raw_title[:50].rsplit(" ", 1)[0]
    if len(raw_title) < 6:
        raw_title = raw_title.ljust(6, " ")
    return raw_title


def _clip_subtitle(raw_sub: str) -> str:
    """Trim a subtitle over 80 characters at a word boundary."""
    if len(raw_sub) > 80:
        raw_sub = raw_sub[:77].rsplit(" ", 1)[0]
    return raw_sub


def canonical_license_name(meta: dict[str, Any]) -> str:
    """Kaggle requires exactly one license: map the first declared one to its canonical name."""
    lic_name = "CC0-1.0"
    raw_lic = meta.get("licenses", [])
    if (
        raw_lic
        and isinstance(raw_lic, list)
        and isinstance(raw_lic[0], dict)
        and "name" in raw_lic[0]
    ):
        lic_name = raw_lic[0]["name"]
    lic_lower = lic_name.lower().replace("_", "-")
    for needle, canonical in _LICENSE_ALIASES:
        if needle in lic_lower:
            return canonical
    return lic_name


def _as_str_list(value: Any) -> list[str]:
    """Normalize a list-or-string metadata field to a list of strings."""
    if isinstance(value, list):
        return [str(p) for p in value]
    if isinstance(value, str):
        return [value]
    return []


def user_specified_sources(meta: dict[str, Any]) -> Any:
    """Explicit user sources, else provenance sources and citations joined by commas."""
    sources_val = meta.get("userSpecifiedSources") or meta.get("user_specified_sources")
    if sources_val:
        return sources_val
    sources_list: list[str] = []
    if meta.get("provenanceSources"):
        sources_list.extend(_as_str_list(meta["provenanceSources"]))
    if meta.get("citations"):
        sources_list.extend(_as_str_list(meta["citations"]))
    return ", ".join(sources_list) if sources_list else None


def dedupe_keywords(raw_keywords: list[Any]) -> list[str]:
    """Turn hyphenated keywords into spaced tags, dropping duplicates in order."""
    kws: list[str] = []
    for k in raw_keywords:
        spaced = str(k).replace("-", " ").strip()
        if spaced not in kws:
            kws.append(spaced)
    return kws


def build_dataset_settings(meta: dict[str, Any], slug: str) -> DatasetSettings:
    """Dataset-level settings: title, subtitle, description, license, frequency, sources."""
    settings = DatasetSettings()
    settings.title = clip_title(meta.get("title", slug))
    settings.subtitle = _clip_subtitle(meta.get("subtitle", ""))
    settings.description = meta.get("description", "")

    lic = SettingsLicense()
    lic.name = canonical_license_name(meta)
    settings.licenses = [lic]

    settings.expected_update_frequency = meta.get(
        "expectedUpdateFrequency", meta.get("expected_update_frequency", "never")
    )
    settings.is_private = meta.get("isPrivate", meta.get("is_private", False))

    sources_val = user_specified_sources(meta)
    if sources_val:
        settings.user_specified_sources = str(sources_val)
    return settings


def header_image_name(meta: dict[str, Any], meta_folder: Path) -> str | None:
    """Header image named in metadata, else the first image file in the manifold folder."""
    name = meta.get("headerImage") or meta.get("header_image")
    if name:
        return name
    for f in meta_folder.iterdir():
        if f.is_file() and f.suffix.lower() in _IMAGE_SUFFIXES:
            return f.name
    return None


def _resolve_remote_name(raw_name: str, remote_files: set[str], prefix: str) -> str:
    """Map a local file path onto the path Kaggle stores it under."""
    if remote_files:
        if raw_name in remote_files:
            return raw_name
        if f"{prefix}/{raw_name}" in remote_files:
            return f"{prefix}/{raw_name}"
        for rf in remote_files:
            if rf.endswith(f"/{raw_name}"):
                return rf
        return raw_name
    if not raw_name.startswith(f"{prefix}/"):
        return f"{prefix}/{raw_name}"
    return raw_name


def _build_columns(fields: list[dict[str, Any]]) -> list[DatasetSettingsFileColumn | None]:
    """Column descriptors (name, description, type defaulting to string)."""
    cols: list[DatasetSettingsFileColumn | None] = []
    for c in fields:
        col = DatasetSettingsFileColumn()
        col.name = c.get("name", "")
        col.description = c.get("description", "")
        col.type = c.get("type", "string")
        cols.append(col)
    return cols


def _settings_file(
    name: str, description: str, fields: list[dict[str, Any]]
) -> DatasetSettingsFile:
    """One file entry with its column descriptors."""
    df = DatasetSettingsFile()
    df.name = name
    df.description = description
    df.columns = _build_columns(fields)
    return df


def _files_from_data(
    entries: list[dict[str, Any]], remote_files: set[str], prefix: str
) -> list[DatasetSettingsFile | None]:
    """File entries from the Kaggle-style ``data`` array."""
    return [
        _settings_file(
            _resolve_remote_name(entry.get("name", ""), remote_files, prefix),
            entry.get("description", ""),
            entry.get("columns", []),
        )
        for entry in entries
    ]


def _files_from_resources(
    resources: list[dict[str, Any]], meta_folder: Path, remote_files: set[str]
) -> list[DatasetSettingsFile | None]:
    """File entries from the Frictionless-style ``resources`` array (files only)."""
    prefix = meta_folder.name
    files_data: list[DatasetSettingsFile | None] = []
    for r in resources:
        res_path = r.get("path", "")
        # Skip directories and extensionless paths; Kaggle file settings take files only
        if (
            not res_path
            or (meta_folder / res_path).is_dir()
            or "." not in os.path.basename(res_path)
        ):
            continue
        target_name = _resolve_remote_name(res_path, remote_files, prefix)
        # Only register if file actually exists remotely (when remote_files available)
        if remote_files and target_name not in remote_files:
            continue
        files_data.append(
            _settings_file(
                target_name, r.get("description", ""), r.get("schema", {}).get("fields", [])
            )
        )
    return files_data


def build_file_settings(
    meta: dict[str, Any], meta_folder: Path, remote_files: set[str]
) -> list[DatasetSettingsFile | None]:
    """Per-file settings from ``data`` (preferred) or ``resources``."""
    if "data" in meta and isinstance(meta["data"], list):
        return _files_from_data(meta["data"], remote_files, meta_folder.name)
    if "resources" in meta and isinstance(meta["resources"], list):
        return _files_from_resources(meta["resources"], meta_folder, remote_files)
    return []
