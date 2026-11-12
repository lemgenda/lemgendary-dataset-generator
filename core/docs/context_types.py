"""Data containers describing a resolved LemGendary dataset manifold.

``ManifoldContext`` groups the resolved metadata into small cohesive records
(identity, task profile, Kaggle listing, lineage, physical inventory and the raw
specification inputs) so each generator only reaches for the part it needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ManifoldIdentity:
    """Where the manifold lives and how it is named and classified."""

    path: Path
    name: str
    slug: str
    task: str


@dataclass
class TaskProfile:
    """Task-level descriptors: category, architecture, objective and model profile."""

    category: str
    arch_base: str
    objective: str
    description: str
    model_profile: dict[str, Any] | None = None


@dataclass
class AuthorInfo:
    """Dataset author shown on the Kaggle listing."""

    name: str
    bio: str


@dataclass
class ListingMeta:
    """Kaggle listing fields: title, subtitle, license, keywords and visibility."""

    title: str
    subtitle: str
    license_name: str
    keywords: list[str]
    is_private: bool
    update_frequency: str
    author: AuthorInfo


@dataclass(frozen=True)
class SplitCounts:
    """Train/val sample counts of the whole manifold; ``None`` when not recorded."""

    train: int | None
    val: int | None


@dataclass
class Lineage:
    """Sample tally and provenance metadata of the manifold.

    ``sources`` maps each source to its ``train``/``val``/``total`` counts; a
    count is ``None`` when the manifold does not record it, so nothing is
    estimated. ``splits`` holds the real split counts of the whole manifold.
    """

    total_samples: int
    sources: dict[str, dict[str, int | None]]
    splits: SplitCounts
    provenance_sources: list[str]
    collection_methodology: str
    citations: list[str]
    coverage: dict[str, Any]


@dataclass
class Inventory:
    """Physical root-level files, folders and cover image of the manifold."""

    cover_image_name: str | None
    cover_image_path: Path | None
    root_files: list[Path] = field(default_factory=list)
    root_dirs: list[Path] = field(default_factory=list)


@dataclass
class SpecInputs:
    """Raw specification inputs: the unified_data.yaml entry and caller overrides."""

    unified_entry: dict[str, Any]
    overrides: dict[str, Any] = field(default_factory=dict)

    def pick(self, key: str, default: Any = None) -> Any:
        """Return the override for ``key``, else the unified entry value (or ``default``)."""
        return self.overrides.get(key) or self.unified_entry.get(key, default)

    def pick_truthy(self, key: str, default: Any = None) -> Any:
        """Return the first truthy value of override, unified entry, then ``default``."""
        return self.overrides.get(key) or self.unified_entry.get(key) or default


@dataclass
class ManifoldContext:
    """Encapsulates all resolved metadata and on-disk assets for a dataset manifold."""

    identity: ManifoldIdentity
    profile: TaskProfile
    listing: ListingMeta
    lineage: Lineage
    inventory: Inventory
    spec: SpecInputs

    @property
    def manifold_path(self) -> Path:
        """Absolute path of the manifold root directory."""
        return self.identity.path

    @property
    def manifold_name(self) -> str:
        """Canonical (PascalCase) manifold name."""
        return self.identity.name

    @property
    def slug(self) -> str:
        """Lower-case Kaggle slug of the manifold."""
        return self.identity.slug

    @property
    def task(self) -> str:
        """Authoritative task classification."""
        return self.identity.task

    @property
    def is_llm(self) -> bool:
        """True when the manifold is an instruction (JSONL) dataset."""
        return self.task in ("ai_helper", "llm") or self.spec.unified_entry.get(
            "dataset_type"
        ) in ("llm", "ai_helper")
