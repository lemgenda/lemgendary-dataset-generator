"""
MigrationService — In-process container and image transcoding migration service.

Single Responsibility: Coordinate and execute container format conversion,
image format transcoding, and manifold name modernization.

Every method keeps its keyword-argument call style; the options are gathered
into small frozen dataclasses (one per concern) before dispatch.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from rich.console import Console

from core.lazy_imports import load_manifold_reduce
from formats.base import parse_also_format
from tools import (
    migrate_manifold_format,
    migrate_manifold_image_format,
    modernization_queue,
    modernize_manifold,
)

from ._errors import SERVICE_ERRORS

console = Console()
logger = logging.getLogger(__name__)


def _take(kwargs: dict[str, Any], cls: Any) -> Any:
    """Pop the ``kwargs`` named like fields of dataclass ``cls`` and build it from them."""
    names = {f.name for f in fields(cls)}
    return cls(**{k: kwargs.pop(k) for k in list(kwargs) if k in names})


def _options(kwargs: dict[str, Any], cls: Any) -> Any:
    """Build dataclass ``cls`` from ``kwargs``; an unknown keyword raises TypeError."""
    remaining = dict(kwargs)
    options = _take(remaining, cls)
    _reject_unknown(remaining)
    return options


def _reject_unknown(remaining: dict[str, Any]) -> None:
    """Raise TypeError for leftover keywords, like a plain signature would."""
    if remaining:
        raise TypeError(f"unexpected keyword argument(s): {', '.join(sorted(remaining))}")


def _guarded(label: str, action: Callable[[], int]) -> int:
    """Run ``action``; report an expected pipeline failure as exit code 1."""
    try:
        return action()
    except SERVICE_ERRORS as exc:
        logger.exception("%s error", label)
        console.print(f"[bold red]{label} error:[/bold red] {exc}")
        return 1


def _flag_args(options: Any, flags: dict[str, str]) -> list[str]:
    """Return the CLI switches (``flags``: field -> switch) whose option is set."""
    return [switch for name, switch in flags.items() if getattr(options, name)]


@dataclass(frozen=True)
class ContainerMigrationFlags:
    """Safety and verification switches of a container migration."""

    force_duplicate: bool = False
    accept_space_loss: bool = False
    purge_source: bool = False
    verify: bool = False
    dry_run: bool = False


@dataclass(frozen=True)
class ImageTranscodeOptions:
    """Image transcoding policy of an in-place manifold transcode."""

    image_format: str = "webp"
    image_quality: int = 92
    target_quality: int = 95
    mask_format: str = "webp-lossless"
    dry_run: bool = False

    def to_argv(self, manifold_path: Path) -> list[str]:
        """Return the ``tools.migrate_manifold_image_format`` arguments."""
        args = [
            "--manifold", str(manifold_path),
            "--image-format", self.image_format,
            "--image-quality", str(self.image_quality),
            "--target-quality", str(self.target_quality),
            "--mask-format", self.mask_format,
        ]
        return args + _flag_args(self, {"dry_run": "--dry-run"})


@dataclass(frozen=True)
class ModernizeSelection:
    """Which manifolds a modernization run picks and how it confirms/publishes."""

    all_: bool = False
    yes: bool = False
    dry_run: bool = False
    datasets: str | None = None
    skip_kaggle: bool = False

    def to_argv(self) -> list[str]:
        """Return the ``tools.modernize_manifold`` selection arguments."""
        args = _flag_args(self, {"all_": "--all", "yes": "--yes", "dry_run": "--dry-run"})
        if self.datasets:
            args.extend(["--datasets", self.datasets])
        return args + _flag_args(self, {"skip_kaggle": "--skip-kaggle"})


@dataclass(frozen=True)
class ModernizeConversion:
    """Image and container conversion settings of a modernization run."""

    image_format: str = "webp"
    image_quality: int = 92
    also_format: str | None = None
    skip_transcode: bool = False
    skip_container: bool = False

    def to_argv(self) -> list[str]:
        """Return the ``tools.modernize_manifold`` conversion arguments."""
        args = ["--image-format", self.image_format, "--image-quality", str(self.image_quality)]
        if self.also_format:
            args.extend(["--also-format", self.also_format])
        return args + _flag_args(
            self, {"skip_transcode": "--skip-transcode", "skip_container": "--skip-container"}
        )


@dataclass(frozen=True)
class ReduceOptions:
    """Options of a reduced manifold variant."""

    max_gb: float | None = None
    suffix: str | None = None
    dataset: str | None = None
    fraction: float | None = None
    yes: bool = False
    dry_run: bool = False

    def to_kwargs(self) -> dict[str, Any]:
        """Return the ``manifold_reduce.reduce_dataset`` keyword arguments."""
        return {
            "max_gb_override": self.max_gb,
            "suffix_override": self.suffix,
            "dataset": self.dataset,
            "fraction": self.fraction,
            "assume_yes": self.yes,
            "dry_run": self.dry_run,
        }


class MigrationService:
    """Encapsulates manifold transformation and format migration workflows."""

    @staticmethod
    def migrate_containers(manifold_path: Path, to_formats: str, **flags: bool) -> int:
        """Migrate directory layout to additional container formats (MDS, LitData, etc.).

        ``flags`` are the fields of :class:`ContainerMigrationFlags`.
        """
        options = _options(flags, ContainerMigrationFlags)

        def _migrate() -> int:
            return migrate_manifold_format.migrate_manifold(
                root=manifold_path, formats=parse_also_format(to_formats), **asdict(options)
            )

        return _guarded("Container migration", _migrate)

    @staticmethod
    def transcode_images(manifold_path: Path, **options: Any) -> int:
        """Transcode images and masks in an existing manifold in place.

        ``options`` are the fields of :class:`ImageTranscodeOptions`.
        """
        opts = _options(options, ImageTranscodeOptions)

        def _transcode() -> int:
            parser = migrate_manifold_image_format.build_parser()
            return migrate_manifold_image_format.run(parser.parse_args(opts.to_argv(manifold_path)))

        return _guarded("Image transcoding", _transcode)

    @staticmethod
    def modernize_manifolds(**options: Any) -> int:
        """Retire the legacy Large suffix, transcode images and convert to container formats.

        ``options`` are the fields of :class:`ModernizeSelection` and
        :class:`ModernizeConversion`.
        """
        remaining = dict(options)
        selection = _take(remaining, ModernizeSelection)
        conversion = _take(remaining, ModernizeConversion)
        _reject_unknown(remaining)

        def _modernize() -> int:
            parser = modernize_manifold.build_parser()
            parsed = parser.parse_args(selection.to_argv() + conversion.to_argv())
            return modernize_manifold.run(parsed)

        return _guarded("Manifold modernization", _modernize)

    @staticmethod
    def reduce_manifolds(**options: Any) -> int:
        """Create a reduced manifold variant; prompts unless ``yes``, which requires ``dataset``.

        ``options`` are the fields of :class:`ReduceOptions`.
        """
        opts = _options(options, ReduceOptions)
        manifold_reduce = load_manifold_reduce()
        return _guarded(
            "Manifold reduction", lambda: manifold_reduce.reduce_dataset(**opts.to_kwargs())
        )

    @staticmethod
    def run_modernization_queue(dataset: str | None = None, **options: Any) -> int:
        """Execute the modernization queue in-process with the same arguments the API job uses.

        ``options`` are the :func:`tools.modernization_queue.build_queue_args`
        options workers, push_kaggle, dry_run, force and purge_source.
        """

        def _run() -> int:
            args = modernization_queue.build_queue_args(
                datasets=dataset.split(",") if dataset else None, **options
            )
            return modernization_queue.main(args)

        return _guarded("Modernization queue", _run)
