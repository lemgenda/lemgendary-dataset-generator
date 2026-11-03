"""
MigrationService — In-process container and image transcoding migration service.

Single Responsibility: Coordinate and execute container format conversion,
image format transcoding, and manifold name modernization.
"""

from __future__ import annotations

import logging
from pathlib import Path
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


class MigrationService:
    """Encapsulates manifold transformation and format migration workflows."""

    @staticmethod
    def migrate_containers(
        manifold_path: Path,
        to_formats: str,
        force_duplicate: bool = False,
        accept_space_loss: bool = False,
        purge_source: bool = False,
        verify: bool = False,
        dry_run: bool = False,
    ) -> int:
        """Migrate directory layout to additional container formats (MDS, LitData, etc.)."""

        try:
            formats = parse_also_format(to_formats)
            ret = migrate_manifold_format.migrate_manifold(
                root=manifold_path,
                formats=formats,
                force_duplicate=force_duplicate,
                accept_space_loss=accept_space_loss,
                purge_source=purge_source,
                verify=verify,
                dry_run=dry_run,
            )
            return ret
        except SERVICE_ERRORS as exc:
            logger.exception("Container migration error")
            console.print(f"[bold red]Container migration error:[/bold red] {exc}")
            return 1

    @staticmethod
    def transcode_images(
        manifold_path: Path,
        image_format: str = "webp",
        image_quality: int = 92,
        target_quality: int = 95,
        mask_format: str = "webp-lossless",
        dry_run: bool = False,
    ) -> int:
        """Transcode images and masks in an existing manifold in place."""

        try:
            args = [
                "--manifold", str(manifold_path),
                "--image-format", image_format,
                "--image-quality", str(image_quality),
                "--target-quality", str(target_quality),
                "--mask-format", mask_format,
            ]
            if dry_run:
                args.append("--dry-run")
            parser = migrate_manifold_image_format.build_parser()
            parsed = parser.parse_args(args)
            return migrate_manifold_image_format.run(parsed)
        except SERVICE_ERRORS as exc:
            logger.exception("Image transcoding error")
            console.print(f"[bold red]Image transcoding error:[/bold red] {exc}")
            return 1

    @staticmethod
    def modernize_manifolds(
        all_: bool = False,
        yes: bool = False,
        dry_run: bool = False,
        datasets: str | None = None,
        skip_kaggle: bool = False,
        image_format: str = "webp",
        image_quality: int = 92,
        also_format: str | None = None,
        skip_transcode: bool = False,
        skip_container: bool = False,
    ) -> int:
        """Retire the legacy Large suffix across manifolds, transcode images, and convert to container formats."""

        try:
            args = []
            if all_:
                args.append("--all")
            if yes:
                args.append("--yes")
            if dry_run:
                args.append("--dry-run")
            if datasets:
                args.extend(["--datasets", datasets])
            if skip_kaggle:
                args.append("--skip-kaggle")
            args.extend(["--image-format", image_format])
            args.extend(["--image-quality", str(image_quality)])
            if also_format:
                args.extend(["--also-format", also_format])
            if skip_transcode:
                args.append("--skip-transcode")
            if skip_container:
                args.append("--skip-container")
            parser = modernize_manifold.build_parser()
            parsed = parser.parse_args(args)
            return modernize_manifold.run(parsed)
        except SERVICE_ERRORS as exc:
            logger.exception("Manifold modernization error")
            console.print(f"[bold red]Manifold modernization error:[/bold red] {exc}")
            return 1

    @staticmethod
    def reduce_manifolds(
        max_gb: float | None = None,
        suffix: str | None = None,
        dataset: str | None = None,
        fraction: float | None = None,
        yes: bool = False,
        dry_run: bool = False,
    ) -> int:
        """Create a reduced manifold variant; prompts unless ``yes`` (then ``dataset`` is required)."""
        manifold_reduce = load_manifold_reduce()

        try:
            return manifold_reduce.reduce_dataset(
                max_gb_override=max_gb,
                suffix_override=suffix,
                dataset=dataset,
                fraction=fraction,
                assume_yes=yes,
                dry_run=dry_run,
            )
        except SERVICE_ERRORS as exc:
            logger.exception("Manifold reduction error")
            console.print(f"[bold red]Manifold reduction error:[/bold red] {exc}")
            return 1

    @staticmethod
    def run_modernization_queue(
        dataset: str | None = None,
        workers: int | None = None,
        push_kaggle: bool = False,
        dry_run: bool = False,
        force: bool = False,
        purge_source: bool = False,
    ) -> int:
        """Execute the modernization queue in-process with the same arguments the API job uses."""

        try:
            args = modernization_queue.build_queue_args(
                datasets=dataset.split(",") if dataset else None,
                workers=workers,
                purge_source=purge_source,
                push_kaggle=push_kaggle,
                dry_run=dry_run,
                force=force,
            )
            return modernization_queue.main(args)
        except SERVICE_ERRORS as exc:
            logger.exception("Modernization queue error")
            console.print(f"[bold red]Modernization queue error:[/bold red] {exc}")
            return 1
