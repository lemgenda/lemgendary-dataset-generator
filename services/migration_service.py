"""
MigrationService — In-process container and image transcoding migration service.

Single Responsibility: Coordinate and execute container format conversion,
image format transcoding, and manifold name modernization.
"""

from __future__ import annotations

from pathlib import Path
from rich.console import Console

console = Console()


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
        from tools import migrate_manifold_format
        from formats.base import parse_also_format

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
        except Exception as exc:
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
        from tools import migrate_manifold_image_format

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
        except Exception as exc:
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
        from tools import modernize_manifold

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
        except Exception as exc:
            console.print(f"[bold red]Manifold modernization error:[/bold red] {exc}")
            return 1

    @staticmethod
    def reduce_manifolds(
        max_gb: float | None = None,
        suffix: str | None = None,
    ) -> int:
        """Launch interactive manifold downsampling and scale reduction."""
        from tools import manifold_reduce

        try:
            manifold_reduce.reduce_dataset(max_gb_override=max_gb, suffix_override=suffix)
            return 0
        except Exception as exc:
            console.print(f"[bold red]Manifold reduction error:[/bold red] {exc}")
            return 1

    @staticmethod
    def run_modernization_queue(
        dataset: str | None = None,
        workers: int | None = None,
        push_kaggle: bool = False,
        dry_run: bool = False,
        force: bool = False,
    ) -> int:
        """Execute the batch modernization queue conversion in-process."""
        from tools import modernization_queue

        try:
            datasets_root = (Path(__file__).resolve().parent.parent.parent / "LemGendaryDatasets").resolve()
            registry_yaml = Path(__file__).resolve().parent.parent / "unified_data.yaml"
            if not datasets_root.exists():
                console.print(f"[red]LemGendaryDatasets directory not found at {datasets_root}[/red]")
                return 1
            items = modernization_queue.load_queue_manifest(datasets_root, registry_yaml)
            filter_keys = [k.strip() for k in dataset.split(",") if k.strip()] if dataset else None
            return modernization_queue.run_queue(
                items=items,
                priority_keys=["mirnet_exposure", "upn_v2"],
                filter_keys=filter_keys,
                push_kaggle=push_kaggle,
                workers=workers,
                dry_run=dry_run,
                force=force,
            )
        except Exception as exc:
            console.print(f"[bold red]Modernization queue error:[/bold red] {exc}")
            return 1
