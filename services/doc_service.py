"""
DocService — In-process documentation and specification generator service.

Single Responsibility: Coordinate and execute generation of manifold READMEs,
dataset_info.yaml manifests, category maps, and the master manifolds.md table.
"""

from __future__ import annotations

from pathlib import Path
from rich.console import Console

console = Console()


class DocService:
    """Encapsulates documentation generation and synchronization workflows."""

    @staticmethod
    def regenerate_all_docs(datasets_dir: Path | str | None = None) -> int:
        """Regenerate per-manifold README and dataset_info files."""
        import doc_generator

        try:
            doc_generator.regenerate_all_docs(datasets_dir=datasets_dir)
            return 0
        except Exception as exc:
            console.print(f"[bold red]Documentation generation error:[/bold red] {exc}")
            return 1

    @staticmethod
    def regenerate_single_doc(manifold_path: Path, manifold_name: str) -> int:
        """Regenerate documentation for a single manifold."""
        import doc_generator

        try:
            doc_generator.generate_dataset_docs(manifold_path, None, manifold_name)
            return 0
        except Exception as exc:
            console.print(f"[bold red]Documentation generation error:[/bold red] {exc}")
            return 1

    @staticmethod
    def rebuild_manifolds_md(check_only: bool = False) -> int:
        """Rebuild top-level manifolds.md from registry databases."""
        import regenerate_manifolds_md

        try:
            parser = regenerate_manifolds_md.argparse.ArgumentParser()
            parser.add_argument("--check", action="store_true")
            args = parser.parse_args(["--check"] if check_only else [])
            # Run main logic
            ret = regenerate_manifolds_md.main()
            return ret if isinstance(ret, int) else 0
        except SystemExit as exc:
            return int(exc.code) if isinstance(exc.code, int) else 0
        except Exception as exc:
            console.print(f"[bold red]manifolds.md rebuild error:[/bold red] {exc}")
            return 1
