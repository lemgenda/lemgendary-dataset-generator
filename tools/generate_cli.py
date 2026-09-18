"""
Standalone generation runner.

Phase 5 of the 2026 modernization roadmap.

Runs label / prompt / mask generation over an existing manifold and writes
the artifacts alongside the directory layout.

Usage:
    python generate_cli.py --manifold <path> --kind label  --strategy blip_caption
    python generate_cli.py --manifold <path> --kind prompt --template diffusers-v1
    python generate_cli.py --manifold <path> --kind mask   --strategy parsenet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator

_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).parent.name == "tools" else Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PIL import Image

from generators import LabelGenerator, MaskGenerator, PromptGenerator


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


# ─── Model loading (standalone context) ─────────────────────────────────────
def _load_generator(
    kind: str,
    strategy: str,
    template: str,
    device: str,
) -> Any:
    """Instantiate a generator with the models it needs.

    Loads only the models required for the requested strategy; no others
    are touched. Raises ImportError with a clear message if a required
    model package is missing.
    """
    if kind == "label":
        captioner = clip = labeler = sentry = None
        if strategy == "blip_caption":
            from models.diffusion import CaptionSentry
            captioner = CaptionSentry(device=device)
        elif strategy == "clip_zeroshot":
            from models.encoder import CLIPManifold
            clip = CLIPManifold(device=device)
        elif strategy in ("yolo_detection", "parsenet_segmentation"):
            from models.detection import AutoLabeler
            mode = "segmentation" if strategy == "parsenet_segmentation" else "detection"
            labeler = AutoLabeler(mode=mode, device=device)
        elif strategy == "nima_quality":
            from models.quality_scorer import QualitySentry
            base_dir = _ROOT
            model_path = base_dir / "models" / "nima_aesthetic_best.pth"
            sentry = QualitySentry(str(model_path), model_name="aesthetic", device=device)
        return LabelGenerator(
            strategy,
            captioner=captioner,
            clip=clip,
            labeler=labeler,
            sentry=sentry,
        )

    if kind == "prompt":
        from models.diffusion import CaptionSentry
        from models.encoder import CLIPManifold
        return PromptGenerator(
            template=template,
            captioner=CaptionSentry(device=device),
            clip=CLIPManifold(device=device),
        )

    if kind == "mask":
        labeler = None
        if strategy == "parsenet":
            from models.detection import AutoLabeler
            labeler = AutoLabeler(mode="segmentation", device=device)
        return MaskGenerator(strategy, labeler=labeler)

    raise ValueError(f"Unknown kind: {kind}")


# ─── Manifold iteration ─────────────────────────────────────────────────────
def _iter_images(manifold: Path, sample_limit: int | None) -> Iterator[tuple[Path, str, str]]:
    """Yield (path, name, split) for every image under images/. Name is
    the stem (matching index.json's 'name' field)."""
    count = 0
    for split in ("train", "val", "test"):
        d = manifold / "images" / split
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
                yield p, p.stem, split
                count += 1
                if sample_limit is not None and count >= sample_limit:
                    return


# ─── Output writers ─────────────────────────────────────────────────────────
def _write_label(manifold: Path, name: str, split: str, value: dict[str, Any]) -> None:
    out = manifold / "labels" / split / f"{name}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)

    if "nima_probs" in value:
        probs = value["nima_probs"]
        with open(out, "w", encoding="utf-8") as f:
            f.write(" ".join(f"{p:.6f}" for p in probs) + "\n")
        return

    if "class_label" in value:
        with open(out, "w", encoding="utf-8") as f:
            f.write(str(int(value["class_label"])) + "\n")
        return

    if "caption" in value:
        with open(out, "w", encoding="utf-8") as f:
            f.write(value["caption"] + "\n")
        return

    annotations = value.get("annotations")
    if annotations:
        with open(out, "w", encoding="utf-8") as f:
            for ann in annotations:
                cls = ann.get("cls", 0)
                data = ann.get("data", [])
                f.write(f"{cls} {' '.join(str(x) for x in data)}\n")


def _write_prompt(manifold: Path, name: str, split: str, prompt: str) -> None:
    out = manifold / "prompts" / split / f"{name}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(prompt + "\n", encoding="utf-8")


def _write_mask(manifold: Path, name: str, split: str, mask: Image.Image) -> None:
    out = manifold / "masks" / split / f"{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    mask.save(out, format="PNG", optimize=True)


# ─── Runner ─────────────────────────────────────────────────────────────────
def run(
    manifold: Path,
    kind: str,
    strategy: str,
    template: str,
    device: str,
    sample_limit: int | None,
    dry_run: bool,
) -> int:
    if not manifold.exists():
        print(f"[ERROR] Manifold not found: {manifold}")
        return 1

    print(f"[GENERATE] {manifold.name} | kind={kind} | strategy={strategy}")

    generator = _load_generator(kind, strategy, template, device)
    images = list(_iter_images(manifold, sample_limit))
    print(f"[SCAN] {len(images)} image(s) discovered" + (f" (capped at {sample_limit})" if sample_limit else ""))

    if dry_run:
        print("[DRY-RUN] Not writing any files.")
        return 0

    written = 0
    failed = 0
    for path, name, split in images:
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                result = generator.generate(img, context={"task": kind, "slug": name})
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {name}: {e}")
            continue

        if result.value is None:
            continue

        if kind == "label":
            _write_label(manifold, name, split, result.value)
        elif kind == "prompt":
            _write_prompt(manifold, name, split, str(result.value))
        elif kind == "mask":
            _write_mask(manifold, name, split, result.value)
        written += 1

        if written % 500 == 0:
            print(f"  [{written}/{len(images)}] written...")

    print()
    print(f"[DONE] written: {written}  failed: {failed}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Standalone label / prompt / mask generator"
    )
    parser.add_argument("--manifold", type=str, required=True)
    parser.add_argument("--kind", type=str, required=True,
                        choices=["label", "prompt", "mask"])
    parser.add_argument("--strategy", type=str, default=None,
                        help="blip_caption | clip_zeroshot | yolo_detection | "
                             "parsenet_segmentation | nima_quality | parsenet | sam | modnet")
    parser.add_argument("--template", type=str, default="diffusers-v1",
                        help="prompt template name (for --kind prompt)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--sample", type=int, default=None,
                        help="cap the image scan for a quick preview")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    strategy = args.strategy
    if strategy is None:
        strategy = {
            "label": "blip_caption",
            "prompt": "diffusers-v1",
            "mask": "parsenet",
        }[args.kind]

    return run(
        Path(args.manifold),
        args.kind,
        strategy,
        args.template,
        args.device,
        args.sample,
        args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())