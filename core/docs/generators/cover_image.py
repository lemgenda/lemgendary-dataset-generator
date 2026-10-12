#!/usr/bin/env python3
"""LemGendary Dataset Cover Image Generator Script.

Validates and generates the canonical 564x284 architecture header banner (<ModelName>.jpg)
required for Kaggle dataset pages and visual previews across all manifolds.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

_MODULE_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE_ROOT))

from PIL import Image, ImageDraw, ImageFont

from core.docs.context import ManifoldContext, resolve_manifold_context

logger = logging.getLogger(__name__)

STANDARD_BANNER_WIDTH = 564
STANDARD_BANNER_HEIGHT = 284


def _create_fallback_banner(
    target_path: Path,
    title: str,
    category: str,
    task: str,
) -> Path:
    """Create a high-quality 564x284 synthetic branding banner using Pillow."""
    width, height = STANDARD_BANNER_WIDTH, STANDARD_BANNER_HEIGHT
    img = Image.new("RGB", (width, height), color=(15, 20, 28))
    draw = ImageDraw.Draw(img)

    # 1. Subtle horizontal gradient
    for y in range(height):
        ratio = y / height
        r = int(15 + ratio * 20)
        g = int(22 + ratio * 24)
        b = int(35 + ratio * 35)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # 2. Tech accent frame
    border_color = (56, 139, 253)
    accent_green = (46, 160, 67)
    draw.rectangle([6, 6, width - 7, height - 7], outline=border_color, width=2)
    draw.rectangle([10, 10, width - 11, height - 11], outline=(30, 41, 59), width=1)

    # Corner decorations
    corner_len = 24
    draw.line([(6, 6), (6 + corner_len, 6)], fill=accent_green, width=3)
    draw.line([(6, 6), (6, 6 + corner_len)], fill=accent_green, width=3)
    draw.line([(width - 7, 6), (width - 7 - corner_len, 6)], fill=accent_green, width=3)
    draw.line([(width - 7, 6), (width - 7, 6 + corner_len)], fill=accent_green, width=3)
    draw.line([(6, height - 7), (6 + corner_len, height - 7)], fill=accent_green, width=3)
    draw.line([(6, height - 7), (6, height - 7 - corner_len)], fill=accent_green, width=3)
    draw.line([(width - 7, height - 7), (width - 7 - corner_len, height - 7)], fill=accent_green, width=3)
    draw.line([(width - 7, height - 7), (width - 7, height - 7 - corner_len)], fill=accent_green, width=3)

    # 3. Badges and Text
    font_large = None
    font_small = None
    font_badge = None
    try:
        font_large = ImageFont.truetype("arial.ttf", 26)
        font_small = ImageFont.truetype("arial.ttf", 15)
        font_badge = ImageFont.truetype("arialbd.ttf", 11)
    except OSError:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
        font_badge = ImageFont.load_default()

    # Brand badge top-left
    draw.rectangle([20, 20, 190, 42], fill=(22, 27, 34), outline=(48, 54, 61), width=1)
    draw.text((28, 25), "LEMGENDARY ECOSYSTEM", fill=(88, 166, 255), font=font_badge)

    # Status badge top-right
    draw.rectangle([width - 150, 20, width - 20, 42], fill=(22, 27, 34), outline=accent_green, width=1)
    draw.text((width - 138, 25), "PRODUCTION V26", fill=accent_green, font=font_badge)

    # Main Title
    display_title = title if len(title) <= 32 else title[:30] + "..."
    draw.text((24, 90), display_title, fill=(240, 246, 252), font=font_large)

    # Subtitle / Category
    draw.text((24, 135), f"Category: {category}", fill=(139, 148, 158), font=font_small)
    draw.text((24, 162), f"Archetype: {task.upper()} | High-Fidelity Manifold", fill=(110, 118, 129), font=font_small)

    # Lower status line
    draw.line([(24, 210), (width - 24, 210)], fill=(48, 54, 61), width=1)
    draw.text((24, 225), "Standardized Streaming Container Topology  WebDataset / Parquet / MDS", fill=(88, 166, 255), font=font_badge)
    draw.text((24, 245), "Verified Lineage  Curated Benchmark Splits  Deterministic Pipeline", fill=(139, 148, 158), font=font_badge)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(target_path, "JPEG", quality=95)
    return target_path


def generate_cover_image(
    context_or_path: ManifoldContext | str | Path,
    force_regenerate: bool = False,
) -> Path:
    """Validate or generate canonical 564x284 cover image (<ModelName>.jpg) for a manifold.

    Args:
        context_or_path: Resolved ManifoldContext or directory path.
        force_regenerate: If True, overwrite existing image with newly generated banner.

    Returns:
        Path to the valid cover image file.
    """
    if isinstance(context_or_path, ManifoldContext):
        ctx = context_or_path
    else:
        ctx = resolve_manifold_context(Path(context_or_path))

    target_dir = ctx.manifold_path
    image_name = ctx.cover_image_name or f"{ctx.manifold_name}.jpg"
    image_path = target_dir / image_name

    # Check for any existing jpg/png if specific name doesn't exist
    if not image_path.exists():
        for cand in target_dir.iterdir():
            if cand.is_file() and cand.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                image_path = cand
                image_name = cand.name
                break

    if image_path.exists() and not force_regenerate:
        try:
            with Image.open(image_path) as im:
                im.verify()
            logger.info("Validated existing cover image: %s", image_path)
            return image_path
        except Exception as exc:
            logger.warning("Corrupted image at %s (%s); regenerating...", image_path, exc)

    # Generate banner
    _create_fallback_banner(
        target_path=image_path,
        title=ctx.title,
        category=ctx.category,
        task=ctx.task,
    )
    return image_path


def main() -> None:
    """CLI entrypoint for cover image generation."""
    parser = argparse.ArgumentParser(description="Validate/Generate cover image (<ModelName>.jpg) for dataset manifolds")
    parser.add_argument("--manifold", type=str, default=None, help="Manifold directory name or path")
    parser.add_argument("--all", action="store_true", help="Process cover images for all manifolds")
    parser.add_argument("--force", action="store_true", help="Force regenerate existing images")
    args = parser.parse_args()

    cand = _MODULE_ROOT.parent / "LemGendaryDatasets"
    datasets_root = cand if cand.exists() else _MODULE_ROOT / "LemGendaryDatasets"

    if args.all:
        if not datasets_root.exists():
            print(f"[ERROR] Datasets directory not found at {datasets_root}")
            sys.exit(1)
        count = 0
        for m_dir in sorted(datasets_root.iterdir()):
            if m_dir.is_dir() and not m_dir.name.startswith("."):
                out = generate_cover_image(m_dir, force_regenerate=args.force)
                with Image.open(out) as im:
                    sz = im.size
                print(f"[OK] {m_dir.name} -> {out.name} ({sz[0]}x{sz[1]})")
                count += 1
        print(f"Processed cover images for {count} manifolds.")
    elif args.manifold:
        p = Path(args.manifold)
        if not p.is_dir() and (datasets_root / args.manifold).is_dir():
            p = datasets_root / args.manifold
        out = generate_cover_image(p, force_regenerate=args.force)
        with Image.open(out) as im:
            sz = im.size
        print(f"[OK] {p.name} -> {out.name} ({sz[0]}x{sz[1]})")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
