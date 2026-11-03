#!/usr/bin/env python3
"""LemGendary Dataset Cover Image Generator Script.

Validates and generates the canonical 564x284 architecture header banner (<ModelName>.jpg)
required for Kaggle dataset pages and visual previews across all manifolds.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from core.docs.context import ManifoldContext
from core.docs.generators.common import (
    ensure_context,
    manifold_parser,
    run_manifold_cli,
)

logger = logging.getLogger(__name__)

STANDARD_BANNER_WIDTH = 564
STANDARD_BANNER_HEIGHT = 284


_BANNER_BG = (15, 20, 28)
_BORDER_BLUE = (56, 139, 253)
_ACCENT_GREEN = (46, 160, 67)
_BADGE_FILL = (22, 27, 34)
_RULE_GREY = (48, 54, 61)
_LINK_BLUE = (88, 166, 255)
_MUTED_GREY = (139, 148, 158)


@dataclass
class _BannerFonts:
    """Fonts used on the banner (Arial when available, Pillow's default otherwise)."""

    large: Any
    small: Any
    badge: Any


def _load_fonts() -> _BannerFonts:
    """Load the banner fonts, falling back to Pillow's built-in bitmap font."""
    try:
        return _BannerFonts(
            large=ImageFont.truetype("arial.ttf", 26),
            small=ImageFont.truetype("arial.ttf", 15),
            badge=ImageFont.truetype("arialbd.ttf", 11),
        )
    except OSError:
        return _BannerFonts(
            large=ImageFont.load_default(),
            small=ImageFont.load_default(),
            badge=ImageFont.load_default(),
        )


def _draw_background(draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
    """Paint the subtle vertical gradient background."""
    for y in range(height):
        ratio = y / height
        r = int(15 + ratio * 20)
        g = int(22 + ratio * 24)
        b = int(35 + ratio * 35)
        draw.line([(0, y), (width, y)], fill=(r, g, b))


def _draw_frame(draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
    """Draw the double accent frame and the green corner brackets."""
    draw.rectangle([6, 6, width - 7, height - 7], outline=_BORDER_BLUE, width=2)
    draw.rectangle([10, 10, width - 11, height - 11], outline=(30, 41, 59), width=1)

    corner_len = 24
    left, top, right, bottom = 6, 6, width - 7, height - 7
    for x, y, dx, dy in (
        (left, top, 1, 1),
        (right, top, -1, 1),
        (left, bottom, 1, -1),
        (right, bottom, -1, -1),
    ):
        draw.line([(x, y), (x + dx * corner_len, y)], fill=_ACCENT_GREEN, width=3)
        draw.line([(x, y), (x, y + dy * corner_len)], fill=_ACCENT_GREEN, width=3)


def _draw_text(
    draw: ImageDraw.ImageDraw, width: int, title: str, subtitle_lines: tuple[str, str]
) -> None:
    """Draw the badges, title, category/archetype lines and the footer."""
    fonts = _load_fonts()

    # Brand badge top-left
    draw.rectangle([20, 20, 190, 42], fill=_BADGE_FILL, outline=_RULE_GREY, width=1)
    draw.text((28, 25), "LEMGENDARY ECOSYSTEM", fill=_LINK_BLUE, font=fonts.badge)

    # Status badge top-right
    draw.rectangle(
        [width - 150, 20, width - 20, 42], fill=_BADGE_FILL, outline=_ACCENT_GREEN, width=1
    )
    draw.text((width - 138, 25), "PRODUCTION V26", fill=_ACCENT_GREEN, font=fonts.badge)

    # Main Title
    display_title = title if len(title) <= 32 else title[:30] + "..."
    draw.text((24, 90), display_title, fill=(240, 246, 252), font=fonts.large)

    # Subtitle / Category
    draw.text((24, 135), subtitle_lines[0], fill=_MUTED_GREY, font=fonts.small)
    draw.text((24, 162), subtitle_lines[1], fill=(110, 118, 129), font=fonts.small)

    # Lower status line
    draw.line([(24, 210), (width - 24, 210)], fill=_RULE_GREY, width=1)
    draw.text(
        (24, 225),
        "Standardized Streaming Container Topology  WebDataset / Parquet / MDS",
        fill=_LINK_BLUE,
        font=fonts.badge,
    )
    draw.text(
        (24, 245),
        "Verified Lineage  Curated Benchmark Splits  Deterministic Pipeline",
        fill=_MUTED_GREY,
        font=fonts.badge,
    )


def _create_fallback_banner(
    target_path: Path,
    title: str,
    category: str,
    task: str,
) -> Path:
    """Create a high-quality 564x284 synthetic branding banner using Pillow."""
    width, height = STANDARD_BANNER_WIDTH, STANDARD_BANNER_HEIGHT
    img = Image.new("RGB", (width, height), color=_BANNER_BG)
    draw = ImageDraw.Draw(img)

    _draw_background(draw, width, height)
    _draw_frame(draw, width, height)
    _draw_text(
        draw,
        width,
        title,
        (f"Category: {category}", f"Archetype: {task.upper()} | High-Fidelity Manifold"),
    )

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
    ctx = ensure_context(context_or_path)

    target_dir = ctx.manifold_path
    image_name = ctx.inventory.cover_image_name or f"{ctx.manifold_name}.jpg"
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
        except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
            logger.warning("Corrupted image at %s (%s); regenerating...", image_path, exc)

    # Generate banner
    _create_fallback_banner(
        target_path=image_path,
        title=ctx.listing.title,
        category=ctx.profile.category,
        task=ctx.task,
    )
    return image_path


def _cli_process(manifold_dir: Path, args: argparse.Namespace) -> str:
    """Validate or generate the cover image of one manifold and describe the result."""
    out = generate_cover_image(manifold_dir, force_regenerate=args.force)
    with Image.open(out) as im:
        sz = im.size
    return f"{out.name} ({sz[0]}x{sz[1]})"


def main() -> None:
    """CLI entrypoint for cover image generation."""
    parser = manifold_parser(
        "Validate/Generate cover image (<ModelName>.jpg) for dataset manifolds",
        "Process cover images for all manifolds",
    )
    parser.add_argument("--force", action="store_true", help="Force regenerate existing images")
    run_manifold_cli(parser, _cli_process, "Processed cover images")


if __name__ == "__main__":
    main()
