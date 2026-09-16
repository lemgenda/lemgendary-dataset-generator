"""
Prompt generation strategies.

Single responsibility: produce structured prompts for diffusion training
from image + existing caption + CLIP features.

Output format template (planned):
    {subject}, {style}, {lighting}, {camera}, {quality_tokens}

Phase 5. Populated when a diffusion manifold config declares
`prompt_strategy: <name>`.
"""

from __future__ import annotations