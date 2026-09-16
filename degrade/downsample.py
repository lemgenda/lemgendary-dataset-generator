"""
Downsampling kernels for super-resolution training pairs.

Single responsibility: downsample an HR image to LR using a configurable
interpolation kernel (bicubic / bilinear / Lanczos) and scale factor.
Used to build LR/HR pairs from clean sources.

Phase 6.
"""

from __future__ import annotations