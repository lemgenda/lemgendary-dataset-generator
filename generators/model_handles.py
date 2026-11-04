"""
Shared container for the pre-loaded model wrappers a generator composes.

Generators never load models themselves; the caller hands in the already
constructed CaptionSentry / CLIPManifold / AutoLabeler / QualitySentry
instances as keyword arguments (``captioner=``, ``clip=``, ``labeler=``,
``sentry=``). ``ModelHandles.from_kwargs`` validates those keywords so a
typo raises ``TypeError`` just like an unknown keyword argument would.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True)
class ModelHandles:
    """Optional model wrappers injected into a generator (None = unavailable)."""

    captioner: Any = None
    clip: Any = None
    labeler: Any = None
    sentry: Any = None

    @classmethod
    def from_kwargs(
        cls,
        owner: str,
        kwargs: dict[str, Any],
        allowed: tuple[str, ...] | None = None,
    ) -> ModelHandles:
        """Build handles from keyword arguments, rejecting unknown names.

        ``owner`` names the calling class for the error message; ``allowed``
        restricts the accepted keywords (default: every field).
        """
        names = allowed if allowed is not None else tuple(f.name for f in fields(cls))
        unknown = sorted(set(kwargs) - set(names))
        if unknown:
            raise TypeError(
                f"{owner}.__init__() got unexpected keyword argument(s): "
                f"{', '.join(unknown)}"
            )
        return cls(**kwargs)
