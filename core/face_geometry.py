"""Face box and five landmarks from a dense landmark set (RetinaFace / WIDER FACE convention).

Five points, in the order of the RetinaFace WIDER FACE annotations and the
ArcFace alignment template: left eye centre, right eye centre, nose tip,
left mouth corner, right mouth corner, where left and right are image left
and image right (the first eye has the smaller x on a frontal face; it is
the subject's right eye). Eye centres are the means of the eye-contour
points.

Face box (WIDER FACE style, tight on the face, not the whole head): x from
the leftmost to the rightmost face point (jaw, cheek to cheek), bottom at
the lowest face point (chin), top above the highest brow point by
``top_margin`` times the brow-to-chin height (the forehead, without hair),
clipped to the image.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.source_labels import LandmarkScheme

__all__ = ["FaceGeometry", "face_geometry", "plausibility_problems"]

_DIGITS = 6


@dataclass(frozen=True)
class FaceGeometry:
    """A face box ``(x0, y0, x1, y1)``, five ``(x, y)`` points and the mouth centre, in pixels."""

    box: tuple[float, float, float, float]
    points: tuple[tuple[float, float], ...]
    mouth_centre: tuple[float, float]

    def box_cxcywh(self, width: int, height: int) -> list[float]:
        """The box as normalised ``cx cy w h``, rounded to 6 digits."""
        x0, y0, x1, y1 = self.box
        values = ((x0 + x1) / 2 / width, (y0 + y1) / 2 / height,
                  (x1 - x0) / width, (y1 - y0) / height)
        return [round(v, _DIGITS) for v in values]

    def landmarks(self, width: int, height: int) -> list[float]:
        """The five points as ten normalised values ``x0 y0 ... x4 y4``, rounded to 6 digits."""
        return [round(v, _DIGITS) for x, y in self.points for v in (x / width, y / height)]


def face_geometry(points: np.ndarray, scheme: LandmarkScheme, width: int,
                  height: int) -> FaceGeometry:
    """Box and five points of one face from its ``scheme`` landmark array ``[N, 2]``.

    Raises ``ValueError`` when the array does not have the scheme's shape or
    holds non-finite values.
    """
    if points.shape != (scheme.points, 2) or not np.isfinite(points).all():
        raise ValueError(f"expected {scheme.points} finite (x, y) points, got {points.shape}")
    rule, idx = scheme.box, scheme.five
    face = points[list(rule.face_points)]
    brow_top = float(points[list(rule.brow_points), 1].min())
    chin = float(face[:, 1].max())
    top = brow_top - rule.top_margin * (chin - brow_top)
    box = (max(0.0, float(face[:, 0].min())), max(0.0, top),
           min(float(width), float(face[:, 0].max())), min(float(height), chin))
    five = (points[list(idx.left_eye)].mean(axis=0), points[list(idx.right_eye)].mean(axis=0),
            points[idx.nose_tip], points[idx.mouth_left], points[idx.mouth_right])
    mouth = points[list(idx.mouth_outline)].mean(axis=0)
    return FaceGeometry(box, tuple((float(p[0]), float(p[1])) for p in five),
                        (float(mouth[0]), float(mouth[1])))


def plausibility_problems(face: FaceGeometry) -> list[str]:
    """Why the face is implausible: eye order, eyes above nose above mouth, points in the box.

    The vertical order uses the mouth centre (outer lip contour), not the
    corners, which rise to nose height in a wide smile.
    """
    (lex, ley), (rex, rey), (_, ny) = face.points[:3]
    x0, y0, x1, y1 = face.box
    problems = []
    if x1 <= x0 or y1 <= y0:
        problems.append("empty box")
    if lex >= rex:
        problems.append("left eye not left of the right eye")
    if not (ley + rey) / 2 < ny < face.mouth_centre[1]:
        problems.append("eyes, nose and mouth not top to bottom")
    if not all(x0 <= x <= x1 and y0 <= y <= y1 for x, y in face.points):
        problems.append("a point outside the box")
    return problems
