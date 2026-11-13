"""Per-file source label schemes (SSOT): mask class schemes and landmark layouts.

Some sources ship one label file per image next to the images (SFHQ:
``segmentations/<name>.png`` face-parsing masks and ``landmarks/<name>.npz``
110-point landmarks). A registry entry names the scheme of those files in
``source_labels.scheme`` and the tag its JSON labels record in
``source_labels.source``; ``repair masks`` and ``repair face-landmarks``
read the files through the scheme defined here.

``celebamask_hq_19``: 8-bit PNG masks whose pixel values are the 19
CelebAMask-HQ class ids (0 background ... 18 hat), as written by the
zllrunning face-parsing BiSeNet (the SFHQ ``segmentations/`` folder).

``sfhq_110``: ``.npz`` files with one ``landmarks`` array of 110 ``(x, y)``
pixel points. The SFHQ card documents 110 points from Dlib, Face Alignment
and BiSeNet post-processing, incl. inner and outer hairlines, without an
index table. Points 0-67 follow the iBUG 300-W 68-point order (checked on
the data: jaw 0-16, brows 17-26, nose 27-35 with the tip at 30, eyes 36-41
and 42-47, outer lips 48-59 with the corners at 48 and 54, inner lips
60-67); points 68-109 are the SFHQ extensions (cheek, hairline and head
contour points), not used here. iBUG numbers the points as they appear in
the image: 36-41 is the eye on the image left (the subject's right eye).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "CELEBAMASK_HQ_19",
    "LANDMARK_SCHEMES",
    "MASK_SCHEMES",
    "SOURCE_LABEL_SCHEMES",
    "BoxRule",
    "FivePoints",
    "LandmarkScheme",
    "MaskScheme",
    "landmark_scheme",
    "mask_scheme",
]

CELEBAMASK_HQ_19: tuple[str, ...] = (
    "background", "skin", "l_brow", "r_brow", "l_eye", "r_eye", "eye_g", "l_ear", "r_ear",
    "ear_r", "nose", "mouth", "u_lip", "l_lip", "neck", "neck_l", "cloth", "hair", "hat",
)


@dataclass(frozen=True)
class MaskScheme:
    """Class masks: one file per image whose pixel values are class ids."""

    name: str
    file_suffix: str
    class_names: tuple[str, ...]

    @property
    def num_classes(self) -> int:
        """Number of classes (ids 0 .. num_classes - 1)."""
        return len(self.class_names)


@dataclass(frozen=True)
class FivePoints:
    """Point indices of the five RetinaFace points, in image order (image-left first).

    Each eye is the mean of its contour points; nose tip and mouth corners
    are single points. ``mouth_outline`` (the outer lip contour) gives the
    mouth centre for the plausibility check: in a wide smile the corners
    rise to the height of the nose tip, the mouth centre does not.
    """

    left_eye: tuple[int, ...]
    right_eye: tuple[int, ...]
    nose_tip: int
    mouth_left: int
    mouth_right: int
    mouth_outline: tuple[int, ...]


@dataclass(frozen=True)
class BoxRule:
    """The face box spans ``face_points``; its top is raised above the highest
    ``brow_points`` point by ``top_margin`` times the brow-to-chin height."""

    face_points: tuple[int, ...]
    brow_points: tuple[int, ...]
    top_margin: float


@dataclass(frozen=True)
class LandmarkScheme:
    """Landmark files (``points`` (x, y) pixels under ``array_key``) and how the
    5-point face label is built from them."""

    name: str
    file_suffix: str
    array_key: str
    points: int
    five: FivePoints
    box: BoxRule


MASK_SCHEMES: dict[str, MaskScheme] = {
    scheme.name: scheme for scheme in (
        MaskScheme("celebamask_hq_19", ".png", CELEBAMASK_HQ_19),
    )
}
LANDMARK_SCHEMES: dict[str, LandmarkScheme] = {
    scheme.name: scheme for scheme in (
        LandmarkScheme(
            "sfhq_110", ".npz", "landmarks", 110,
            five=FivePoints(left_eye=tuple(range(36, 42)), right_eye=tuple(range(42, 48)),
                            nose_tip=30, mouth_left=48, mouth_right=54,
                            mouth_outline=tuple(range(48, 60))),
            box=BoxRule(face_points=tuple(range(68)), brow_points=tuple(range(17, 27)),
                        top_margin=0.2),
        ),
    )
}
SOURCE_LABEL_SCHEMES: dict[str, MaskScheme | LandmarkScheme] = {**MASK_SCHEMES,
                                                                 **LANDMARK_SCHEMES}


def mask_scheme(name: str) -> MaskScheme:
    """The mask scheme ``name``; ``ValueError`` when it is unknown or not a mask scheme."""
    scheme = MASK_SCHEMES.get(name)
    if scheme is None:
        raise ValueError(f"'{name}' is not a mask scheme")
    return scheme


def landmark_scheme(name: str) -> LandmarkScheme:
    """The landmark scheme ``name``; ``ValueError`` when it is unknown or not one."""
    scheme = LANDMARK_SCHEMES.get(name)
    if scheme is None:
        raise ValueError(f"'{name}' is not a landmark scheme")
    return scheme
