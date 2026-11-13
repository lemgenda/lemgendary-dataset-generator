"""Per-file source label schemes (SSOT): mask class schemes.

Some sources ship one label file per image next to the images (SFHQ:
``segmentations/<name>.png`` face-parsing masks and ``landmarks/<name>.npz``
110-point landmarks). A registry entry names the scheme of those files in
``source_labels.scheme`` and the tag its JSON labels record in
``source_labels.source``; ``repair masks`` and ``repair face-landmarks``
read the files through the scheme defined here.

``celebamask_hq_19``: 8-bit PNG masks whose pixel values are the 19
CelebAMask-HQ class ids (0 background ... 18 hat), as written by the
zllrunning face-parsing BiSeNet (the SFHQ ``segmentations/`` folder).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "CELEBAMASK_HQ_19",
    "SOURCE_LABEL_SCHEMES",
    "MaskScheme",
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


SOURCE_LABEL_SCHEMES: dict[str, MaskScheme] = {
    scheme.name: scheme for scheme in (
        MaskScheme("celebamask_hq_19", ".png", CELEBAMASK_HQ_19),
    )
}


def mask_scheme(name: str) -> MaskScheme:
    """The mask scheme ``name``; ``ValueError`` when it is unknown or not a mask scheme."""
    scheme = SOURCE_LABEL_SCHEMES.get(name)
    if scheme is None:
        raise ValueError(f"'{name}' is not a mask scheme")
    return scheme
