"""
Ground-truth loaders for quality / technical / aesthetic datasets.

Extracted from compiler_core.load_ground_truth() in Phase 1.5.5, which was a
single 120-line function walking 8 hardcoded dataset layouts. This module uses
a registry pattern so new sources can be added without editing dispatch.

Public API:
    cache = GroundTruthCaches()
    cache.load(root=Path("./raw-sets"), model_name="nima_technical")
    cache.ava           # dict[image_num -> 10 vote keys]
    cache.aadb          # dict[filename -> scalar 0-1]
    cache.tid           # dict[lowercased filename -> score 1-10]
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, cast

# ─── Shared state ───────────────────────────────────────────────────────────
@dataclass
class GroundTruthCaches:
    ava: dict = field(default_factory=dict)
    aadb: dict = field(default_factory=dict)
    tid: dict = field(default_factory=dict)

    def load(self, root: Path, model_name: str = "") -> None:
        for loader in _LOADERS:
            try:
                loader(self, Path(root))
            except Exception as e:
                print(f"[GT] loader {loader.__name__} skipped: {e}")


_LOADERS: list[Callable[[GroundTruthCaches, Path], None]] = []
def register(fn):
    _LOADERS.append(fn)
    return fn


# ─── Path resolution helper ─────────────────────────────────────────────────
def _find(root: Path, base_name: str, relative_target: str) -> Path | None:
    """Try the three layout conventions used across quality datasets."""
    for candidate in (
        root / "IQA-PyTorch-Datasets" / base_name / relative_target,
        root / base_name / relative_target,
        root / base_name / base_name / relative_target,
    ):
        if candidate.exists():
            return candidate
    return None


# ─── Individual loaders ─────────────────────────────────────────────────────
@register
def _load_ava(c: GroundTruthCaches, root: Path) -> None:
    path = root / "ava-aesthetic-visual-assessment" / "ground_truth_dataset.csv"
    if not path.exists():
        return
    import pandas as pd
    df = pd.read_csv(path)
    vote_cols = [f"vote_{i}" for i in range(1, 11)]
    c.ava = cast(dict, cast(Any, df.set_index("image_num")[vote_cols]).to_dict("index"))
    print(f"[GT] {len(c.ava)} AVA Aesthetic ratings cached.")


@register
def _load_aadb(c: GroundTruthCaches, root: Path) -> None:
    path = root / "aadb-imagedatabase" / "Dataset.csv"
    if not path.exists():
        return
    import pandas as pd
    df = pd.read_csv(path)
    c.aadb = df.set_index("ImageFile")["score"].to_dict()
    print(f"[GT] {len(c.aadb)} AADB Aesthetic ratings cached.")


@register
def _load_koniq(c: GroundTruthCaches, root: Path) -> None:
    csv = _find(root, "koniq-10k-dataset", "koniq10k_distributions_sets.csv") \
        or _find(root, "koniq10k", "koniq10k_scores.csv")
    if not csv:
        return
    import pandas as pd
    df = pd.read_csv(csv)
    for _, row in df.iterrows():
        val = float(cast(Any, row["MOS"])) / 10.0
        c.tid[str(row["image_name"]).lower()] = max(1.0, min(10.0, val))
    print(f"[GT] KonIQ-10k ratings cached.")


@register
def _load_spaq(c: GroundTruthCaches, root: Path) -> None:
    csv = _find(root, "spaq", "SPAQ/Annotations/MOS_Average.csv") \
        or _find(root, "spaq", "Annotations/MOS_Average.csv")
    if not csv:
        return
    import pandas as pd
    df = pd.read_csv(csv)
    for _, row in df.iterrows():
        c.tid[str(row["Image name"]).lower()] = 1.0 + float(cast(Any, row["MOS"])) * 0.09
    print(f"[GT] SPAQ ratings cached.")


@register
def _load_tid2013(c: GroundTruthCaches, root: Path) -> None:
    txt = _find(root, "tid2013", "mos_with_names.txt")
    if not txt:
        return
    with open(txt) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                c.tid[parts[1].strip().lower()] = float(parts[0]) + 1.0
    print(f"[GT] TID2013 ratings cached.")


@register
def _load_live(c: GroundTruthCaches, root: Path) -> None:
    csv = _find(root, "live", "live_scores.csv")
    if not csv:
        return
    import pandas as pd
    df = pd.read_csv(csv)
    for _, row in df.iterrows():
        orig = min(100.0, float(cast(Any, row["dmos"])))
        c.tid[str(row["image_name"]).lower()] = 1.0 + (1.0 - orig / 100.0) * 9.0
    print(f"[GT] LIVE IQA ratings cached.")


@register
def _load_csiq(c: GroundTruthCaches, root: Path) -> None:
    csv = _find(root, "csiq", "csiq_scores.csv")
    if not csv:
        return
    import pandas as pd
    df = pd.read_csv(csv)
    for _, row in df.iterrows():
        c.tid[str(row["image_name"]).lower()] = 1.0 + (1.0 - float(cast(Any, row["dmos"]))) * 9.0
    print(f"[GT] CSIQ ratings cached.")


@register
def _load_tad66k(c: GroundTruthCaches, root: Path) -> None:
    labels = _find(root, "TAD66K_for_Image_Aesthetics_Assessment", "labels/unmerge")
    if not labels or not labels.exists():
        labels = _find(root, "TAD66K_for_Image_Aesthetics_Assessment", "labels/labels/unmerge")
    if not labels or not labels.exists():
        return
    import pandas as pd
    count = 0
    for dirpath, _, files in os.walk(labels):
        for f in files:
            if not f.endswith(".csv"):
                continue
            df = pd.read_csv(os.path.join(dirpath, f))
            for _, row in df.iterrows():
                if "image" in row and "score" in row:
                    c.tid[str(row["image"]).lower()] = max(1.0, min(10.0, float(cast(Any, row["score"]))))
                    count += 1
    print(f"[GT] {count} TAD66K ratings cached.")