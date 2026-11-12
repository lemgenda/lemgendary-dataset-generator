"""Paired samples need a real target; label geometry is normalized once (audit 2.1.6)."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

from core import sample_processor
from core.sample_annotations import (
    AnnotationSource,
    LabelContent,
    _face_pose_from_landmarks,
    _parse_parquet_row,
    _parse_safetensors,
    label_lines,
)
from core.sample_processor import _ImageSample, _Target, _write_outputs


def _src(fmt: str, ann_data: object, width: int = 200, height: int = 100) -> AnnotationSource:
    return AnnotationSource(fmt, ann_data, "", "detection", width, height)


def _sample(task: str) -> mock.MagicMock:
    sample = mock.MagicMock(spec=_ImageSample)
    sample.configure_mock(task=task, name="s1", slug="src", img=object(),
                          img_path=Path("in.png"), img_data=b"")
    sample.image_out_path.return_value = Path("out/images/train/s1.png")
    sample.target_out_path.return_value = Path("out/targets/train/s1.png")
    return sample


class TestPairedTargets(unittest.TestCase):
    """The degraded input is never written as its own target."""

    def test_restoration_without_target_is_rejected(self) -> None:
        """No target, no clean-only flag: nothing is written and None is returned."""
        with (
            mock.patch.object(sample_processor, "write_image") as write,
            self.assertLogs("core.sample_processor", level="WARNING"),
        ):
            out = _write_outputs(_sample("restoration"), _Target(None, None, False))
        self.assertIsNone(out)
        write.assert_not_called()

    def test_segmentation_uses_generated_mask(self) -> None:
        """A generated mask becomes the target; without one the sample is rejected."""
        mask_gen = mock.MagicMock()
        mask_gen.generate.return_value = SimpleNamespace(value="MASK")
        generators = SimpleNamespace(mask_gen=mask_gen, label_gen=None)
        with (
            mock.patch.object(sample_processor.WORKER, "generators", generators),
            mock.patch.object(sample_processor.WORKER, "physical_index", set()),
            mock.patch.object(sample_processor, "write_image", return_value=Path("o.png")) as w,
        ):
            sample = _sample("segmentation")
            self.assertIsNotNone(_write_outputs(sample, _Target(None, None, False)))
            self.assertEqual(w.call_args_list[-1].kwargs["kind"], "mask")
            self.assertEqual(w.call_args_list[-1].kwargs["loaded_img"], "MASK")
            mask_gen.generate.return_value = SimpleNamespace(value=None)
            w.reset_mock()
            with self.assertLogs("core.sample_processor", level="WARNING"):
                self.assertIsNone(
                    _write_outputs(_sample("segmentation"), _Target(None, None, False))
                )
            w.assert_not_called()


class TestAnnotationGeometry(unittest.TestCase):
    """Boxes and keypoints end up normalized exactly once."""

    def test_safetensors_full_frame_box(self) -> None:
        """The full-frame tag box renders as 0.5 0.5 1 1, not divided by the width again."""
        anns = _parse_safetensors(_src("safetensors", {"ss_tag_frequency": '{"b": {"cat": 1}}'}))
        line = label_lines(LabelContent("detection", [], anns, None, Path("x")), 200, 100)
        self.assertEqual(line[0].split()[1:], ["0.5", "0.5", "1.0", "1.0"])

    def test_parquet_pose_row_gets_a_box(self) -> None:
        """A pose row uses its bbox columns, else the box around its visible keypoints."""
        mapping = {"class": "class", "keypoints": "kp", "segmentation": "seg"}
        row = pd.Series({"class": 0, "kp": [20, 10, 2, 60, 50, 2, 199, 99, 0], "seg": None})
        ann = _parse_parquet_row(_src("parquet", None), row, mapping)
        self.assertEqual(ann["data"][:4], [20.0, 10.0, 40.0, 40.0])
        with_box = pd.Series({"class": 0, "kp": [20, 10, 2], "seg": None, "xmin": 5,
                              "ymin": 6, "width": 7, "height": 8})
        ann = _parse_parquet_row(_src("parquet", None), with_box, mapping)
        self.assertEqual(ann["data"][:4], [5, 6, 7, 8])

    def test_face_keypoints_are_normalized(self) -> None:
        """NPZ face landmarks produce keypoints in 0..1, like every other pose source."""
        landmarks = np.full((110, 2), 50.0)
        landmarks[:68, 0] = np.linspace(40, 160, 68)
        ann = _face_pose_from_landmarks(_src("npz", None), landmarks)
        kpts = ann["data"][4:]
        self.assertEqual(len(kpts), 10)
        self.assertTrue(all(0.0 <= v <= 1.0 for v in kpts), kpts)


if __name__ == "__main__":
    unittest.main()
