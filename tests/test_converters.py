"""Unit tests for LemGendary Dataset Compiler annotation converters and dispatch."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import scipy.io as sio

from converters.coco import parse_coco
from converters.dispatch import detect_annotations
from converters.matlab import parse_matlab
from converters.xml import parse_xml
from converters.yolo import parse_yolo


class TestAnnotationConverters(unittest.TestCase):
    """Test suite for annotation converters and detection dispatch."""

    def test_matlab_converter(self) -> None:
        """Verify MATLAB .mat file parsing and primary key resolution."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            mat_file = Path(tmp_dir) / "annotations.mat"
            expected_array = np.array([[15.0, 25.0, 100.0, 200.0], [50.0, 60.0, 80.0, 90.0]])
            sio.savemat(str(mat_file), {"bounding_boxes": expected_array, "version": 1})

            data, primary_key = parse_matlab(mat_file)
            self.assertIn("bounding_boxes", data)
            self.assertFalse(primary_key.startswith("__"))
            np.testing.assert_array_equal(data["bounding_boxes"], expected_array)

    def test_coco_converter(self) -> None:
        """Verify COCO JSON parsing into images_by_id and annotations_by_image_id."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_file = Path(tmp_dir) / "instances_train.json"
            coco_payload = {
                "images": [
                    {"id": 101, "file_name": "sample_101.jpg", "width": 800, "height": 600},
                    {"id": 102, "file_name": "sample_102.jpg", "width": 1024, "height": 768},
                ],
                "annotations": [
                    {"id": 1, "image_id": 101, "category_id": 3, "bbox": [50, 60, 200, 150]},
                    {"id": 2, "image_id": 101, "category_id": 5, "bbox": [300, 400, 100, 80]},
                    {"id": 3, "image_id": 102, "category_id": 3, "bbox": [10, 20, 50, 50]},
                ],
            }
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(coco_payload, f)

            images, anns = parse_coco(json_file)
            self.assertEqual(len(images), 2)
            self.assertIn(101, images)
            self.assertEqual(images[101]["file_name"], "sample_101.jpg")
            self.assertEqual(len(anns[101]), 2)
            self.assertEqual(len(anns[102]), 1)
            self.assertEqual(anns[101][0]["bbox"], [50, 60, 200, 150])

    def test_yolo_converter(self) -> None:
        """Verify YOLO normalized space-delimited text label parsing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            txt_file = Path(tmp_dir) / "sample_001.txt"
            # format: class cx cy nw nh [optional keypoints]
            content = "0 0.5 0.5 0.2 0.4\n1 0.25 0.75 0.1 0.2 10.0 20.0 1.0\n"
            txt_file.write_text(content, encoding="utf-8")

            # Parse with spatial dimensions 1000x800
            img_w = 1000
            img_h = 800
            anns = parse_yolo(txt_file, img_w=img_w, img_h=img_h)

            self.assertEqual(len(anns), 2)
            # Item 0: cx=500, cy=400, w=200, h=320 -> xmin = 500-100=400, ymin = 400-160=240
            self.assertEqual(anns[0]["class"], "0")
            self.assertAlmostEqual(anns[0]["bbox"][0], 400.0)
            self.assertAlmostEqual(anns[0]["bbox"][1], 240.0)
            self.assertAlmostEqual(anns[0]["bbox"][2], 200.0)
            self.assertAlmostEqual(anns[0]["bbox"][3], 320.0)

            # Item 1: includes keypoints
            self.assertEqual(anns[1]["class"], "1")
            self.assertIn("keypoints", anns[1])
            self.assertEqual(anns[1]["keypoints"], [10.0, 20.0, 1.0])

    def test_xml_converter(self) -> None:
        """Verify Pascal VOC XML node parsing into bounding box dicts."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            xml_file = Path(tmp_dir) / "sample_001.xml"
            xml_content = """<annotation>
                <folder>VOC2012</folder>
                <filename>sample_001.jpg</filename>
                <object>
                    <name>dog</name>
                    <bndbox>
                        <xmin>50</xmin>
                        <ymin>100</ymin>
                        <xmax>350</xmax>
                        <ymax>500</ymax>
                    </bndbox>
                </object>
                <object>
                    <name>frisbee</name>
                    <bndbox>
                        <xmin>400</xmin>
                        <ymin>120</ymin>
                        <xmax>480</xmax>
                        <ymax>200</ymax>
                    </bndbox>
                </object>
            </annotation>"""
            xml_file.write_text(xml_content, encoding="utf-8")

            anns = parse_xml(xml_file)
            self.assertEqual(len(anns), 2)
            self.assertEqual(anns[0]["class"], "dog")
            # xmin=50, ymin=100, width=300, height=400
            self.assertEqual(anns[0]["bbox"], [50.0, 100.0, 300.0, 400.0])
            self.assertEqual(anns[1]["class"], "frisbee")
            # xmin=400, ymin=120, width=80, height=80
            self.assertEqual(anns[1]["bbox"], [400.0, 120.0, 80.0, 80.0])

    def test_detect_annotations_dispatch(self) -> None:
        """Verify detect_annotations auto-discovery across diverse directory layouts."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_path = Path(tmp_dir)

            # Test MATLAB detection in root
            mat_dir = base_path / "mat_set"
            mat_dir.mkdir()
            (mat_dir / "labels.mat").write_bytes(b"dummy")
            fmt, match = detect_annotations(mat_dir)
            self.assertEqual(fmt, "matlab")
            self.assertIsNotNone(match)

            # Test COCO detection in annotations/
            coco_dir = base_path / "coco_set"
            (coco_dir / "annotations").mkdir(parents=True)
            (coco_dir / "annotations" / "instances_val.json").write_text("{}", encoding="utf-8")
            fmt, match = detect_annotations(coco_dir)
            self.assertEqual(fmt, "coco")
            self.assertIsNotNone(match)

            # Test YOLO detection in labels/
            yolo_dir = base_path / "yolo_set"
            (yolo_dir / "labels").mkdir(parents=True)
            (yolo_dir / "labels" / "001.txt").write_text("0 0.5 0.5 0.1 0.1", encoding="utf-8")
            fmt, match = detect_annotations(yolo_dir)
            self.assertEqual(fmt, "yolo")
            self.assertIsNotNone(match)

            # Test Empty Directory
            empty_dir = base_path / "empty_set"
            empty_dir.mkdir()
            fmt, match = detect_annotations(empty_dir)
            self.assertIsNone(fmt)
            self.assertIsNone(match)


if __name__ == "__main__":
    unittest.main()
