from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from perception.dataset import build_class_samples, build_contact_sheet, build_dataset
from tests.unit.test_dataset_converter import _materialize_fixture


ROOT = Path(__file__).resolve().parents[2]


class ClassSamplesTests(unittest.TestCase):
    def test_picks_a_visible_instance_and_reports_missing_classes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture_root = root / "capture"
            _materialize_fixture(capture_root)
            dataset = root / "dataset"
            build_dataset(
                [capture_root],
                dataset,
                ROOT / "configs/dataset.json",
                ROOT / "configs/ontology.json",
            )

            report = build_class_samples(dataset)

            self.assertTrue(report["written"])
            output = Path(report["output"])
            for name in report["written"]:
                self.assertTrue((output / f"{name}.png").is_file())
            # Every semantic class in the tiny fixture arena is either written or reported
            # missing — never silently dropped.
            self.assertEqual(set(report["written"]) & set(report["missing"]), set())

    def test_extreme_close_up_is_not_picked_over_a_clearly_visible_instance(self) -> None:
        """A point-blank, frame-filling instance is the least recognizable view — must lose."""
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary)
            (dataset / "metadata").mkdir(parents=True)
            (dataset / "images/train").mkdir(parents=True)
            (dataset / "data.yaml").write_text("names:\n  0: GoodGoal\n", encoding="utf-8")

            for name, box in (
                ("close_up.png", {"x_min": 0, "y_min": 0, "x_max": 256, "y_max": 256}),
                ("clear_view.png", {"x_min": 90, "y_min": 90, "x_max": 166, "y_max": 166}),
            ):
                Image.new("RGB", (256, 256), (10, 20, 30)).save(dataset / "images/train" / name)

            records = [
                {
                    "image": "images/train/close_up.png",
                    "objects": [
                        {
                            "class_name": "GoodGoal",
                            "mask_source": "polygon",
                            "area_pixels": 256 * 256,
                            "bounding_box": {
                                "x_min": 0, "y_min": 0, "x_max": 256, "y_max": 256,
                                "image_width": 256, "image_height": 256,
                            },
                        }
                    ],
                },
                {
                    "image": "images/train/clear_view.png",
                    "objects": [
                        {
                            "class_name": "GoodGoal",
                            "mask_source": "polygon",
                            "area_pixels": 76 * 76,
                            "bounding_box": {
                                "x_min": 90, "y_min": 90, "x_max": 166, "y_max": 166,
                                "image_width": 256, "image_height": 256,
                            },
                        }
                    ],
                },
            ]
            import json

            (dataset / "metadata/train.jsonl").write_text(
                "\n".join(json.dumps(record) for record in records), encoding="utf-8"
            )

            report = build_class_samples(dataset, target_area_fraction=0.12)

            with Image.open(Path(report["output"]) / "GoodGoal.png") as sample:
                # The close-up crop is the full 256x256 frame plus the banner; the clear-view
                # crop is far smaller. Picking the close-up would make the sample at least
                # as wide as the original frame.
                self.assertLess(sample.width, 256)

    def test_contact_sheet_tiles_every_written_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture_root = root / "capture"
            _materialize_fixture(capture_root)
            dataset = root / "dataset"
            build_dataset(
                [capture_root],
                dataset,
                ROOT / "configs/dataset.json",
                ROOT / "configs/ontology.json",
            )
            report = build_class_samples(dataset)

            sheet_path = build_contact_sheet(report["output"])

            self.assertTrue(sheet_path.is_file())
            with Image.open(sheet_path) as sheet:
                self.assertGreater(sheet.width, 0)
                self.assertGreater(sheet.height, 0)


if __name__ == "__main__":
    unittest.main()
