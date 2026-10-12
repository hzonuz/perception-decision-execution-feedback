from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from perception.dataset import augment_dataset, build_dataset
from tests.unit.test_dataset_converter import _materialize_fixture


ROOT = Path(__file__).resolve().parents[2]


class DatasetAugmentTests(unittest.TestCase):
    def _build(self, root: Path) -> Path:
        capture_root = root / "capture"
        _materialize_fixture(capture_root)
        dataset_root = root / "dataset"
        build_dataset(
            [capture_root],
            dataset_root,
            ROOT / "configs/dataset.json",
            ROOT / "configs/ontology.json",
        )
        return dataset_root

    def _records(self, dataset_root: Path, split: str) -> list[dict]:
        path = dataset_root / "metadata" / f"{split}.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def test_adds_fliplr_and_zoom_variants_to_train_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_root = self._build(root)

            train_before = self._records(dataset_root, "train")
            val_before = self._records(dataset_root, "val")
            test_before = self._records(dataset_root, "test")
            self.assertEqual(len(train_before), 2)

            report = augment_dataset(dataset_root, fliplr=True, zoom_max=1.3, seed=7)

            self.assertEqual(report["variants_added"], {"fliplr": 2, "zoom": 2})
            self.assertEqual(report["objects_dropped"] + report["objects_kept"], 4)

            train_after = self._records(dataset_root, "train")
            self.assertEqual(len(train_after), len(train_before) + 4)
            self.assertEqual(self._records(dataset_root, "val"), val_before)
            self.assertEqual(self._records(dataset_root, "test"), test_before)

            for record in train_after[len(train_before):]:
                self.assertTrue((dataset_root / record["image"]).is_file())
                self.assertTrue((dataset_root / record["label"]).is_file())
                self.assertIn(record["augmented"], ("fliplr", "zoom"))
                for item in record["objects"]:
                    box = item["bounding_box"]
                    self.assertGreaterEqual(box["x_min"], 0)
                    self.assertGreaterEqual(box["y_min"], 0)
                    self.assertLessEqual(box["x_max"], box["image_width"])
                    self.assertLessEqual(box["y_max"], box["image_height"])

    def test_fliplr_mirrors_the_bounding_box_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_root = self._build(root)
            train_before = self._records(dataset_root, "train")
            width = train_before[0]["objects"][0]["bounding_box"]["image_width"]

            augment_dataset(dataset_root, fliplr=True, zoom_max=1.3, seed=7)

            flipped = next(
                record
                for record in self._records(dataset_root, "train")
                if record.get("augmented") == "fliplr"
            )
            original_box = train_before[0]["objects"][0]["bounding_box"]
            flipped_box = flipped["objects"][0]["bounding_box"]
            self.assertAlmostEqual(flipped_box["x_min"], width - original_box["x_max"], places=3)
            self.assertAlmostEqual(flipped_box["x_max"], width - original_box["x_min"], places=3)
            self.assertAlmostEqual(flipped_box["y_min"], original_box["y_min"], places=3)
            self.assertAlmostEqual(flipped_box["y_max"], original_box["y_max"], places=3)

    def test_rerun_without_overwrite_is_a_no_op(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_root = self._build(root)

            first = augment_dataset(dataset_root, seed=1)
            count_after_first = len(self._records(dataset_root, "train"))

            second = augment_dataset(dataset_root, seed=99)
            self.assertEqual(first, second)
            self.assertEqual(len(self._records(dataset_root, "train")), count_after_first)

    def test_rejects_non_positive_zoom(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_root = self._build(root)
            with self.assertRaises(ValueError):
                augment_dataset(dataset_root, zoom_max=1.0)


if __name__ == "__main__":
    unittest.main()
