from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from perception.dataset import ExternalSource, build_dataset, merge_datasets
from perception.dataset.models import DatasetValidationError
from tests.unit.test_dataset_converter import _materialize_fixture


ROOT = Path(__file__).resolve().parents[2]


def _external_source(root: Path) -> None:
    """A three-arena YOLO export: box-only labels plus one polygon label."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(
        "train: ../train/images\nnc: 2\nnames: ['GoodGoal', 'Wall']\n", encoding="utf-8"
    )
    labels = {
        "arena-1_ep0_0": "0 0.5 0.5 0.25 0.5\n1 0.1 0.1 0.2 0.2\n",
        "arena-1_ep0_1": "0 0.4 0.4 0.2 0.2\n",
        "arena-2_ep0_0": "1 0.2 0.3 0.4 0.4 0.6 0.3 0.4 0.1\n",
        "arena-3_ep0_0": "0 0.5 0.5 0.5 0.5\n",
    }
    for split in ("train", "valid", "test"):
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)
    for index, (stem, label) in enumerate(labels.items()):
        split = ("train", "valid", "test", "train")[index]
        name = f"{stem}_png.rf.{index:032x}"
        Image.new("RGB", (16, 8), (10, 20, 30)).save(root / split / "images" / f"{name}.jpg")
        (root / split / "labels" / f"{name}.txt").write_text(label, encoding="utf-8")


class DatasetMergeTests(unittest.TestCase):
    def _primary(self, root: Path) -> Path:
        capture_root = root / "capture"
        _materialize_fixture(capture_root)
        primary = root / "primary"
        build_dataset(
            [capture_root],
            primary,
            ROOT / "configs/dataset.json",
            ROOT / "configs/ontology.json",
        )
        return primary

    def test_merge_remaps_classes_widens_boxes_and_groups_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            primary = self._primary(root)
            external = root / "external"
            _external_source(external)

            report = merge_datasets(
                primary,
                [
                    ExternalSource(
                        name="ext",
                        root=external,
                        class_map={"GoodGoal": "GoodGoal", "Wall": "Wall"},
                        group_pattern=r"^(arena-\d+)",
                    )
                ],
                root / "merged",
            )
            merged = root / "merged"

            self.assertEqual(report["leakage"], "none")
            self.assertEqual(report["sources"]["ext"]["groups"], 3)
            self.assertEqual(
                report["objects_by_mask_source"]["box_rectangle"], 4
            )

            records = [
                json.loads(line)
                for split in ("train", "val", "test")
                for line in (merged / "metadata" / f"{split}.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            external_records = [item for item in records if item["source"] == "ext"]
            self.assertEqual(len(external_records), 4)

            # Both frames of arena-1 land in the same split, never straddling the boundary.
            arena_one = {
                item["split"] for item in external_records if item["arena_id"] == "arena-1"
            }
            self.assertEqual(len(arena_one), 1)

            # GoodGoal -> GoodGoal (semantic index 0), Wall (semantic index 10), and a box
            # label becomes a four-corner polygon with eight coordinates.
            box_record = next(
                item for item in external_records if len(item["objects"]) == 2
            )
            self.assertEqual(
                [entry["class_id"] for entry in box_record["objects"]], [0, 10]
            )
            self.assertEqual(
                [entry["mask_source"] for entry in box_record["objects"]],
                ["box_rectangle", "box_rectangle"],
            )
            first = box_record["objects"][0]["bounding_box"]
            self.assertAlmostEqual(first["x_min"], 0.375 * 16)
            self.assertAlmostEqual(first["y_max"], 0.75 * 8)
            self.assertEqual(first["image_width"], 16)
            label_line = (
                (merged / box_record["label"]).read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertEqual(len(label_line.split()), 9)

            # The real polygon survives as a polygon.
            polygon_record = next(
                item
                for item in external_records
                if item["objects"] and item["objects"][0]["mask_source"] == "polygon"
            )
            self.assertEqual(polygon_record["objects"][0]["class_name"], "Wall")

            # Capture frames keep their own metadata and provenance.
            capture_records = [item for item in records if item["source"] == "capture"]
            self.assertTrue(capture_records)
            self.assertTrue(
                all(item["mask_source"] == "polygon" for item in capture_records)
            )
            for item in records:
                self.assertTrue((merged / item["image"]).is_file())
                self.assertTrue((merged / item["label"]).is_file())

    def test_unmapped_class_and_non_empty_output_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            primary = self._primary(root)
            external = root / "external"
            _external_source(external)
            source = ExternalSource(
                name="ext", root=external, class_map={"GoodGoal": "GoodGoal"}
            )

            with self.assertRaises(DatasetValidationError):
                merge_datasets(primary, [source], root / "merged")

            occupied = root / "occupied"
            occupied.mkdir()
            (occupied / "stale.txt").write_text("", encoding="utf-8")
            with self.assertRaises(DatasetValidationError):
                merge_datasets(primary, [], occupied)


if __name__ == "__main__":
    unittest.main()
