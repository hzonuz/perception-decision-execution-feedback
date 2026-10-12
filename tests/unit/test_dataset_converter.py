from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from perception.dataset import DatasetValidationError, build_dataset


ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = ROOT / "tests/fixtures/dataset/spec.json"
BASE_CAPTURE_PATH = ROOT / "tests/fixtures/contracts/capture_frame.json"


class DatasetConverterTests(unittest.TestCase):
    def test_conversion_has_group_disjoint_splits_and_training_formats(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture_root = root / "capture"
            _materialize_fixture(capture_root)

            first = root / "dataset-first"
            second = root / "dataset-second"
            result = build_dataset(
                [capture_root],
                first,
                ROOT / "configs/dataset.json",
                ROOT / "configs/ontology.json",
            )
            build_dataset(
                [capture_root],
                second,
                ROOT / "configs/dataset.json",
                ROOT / "configs/ontology.json",
            )

            self.assertEqual(result["split_counts"], {"train": 2, "val": 2, "test": 2})
            self.assertEqual(result["validation_status"], "passed")
            splits = json.loads((first / "splits.json").read_text(encoding="utf-8"))
            episode_sets = [set(splits["episodes"][split]) for split in ("train", "val", "test")]
            self.assertFalse(episode_sets[0] & episode_sets[1])
            self.assertFalse(episode_sets[0] & episode_sets[2])
            self.assertFalse(episode_sets[1] & episode_sets[2])
            self.assertEqual(
                (first / "splits.json").read_text(encoding="utf-8"),
                (second / "splits.json").read_text(encoding="utf-8"),
            )

            labels = list((first / "labels").glob("*/*.txt"))
            self.assertEqual(len(labels), 6)
            for path in labels:
                values = path.read_text(encoding="utf-8").split()
                self.assertGreaterEqual(len(values), 7)
                self.assertEqual(values[0], "0")
                self.assertTrue(all(0 <= float(value) <= 1 for value in values[1:]))
            for split in ("train", "val", "test"):
                self.assertTrue((first / f"coco/instances_{split}.json").is_file())
                self.assertTrue(list((first / "visualizations" / split).glob("*.png")))
            statistics = json.loads((first / "statistics.json").read_text(encoding="utf-8"))
            self.assertEqual(statistics["samples_per_class"]["GoodGoal"], 6)

    def test_duplicate_frame_is_rejected_and_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture_root = root / "capture"
            output = root / "dataset"
            _materialize_fixture(capture_root)

            with self.assertRaises(DatasetValidationError):
                build_dataset(
                    [capture_root, capture_root],
                    output,
                    ROOT / "configs/dataset.json",
                    ROOT / "configs/ontology.json",
                )

            report = json.loads(
                (output / "validation_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["status"], "failed")
            self.assertIn(
                "duplicate_frame",
                {item["code"] for item in report["errors"]},
            )

    def test_arena_and_seed_grouping_also_keep_episodes_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture_root = root / "capture"
            _materialize_fixture(capture_root)
            base_config = json.loads(
                (ROOT / "configs/dataset.json").read_text(encoding="utf-8")
            )

            for group_by in ("arena", "seed"):
                with self.subTest(group_by=group_by):
                    config = copy.deepcopy(base_config)
                    config["split"]["group_by"] = group_by
                    config_path = root / f"{group_by}.json"
                    config_path.write_text(json.dumps(config), encoding="utf-8")
                    output = root / f"dataset-{group_by}"
                    build_dataset(
                        [capture_root],
                        output,
                        config_path,
                        ROOT / "configs/ontology.json",
                    )
                    splits = json.loads(
                        (output / "splits.json").read_text(encoding="utf-8")
                    )
                    episode_sets = [
                        set(splits["episodes"][split])
                        for split in ("train", "val", "test")
                    ]
                    self.assertFalse(episode_sets[0] & episode_sets[1])
                    self.assertFalse(episode_sets[0] & episode_sets[2])
                    self.assertFalse(episode_sets[1] & episode_sets[2])


def _materialize_fixture(root: Path) -> None:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    template = json.loads(BASE_CAPTURE_PATH.read_text(encoding="utf-8"))
    width = spec["image_width"]
    height = spec["image_height"]
    for episode in spec["episodes"]:
        for frame_index in range(spec["frames_per_episode"]):
            payload = copy.deepcopy(template)
            episode_id = episode["episode_id"]
            instance_id = f"{episode_id}:{spec['instance_id_suffix']}"
            rgb_relative = f"rgb/{episode_id}/{frame_index:06d}.png"
            mask_relative = f"masks/{episode_id}/{frame_index:06d}.png"

            payload["episode_id"] = episode_id
            payload["arena_id"] = episode["arena_id"]
            payload["random_seed"] = episode["random_seed"]
            payload["stamp"] = _stamp(episode_id, frame_index)
            payload["rgb"] = {
                "path": rgb_relative,
                "width": width,
                "height": height,
                "encoding": "png",
                "sha256": None,
            }
            payload["camera"]["intrinsics"].update(
                {
                    "width": width,
                    "height": height,
                    "fx": 12.0,
                    "fy": 12.0,
                    "cx": width / 2,
                    "cy": height / 2,
                }
            )
            item = payload["objects"][0]
            item["instance_id"] = instance_id
            item["bounding_box"] = {
                "x_min": 4.0,
                "y_min": 4.0,
                "x_max": 10.0,
                "y_max": 10.0,
                "image_width": width,
                "image_height": height,
            }
            item["instance_mask"] = {
                "path": mask_relative,
                "encoding": "png",
                "width": width,
                "height": height,
                "instance_value": 1,
            }
            if frame_index == 0:
                payload["lifecycle_events"][0]["instance_id"] = instance_id
                payload["lifecycle_events"][0]["stamp"] = _stamp(episode_id, frame_index)
            else:
                payload["lifecycle_events"] = []

            image_path = root / rgb_relative
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (width, height), (64, 64, 64)).save(image_path)
            mask = Image.new("RGB", (width, height), (0, 0, 0))
            for y in range(4, 10):
                for x in range(4, 10):
                    mask.putpixel((x, y), (1, 0, 0))
            mask_path = root / mask_relative
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            mask.save(mask_path)

            annotation_path = (
                root / "annotations" / episode_id / f"{frame_index:06d}.json"
            )
            annotation_path.parent.mkdir(parents=True, exist_ok=True)
            annotation_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )


def _stamp(episode_id: str, frame_index: int) -> dict[str, object]:
    return {
        "episode_id": episode_id,
        "frame_index": frame_index,
        "timestamp_seconds": frame_index * 0.02,
    }


if __name__ == "__main__":
    unittest.main()
