from __future__ import annotations

import unittest

from perception.detection import DetectorConfig, evaluate_predictions


class DetectorTrainingTests(unittest.TestCase):
    def test_metadata_metrics_without_model_download(self) -> None:
        config = DetectorConfig(
            data="data.yaml",
            model="yolo26n-seg.yaml",
            pretrained_checkpoint=None,
            resume_checkpoint=None,
            output_root="runs",
            experiment_name="test",
            seed=7,
            epochs=1,
            image_size=64,
            batch_size=1,
            workers=0,
            device="cpu",
            confidence_threshold=0.25,
            nms_iou=0.7,
            match_iou=0.5,
            small_area_fraction=0.1,
            distant_depth_threshold=10.0,
            calibration_bins=5,
            augmentation={},
        )
        metadata = [
            {
                "image": "images/test/frame.png",
                "objects": [
                    {
                        "instance_id": "stable-1",
                        "class_id": 0,
                        "class_name": "positive_goal",
                        "bounding_box": {
                            "x_min": 4.0,
                            "y_min": 4.0,
                            "x_max": 10.0,
                            "y_max": 10.0,
                            "image_width": 16,
                            "image_height": 16,
                        },
                        "area_pixels": 36,
                        "visibility_fraction": 0.9,
                        "occlusion_fraction": 0.1,
                        "relative_depth": 12.0,
                    }
                ],
            }
        ]
        predictions = {
            "images/test/frame.png": [
                {
                    "class_id": 0,
                    "class_name": "positive_goal",
                    "confidence": 0.8,
                    "bbox": [4.0, 4.0, 10.0, 10.0],
                }
            ]
        }
        metrics, errors = evaluate_predictions(
            metadata, predictions, ["positive_goal"], config
        )
        self.assertEqual(metrics["per_class"]["positive_goal"]["recall"], 1.0)
        self.assertEqual(metrics["recall_by_visibility_fraction"][3]["recall"], 1.0)
        self.assertEqual(metrics["calibration"]["predictions"], 1)
        self.assertEqual(errors["false_positives"], [])
        self.assertEqual(errors["false_negatives"], [])

    def test_unknown_visibility_lands_in_the_unknown_bucket(self) -> None:
        """Merged external labels carry explicit nulls; unknown must not crash or count as 0.0."""
        config = DetectorConfig(
            data="data.yaml",
            model="yolo26n-seg.yaml",
            pretrained_checkpoint=None,
            resume_checkpoint=None,
            output_root="runs",
            experiment_name="test",
            seed=7,
            epochs=1,
            image_size=64,
            batch_size=1,
            workers=0,
            device="cpu",
            confidence_threshold=0.25,
            nms_iou=0.7,
            match_iou=0.5,
            small_area_fraction=0.1,
            distant_depth_threshold=10.0,
            calibration_bins=5,
            augmentation={},
        )
        metadata = [
            {
                "image": "images/test/frame.png",
                "objects": [
                    {
                        "class_id": 0,
                        "class_name": "positive_goal",
                        "bounding_box": {
                            "x_min": 0.0,
                            "y_min": 0.0,
                            "x_max": 8.0,
                            "y_max": 8.0,
                            "image_width": 16,
                            "image_height": 16,
                        },
                        "area_pixels": 64,
                        "visibility_fraction": None,
                        "occlusion_fraction": None,
                        "relative_depth": None,
                    }
                ],
            }
        ]

        metrics, _ = evaluate_predictions(metadata, {}, ["positive_goal"], config)

        buckets = {
            item["bucket"]: item for item in metrics["recall_by_visibility_fraction"]
        }
        self.assertEqual(buckets["unknown"]["objects"], 1)
        self.assertEqual(buckets["0.00-0.25"]["objects"], 0)
        self.assertIsNone(buckets["0.00-0.25"]["recall"])

    def test_recall_by_exact_class_exposes_a_starved_sibling_a_merged_class_hides(self) -> None:
        """configs/ontology.json merges GoodGoalMulti/GoodGoalMultiBounce/GrowGoal/ShrinkGoal into
        one detector class 'GoodGoalMulti' because they are pixel-identical. per_class recall for
        that merged class can look perfect while one rare sibling is never actually found —
        recall_by_exact_class is what catches that."""
        config = DetectorConfig(
            data="data.yaml",
            model="yolo26n-seg.yaml",
            pretrained_checkpoint=None,
            resume_checkpoint=None,
            output_root="runs",
            experiment_name="test",
            seed=7,
            epochs=1,
            image_size=64,
            batch_size=1,
            workers=0,
            device="cpu",
            confidence_threshold=0.25,
            nms_iou=0.7,
            match_iou=0.5,
            small_area_fraction=0.1,
            distant_depth_threshold=10.0,
            calibration_bins=5,
            augmentation={},
        )

        def merged_object(instance_id: str, exact_class: str, x_min: float) -> dict:
            return {
                "instance_id": instance_id,
                "class_id": 0,
                "class_name": "GoodGoalMulti",
                "exact_simulator_class": exact_class,
                "bounding_box": {
                    "x_min": x_min,
                    "y_min": 0.0,
                    "x_max": x_min + 8.0,
                    "y_max": 8.0,
                    "image_width": 32,
                    "image_height": 16,
                },
                "area_pixels": 64,
                "visibility_fraction": None,
                "occlusion_fraction": None,
                "relative_depth": None,
            }

        metadata = [
            {
                "image": "images/test/frame.png",
                "objects": [
                    merged_object("a", "GoodGoalMulti", 0.0),
                    merged_object("b", "ShrinkGoal", 16.0),
                ],
            }
        ]
        # Only the GoodGoalMulti box gets a matching prediction; ShrinkGoal is never detected.
        predictions = {
            "images/test/frame.png": [
                {"class_id": 0, "class_name": "GoodGoalMulti", "confidence": 0.9,
                 "bbox": [0.0, 0.0, 8.0, 8.0]},
            ]
        }

        metrics, _ = evaluate_predictions(metadata, predictions, ["GoodGoalMulti"], config)

        self.assertEqual(metrics["per_class"]["GoodGoalMulti"]["recall"], 0.5)
        by_exact = metrics["recall_by_exact_class"]
        self.assertEqual(by_exact["GoodGoalMulti"]["recall"], 1.0)
        self.assertEqual(by_exact["ShrinkGoal"]["recall"], 0.0)


if __name__ == "__main__":
    unittest.main()
