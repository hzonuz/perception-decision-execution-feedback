from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from perception.contracts import DetectionFrame, FrameStamp
from perception.detection import (
    DetectorConfig,
    detection_frame_from_result,
    evaluate_predictions,
)


class DetectorTrainingTests(unittest.TestCase):
    def test_contract_conversion_and_metadata_metrics_without_model_download(self) -> None:
        config = DetectorConfig(
            data="data.yaml",
            model="yolo11n-seg.yaml",
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
        result = SimpleNamespace(
            orig_shape=(16, 16),
            boxes=SimpleNamespace(
                xyxy=[[4.0, 4.0, 10.0, 10.0]],
                conf=[0.8],
                cls=[0],
            ),
            masks=None,
            names={0: "positive_goal"},
        )
        with tempfile.TemporaryDirectory() as temporary:
            frame = detection_frame_from_result(
                result,
                FrameStamp("episode-test", 2, 0.04),
                Path(temporary),
                configuration_id="unit-test",
                producer_version="8.4.102",
            )
            self.assertEqual(DetectionFrame.from_json(frame.to_json()), frame)
            self.assertAlmostEqual(
                frame.objects[0].semantic_distribution.unknown_probability, 0.2
            )
            self.assertFalse(frame.source.privileged)

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


if __name__ == "__main__":
    unittest.main()
