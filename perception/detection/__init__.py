"""YOLO instance-segmentation training and evaluation."""

from perception.detection.config import DetectorConfig
from perception.detection.evaluation import evaluate_predictions
from perception.detection.results import detection_frame_from_result
from perception.detection.training import run_experiment

__all__ = [
    "DetectorConfig",
    "detection_frame_from_result",
    "evaluate_predictions",
    "run_experiment",
]
