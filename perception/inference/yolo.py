from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ultralytics.utils.ops import scale_masks

from perception.contracts import FrameStamp, SourceMetadata, SourceType
from perception.inference.detector import (
    Detection,
    Detector,
    Frame,
    Timing,
    build_detection,
    resolve_device,
)


class YoloDetector(Detector):
    name = "ultralytics_yolo_segmentation"

    def __init__(
        self,
        weights: str | Path,
        *,
        device: str = "auto",
        image_size: int = 256,
        confidence_threshold: float = 0.25,
        max_detections: int = 300,
        deterministic: bool = False,
        seed: int = 0,
    ) -> None:
        import torch
        import ultralytics
        from ultralytics import YOLO

        if deterministic:
            torch.manual_seed(seed)
            torch.use_deterministic_algorithms(True, warn_only=True)
        self.device = resolve_device(device, deterministic=deterministic)
        self.model = YOLO(str(weights))
        if self.model.task != "segment":
            raise ValueError(
                f"{weights} is a {self.model.task!r} model, expected 'segment'"
            )
        self.class_names = tuple(self.model.names[i] for i in sorted(self.model.names))
        self.version = ultralytics.__version__
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold
        self.max_detections = max_detections
        self.source = SourceMetadata(
            SourceType.DETECTOR,
            self.name,
            self.version,
            configuration_id=Path(weights).as_posix(),
        )
        # Warm up so first-call setup (weights to device, kernel compilation) stays out of latency.
        self.model.predict(
            np.zeros((image_size, image_size, 3), np.uint8),
            imgsz=image_size,
            device=self.device,
            verbose=False,
        )

    def detect_batch(self, frames: Sequence[Frame]) -> list[Detection]:
        if not frames:
            return []
        results = self.model.predict(
            # Ultralytics reads numpy arrays as BGR (OpenCV order) — the order it trained on.
            [np.ascontiguousarray(frame.image[..., ::-1]) for frame in frames],
            imgsz=self.image_size,
            conf=self.confidence_threshold,
            max_det=self.max_detections,
            device=self.device,
            retina_masks=True,
            verbose=False,
        )
        detections = []
        for frame, result in zip(frames, results, strict=True):
            start = time.perf_counter()
            detection = detection_from_result(result, frame.stamp, self.source)
            convert_ms = (time.perf_counter() - start) * 1000
            timing = replace(
                detection.timing,
                postprocess_ms=detection.timing.postprocess_ms + convert_ms,
            )
            detections.append(replace(detection, timing=timing))
        return detections


def detection_from_result(
    result: Any, stamp: FrameStamp, source: SourceMetadata
) -> Detection:
    height, width = (int(value) for value in result.orig_shape)
    boxes = _numpy(result.boxes.xyxy) if result.boxes is not None else np.zeros((0, 4))
    confidences = _numpy(result.boxes.conf) if result.boxes is not None else np.zeros(0)
    classes = (
        _numpy(result.boxes.cls).astype(int)
        if result.boxes is not None
        else np.zeros(0, int)
    )
    masks = None
    if result.masks is not None and len(result.masks.data):
        masks = _numpy(result.masks.data) > 0.5
        if masks.shape[1:] != (
            height,
            width,
        ):  # predicted without retina_masks: undo the letterbox

            scaled = scale_masks(
                torch.as_tensor(masks[None], dtype=torch.float32), (height, width)
            )
            masks = scaled[0].numpy() > 0.5

    grouped: dict[
        tuple[float, ...], tuple[tuple[float, ...], dict[str, float], np.ndarray | None]
    ] = {}
    for index, (box, confidence, class_id) in enumerate(
        zip(boxes, confidences, classes, strict=True)
    ):
        key = tuple(float(v) for v in box)
        _, scores, _ = grouped.setdefault(
            key, (key, {}, None if masks is None else masks[index])
        )
        label = str(result.names[int(class_id)])
        scores[label] = max(scores.get(label, 0.0), float(confidence))
    speed = getattr(result, "speed", None) or {}  # Ultralytics' per-image stage split
    timing = Timing(
        *(
            float(speed.get(k) or 0.0)
            for k in ("preprocess", "inference", "postprocess")
        )
    )
    return build_detection(stamp, width, height, list(grouped.values()), source, timing)


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "cpu"):
        value = value.cpu().numpy()
    return (
        np.asarray(value, dtype=float) if not isinstance(value, np.ndarray) else value
    )
