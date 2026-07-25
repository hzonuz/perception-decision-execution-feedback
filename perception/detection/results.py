"""Ultralytics result conversion to shared detector contracts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from perception.contracts import (
    SCHEMA_VERSION,
    BoundingBox2D,
    CategoricalDistribution,
    DetectionFrame,
    DetectorObject,
    FrameStamp,
    MaskEncoding,
    MaskReference,
    NormalizedPoint,
    ProbabilityEntry,
    SourceMetadata,
    SourceType,
    Uncertainty,
)


def detection_frame_from_result(
    result: Any,
    stamp: FrameStamp,
    artifact_root: str | Path,
    *,
    configuration_id: str,
    producer_version: str,
) -> DetectionFrame:
    """Convert one result without exposing privileged simulator state."""
    height, width = (int(value) for value in result.orig_shape)
    source = SourceMetadata(
        source=SourceType.DETECTOR,
        producer="ultralytics_yolo_segmentation",
        producer_version=producer_version,
        privileged=False,
        configuration_id=configuration_id,
    )
    root = Path(artifact_root).resolve()
    masks_dir = root / "inference" / "masks"
    boxes = to_list(result.boxes.xyxy) if result.boxes is not None else []
    confidences = to_list(result.boxes.conf) if result.boxes is not None else []
    classes = to_list(result.boxes.cls) if result.boxes is not None else []
    objects: list[DetectorObject] = []
    for index, (raw_box, raw_confidence, raw_class) in enumerate(
        zip(boxes, confidences, classes, strict=True)
    ):
        x_min, y_min, x_max, y_max = (
            max(0.0, min(float(value), width if axis % 2 == 0 else height))
            for axis, value in enumerate(raw_box)
        )
        if x_min >= x_max or y_min >= y_max:
            continue
        confidence = max(0.0, min(float(raw_confidence), 1.0))
        class_id = int(raw_class)
        class_name = str(result.names[class_id])
        mask_reference = None
        if result.masks is not None and index < len(result.masks.data):
            from PIL import Image

            masks_dir.mkdir(parents=True, exist_ok=True)
            mask_path = masks_dir / f"det_{index:04d}.png"
            mask = result.masks.data[index]
            if hasattr(mask, "cpu"):
                mask = mask.cpu()
            if hasattr(mask, "numpy"):
                mask = mask.numpy()
            Image.fromarray((mask > 0.5).astype("uint8") * 255).resize(
                (width, height), Image.Resampling.NEAREST
            ).save(mask_path)
            mask_reference = MaskReference(
                path=mask_path.relative_to(root).as_posix(),
                encoding=MaskEncoding.PNG,
                width=width,
                height=height,
                instance_value=1,
            )
        entropy = -sum(
            probability * math.log(probability)
            for probability in (confidence, 1.0 - confidence)
            if probability > 0
        )
        objects.append(
            DetectorObject(
                detector_id=f"det_{index:04d}",
                semantic_distribution=CategoricalDistribution(
                    entries=(
                        ProbabilityEntry(label=class_name, probability=confidence),
                    ),
                    unknown_probability=1.0 - confidence,
                ),
                bounding_box=BoundingBox2D(
                    x_min=x_min,
                    y_min=y_min,
                    x_max=x_max,
                    y_max=y_max,
                    image_width=width,
                    image_height=height,
                ),
                instance_mask=mask_reference,
                normalized_location=NormalizedPoint(
                    x=(x_min + x_max) / (2 * width),
                    y=(y_min + y_max) / (2 * height),
                ),
                confidence=confidence,
                uncertainty=Uncertainty(entropy=entropy),
                source=source,
            )
        )
    return DetectionFrame(
        schema_version=SCHEMA_VERSION,
        stamp=stamp,
        image_width=width,
        image_height=height,
        objects=tuple(objects),
        source=source,
    )


def result_predictions(result: Any) -> list[dict[str, Any]]:
    if result.boxes is None:
        return []
    return [
        {
            "class_id": int(class_id),
            "class_name": str(result.names[int(class_id)]),
            "confidence": float(confidence),
            "bbox": [float(value) for value in box],
        }
        for box, confidence, class_id in zip(
            to_list(result.boxes.xyxy),
            to_list(result.boxes.conf),
            to_list(result.boxes.cls),
            strict=True,
        )
    ]


def to_list(value: Any) -> list[Any]:
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)
