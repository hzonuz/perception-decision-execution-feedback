from __future__ import annotations

from typing import Any


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
