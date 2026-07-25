from __future__ import annotations

import math
from typing import Any

from perception.detection.config import DetectorConfig


def evaluate_predictions(
    metadata_records: list[dict[str, Any]],
    predictions_by_image: dict[str, list[dict[str, Any]]],
    class_names: list[str],
    config: DetectorConfig,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]] | dict[str, Any]]]:
    ground_truths: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    image_sizes: dict[str, tuple[int, int]] = {}
    for record in metadata_records:
        image = record["image"]
        for item in record.get("objects", []):
            box = item.get("bounding_box")
            if box:
                image_sizes[image] = (box["image_width"], box["image_height"])
                ground_truths.append({**item, "image": image, "matched": False})
        for prediction in predictions_by_image.get(image, []):
            predictions.append({**prediction, "image": image, "matched": False})

    for image in {item["image"] for item in ground_truths + predictions}:
        gt_indices = [
            i for i, item in enumerate(ground_truths) if item["image"] == image
        ]
        pred_indices = [
            i for i, item in enumerate(predictions) if item["image"] == image
        ]
        pairs = sorted(
            (
                _box_iou(predictions[p]["bbox"], _metadata_box(ground_truths[g])),
                p,
                g,
            )
            for p in pred_indices
            for g in gt_indices
            if predictions[p]["class_id"] == ground_truths[g]["class_id"]
        )
        for iou, pred_index, gt_index in reversed(pairs):
            if iou < config.match_iou:
                break
            if predictions[pred_index]["matched"] or ground_truths[gt_index]["matched"]:
                continue
            predictions[pred_index]["matched"] = True
            ground_truths[gt_index]["matched"] = True

    per_class: dict[str, dict[str, int | float | None]] = {}
    for class_id, class_name in enumerate(class_names):
        truth = [item for item in ground_truths if item["class_id"] == class_id]
        predicted = [item for item in predictions if item["class_id"] == class_id]
        true_positive = sum(bool(item["matched"]) for item in predicted)
        false_positive = len(predicted) - true_positive
        false_negative = len(truth) - sum(bool(item["matched"]) for item in truth)
        per_class[class_name] = {
            "true_positives": true_positive,
            "false_positives": false_positive,
            "false_negatives": false_negative,
            "precision": _ratio(true_positive, true_positive + false_positive),
            "recall": _ratio(true_positive, true_positive + false_negative),
        }

    crowded_keys = {
        (item["image"], item["class_id"])
        for item in ground_truths
        if sum(
            other["image"] == item["image"] and other["class_id"] == item["class_id"]
            for other in ground_truths
        )
        > 1
    }
    small = [
        item
        for item in ground_truths
        if item.get("area_pixels") is not None
        and item["image"] in image_sizes
        and item["area_pixels"] / math.prod(image_sizes[item["image"]])
        < config.small_area_fraction
    ]
    distant = [
        item
        for item in ground_truths
        if item.get("relative_depth") is not None
        and item["relative_depth"] >= config.distant_depth_threshold
    ]
    calibration = _calibration(predictions, config.calibration_bins)
    metrics = {
        "confidence_threshold": config.confidence_threshold,
        "match_iou": config.match_iou,
        "per_class": per_class,
        "small_object_recall": _recall(small),
        "recall_by_visibility_fraction": _fraction_recall(
            ground_truths, "visibility_fraction"
        ),
        "recall_by_occlusion_level": _fraction_recall(
            ground_truths, "occlusion_fraction"
        ),
        "crowded_same_class_recall": _recall(
            [
                item
                for item in ground_truths
                if (item["image"], item["class_id"]) in crowded_keys
            ]
        ),
        "calibration": calibration,
    }
    false_positives = [
        _public_prediction(item) for item in predictions if not item["matched"]
    ]
    false_negatives = [
        _public_truth(item) for item in ground_truths if not item["matched"]
    ]
    errors: dict[str, list[dict[str, Any]] | dict[str, Any]] = {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "class_confusion": _confusion(
            ground_truths, predictions, class_names, config.match_iou
        ),
        "transparent_object_failures": [
            item
            for item in false_negatives + false_positives
            if item.get("class_name") == "transparent_wall"
        ],
        "small_and_distant_object_failures": {
            "small_false_negatives": [
                _public_truth(item) for item in small if not item["matched"]
            ],
            "distant_false_negatives": [
                _public_truth(item) for item in distant if not item["matched"]
            ],
        },
        "calibration": calibration,
    }
    return metrics, errors


def _fraction_recall(
    items: list[dict[str, Any]], field: str
) -> list[dict[str, int | float | str | None]]:
    buckets: list[tuple[str, float | None, float | None]] = [
        ("0.00-0.25", 0.0, 0.25),
        ("0.25-0.50", 0.25, 0.5),
        ("0.50-0.75", 0.5, 0.75),
        ("0.75-1.00", 0.75, 1.0000001),
        ("unknown", None, None),
    ]
    output = []
    for label, lower, upper in buckets:
        selected = [
            item
            for item in items
            if (
                item.get(field) is None
                if lower is None
                else lower <= item.get(field, -1) < upper
            )
        ]
        output.append({"bucket": label, **_recall(selected)})
    return output


def _recall(items: list[dict[str, Any]]) -> dict[str, int | float | None]:
    matched = sum(bool(item["matched"]) for item in items)
    return {
        "objects": len(items),
        "matched": matched,
        "recall": _ratio(matched, len(items)),
    }


def _calibration(predictions: list[dict[str, Any]], bins: int) -> dict[str, Any]:
    if not predictions:
        return {
            "predictions": 0,
            "expected_calibration_error": None,
            "brier_score": None,
            "bins": [],
        }
    output = []
    ece = 0.0
    for index in range(bins):
        selected = [
            item
            for item in predictions
            if min(int(item["confidence"] * bins), bins - 1) == index
        ]
        if not selected:
            continue
        accuracy = sum(bool(item["matched"]) for item in selected) / len(selected)
        mean_confidence = sum(item["confidence"] for item in selected) / len(selected)
        ece += len(selected) / len(predictions) * abs(accuracy - mean_confidence)
        output.append(
            {
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "count": len(selected),
                "accuracy": accuracy,
                "mean_confidence": mean_confidence,
            }
        )
    brier = sum(
        (item["confidence"] - float(bool(item["matched"]))) ** 2 for item in predictions
    ) / len(predictions)
    return {
        "predictions": len(predictions),
        "expected_calibration_error": ece,
        "brier_score": brier,
        "bins": output,
    }


def _confusion(
    truths: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    class_names: list[str],
    threshold: float,
) -> dict[str, Any]:
    labels = [*class_names, "__background__", "__missed__"]
    matrix = {truth: {predicted: 0 for predicted in labels} for truth in labels}
    used_truth: set[int] = set()
    used_prediction: set[int] = set()
    pairs = sorted(
        (
            _box_iou(prediction["bbox"], _metadata_box(truth)),
            pred_index,
            truth_index,
        )
        for pred_index, prediction in enumerate(predictions)
        for truth_index, truth in enumerate(truths)
        if prediction["image"] == truth["image"]
    )
    for iou, pred_index, truth_index in reversed(pairs):
        if iou < threshold:
            break
        if pred_index in used_prediction or truth_index in used_truth:
            continue
        used_prediction.add(pred_index)
        used_truth.add(truth_index)
        matrix[truths[truth_index]["class_name"]][
            predictions[pred_index]["class_name"]
        ] += 1
    for index, truth in enumerate(truths):
        if index not in used_truth:
            matrix[truth["class_name"]]["__missed__"] += 1
    for index, prediction in enumerate(predictions):
        if index not in used_prediction:
            matrix["__background__"][prediction["class_name"]] += 1
    return {"labels": labels, "matrix": matrix}


def _metadata_box(item: dict[str, Any]) -> list[float]:
    box = item["bounding_box"]
    return [box["x_min"], box["y_min"], box["x_max"], box["y_max"]]


def _box_iou(left: list[float], right: list[float]) -> float:
    x_min = max(left[0], right[0])
    y_min = max(left[1], right[1])
    x_max = min(left[2], right[2])
    y_max = min(left[3], right[3])
    intersection = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _public_prediction(item: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item[key]
        for key in ("image", "class_id", "class_name", "confidence", "bbox")
    }


def _public_truth(item: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item.get(key)
        for key in (
            "image",
            "instance_id",
            "class_id",
            "class_name",
            "bounding_box",
            "area_pixels",
            "visibility_fraction",
            "occlusion_fraction",
            "relative_depth",
        )
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None
