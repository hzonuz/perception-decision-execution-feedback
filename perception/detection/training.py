"""YOLO segmentation training orchestration."""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from perception.contracts import DetectionFrame, FrameStamp
from perception.detection.config import DetectorConfig
from perception.detection.evaluation import evaluate_predictions
from perception.detection.results import detection_frame_from_result, result_predictions


def run_experiment(
    config_path: str | Path,
    *,
    data: str | None = None,
    experiment_name: str | None = None,
    resume_checkpoint: str | None = None,
) -> Path:
    """Train, evaluate on the held-out test split, and export one detector record."""
    config = DetectorConfig.from_json_file(config_path)
    config = replace(
        config,
        data=data or config.data,
        experiment_name=experiment_name or config.experiment_name,
        resume_checkpoint=resume_checkpoint or config.resume_checkpoint,
    )
    config.validate()
    data_yaml = Path(config.data).resolve()
    dataset_root, class_names, test_records = _validate_dataset(data_yaml)

    cache = Path(".cache/detector").resolve()
    (cache / "Ultralytics").mkdir(parents=True, exist_ok=True)
    (cache / "matplotlib").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(cache))
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))

    import numpy as np
    import torch
    import torchvision
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils import SETTINGS

    SETTINGS.update({"sync": False})
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    output_root = Path(config.output_root).resolve()
    expected = output_root / config.experiment_name
    if not config.resume_checkpoint and expected.exists():
        raise FileExistsError(f"experiment directory already exists: {expected}")

    model = YOLO(
        config.resume_checkpoint or config.pretrained_checkpoint or config.model
    )
    train_args: dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": config.epochs,
        "imgsz": config.image_size,
        "batch": config.batch_size,
        "workers": config.workers,
        "device": config.device,
        "seed": config.seed,
        "deterministic": True,
        "project": str(output_root),
        "name": config.experiment_name,
        "exist_ok": False,
        "plots": True,
        "save": True,
        **config.augmentation,
    }
    if config.resume_checkpoint:
        train_args["resume"] = config.resume_checkpoint
    else:
        train_args["pretrained"] = config.pretrained_checkpoint is not None
    model.train(**train_args)

    experiment = Path(model.trainer.save_dir).resolve()
    checkpoint = experiment / "weights" / "best.pt"
    if not checkpoint.is_file():
        checkpoint = experiment / "weights" / "last.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError("training did not produce a checkpoint")

    inference_model = YOLO(str(checkpoint))
    builtin = inference_model.val(
        data=str(data_yaml),
        split="test",
        imgsz=config.image_size,
        batch=config.batch_size,
        workers=config.workers,
        device=config.device,
        project=str(experiment),
        name="evaluation",
        exist_ok=True,
        plots=True,
    )
    image_paths = [dataset_root / record["image"] for record in test_records]
    results = inference_model.predict(
        source=[str(path) for path in image_paths],
        imgsz=config.image_size,
        conf=config.confidence_threshold,
        iou=config.nms_iou,
        device=config.device,
        verbose=False,
    )
    predictions = {
        record["image"]: result_predictions(result)
        for record, result in zip(test_records, results, strict=True)
    }
    sliced_metrics, errors = evaluate_predictions(
        test_records, predictions, class_names, config
    )
    _write_json(
        experiment / "metrics.json",
        {
            "ultralytics": _ultralytics_metrics(builtin),
            "object_level_at_confidence_threshold": sliced_metrics,
        },
    )
    error_root = experiment / "error_analysis"
    for name, payload in errors.items():
        _write_json(error_root / f"{name}.json", payload)
    rendered = error_root / "rendered_predictions"
    rendered.mkdir(parents=True, exist_ok=True)
    for record, result in zip(test_records, results, strict=True):
        result.save(filename=str(rendered / Path(record["image"]).name))

    artifact_root = experiment / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, artifact_root / "inference.pt")
    first_record = test_records[0]
    first_frame = detection_frame_from_result(
        results[0],
        FrameStamp(
            episode_id=first_record["episode_id"],
            frame_index=first_record["frame_index"],
            timestamp_seconds=first_record["timestamp_seconds"],
        ),
        experiment,
        configuration_id=config.experiment_name,
        producer_version=ultralytics.__version__,
    )
    inference_root = experiment / "inference"
    inference_root.mkdir(parents=True, exist_ok=True)
    (inference_root / "detection_frame.json").write_text(
        first_frame.to_json(), encoding="utf-8"
    )
    DetectionFrame.from_json_file(inference_root / "detection_frame.json")

    _write_json(
        experiment / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "ultralytics": ultralytics.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "mps_built": bool(
                getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_built()
            ),
            "mps_available": bool(
                getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
            ),
            "packages": {
                distribution.metadata["Name"]: distribution.version
                for distribution in importlib.metadata.distributions()
                if distribution.metadata["Name"]
            },
        },
    )
    _write_json(
        experiment / "manifest.json",
        {
            "config": {
                field: getattr(config, field)
                for field in config.__dataclass_fields__
            },
            "data_yaml": str(data_yaml),
            "checkpoint": str(artifact_root / "inference.pt"),
            "test_frames": len(test_records),
            "seed": config.seed,
            "command": " ".join(sys.argv),
        },
    )
    return experiment


def _validate_dataset(data_yaml: Path) -> tuple[Path, list[str], list[dict[str, Any]]]:
    if not data_yaml.is_file():
        raise FileNotFoundError(f"dataset YAML not found: {data_yaml}")
    import yaml

    raw = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("names"), dict):
        raise ValueError("dataset YAML must contain an indexed names mapping")
    root = Path(raw["path"])
    root = (data_yaml.parent / root).resolve() if not root.is_absolute() else root.resolve()
    names = [
        str(name)
        for _, name in sorted(raw["names"].items(), key=lambda item: int(item[0]))
    ]
    splits = json.loads((root / "splits.json").read_text(encoding="utf-8"))
    episode_sets = [
        set(splits["episodes"][split]) for split in ("train", "val", "test")
    ]
    if any(
        episode_sets[left] & episode_sets[right]
        for left, right in ((0, 1), (0, 2), (1, 2))
    ):
        raise ValueError("dataset has episode leakage across splits")
    if any(not episode_set for episode_set in episode_sets):
        raise ValueError("train, val, and test must each contain an independent episode")
    records = [
        json.loads(line)
        for line in (root / "metadata/test.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("held-out test metadata is empty")
    for record in records:
        if not (root / record["image"]).is_file():
            raise FileNotFoundError(f"test image is missing: {record['image']}")
    return root, names, records


def _ultralytics_metrics(metrics: Any) -> dict[str, Any]:
    return _sanitize(
        {
            "box": {
                "precision": metrics.box.mp,
                "recall": metrics.box.mr,
                "map50": metrics.box.map50,
                "map50_95": metrics.box.map,
            },
            "mask": {
                "precision": metrics.seg.mp,
                "recall": metrics.seg.mr,
                "map50": metrics.seg.map50,
                "map50_95": metrics.seg.map,
            },
            "per_class": metrics.summary(),
        }
    )


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_sanitize(item) for item in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_sanitize(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
