"""Dataset configuration and internal records."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from perception.contracts import CaptureFrame, ObjectAnnotation


SPLITS = ("train", "val", "test")
Point = tuple[int, int]
Polygon = tuple[Point, ...]


class DatasetValidationError(ValueError):
    """Raised after an integrity report has been written."""


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    group_by: str
    random_seed: int
    train_fraction: float
    val_fraction: float
    test_fraction: float
    class_mode: str
    allow_empty_annotations: bool
    box_mask_tolerance_pixels: float
    moving_speed_threshold: float
    small_area_fraction: float
    large_area_fraction: float
    visualization_samples_per_split: int
    export_coco: bool

    @classmethod
    def from_json_file(cls, path: str | Path) -> "DatasetConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        split = _object(raw, "split")
        ratios = _object(split, "ratios")
        config = cls(
            group_by=_string(split, "group_by"),
            random_seed=_integer(split, "random_seed"),
            train_fraction=_number(ratios, "train"),
            val_fraction=_number(ratios, "val"),
            test_fraction=_number(ratios, "test"),
            class_mode=_string(raw, "class_mode"),
            allow_empty_annotations=_boolean(raw, "allow_empty_annotations"),
            box_mask_tolerance_pixels=_number(raw, "box_mask_tolerance_pixels"),
            moving_speed_threshold=_number(raw, "moving_speed_threshold"),
            small_area_fraction=_number(raw, "small_area_fraction"),
            large_area_fraction=_number(raw, "large_area_fraction"),
            visualization_samples_per_split=_integer(
                raw, "visualization_samples_per_split"
            ),
            export_coco=_boolean(raw, "export_coco"),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.group_by not in {"episode", "arena", "seed"}:
            raise ValueError("split.group_by must be episode, arena, or seed")
        if self.class_mode not in {"semantic", "exact"}:
            raise ValueError("class_mode must be semantic or exact")
        ratios = self.ratios
        if any(value < 0 for value in ratios.values()) or not math.isclose(
            sum(ratios.values()), 1.0, abs_tol=1e-9
        ):
            raise ValueError("split ratios must be non-negative and sum to 1")
        if self.box_mask_tolerance_pixels < 0 or self.moving_speed_threshold < 0:
            raise ValueError("pixel tolerance and moving threshold must be non-negative")
        if not 0 < self.small_area_fraction < self.large_area_fraction <= 1:
            raise ValueError("object area thresholds must satisfy 0 < small < large <= 1")
        if self.visualization_samples_per_split < 0:
            raise ValueError("visualization sample count must be non-negative")

    @property
    def ratios(self) -> dict[str, float]:
        return {
            "train": self.train_fraction,
            "val": self.val_fraction,
            "test": self.test_fraction,
        }


@dataclass(frozen=True, slots=True)
class Issue:
    severity: str
    code: str
    message: str
    episode_id: str | None = None
    frame_index: int | None = None
    instance_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "episode_id": self.episode_id,
            "frame_index": self.frame_index,
            "instance_id": self.instance_id,
        }


@dataclass(frozen=True, slots=True)
class DatasetObject:
    annotation: ObjectAnnotation
    class_name: str
    class_id: int
    polygons: tuple[Polygon, ...]
    area_pixels: int
    mask_box: tuple[int, int, int, int]
    motion: str


@dataclass(frozen=True, slots=True)
class Sample:
    capture_root: Path
    annotation_path: Path
    frame: CaptureFrame
    image_path: Path
    objects: tuple[DatasetObject, ...]

    @property
    def stem(self) -> str:
        safe_episode = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in self.frame.episode_id
        )
        return f"{safe_episode}__{self.frame.stamp.frame_index:06d}"


def _object(value: dict[str, Any], name: str) -> dict[str, Any]:
    item = value.get(name)
    if not isinstance(item, dict):
        raise ValueError(f"{name} must be an object")
    return item


def _string(value: dict[str, Any], name: str) -> str:
    item = value.get(name)
    if not isinstance(item, str) or not item:
        raise ValueError(f"{name} must be a non-empty string")
    return item


def _integer(value: dict[str, Any], name: str) -> int:
    item = value.get(name)
    if type(item) is not int:
        raise ValueError(f"{name} must be an integer")
    return item


def _number(value: dict[str, Any], name: str) -> float:
    item = value.get(name)
    if type(item) not in (int, float) or not math.isfinite(item):
        raise ValueError(f"{name} must be a finite number")
    return float(item)


def _boolean(value: dict[str, Any], name: str) -> bool:
    item = value.get(name)
    if type(item) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return item
