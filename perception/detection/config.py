from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


_AUGMENTATION_KEYS = {
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "degrees",
    "translate",
    "scale",
    "shear",
    "perspective",
    "flipud",
    "fliplr",
    "mosaic",
    "mixup",
    "copy_paste",
    "erasing",
}


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    data: str
    model: str
    pretrained_checkpoint: str | None
    resume_checkpoint: str | None
    output_root: str
    experiment_name: str
    seed: int
    epochs: int
    image_size: int
    batch_size: int
    workers: int
    device: str
    confidence_threshold: float
    nms_iou: float
    match_iou: float
    small_area_fraction: float
    distant_depth_threshold: float
    calibration_bins: int
    augmentation: dict[str, float]

    @classmethod
    def from_json_file(cls, path: str | Path) -> "DetectorConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("detector configuration must be a JSON object")
        try:
            config = cls(**raw)
        except TypeError as exc:
            raise ValueError(f"invalid detector configuration: {exc}") from exc
        config.validate()
        return config

    def validate(self) -> None:
        for name in ("data", "model", "output_root", "experiment_name", "device"):
            if (
                not isinstance(getattr(self, name), str)
                or not getattr(self, name).strip()
            ):
                raise ValueError(f"{name} must be a non-empty string")
        for name in (
            "seed",
            "epochs",
            "image_size",
            "batch_size",
            "workers",
            "calibration_bins",
        ):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.epochs <= 0 or self.image_size <= 0 or self.batch_size == 0:
            raise ValueError(
                "epochs and image_size must be positive; batch_size must be non-zero"
            )
        if self.workers < 0 or self.calibration_bins <= 0:
            raise ValueError(
                "workers must be non-negative and calibration_bins positive"
            )
        for name in ("confidence_threshold", "nms_iou", "match_iou"):
            value = getattr(self, name)
            if not isinstance(value, int | float) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 0 < self.small_area_fraction <= 1:
            raise ValueError("small_area_fraction must be in (0, 1]")
        if self.distant_depth_threshold < 0:
            raise ValueError("distant_depth_threshold must be non-negative")
        if not isinstance(self.augmentation, dict):
            raise ValueError("augmentation must be an object")
        unknown = set(self.augmentation) - _AUGMENTATION_KEYS
        if unknown:
            raise ValueError(f"unknown augmentation settings: {sorted(unknown)}")
        if any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in self.augmentation.values()
        ):
            raise ValueError("augmentation values must be finite numbers")
