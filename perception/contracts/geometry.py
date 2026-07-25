from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from perception.contracts.base import ContractValidationError, JsonContract
from perception.contracts.common import (
    _require_finite,
    _require_nonempty,
    _require_nonnegative_finite,
)


@dataclass(frozen=True, slots=True)
class Vector2(JsonContract):
    x: float
    y: float

    def __post_init__(self) -> None:
        _require_finite(self.x, "x")
        _require_finite(self.y, "y")


@dataclass(frozen=True, slots=True)
class Vector3(JsonContract):
    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        _require_finite(self.x, "x")
        _require_finite(self.y, "y")
        _require_finite(self.z, "z")


@dataclass(frozen=True, slots=True)
class Quaternion(JsonContract):
    x: float
    y: float
    z: float
    w: float

    def __post_init__(self) -> None:
        _require_finite(self.x, "x")
        _require_finite(self.y, "y")
        _require_finite(self.z, "z")
        _require_finite(self.w, "w")
        norm_squared = self.x**2 + self.y**2 + self.z**2 + self.w**2
        if norm_squared <= 1e-12:
            raise ContractValidationError("quaternion must have non-zero norm")


@dataclass(frozen=True, slots=True)
class NormalizedPoint(JsonContract):
    x: float
    y: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.x <= 1.0 or not 0.0 <= self.y <= 1.0:
            raise ContractValidationError(
                "normalized point coordinates must be in [0, 1]"
            )


@dataclass(frozen=True, slots=True)
class BoundingBox2D(JsonContract):
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    image_width: int
    image_height: int

    def __post_init__(self) -> None:
        if self.image_width <= 0 or self.image_height <= 0:
            raise ContractValidationError("image dimensions must be positive")
        for name in ("x_min", "y_min", "x_max", "y_max"):
            _require_finite(getattr(self, name), name)
        if not 0 <= self.x_min < self.x_max <= self.image_width:
            raise ContractValidationError(
                "horizontal box coordinates are outside the image"
            )
        if not 0 <= self.y_min < self.y_max <= self.image_height:
            raise ContractValidationError(
                "vertical box coordinates are outside the image"
            )


class ImageEncoding(str, Enum):
    PNG = "png"
    JPEG = "jpeg"


@dataclass(frozen=True, slots=True)
class ImageReference(JsonContract):
    path: str
    width: int
    height: int
    encoding: ImageEncoding
    sha256: str | None = None

    def __post_init__(self) -> None:
        _require_relative_path(self.path, "path")
        if self.width <= 0 or self.height <= 0:
            raise ContractValidationError("image dimensions must be positive")
        if self.sha256 is not None:
            if len(self.sha256) != 64 or any(
                ch not in "0123456789abcdef" for ch in self.sha256
            ):
                raise ContractValidationError(
                    "sha256 must be 64 lowercase hexadecimal characters"
                )


class MaskEncoding(str, Enum):
    PNG = "png"
    POLYGON = "polygon"
    COCO_RLE = "coco_rle"


@dataclass(frozen=True, slots=True)
class MaskReference(JsonContract):
    path: str
    encoding: MaskEncoding
    width: int
    height: int
    instance_value: int | None = None

    def __post_init__(self) -> None:
        _require_relative_path(self.path, "path")
        if self.width <= 0 or self.height <= 0:
            raise ContractValidationError("mask dimensions must be positive")
        if self.instance_value is not None and self.instance_value <= 0:
            raise ContractValidationError("instance_value must be positive")


@dataclass(frozen=True, slots=True)
class CameraIntrinsics(JsonContract):
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    near_clip: float | None = None
    far_clip: float | None = None

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ContractValidationError("camera dimensions must be positive")
        for name in ("fx", "fy", "cx", "cy"):
            _require_finite(getattr(self, name), name)
        if self.fx <= 0 or self.fy <= 0:
            raise ContractValidationError("camera focal lengths must be positive")
        if self.near_clip is not None:
            _require_nonnegative_finite(self.near_clip, "near_clip")
        if self.far_clip is not None:
            _require_nonnegative_finite(self.far_clip, "far_clip")
        if (
            self.near_clip is not None
            and self.far_clip is not None
            and self.near_clip >= self.far_clip
        ):
            raise ContractValidationError("near_clip must be less than far_clip")


@dataclass(frozen=True, slots=True)
class CameraExtrinsics(JsonContract):
    world_position: Vector3
    world_rotation: Quaternion


@dataclass(frozen=True, slots=True)
class CameraCalibration(JsonContract):
    camera_id: str
    intrinsics: CameraIntrinsics | None
    extrinsics: CameraExtrinsics | None

    def __post_init__(self) -> None:
        _require_nonempty(self.camera_id, "camera_id")


def _require_relative_path(value: str, name: str) -> None:
    _require_nonempty(value, name)
    normalized = value.replace("\\", "/")
    if (
        normalized.startswith("/")
        or normalized.startswith("../")
        or "/../" in normalized
    ):
        raise ContractValidationError(f"{name} must be a dataset-relative path")
