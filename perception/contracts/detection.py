from __future__ import annotations

from dataclasses import dataclass

from perception.contracts.base import ContractValidationError, JsonContract
from perception.contracts.common import (
    SCHEMA_VERSION,
    CategoricalDistribution,
    FrameStamp,
    SourceMetadata,
    SourceType,
    Uncertainty,
    _require_nonempty,
    _require_probability,
)
from perception.contracts.geometry import BoundingBox2D, MaskReference, NormalizedPoint


@dataclass(frozen=True, slots=True)
class DetectorObject(JsonContract):
    detector_id: str
    semantic_distribution: CategoricalDistribution
    bounding_box: BoundingBox2D
    instance_mask: MaskReference | None
    normalized_location: NormalizedPoint
    confidence: float
    uncertainty: Uncertainty | None
    source: SourceMetadata

    def __post_init__(self) -> None:
        _require_nonempty(self.detector_id, "detector_id")
        _require_probability(self.confidence, "confidence")
        if self.source.source is not SourceType.DETECTOR or self.source.privileged:
            raise ContractValidationError(
                "detector objects require a non-privileged detector source"
            )


@dataclass(frozen=True, slots=True)
class DetectionFrame(JsonContract):
    schema_version: str
    stamp: FrameStamp
    image_width: int
    image_height: int
    objects: tuple[DetectorObject, ...]
    source: SourceMetadata

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ContractValidationError(
                f"unsupported schema_version {self.schema_version!r}"
            )
        if self.image_width <= 0 or self.image_height <= 0:
            raise ContractValidationError("image dimensions must be positive")
        ids = [item.detector_id for item in self.objects]
        if len(ids) != len(set(ids)):
            raise ContractValidationError("detector IDs must be unique within a frame")
        for item in self.objects:
            box = item.bounding_box
            if (
                box.image_width != self.image_width
                or box.image_height != self.image_height
            ):
                raise ContractValidationError(
                    "detector box dimensions must match the frame"
                )
        if self.source.source is not SourceType.DETECTOR or self.source.privileged:
            raise ContractValidationError(
                "detection frames require a non-privileged detector source"
            )
