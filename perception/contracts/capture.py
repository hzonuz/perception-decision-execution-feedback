from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from perception.contracts.base import ContractValidationError, JsonContract
from perception.contracts.common import (
    SCHEMA_VERSION,
    AttributeValue,
    CategoricalDistribution,
    FrameStamp,
    NamedBelief,
    SourceMetadata,
    SourceType,
    _require_finite,
    _require_nonempty,
    _require_nonnegative_finite,
    _require_probability,
)
from perception.contracts.geometry import (
    BoundingBox2D,
    CameraCalibration,
    ImageReference,
    MaskReference,
    Quaternion,
    Vector3,
)


@dataclass(frozen=True, slots=True)
class PrimitiveAction(JsonContract):
    forward: int | None
    rotate: int | None
    name: str | None = None

    def __post_init__(self) -> None:
        # Verified TrainingAgent action branches are 0, 1, and 2.
        for field_name in ("forward", "rotate"):
            value = getattr(self, field_name)
            if value is not None and value not in (0, 1, 2):
                raise ContractValidationError(
                    f"{field_name} action must be 0, 1, 2, or unknown"
                )


@dataclass(frozen=True, slots=True)
class AgentState(JsonContract):
    node_id: str
    world_position: Vector3 | None
    world_rotation: Quaternion | None
    velocity: Vector3 | None
    recent_action: PrimitiveAction | None
    health: float | None
    current_option: str | None = None
    option_status: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty(self.node_id, "node_id")
        if self.health is not None:
            _require_finite(self.health, "health")


class LifecycleEventType(str, Enum):
    SPAWNED = "spawned"
    ENABLED = "enabled"
    DISABLED = "disabled"
    DESTROYED = "destroyed"
    TERMINATED = "terminated"


@dataclass(frozen=True, slots=True)
class LifecycleEvent(JsonContract):
    instance_id: str
    event: LifecycleEventType
    stamp: FrameStamp
    reason: str | None
    source: SourceMetadata

    def __post_init__(self) -> None:
        _require_nonempty(self.instance_id, "instance_id")
        _require_oracle_source(self.source)


@dataclass(frozen=True, slots=True)
class ContactEvent(JsonContract):
    subject_instance_id: str
    object_instance_id: str
    stamp: FrameStamp
    impulse: float | None
    contact_point: Vector3 | None
    source: SourceMetadata

    def __post_init__(self) -> None:
        _require_nonempty(self.subject_instance_id, "subject_instance_id")
        _require_nonempty(self.object_instance_id, "object_instance_id")
        if self.subject_instance_id == self.object_instance_id:
            raise ContractValidationError("contact endpoints must differ")
        if self.impulse is not None:
            _require_nonnegative_finite(self.impulse, "impulse")
        _require_oracle_source(self.source)


@dataclass(frozen=True, slots=True)
class ObjectAnnotation(JsonContract):
    instance_id: str
    exact_simulator_class: str
    semantic_distribution: CategoricalDistribution
    functional_types: tuple[NamedBelief, ...]
    world_position: Vector3
    world_rotation: Quaternion
    dimensions: Vector3
    velocity: Vector3 | None
    bounding_box: BoundingBox2D | None
    instance_mask: MaskReference | None
    relative_depth: float | None
    visibility_fraction: float | None
    occlusion_fraction: float | None
    attributes: tuple[AttributeValue, ...]
    source: SourceMetadata

    def __post_init__(self) -> None:
        _require_nonempty(self.instance_id, "instance_id")
        _require_nonempty(self.exact_simulator_class, "exact_simulator_class")
        if min(self.dimensions.x, self.dimensions.y, self.dimensions.z) < 0:
            raise ContractValidationError("object dimensions must be non-negative")
        if self.relative_depth is not None:
            _require_nonnegative_finite(self.relative_depth, "relative_depth")
        if self.visibility_fraction is not None:
            _require_probability(self.visibility_fraction, "visibility_fraction")
        if self.occlusion_fraction is not None:
            _require_probability(self.occlusion_fraction, "occlusion_fraction")
        names = [item.name for item in self.functional_types]
        if len(names) != len(set(names)):
            raise ContractValidationError("functional type names must be unique")
        _require_oracle_source(self.source)


@dataclass(frozen=True, slots=True)
class CaptureFrame(JsonContract):
    schema_version: str
    episode_id: str
    arena_id: str
    random_seed: int | None
    stamp: FrameStamp
    rgb: ImageReference
    camera: CameraCalibration | None
    agent: AgentState
    objects: tuple[ObjectAnnotation, ...]
    lifecycle_events: tuple[LifecycleEvent, ...]
    contact_events: tuple[ContactEvent, ...]
    source: SourceMetadata

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ContractValidationError(
                f"unsupported schema_version {self.schema_version!r}"
            )
        _require_nonempty(self.episode_id, "episode_id")
        _require_nonempty(self.arena_id, "arena_id")
        if self.stamp.episode_id != self.episode_id:
            raise ContractValidationError(
                "stamp episode_id must match frame episode_id"
            )
        ids = [item.instance_id for item in self.objects]
        if len(ids) != len(set(ids)):
            raise ContractValidationError(
                "object instance IDs must be unique within a frame"
            )
        for item in self.objects:
            box = item.bounding_box
            if box is not None and (
                box.image_width != self.rgb.width or box.image_height != self.rgb.height
            ):
                raise ContractValidationError(
                    "object box dimensions must match the RGB image"
                )
            mask = item.instance_mask
            if mask is not None and (
                mask.width != self.rgb.width or mask.height != self.rgb.height
            ):
                raise ContractValidationError(
                    "object mask dimensions must match the RGB image"
                )
        for event in (*self.lifecycle_events, *self.contact_events):
            if event.stamp.episode_id != self.episode_id:
                raise ContractValidationError(
                    "event episode_id must match frame episode_id"
                )
        _require_oracle_source(self.source)


def _require_oracle_source(source: SourceMetadata) -> None:
    if source.source is not SourceType.SIMULATOR_ORACLE or not source.privileged:
        raise ContractValidationError(
            "capture records require a privileged simulator_oracle source"
        )
