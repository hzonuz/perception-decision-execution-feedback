"""Stable identity and temporal-memory contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from perception.contracts.base import ContractValidationError, JsonContract
from perception.contracts.common import (
    SCHEMA_VERSION,
    CategoricalDistribution,
    FrameStamp,
    MemoryState,
    MotionState,
    SourceMetadata,
    SourceType,
    Uncertainty,
    _require_nonempty,
    _require_nonnegative_finite,
    _require_nonnegative_int,
    _require_probability,
)
from perception.contracts.geometry import BoundingBox2D, MaskReference, Vector2, Vector3


class TrackLifecycle(str, Enum):
    CREATED = "created"
    UPDATED = "updated"
    MISSED = "missed"
    REAPPEARED = "reappeared"
    ARCHIVED = "archived"
    TERMINATED = "terminated"


@dataclass(frozen=True, slots=True)
class TrackRecord(JsonContract):
    track_id: str
    detector_id: str | None
    stamp: FrameStamp
    lifecycle: TrackLifecycle
    association_confidence: float | None
    age_frames: int
    missed_frames: int
    relative_velocity: Vector3 | None
    motion_state: MotionState
    source: SourceMetadata
    uncertainty: Uncertainty | None = None

    def __post_init__(self) -> None:
        _require_nonempty(self.track_id, "track_id")
        if self.association_confidence is not None:
            _require_probability(self.association_confidence, "association_confidence")
        _require_nonnegative_int(self.age_frames, "age_frames")
        _require_nonnegative_int(self.missed_frames, "missed_frames")
        if self.missed_frames > self.age_frames:
            raise ContractValidationError("missed_frames cannot exceed age_frames")
        if self.source.source is not SourceType.TRACKER or self.source.privileged:
            raise ContractValidationError("track records require a non-privileged tracker source")


@dataclass(frozen=True, slots=True)
class ObjectMemory(JsonContract):
    track_id: str
    state: MemoryState
    visibility_probability: float | None
    existence_probability: float | None
    last_seen: FrameStamp | None
    updated_at: FrameStamp
    age_frames: int
    source: SourceMetadata
    uncertainty: Uncertainty | None = None

    def __post_init__(self) -> None:
        _require_nonempty(self.track_id, "track_id")
        for name in ("visibility_probability", "existence_probability"):
            value = getattr(self, name)
            if value is not None:
                _require_probability(value, name)
        _require_nonnegative_int(self.age_frames, "age_frames")
        if self.last_seen is not None:
            if self.last_seen.episode_id != self.updated_at.episode_id:
                raise ContractValidationError("memory stamps must belong to the same episode")
            if self.last_seen.frame_index > self.updated_at.frame_index:
                raise ContractValidationError("last_seen cannot be later than updated_at")
        if self.state is MemoryState.TERMINATED and self.existence_probability not in (0.0, None):
            raise ContractValidationError("terminated memory cannot assert positive existence")
        if self.source.source is not SourceType.TEMPORAL_MEMORY or self.source.privileged:
            raise ContractValidationError(
                "object memory requires a non-privileged temporal_memory source"
            )



@dataclass(frozen=True, slots=True)
class TrackState(JsonContract):
    """Full per-frame state of one track: the association `record`, the visibility/existence
    `memory`, and what neither carries. `bounding_box`/`instance_mask` are the last *observed*
    ones (`memory.last_seen`); `image_velocity` is px/s in image coordinates (includes ego-motion);
    `relative_depth` is `None` when not estimated."""

    record: TrackRecord
    memory: ObjectMemory
    semantic_distribution: CategoricalDistribution
    bounding_box: BoundingBox2D | None
    instance_mask: MaskReference | None
    image_velocity: Vector2 | None
    relative_depth: float | None
    hits: int
    tracker_confidence: float | None
    occluder_track_id: str | None
    termination_reason: str | None = None

    def __post_init__(self) -> None:
        if self.record.track_id != self.memory.track_id:
            raise ContractValidationError("record and memory must describe the same track")
        if self.record.stamp != self.memory.updated_at:
            raise ContractValidationError("record and memory must share a stamp")
        if self.record.age_frames != self.memory.age_frames:
            raise ContractValidationError("record and memory ages must match")
        _require_nonnegative_int(self.hits, "hits")
        if self.hits + self.record.missed_frames > self.record.age_frames:
            raise ContractValidationError("hits plus misses cannot exceed age_frames")
        if self.relative_depth is not None:
            _require_nonnegative_finite(self.relative_depth, "relative_depth")
        if self.tracker_confidence is not None:
            _require_probability(self.tracker_confidence, "tracker_confidence")
        if self.occluder_track_id == self.record.track_id:
            raise ContractValidationError("a track cannot occlude itself")
        if (self.memory.state is MemoryState.TERMINATED) != (self.termination_reason is not None):
            raise ContractValidationError("termination_reason is required exactly when terminated")
        if (self.bounding_box is None) != (self.memory.last_seen is None):
            raise ContractValidationError("a last box exists exactly when the track has been seen")


@dataclass(frozen=True, slots=True)
class TrackFrame(JsonContract):
    """Tracker output for one frame, independent of any scene graph. Tracks archived or terminated
    at this frame appear here one last time."""

    schema_version: str
    stamp: FrameStamp
    observable: bool
    tracks: tuple[TrackState, ...]
    source: SourceMetadata

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ContractValidationError(f"unsupported schema_version {self.schema_version!r}")
        ids = [item.record.track_id for item in self.tracks]
        if len(ids) != len(set(ids)):
            raise ContractValidationError("track IDs must be unique within a frame")
        for item in self.tracks:
            if item.record.stamp != self.stamp:
                raise ContractValidationError("every track must be stamped with the frame")
            if item.occluder_track_id is not None and item.occluder_track_id not in ids:
                raise ContractValidationError("occluder must be a track in the same frame")
            if not self.observable and item.record.lifecycle in (TrackLifecycle.CREATED, TrackLifecycle.UPDATED):
                raise ContractValidationError("an unobservable frame cannot create or update tracks")
        if self.source.source is not SourceType.TRACKER or self.source.privileged:
            raise ContractValidationError("track frames require a non-privileged tracker source")
