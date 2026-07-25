"""Stable identity and temporal-memory contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from perception.contracts.base import ContractValidationError, JsonContract
from perception.contracts.common import (
    FrameStamp,
    MemoryState,
    MotionState,
    SourceMetadata,
    SourceType,
    Uncertainty,
    _require_nonempty,
    _require_nonnegative_int,
    _require_probability,
)
from perception.contracts.geometry import Vector3


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

