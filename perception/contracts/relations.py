"""Probabilistic spatial, temporal, and task-relation contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from perception.contracts.base import ContractValidationError, JsonContract
from perception.contracts.common import (
    FrameStamp,
    JsonScalar,
    SourceMetadata,
    SourceType,
    Uncertainty,
    _require_nonempty,
    _require_nonnegative_int,
    _require_probability,
)


class RelationPredicate(str, Enum):
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    ABOVE = "above"
    BELOW = "below"
    NEAR = "near"
    FAR = "far"
    OVERLAPS = "overlaps"
    ADJACENT = "adjacent"
    IN_FRONT_OF = "in_front_of"
    BEHIND = "behind"
    ON = "on"
    SUPPORTS = "supports"
    OCCLUDES = "occludes"
    SAME_TRACK = "same_track"
    MOVING_TOWARD = "moving_toward"
    MOVING_AWAY = "moving_away"
    PREVIOUSLY_VISIBLE = "previously_visible"
    DISAPPEARED_AFTER = "disappeared_after"
    LAST_SEEN_BEHIND = "last_seen_behind"
    BELIEVED_PRESENT = "believed_present"
    REAPPEARED_AFTER = "reappeared_after"
    GOAL_ON_PLATFORM = "goal_on_platform"
    DANGER_BETWEEN = "danger_between"
    PATH_BLOCKED_BY = "path_blocked_by"
    LIKELY_OCCLUDER = "likely_occluder"
    MORE_GOALS_LEFT = "more_goals_left"
    MORE_GOALS_RIGHT = "more_goals_right"
    SAFE_TO_INTERACT = "safe_to_interact"
    LIKELY_REACHABLE = "likely_reachable"


ALLOWED_RELATION_SOURCES = frozenset(
    {
        SourceType.GEOMETRY,
        SourceType.DETECTOR,
        SourceType.TRACKER,
        SourceType.TEMPORAL_MEMORY,
        SourceType.LEARNED_RELATION_MODEL,
        SourceType.TASK_RULE,
        SourceType.SIMULATOR_ORACLE,
    }
)


@dataclass(frozen=True, slots=True)
class Relation(JsonContract):
    subject_node_id: str
    predicate: RelationPredicate
    object_node_id: str | None
    symbolic_value: JsonScalar
    probability: float
    source: SourceMetadata
    stamp: FrameStamp
    age_frames: int
    evidence: tuple[str, ...] = ()
    uncertainty: Uncertainty | None = None

    def __post_init__(self) -> None:
        _require_nonempty(self.subject_node_id, "subject_node_id")
        if (self.object_node_id is None) == (self.symbolic_value is None):
            raise ContractValidationError(
                "relation must have exactly one object_node_id or non-null symbolic_value"
            )
        if self.object_node_id is not None:
            _require_nonempty(self.object_node_id, "object_node_id")
        _require_probability(self.probability, "probability")
        _require_nonnegative_int(self.age_frames, "age_frames")
        if self.source.source not in ALLOWED_RELATION_SOURCES:
            raise ContractValidationError("invalid relation source")
        if self.source.privileged != (self.source.source is SourceType.SIMULATOR_ORACLE):
            raise ContractValidationError("relation privileged flag does not match its source")

