"""Probabilistic Symbolic Scene Graph contracts."""

from __future__ import annotations

from dataclasses import dataclass

from perception.contracts.base import ContractValidationError, JsonContract
from perception.contracts.capture import AgentState
from perception.contracts.common import (
    SCHEMA_VERSION,
    AttributeValue,
    CategoricalDistribution,
    FrameStamp,
    MemoryState,
    NamedBelief,
    SourceMetadata,
    Uncertainty,
    _require_nonempty,
    _require_nonnegative_finite,
    _require_nonnegative_int,
    _require_probability,
)
from perception.contracts.geometry import BoundingBox2D, MaskReference, NormalizedPoint, Vector3
from perception.contracts.relations import Relation


@dataclass(frozen=True, slots=True)
class AgentNode(JsonContract):
    node_id: str
    state: AgentState
    stamp: FrameStamp
    confidence: float
    source: SourceMetadata
    uncertainty: Uncertainty | None = None

    def __post_init__(self) -> None:
        _require_nonempty(self.node_id, "node_id")
        if self.state.node_id != self.node_id:
            raise ContractValidationError("agent state node_id must match agent node_id")
        _require_probability(self.confidence, "confidence")


@dataclass(frozen=True, slots=True)
class ObjectNode(JsonContract):
    node_id: str
    track_id: str
    detector_id: str | None
    exact_simulator_class: str | None
    semantic_distribution: CategoricalDistribution
    functional_types: tuple[NamedBelief, ...]
    bounding_box: BoundingBox2D | None
    instance_mask: MaskReference | None
    normalized_location: NormalizedPoint | None
    bearing_degrees: float | None
    relative_distance: float | None
    relative_velocity: Vector3 | None
    detector_confidence: float | None
    visibility_probability: float | None
    existence_probability: float | None
    track_age_frames: int
    last_seen: FrameStamp | None
    memory_state: MemoryState
    confidence: float
    source: SourceMetadata
    stamp: FrameStamp
    attributes: tuple[AttributeValue, ...] = ()
    uncertainty: Uncertainty | None = None

    def __post_init__(self) -> None:
        _require_nonempty(self.node_id, "node_id")
        _require_nonempty(self.track_id, "track_id")
        if self.exact_simulator_class is not None and not self.source.privileged:
            raise ContractValidationError(
                "exact_simulator_class may only appear on a privileged oracle node"
            )
        if self.relative_distance is not None:
            _require_nonnegative_finite(self.relative_distance, "relative_distance")
        for name in (
            "detector_confidence",
            "visibility_probability",
            "existence_probability",
            "confidence",
        ):
            value = getattr(self, name)
            if value is not None:
                _require_probability(value, name)
        _require_nonnegative_int(self.track_age_frames, "track_age_frames")
        if self.last_seen is not None:
            if self.last_seen.episode_id != self.stamp.episode_id:
                raise ContractValidationError("node timestamps must belong to the same episode")
            if self.last_seen.frame_index > self.stamp.frame_index:
                raise ContractValidationError("last_seen cannot be later than node stamp")
        names = [item.name for item in self.functional_types]
        if len(names) != len(set(names)):
            raise ContractValidationError("functional type names must be unique")


@dataclass(frozen=True, slots=True)
class SSGSnapshot(JsonContract):
    schema_version: str
    graph_id: str
    episode_id: str
    stamp: FrameStamp
    is_macro_step: bool
    agent: AgentNode
    objects: tuple[ObjectNode, ...]
    relations: tuple[Relation, ...]
    confidence: float
    source: SourceMetadata
    uncertainty: Uncertainty | None = None

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ContractValidationError(f"unsupported schema_version {self.schema_version!r}")
        _require_nonempty(self.graph_id, "graph_id")
        _require_nonempty(self.episode_id, "episode_id")
        if self.stamp.episode_id != self.episode_id or self.agent.stamp != self.stamp:
            raise ContractValidationError("graph, agent, and stamp must identify the same frame")
        node_ids = [self.agent.node_id, *(item.node_id for item in self.objects)]
        if len(node_ids) != len(set(node_ids)):
            raise ContractValidationError("graph node IDs must be unique")
        track_ids = [item.track_id for item in self.objects]
        if len(track_ids) != len(set(track_ids)):
            raise ContractValidationError("object track IDs must be unique")
        node_id_set = set(node_ids)
        for relation in self.relations:
            if relation.stamp.episode_id != self.episode_id:
                raise ContractValidationError("relation episode_id must match graph episode_id")
            if relation.subject_node_id not in node_id_set:
                raise ContractValidationError("relation subject is not a graph node")
            if relation.object_node_id is not None and relation.object_node_id not in node_id_set:
                raise ContractValidationError("relation object is not a graph node")
        _require_probability(self.confidence, "confidence")


@dataclass(frozen=True, slots=True)
class GraphHistory(JsonContract):
    schema_version: str
    episode_id: str
    snapshots: tuple[SSGSnapshot, ...]

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ContractValidationError(f"unsupported schema_version {self.schema_version!r}")
        _require_nonempty(self.episode_id, "episode_id")
        previous_frame = -1
        graph_ids: set[str] = set()
        for snapshot in self.snapshots:
            if snapshot.episode_id != self.episode_id:
                raise ContractValidationError("history cannot cross episode boundaries")
            if snapshot.stamp.frame_index <= previous_frame:
                raise ContractValidationError("history frames must be strictly increasing")
            if snapshot.graph_id in graph_ids:
                raise ContractValidationError("history graph IDs must be unique")
            previous_frame = snapshot.stamp.frame_index
            graph_ids.add(snapshot.graph_id)

