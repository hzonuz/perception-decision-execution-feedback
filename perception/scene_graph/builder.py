"""
Probabilistic temporal Symbolic Scene Graph construction from tracker output.

One `SSGSnapshot` per frame:

- nodes: the explicit agent, every live track as an `ObjectNode`. Visible nodes (VISIBLE /
  PARTIALLY_VISIBLE) have a `tracker` source and `observation=visible`; memory-only nodes (occluded,
  out of view, lost, unobservable) have a `temporal_memory` source, `observation=memory_only`, and
  keep their last observed box. A node's ID is derived from its track ID, so it is stable for the
  track's life; the graph ID is `<episode>:frame-<index>`.
- edges: the `RelationEngine`'s fresh relations, plus binary relations that lost their evidence only
  because an endpoint is no longer visible. Those are carried for `relation_ttl_frames[predicate]`
  frames with probability x `stale_decay` per frame (`last_seen_behind` is a historical fact: it
  does not decay), a `stale_frames=N` evidence tag and the stamp of their last evidence, then
  dropped. A relation between two
  *visible* nodes is never carried: fresh evidence says it no longer holds; nor is
  `last_seen_behind`/`likely_occluder` once its subject is visible again. `age_frames` counts the
  frames an edge has been continuously present.
- `is_macro_step` every `macro_every_frames` frames (or when the caller says so); `history()` and
  `history(macro_only=True)` return the bounded `GraphHistory`.
"""

from __future__ import annotations

import math
import re
from collections import deque
from dataclasses import dataclass, replace

import numpy as np

from perception.contracts import (
    SCHEMA_VERSION,
    AgentNode,
    AgentState,
    AttributeValue,
    BernoulliBelief,
    GraphHistory,
    MemoryState,
    NamedBelief,
    NormalizedPoint,
    ObjectNode,
    Relation,
    RelationPredicate,
    SourceMetadata,
    SourceType,
    SSGSnapshot,
    TrackFrame,
    TrackLifecycle,
    TrackState,
    Uncertainty,
)
from perception.ontology import Ontology
from perception.scene_graph.relations import (
    VISIBLE_STATES,
    NodeView,
    RelationContext,
    RelationEngine,
    RelationModel,
    SceneGraphConfig,
    ground_position,
)

AGENT_ID = "agent"
# facts about where an unseen subject is: they end the moment the subject is seen again
ABOUT_A_HIDDEN_SUBJECT = (
    RelationPredicate.LAST_SEEN_BEHIND,
    RelationPredicate.LIKELY_OCCLUDER,
)
VISIBLE_SOURCE = SourceMetadata(SourceType.TRACKER, "perception_ssg_builder", "1.0.0")
MEMORY_SOURCE = SourceMetadata(
    SourceType.TEMPORAL_MEMORY, "perception_ssg_builder", "1.0.0"
)
AGENT_SOURCE = SourceMetadata(
    SourceType.PROPRIOCEPTION, "perception_ssg_builder", "1.0.0"
)


@dataclass(slots=True)
class _Edge:
    relation: Relation  # last fresh version
    first_frame: int
    last_fresh: int
    last_emitted: int


class SceneGraphBuilder:
    def __init__(
        self,
        config: SceneGraphConfig,
        ontology: Ontology,
        models: tuple[RelationModel, ...] = (),
    ) -> None:
        self.config = config
        self.ontology = ontology
        self.engine = RelationEngine(config, models)
        self._type_fractions = _functional_fractions(ontology)
        self._reset(None)

    def _reset(self, episode: str | None) -> None:
        self._episode = episode
        self._edges: dict[tuple, _Edge] = {}
        self._last_macro: int | None = None
        self._history: deque[SSGSnapshot] = deque(maxlen=self.config.history_length)
        self._macro: deque[SSGSnapshot] = deque(maxlen=self.config.history_length)

    def history(self, *, macro_only: bool = False) -> GraphHistory:
        snapshots = tuple(self._macro if macro_only else self._history)
        return GraphHistory(SCHEMA_VERSION, self._episode or "none", snapshots)

    def update(
        self,
        tracks: TrackFrame,
        *,
        width: int,
        height: int,
        agent: AgentState | None = None,
        macro_step: bool | None = None,
    ) -> SSGSnapshot:
        stamp = tracks.stamp
        if stamp.episode_id != self._episode:
            self._reset(stamp.episode_id)
        index = stamp.frame_index
        views = [
            self._view(state, stamp, width, height)
            for state in tracks.tracks
            if state.record.lifecycle is not TrackLifecycle.ARCHIVED
        ]
        track_to_node = {v.track.record.track_id: v.node_id for v in views}
        context = RelationContext(
            stamp, width, height, AGENT_ID, tuple(views), track_to_node
        )
        relations = self._with_ages(
            self.engine.infer(context), {v.node_id: v for v in views}, index
        )

        if macro_step is None:
            macro_step = (
                self._last_macro is None
                or index - self._last_macro >= self.config.macro_every_frames
            )
        if macro_step:
            self._last_macro = index
        agent_state = (
            replace(agent, node_id=AGENT_ID)
            if agent is not None
            else AgentState(AGENT_ID, None, None, None, None, None)
        )
        nodes = tuple(v.node for v in views)
        snapshot = SSGSnapshot(
            schema_version=SCHEMA_VERSION,
            graph_id=f"{stamp.episode_id}:frame-{index:06d}",
            episode_id=stamp.episode_id,
            stamp=stamp,
            is_macro_step=macro_step,
            agent=AgentNode(AGENT_ID, agent_state, stamp, 1.0, AGENT_SOURCE),
            objects=nodes,
            relations=tuple(relations),
            confidence=float(np.mean([n.confidence for n in nodes])) if nodes else 1.0,
            source=MEMORY_SOURCE,
        )
        self.ontology.validate_ssg(snapshot)
        self._history.append(snapshot)
        if macro_step:
            self._macro.append(snapshot)
        return snapshot

    def _with_ages(
        self, fresh: list[Relation], views: dict[str, NodeView], index: int
    ) -> list[Relation]:
        out = []
        seen = set()
        for relation in fresh:
            key = _key(relation)
            seen.add(key)
            edge = self._edges.get(key)
            if edge is None or edge.last_emitted != index - 1:
                edge = self._edges[key] = _Edge(relation, index, index, index)
            edge.relation, edge.last_fresh, edge.last_emitted = relation, index, index
            out.append(replace(relation, age_frames=index - edge.first_frame))
        for key, edge in list(self._edges.items()):
            if key in seen:
                continue
            relation = edge.relation
            stale = index - edge.last_fresh
            subject, obj = views.get(relation.subject_node_id), views.get(
                relation.object_node_id or ""
            )
            carry = (
                edge.last_emitted == index - 1
                and not (
                    relation.predicate in ABOUT_A_HIDDEN_SUBJECT
                    and subject is not None
                    and subject.visible
                )
                and relation.object_node_id is not None
                and subject is not None
                and obj is not None
                and MemoryState.TERMINATED
                not in (subject.node.memory_state, obj.node.memory_state)
                and not (subject.visible and obj.visible)
                and stale <= self.config.ttl(relation.predicate)
            )
            decay = (
                1.0
                if relation.predicate.value == "last_seen_behind"
                else self.config.stale_decay**stale
            )
            probability = relation.probability * decay
            if not carry or probability < self.config.min_relation_probability:
                del self._edges[key]
                continue
            edge.last_emitted = index
            out.append(
                replace(
                    relation,
                    probability=probability,
                    age_frames=index - edge.first_frame,
                    evidence=(*relation.evidence, f"stale_frames={stale}"),
                )
            )
        return sorted(
            out,
            key=lambda r: (
                r.predicate.value,
                r.subject_node_id,
                r.object_node_id or "",
            ),
        )

    def _view(self, state: TrackState, stamp, width: int, height: int) -> NodeView:
        c = self.config
        memory = state.memory
        visible = memory.state in VISIBLE_STATES
        box = state.bounding_box
        xyxy = np.array([box.x_min, box.y_min, box.x_max, box.y_max], float)
        ground = ground_position(xyxy, width, height, c)
        distance = None
        if ground is not None:
            d = math.hypot(*ground)
            distance = (
                d,
                d * (1 - c.depth_relative_error),
                d * (1 + c.depth_relative_error),
            )
        beliefs = self._beliefs(state)
        source = VISIBLE_SOURCE if visible else MEMORY_SOURCE
        f = c.camera_focal_fraction * width
        attributes = [
            AttributeValue(
                "observation", "visible" if visible else "memory_only", None, source
            )
        ]
        if ground is not None:
            attributes += [
                AttributeValue("ground_x", round(ground[0], 4), None, source),
                AttributeValue("ground_z", round(ground[1], 4), None, source),
            ]
        node = ObjectNode(
            node_id="obj_" + re.sub(r"[^A-Za-z0-9_]+", "_", state.record.track_id),
            track_id=state.record.track_id,
            detector_id=state.record.detector_id,
            exact_simulator_class=None,
            semantic_distribution=state.semantic_distribution,
            functional_types=tuple(
                NamedBelief(
                    name,
                    BernoulliBelief(
                        point, source, Uncertainty(lower_bound=lo, upper_bound=hi)
                    ),
                )
                for name, (point, lo, hi) in sorted(beliefs.items())
            ),
            bounding_box=box,
            instance_mask=state.instance_mask,
            normalized_location=NormalizedPoint(
                (box.x_min + box.x_max) / (2 * width),
                (box.y_min + box.y_max) / (2 * height),
            ),
            bearing_degrees=math.degrees(
                math.atan2(
                    (box.x_min + box.x_max) / 2 - c.camera_cx_fraction * width, f
                )
            ),
            relative_distance=None if distance is None else distance[0],
            relative_velocity=None,
            detector_confidence=memory.visibility_probability if visible else None,
            visibility_probability=memory.visibility_probability,
            existence_probability=memory.existence_probability,
            track_age_frames=memory.age_frames,
            last_seen=memory.last_seen,
            memory_state=memory.state,
            confidence=memory.existence_probability or 0.0,
            source=source,
            stamp=stamp,
            attributes=tuple(attributes),
            uncertainty=Uncertainty(entropy=_entropy(state)),
        )
        return NodeView(node, state, visible, xyxy, ground, distance, beliefs)

    def _beliefs(
        self, state: TrackState
    ) -> dict[str, tuple[float | None, float, float]]:
        """
        Functional-type belief from the class distribution: `lower` counts only known classes,
        `upper` also gives the type to all unknown mass; the estimate is conditional on the known
        classes, and unknown (None) when they hold < `min_known_class_mass`.
        """
        distribution = state.semantic_distribution
        known = 1.0 - distribution.unknown_probability
        out = {}
        for name in sorted(self.ontology.functional_names):
            lower = sum(
                e.probability * self._type_fractions.get((e.label, name), 0.0)
                for e in distribution.entries
            )
            upper = lower + distribution.unknown_probability
            if lower > 0:
                point = (
                    lower / known if known >= self.config.min_known_class_mass else None
                )
                out[name] = (
                    None if point is None else min(point, 1.0),
                    lower,
                    min(upper, 1.0),
                )
        return out


def to_asp(snapshot: SSGSnapshot, *, min_probability: float = 0.0) -> str:
    """
    ASP-style facts. Probabilities are integer per-mille; unknown values are absent, never 0.

    agent(agent). object(o). visible(o) | memory_only(o). state(o,occluded).
    class(o,"GoodGoal",930). existence(o,970). type(o,collectable,1000).
    rel(left_of,o1,o2,870).  val(previously_visible,o,120,900).
    """
    lines = [
        f"% {snapshot.graph_id} t={snapshot.stamp.timestamp_seconds:.3f} macro={str(snapshot.is_macro_step).lower()}",
        f"agent({_atom(snapshot.agent.node_id)}).",
    ]
    for node in snapshot.objects:
        o = _atom(node.node_id)
        observed = next(
            (a.value for a in node.attributes if a.name == "observation"), None
        )
        lines += [
            f"object({o}).",
            f"{observed}({o})." if observed else f"% {o}: observation unknown",
            f"state({o},{node.memory_state.value}).",
        ]
        if node.existence_probability is not None:
            lines.append(f"existence({o},{_permille(node.existence_probability)}).")
        lines += [
            f'class({o},"{e.label}",{_permille(e.probability)}).'
            for e in node.semantic_distribution.entries
        ]
        lines += [
            f"type({o},{t.name},{_permille(t.belief.probability)})."
            for t in node.functional_types
            if t.belief.probability is not None
        ]
    for r in snapshot.relations:
        if r.probability < min_probability:
            continue
        if r.object_node_id is not None:
            lines.append(
                f"rel({r.predicate.value},{_atom(r.subject_node_id)},{_atom(r.object_node_id)},{_permille(r.probability)})."
            )
        else:
            lines.append(
                f"val({r.predicate.value},{_atom(r.subject_node_id)},{_term(r.symbolic_value)},{_permille(r.probability)})."
            )
    return "\n".join(lines) + "\n"


def _functional_fractions(ontology: Ontology) -> dict[tuple[str, str], float]:
    members: dict[str, list[tuple[str, ...]]] = {}
    for mapping in ontology.simulator_mappings:
        members.setdefault(mapping.semantic_class, []).append(mapping.functional_types)
    return {
        (cls, name): sum(name in types for types in group) / len(group)
        for cls, group in members.items()
        for name in ontology.functional_names
    }


def _entropy(state: TrackState) -> float:
    probabilities = [e.probability for e in state.semantic_distribution.entries] + [
        state.semantic_distribution.unknown_probability
    ]
    return -sum(p * math.log(p) for p in probabilities if p > 0)


def _key(relation: Relation) -> tuple:
    return (relation.subject_node_id, relation.predicate, relation.object_node_id)


def _atom(node_id: str) -> str:
    atom = re.sub(r"[^a-z0-9_]", "_", node_id.lower())
    return atom if atom[:1].isalpha() else f"n_{atom}"


def _permille(p: float) -> int:
    return int(round(p * 1000))


def _term(value) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    return f'"{value}"'
