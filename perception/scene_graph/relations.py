"""
Hybrid relation engine: deterministic geometry, configurable heuristics, pluggable learned models.

Frames of reference. Image relations (`left_of`, `right_of`, `above`, `below`, `overlaps`, `on`,
`occludes`) are egocentric, in the agent's current view, between *visible* nodes only. Metric
relations (`near`, `far`, `adjacent`, `moving_toward`, `moving_away`) use a floor-contact distance:
Animal-AI's camera has zero pitch at `camera_height`, so a box whose bottom edge sits `v - cy` rows
below the horizon rests `z = camera_height * f / (v - cy)` ahead, `x = (u - cx) * z / f` to the side.
Checked on the curriculum captures: median error <= 10% from 5 to 20 units, so every distance
carries +-`depth_relative_error`. It is unknown (and the relation omitted) when the box bottom is
cut by the frame or within `depth_min_rows_below_horizon` of the horizon, and it is wrong for
objects not resting on the floor.

Every relation is emitted only when it holds with probability >= `min_relation_probability`;
nothing is ever asserted false, and a relation whose inputs are unknown is omitted. Probabilities
multiply the endpoints' existence beliefs; heuristic ones also multiply the functional-type belief
(collectable, harmful, barrier, support) derived from the node's class distribution.

The engine is stateful only for temporal relations (distance history, what a node disappeared
behind, when it was last seen) and resets on a new episode.
"""

from __future__ import annotations

import json
import math
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Protocol

import numpy as np

from perception.contracts import (
    FrameStamp,
    MemoryState,
    ObjectNode,
    Relation,
    RelationPredicate,
    SourceMetadata,
    SourceType,
    TrackLifecycle,
    TrackState,
    Uncertainty,
)

P = RelationPredicate
GEOMETRY = SourceMetadata(SourceType.GEOMETRY, "perception_relations", "1.0.0")
TRACKER = SourceMetadata(SourceType.TRACKER, "perception_relations", "1.0.0")
MEMORY = SourceMetadata(SourceType.TEMPORAL_MEMORY, "perception_relations", "1.0.0")
TASK = SourceMetadata(SourceType.TASK_RULE, "perception_relations", "1.0.0")
VISIBLE_STATES = (MemoryState.VISIBLE, MemoryState.PARTIALLY_VISIBLE)


@dataclass(frozen=True, slots=True)
class SceneGraphConfig:
    camera_focal_fraction: float  # focal length / image width (Animal-AI: 107.4 / 256)
    camera_cx_fraction: float
    camera_cy_fraction: float
    camera_height: float
    depth_relative_error: float
    depth_min_rows_below_horizon: float
    lateral_margin_fraction: float
    above_min_horizontal_overlap: float
    overlap_min_iou: float
    near_distance: float
    far_distance: float
    adjacent_max_gap_fraction: float
    adjacent_max_depth_gap: float
    motion_min_speed: float
    motion_window_frames: int
    on_probability: float
    on_vertical_tolerance_fraction: float
    occludes_min_coverage: float
    occludes_probability_tracker: float
    occludes_probability_geometry: float
    likely_occluder_probability_tracker: float
    likely_occluder_probability_geometry: float
    last_seen_behind_probability_tracker: float
    last_seen_behind_probability_geometry: float
    believed_present_min_existence: float
    path_half_width_fraction: float
    path_blocked_probability: float
    danger_between_probability: float
    goal_type: str
    danger_type: str
    barrier_type: str
    support_type: str
    on_subject_types: tuple[
        str, ...
    ]  # what can rest on a support (walls seen above walls are not "on")
    min_known_class_mass: float
    min_relation_probability: float
    relation_ttl_frames: dict[
        str, int
    ]  # frames a relation outlives its last evidence; "default"
    stale_decay: (
        float  # per stale frame; historical facts (last_seen_behind) do not decay
    )
    macro_every_frames: int
    history_length: int

    @classmethod
    def from_json_file(cls, path: str | Path) -> "SceneGraphConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("scene graph configuration must be a JSON object")
        if isinstance(raw.get("on_subject_types"), list):
            raw["on_subject_types"] = tuple(raw["on_subject_types"])
        try:
            config = cls(**raw)
        except TypeError as exc:
            raise ValueError(f"invalid scene graph configuration: {exc}") from exc
        config.validate()
        return config

    def validate(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name == "relation_ttl_frames":
                predicates = {p.value for p in RelationPredicate} | {"default"}
                if "default" not in value or set(value) - predicates:
                    raise ValueError(
                        f"relation_ttl_frames needs 'default' and only known predicates, got {sorted(value)}"
                    )
                if any(type(v) is not int or v < 0 for v in value.values()):
                    raise ValueError(
                        "relation_ttl_frames values must be non-negative integers"
                    )
            elif item.name == "on_subject_types":
                if (
                    not isinstance(value, list | tuple)
                    or not value
                    or not all(isinstance(v, str) and v for v in value)
                ):
                    raise ValueError(
                        "on_subject_types must be a non-empty list of functional type names"
                    )
            elif item.name.endswith("_type"):
                if not isinstance(value, str) or not value:
                    raise ValueError(f"{item.name} must be a functional type name")
            elif (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{item.name} must be a non-negative finite number")
        probabilities = [
            f.name
            for f in fields(self)
            if f.name.endswith("_probability") or "probability_" in f.name
        ]
        for name in probabilities + [
            "min_known_class_mass",
            "stale_decay",
            "depth_relative_error",
        ]:
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.near_distance >= self.far_distance:
            raise ValueError("near_distance must be below far_distance")
        for name in ("motion_window_frames", "macro_every_frames", "history_length"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")

    def ttl(self, predicate: RelationPredicate) -> int:
        return self.relation_ttl_frames.get(
            predicate.value, self.relation_ttl_frames["default"]
        )


@dataclass(frozen=True, slots=True)
class NodeView:

    node: ObjectNode
    track: TrackState
    visible: bool
    box: np.ndarray
    ground: (
        tuple[float, float] | None
    )  # (lateral x, forward z) in arena units, from the last box
    distance: (
        tuple[float, float, float] | None
    )  # (estimate, lower, upper) from the camera
    beliefs: dict[
        str, tuple[float | None, float, float]
    ]  # functional type -> (estimate, lower, upper)

    @property
    def node_id(self) -> str:
        return self.node.node_id

    @property
    def existence(self) -> float:
        return self.node.existence_probability or 0.0

    @property
    def memory_only(self) -> bool:
        return not self.visible and self.node.memory_state is not MemoryState.TERMINATED

    def belief(self, name: str) -> float | None:
        return self.beliefs.get(name, (0.0, 0.0, 0.0))[0]


@dataclass(frozen=True, slots=True)
class RelationContext:
    stamp: FrameStamp
    width: int
    height: int
    agent_id: str
    nodes: tuple[NodeView, ...]
    track_to_node: dict[str, str]


class RelationModel(Protocol):

    name: str

    def predict(self, context: RelationContext) -> Iterable[Relation]: ...


class RelationEngine:
    def __init__(
        self, config: SceneGraphConfig, models: Sequence[RelationModel] = ()
    ) -> None:
        config.validate()
        self.config = config
        self.models = tuple(models)
        self.reset()

    def reset(self) -> None:
        self._episode: str | None = None
        self._distances: dict[str, deque[tuple[float, float]]] = {}
        self._was_visible: dict[str, bool] = {}
        self._last_seen: dict[str, int] = {}

    def infer(self, context: RelationContext) -> list[Relation]:
        if context.stamp.episode_id != self._episode:
            self.reset()
            self._episode = context.stamp.episode_id
        out: dict[tuple[str, P, str | None], Relation] = {}

        def emit(
            subject,
            predicate,
            p,
            source,
            obj=None,
            value=None,
            evidence=(),
            uncertainty=None,
        ):
            if p < self.config.min_relation_probability:
                return
            relation = Relation(
                subject,
                predicate,
                obj,
                value,
                min(float(p), 1.0),
                source,
                context.stamp,
                0,
                tuple(evidence),
                uncertainty,
            )
            key = (subject, predicate, obj)
            if key not in out or out[key].probability < relation.probability:
                out[key] = relation

        visible = [n for n in context.nodes if n.visible]
        self._pairwise(context, visible, emit)
        self._agent_relative(context, visible, emit)
        self._occlusion(context, emit)
        self._temporal(context, emit)
        self._task(context, emit)
        node_ids = {n.node_id for n in context.nodes} | {context.agent_id}
        for model in self.models:
            for relation in model.predict(context):
                if relation.source.source is not SourceType.LEARNED_RELATION_MODEL:
                    raise ValueError(
                        f"{model.name} must emit learned_relation_model relations"
                    )
                if relation.subject_node_id not in node_ids or (
                    relation.object_node_id is not None
                    and relation.object_node_id not in node_ids
                ):
                    raise ValueError(
                        f"{model.name} referenced a node that is not in the graph"
                    )
                emit(
                    relation.subject_node_id,
                    relation.predicate,
                    relation.probability,
                    relation.source,
                    relation.object_node_id,
                    relation.symbolic_value,
                    relation.evidence,
                    relation.uncertainty,
                )
        for node in context.nodes:
            self._was_visible[node.node_id] = node.visible
            if node.node.last_seen is not None:
                self._last_seen[node.node_id] = node.node.last_seen.frame_index
        return sorted(
            out.values(),
            key=lambda r: (
                r.predicate.value,
                r.subject_node_id,
                r.object_node_id or "",
            ),
        )

    def _pairwise(self, context, visible, emit) -> None:
        c, width = self.config, context.width
        margin = c.lateral_margin_fraction * width
        for index, a in enumerate(visible):
            for b in visible[index + 1 :]:
                p = a.existence * b.existence
                (ax0, ay0, ax1, ay1), (bx0, by0, bx1, by1) = a.box, b.box
                acx, bcx = (ax0 + ax1) / 2, (bx0 + bx1) / 2
                if acx + margin < bcx:
                    emit(a.node_id, P.LEFT_OF, p, GEOMETRY, b.node_id)
                    emit(b.node_id, P.RIGHT_OF, p, GEOMETRY, a.node_id)
                elif bcx + margin < acx:
                    emit(b.node_id, P.LEFT_OF, p, GEOMETRY, a.node_id)
                    emit(a.node_id, P.RIGHT_OF, p, GEOMETRY, b.node_id)
                overlap_x = min(ax1, bx1) - max(ax0, bx0)
                min_width = min(ax1 - ax0, bx1 - bx0)
                if overlap_x >= c.above_min_horizontal_overlap * min_width:
                    tolerance = c.on_vertical_tolerance_fraction * context.height
                    if ay1 <= by0 + tolerance:
                        emit(a.node_id, P.ABOVE, p, GEOMETRY, b.node_id)
                        emit(b.node_id, P.BELOW, p, GEOMETRY, a.node_id)
                    elif by1 <= ay0 + tolerance:
                        emit(b.node_id, P.ABOVE, p, GEOMETRY, a.node_id)
                        emit(a.node_id, P.BELOW, p, GEOMETRY, b.node_id)
                first, second = sorted(
                    (a, b), key=lambda n: n.node_id
                )  # symmetric: emitted once
                iou = _iou(a.box, b.box)
                if iou >= c.overlap_min_iou:
                    emit(
                        first.node_id,
                        P.OVERLAPS,
                        p,
                        GEOMETRY,
                        second.node_id,
                        evidence=(f"iou={iou:.2f}",),
                    )
                if a.ground is not None and b.ground is not None:
                    lo, hi = _pair_distance_bounds(
                        a.ground, b.ground, c.depth_relative_error
                    )
                    bounds = Uncertainty(
                        lower_bound=lo,
                        upper_bound=hi,
                        calibration_id="ground_contact_distance",
                    )
                    emit(
                        first.node_id,
                        P.NEAR,
                        p * _prob_below(c.near_distance, lo, hi),
                        GEOMETRY,
                        second.node_id,
                        uncertainty=bounds,
                    )
                    emit(
                        first.node_id,
                        P.FAR,
                        p * (1 - _prob_below(c.far_distance, lo, hi)),
                        GEOMETRY,
                        second.node_id,
                        uncertainty=bounds,
                    )
                    gap = max(bx0 - ax1, ax0 - bx1)
                    if (
                        iou < c.overlap_min_iou
                        and gap <= c.adjacent_max_gap_fraction * width
                        and abs(a.ground[1] - b.ground[1]) <= c.adjacent_max_depth_gap
                    ):
                        emit(
                            first.node_id,
                            P.ADJACENT,
                            p,
                            GEOMETRY,
                            second.node_id,
                            evidence=(f"gap_px={gap:.1f}",),
                        )
                for top, base in ((a, b), (b, a)):
                    self._on(context, top, base, emit)

    def _on(self, context, top: NodeView, base: NodeView, emit) -> None:
        """`top` rests on `base`: its bottom edge sits in the upper band of `base`, it is centred over
        it, sticks out above it, is narrower, and `base` can support things."""
        c = self.config
        support = base.belief(c.support_type)
        rests = max((top.belief(t) or 0.0) for t in c.on_subject_types)
        if not support or rests < 0.5:
            return
        (tx0, ty0, tx1, ty1), (bx0, by0, bx1, by1) = top.box, base.box
        tolerance = c.on_vertical_tolerance_fraction * context.height
        band = max(tolerance, 0.35 * (by1 - by0))
        if (
            bx0 <= (tx0 + tx1) / 2 <= bx1
            and by0 - tolerance <= ty1 <= by0 + band
            and ty0 < by0
            and tx1 - tx0 < bx1 - bx0
        ):
            p = c.on_probability * support * top.existence * base.existence
            emit(
                top.node_id,
                P.ON,
                p,
                TASK,
                base.node_id,
                evidence=("bottom_on_top_edge",),
            )
            emit(
                base.node_id,
                P.SUPPORTS,
                p,
                TASK,
                top.node_id,
                evidence=("bottom_on_top_edge",),
            )

    def _agent_relative(self, context, visible, emit) -> None:
        c, agent, t = self.config, context.agent_id, context.stamp.timestamp_seconds
        for node in visible:
            if node.distance is None:
                continue
            estimate, lo, hi = node.distance
            bounds = Uncertainty(
                lower_bound=lo, upper_bound=hi, calibration_id="ground_contact_distance"
            )
            emit(
                node.node_id,
                P.NEAR,
                node.existence * _prob_below(c.near_distance, lo, hi),
                GEOMETRY,
                agent,
                uncertainty=bounds,
            )
            emit(
                node.node_id,
                P.FAR,
                node.existence * (1 - _prob_below(c.far_distance, lo, hi)),
                GEOMETRY,
                agent,
                uncertainty=bounds,
            )
            history = self._distances.setdefault(
                node.node_id, deque(maxlen=c.motion_window_frames + 1)
            )
            history.append((t, estimate))
            if len(history) > c.motion_window_frames:
                (t0, d0), (t1, d1) = history[0], history[-1]
                rate = (d1 - d0) / (t1 - t0) if t1 > t0 else 0.0
                evidence = (f"range_rate={rate:.2f}/s",)
                if rate <= -c.motion_min_speed:
                    emit(
                        node.node_id,
                        P.MOVING_TOWARD,
                        node.existence,
                        GEOMETRY,
                        agent,
                        evidence=evidence,
                    )
                elif rate >= c.motion_min_speed:
                    emit(
                        node.node_id,
                        P.MOVING_AWAY,
                        node.existence,
                        GEOMETRY,
                        agent,
                        evidence=evidence,
                    )
        for node in context.nodes:
            if not node.visible or node.distance is None:
                self._distances.pop(
                    node.node_id, None
                )  # a gap breaks the motion window

    def _occlusion(self, context, emit) -> None:
        c = self.config
        by_id = {n.node_id: n for n in context.nodes}
        for node in context.nodes:
            occluder = context.track_to_node.get(node.track.occluder_track_id or "")
            if occluder is not None and occluder in by_id:
                p = node.existence * by_id[occluder].existence
                emit(
                    occluder,
                    P.OCCLUDES,
                    c.occludes_probability_tracker * p,
                    TRACKER,
                    node.node_id,
                    evidence=("tracker_occluder",),
                )
                if node.memory_only:
                    emit(
                        node.node_id,
                        P.LIKELY_OCCLUDER,
                        c.likely_occluder_probability_tracker * p,
                        TRACKER,
                        occluder,
                        evidence=("tracker_occluder",),
                    )
            front, coverage = self._front_cover(node, context.nodes)
            if front is None or coverage < c.occludes_min_coverage:
                continue
            p = node.existence * front.existence
            evidence = (f"coverage={coverage:.2f}",)
            if node.visible:
                emit(
                    front.node_id,
                    P.OCCLUDES,
                    c.occludes_probability_geometry * p,
                    GEOMETRY,
                    node.node_id,
                    evidence=evidence,
                )
            elif (
                node.memory_only
                and node.node.memory_state is not MemoryState.OUT_OF_VIEW
            ):
                emit(
                    node.node_id,
                    P.LIKELY_OCCLUDER,
                    c.likely_occluder_probability_geometry * p,
                    GEOMETRY,
                    front.node_id,
                    evidence=evidence,
                )

    def _front_cover(self, node: NodeView, nodes) -> tuple[NodeView | None, float]:
        """The visible node nearer than `node` (lower bottom edge) that covers most of its box."""
        best, best_cover = None, 0.0
        area = _area(node.box)
        for other in nodes:
            if (
                other is node
                or not other.visible
                or other.box[3] <= node.box[3]
                or area <= 0
            ):
                continue
            cover = _intersection(node.box, other.box) / area
            if cover > best_cover:
                best, best_cover = other, cover
        return best, best_cover

    def _temporal(self, context, emit) -> None:
        c = self.config
        for node in context.nodes:
            state, last_seen = node.node.memory_state, node.node.last_seen
            if (
                node.track.record.lifecycle is TrackLifecycle.REAPPEARED
                and node.node_id in self._last_seen
            ):
                gap = context.stamp.frame_index - self._last_seen[node.node_id]
                emit(
                    node.node_id,
                    P.REAPPEARED_AFTER,
                    node.existence,
                    MEMORY,
                    value=gap,
                    evidence=(f"unseen_frames={gap}",),
                )
            if not node.memory_only or last_seen is None:
                continue
            emit(
                node.node_id,
                P.PREVIOUSLY_VISIBLE,
                node.existence,
                MEMORY,
                value=last_seen.frame_index,
            )
            if node.existence >= c.believed_present_min_existence:
                emit(
                    node.node_id,
                    P.BELIEVED_PRESENT,
                    node.existence,
                    MEMORY,
                    value=state.value,
                )
            # It just vanished *in view* (not by leaving the frame or a blackout): what was in front?
            if self._was_visible.get(node.node_id) and state in (
                MemoryState.OCCLUDED,
                MemoryState.LOST,
            ):
                occluder = context.track_to_node.get(node.track.occluder_track_id or "")
                by_id = {n.node_id: n for n in context.nodes}
                if occluder in by_id:
                    p = (
                        c.last_seen_behind_probability_tracker
                        * node.existence
                        * by_id[occluder].existence
                    )
                    emit(
                        node.node_id,
                        P.LAST_SEEN_BEHIND,
                        p,
                        MEMORY,
                        occluder,
                        evidence=("tracker_occluder",),
                    )
                else:
                    front, coverage = self._front_cover(node, context.nodes)
                    if front is not None and coverage >= c.occludes_min_coverage:
                        p = (
                            c.last_seen_behind_probability_geometry
                            * node.existence
                            * front.existence
                        )
                        emit(
                            node.node_id,
                            P.LAST_SEEN_BEHIND,
                            p,
                            MEMORY,
                            front.node_id,
                            evidence=(f"coverage={coverage:.2f}",),
                        )

    def _task(self, context, emit) -> None:
        c, width, height = self.config, context.width, context.height
        goals = []
        for node in context.nodes:
            if node.node.memory_state is MemoryState.TERMINATED:
                continue
            collectable, harmful = node.beliefs.get(c.goal_type), node.beliefs.get(
                c.danger_type, (0.0, 0.0, 0.0)
            )
            if collectable is None or collectable[0] is None or harmful[0] is None:
                continue
            q = (
                collectable[0] * (1 - harmful[0]) * node.existence
            )  # a rewarding goal that exists
            bounds = (
                collectable[1] * (1 - harmful[2]) * node.existence,
                collectable[2] * (1 - harmful[1]) * node.existence,
            )
            if q > 0:
                goals.append((node, q, bounds))
        start = np.array(
            [width / 2, float(height)]
        )  # the agent stands at the bottom centre of its view
        half = c.path_half_width_fraction * width
        for goal, q, (q_lo, q_hi) in goals:
            if goal.node.memory_state is MemoryState.OUT_OF_VIEW:
                continue
            end = np.array([(goal.box[0] + goal.box[2]) / 2, goal.box[3]])
            for other in context.nodes:
                if other is goal or not other.visible or other.box[3] <= goal.box[3]:
                    continue  # only things nearer than the goal can be in the way
                if not _segment_hits_box(start, end, other.box, half):
                    continue
                for predicate, kind, base in (
                    (P.DANGER_BETWEEN, c.danger_type, c.danger_between_probability),
                    (P.PATH_BLOCKED_BY, c.barrier_type, c.path_blocked_probability),
                ):
                    point, lo, hi = other.beliefs.get(kind, (0.0, 0.0, 0.0))
                    if point:
                        k = base * other.existence
                        emit(
                            goal.node_id,
                            predicate,
                            k * q * point,
                            TASK,
                            other.node_id,
                            evidence=("on_straight_path",),
                            uncertainty=Uncertainty(
                                lower_bound=k * q_lo * lo,
                                upper_bound=min(1.0, k * q_hi * hi),
                                calibration_id="class_unknown_mass",
                            ),
                        )
        margin = c.lateral_margin_fraction * width
        left = [
            q
            for node, q, _ in goals
            if (node.box[0] + node.box[2]) / 2 < width / 2 - margin
        ]
        right = [
            q
            for node, q, _ in goals
            if (node.box[0] + node.box[2]) / 2 > width / 2 + margin
        ]
        if left or right:
            p_left, p_right = _p_more(left, right)
            evidence = (
                f"expected_left={sum(left):.2f}",
                f"expected_right={sum(right):.2f}",
                "includes_memory_only",
            )
            emit(
                context.agent_id,
                P.MORE_GOALS_LEFT,
                p_left,
                TASK,
                value=round(sum(left), 3),
                evidence=evidence,
            )
            emit(
                context.agent_id,
                P.MORE_GOALS_RIGHT,
                p_right,
                TASK,
                value=round(sum(right), 3),
                evidence=evidence,
            )


def ground_position(
    box: np.ndarray, width: int, height: int, config: SceneGraphConfig
) -> tuple[float, float] | None:
    f = config.camera_focal_fraction * width
    cx, cy = config.camera_cx_fraction * width, config.camera_cy_fraction * height
    rows = box[3] - cy
    if box[3] >= height - 1 or rows < config.depth_min_rows_below_horizon:
        return None  # bottom edge not visible, or at the horizon: distance unknown
    z = config.camera_height * f / rows
    return (((box[0] + box[2]) / 2 - cx) * z / f, z)


def _p_more(left: list[float], right: list[float]) -> tuple[float, float]:
    """P(#left > #right), P(#right > #left) for independent Bernoulli goals (exact convolution)."""
    offset = len(right)
    dist = np.zeros(len(left) + len(right) + 1)
    dist[offset] = 1.0
    for q in left:
        dist = dist * (1 - q) + np.roll(dist, 1) * q
    for q in right:
        dist = dist * (1 - q) + np.roll(dist, -1) * q
    return float(dist[offset + 1 :].sum()), float(dist[:offset].sum())


def _prob_below(threshold: float, lo: float, hi: float) -> float:
    """P(d < threshold) with d uniform on [lo, hi]."""
    if hi <= lo:
        return 1.0 if lo < threshold else 0.0
    return min(max((threshold - lo) / (hi - lo), 0.0), 1.0)


def _pair_distance_bounds(a, b, error: float) -> tuple[float, float]:
    candidates = [
        math.dist((a[0] * sa, a[1] * sa), (b[0] * sb, b[1] * sb))
        for sa in (1 - error, 1 + error)
        for sb in (1 - error, 1 + error)
    ]
    return min(candidates), max(candidates)


def _segment_hits_box(
    start: np.ndarray, end: np.ndarray, box: np.ndarray, half_width: float
) -> bool:
    x0, y0, x1, y1 = box[0] - half_width, box[1], box[2] + half_width, box[3]
    for t in np.linspace(0.0, 1.0, 33):
        x, y = start + t * (end - start)
        if x0 <= x <= x1 and y0 <= y <= y1:
            return True
    return False


def _area(box: np.ndarray) -> float:
    return max(box[2] - box[0], 0.0) * max(box[3] - box[1], 0.0)


def _intersection(a: np.ndarray, b: np.ndarray) -> float:
    return max(min(a[2], b[2]) - max(a[0], b[0]), 0.0) * max(
        min(a[3], b[3]) - max(a[1], b[1]), 0.0
    )


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = _intersection(a, b)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0
