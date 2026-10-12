"""
Multi-object tracker and temporal object memory over detector-neutral `DetectionFrame`s.

Per frame: predict every track with constant image-space velocity plus the caller's
`camera_shift` (ego-motion: how static scene content moved in the image), associate greedily (1) by IoU
with the predicted box, then (2) re-identify leftover detections against unmatched tracks by class
similarity and distance to the last-seen or predicted box (sizes within `max_scale_change`), then
decide each track's state. Both stages are class-gated: a detection of another class can only take
over a track at `relabel_iou` overlap. An unmatched detection overlapping an assigned one that much
is a duplicate and starts no track. Velocity is only measured between two un-truncated sightings.


    matched                    VISIBLE, or PARTIALLY_VISIBLE when truncated by the frame edge or
                               covered by a nearer detection (>= partial_coverage)
    frame unobservable         UNOBSERVABLE: no miss counted, existence x unobservable_decay (1.0)
    predicted mostly off-frame OUT_OF_VIEW, existence x out_of_view_decay; also when last seen cut
                               by the frame edge (it most likely left through it)
    covered by a nearer object OCCLUDED with `occluder_track_id`, existence x occluded_decay
    large, in view, gone       TERMINATED after `termination_misses` frames, only for tracks whose
                               class belief is mostly `terminable_classes` (e.g. goal collected)
    otherwise                  LOST (unexplained miss), existence x lost_decay

A miss never deletes a track by itself: it is archived only once existence < archive_existence
after >= min_misses_to_archive consecutive observable misses. A match after any non-visible state
is a REAPPEARED lifecycle event. Archived/terminated tracks are reported once more, then dropped.

"Nearer" uses the ground-contact proxy: on a floor seen from an upright camera, a lower box bottom
edge is closer. `relative_depth` reports that proxy, `1 - y_max / height` (0 = at the camera's feet,
larger = farther); it is wrong for objects not resting on the floor. `image_velocity` has the given
`camera_shift` removed; ego-motion that is not passed in (forward motion, uncalibrated turns) stays
in it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np

from perception.contracts import (
    SCHEMA_VERSION,
    BoundingBox2D,
    CategoricalDistribution,
    DetectionFrame,
    DetectorObject,
    FrameStamp,
    MaskReference,
    MemoryState,
    MotionState,
    ObjectMemory,
    PrimitiveAction,
    ProbabilityEntry,
    SourceMetadata,
    SourceType,
    TrackFrame,
    TrackLifecycle,
    TrackRecord,
    TrackState,
    Vector2,
)
from perception.ontology import Ontology

TRACKER_SOURCE = SourceMetadata(SourceType.TRACKER, "perception_tracker", "1.0.0")
MEMORY_SOURCE = SourceMetadata(
    SourceType.TEMPORAL_MEMORY, "perception_tracker", "1.0.0"
)
_SEEN = (MemoryState.VISIBLE, MemoryState.PARTIALLY_VISIBLE)


@dataclass(frozen=True, slots=True)
class TrackerConfig:
    min_iou: float
    reid_distance: (
        float  # in multiples of the track's largest observed box diagonal ...
    )
    reid_image_fraction: (
        float  # ... or this fraction of the image width, whichever is larger
    )
    min_class_similarity: (
        float  # both association stages; below it only relabel_iou can match
    )
    relabel_iou: (
        float  # overlap at which a detection may re-label a track of another class
    )
    max_scale_change: (
        float  # re-identification: max ratio between detection and track box sizes
    )
    initial_existence: (
        float  # new track existence = initial_existence x detector confidence
    )
    hit_gain: float  # on a hit: p += (1 - p) x hit_gain x confidence
    occluded_decay: float
    out_of_view_decay: float
    lost_decay: float
    unobservable_decay: float
    archive_existence: float
    min_misses_to_archive: int
    class_memory: int  # class belief averages at most this many recent observations
    velocity_smoothing: float  # EMA weight on the previous velocity
    partial_coverage: float
    occlusion_coverage: float
    out_of_view_fraction: float  # predicted-box area fraction inside the frame
    edge_margin_px: float
    collection_area_fraction: float | None  # None disables observable termination
    termination_misses: int
    min_hits_to_terminate: int
    moving_speed_px_s: float
    blackout_max_mean: (
        float  # mean pixel value at or below which a frame is unobservable
    )
    # Image shift per rotate action, as a fraction of width. Calibration: 25.5 px of 256 on the
    # curriculum captures (~0.145 s/frame). Re-measure if decision period or frame rate changes.
    rotation_shift_fraction: float

    @classmethod
    def from_json_file(cls, path: str | Path) -> "TrackerConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("tracker configuration must be a JSON object")
        try:
            config = cls(**raw)
        except TypeError as exc:
            raise ValueError(f"invalid tracker configuration: {exc}") from exc
        config.validate()
        return config

    def validate(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if value is None and item.name == "collection_area_fraction":
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
            ):
                raise ValueError(f"{item.name} must be a finite number")
        unit = (
            "rotation_shift_fraction",
            "min_iou",
            "reid_image_fraction",
            "min_class_similarity",
            "relabel_iou",
            "initial_existence",
            "hit_gain",
            "occluded_decay",
            "out_of_view_decay",
            "lost_decay",
            "unobservable_decay",
            "archive_existence",
            "velocity_smoothing",
            "partial_coverage",
            "occlusion_coverage",
            "out_of_view_fraction",
        )
        for name in unit:
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in (
            "min_misses_to_archive",
            "class_memory",
            "termination_misses",
            "min_hits_to_terminate",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_scale_change < 1:
            raise ValueError("max_scale_change must be >= 1")
        if self.min_misses_to_archive < 2:
            raise ValueError(
                "min_misses_to_archive must be >= 2: one missed frame never deletes a track"
            )


def collectable_classes(ontology: Ontology) -> frozenset[str]:
    """Semantic classes with a `collectable` member: the ones that vanish when the agent takes them."""
    return frozenset(
        m.semantic_class
        for m in ontology.simulator_mappings
        if "collectable" in m.functional_types
    )


def action_camera_shift(
    action: PrimitiveAction | None, width: int, config: TrackerConfig
) -> tuple[float, float]:
    """Ego-motion from the agent's own last action (Animal-AI rotate: 1 = right, 2 = left).
    Turning right moves the scene left. Unknown action = no shift."""
    rotate = None if action is None else action.rotate
    step = config.rotation_shift_fraction * width
    return ({1: -step, 2: step}.get(rotate, 0.0), 0.0)


def is_blackout(image: np.ndarray, config: TrackerConfig) -> bool:
    """Animal-AI blackouts render the frame (near) black: nothing is observable, so nothing is missed."""
    return float(image.mean()) <= config.blackout_max_mean


@dataclass(slots=True)
class _Track:
    track_id: str
    box: BoundingBox2D  # last observed
    mask: MaskReference | None
    predicted: np.ndarray  # xyxy, extrapolated through misses
    size: float  # largest observed box diagonal
    labels: dict[str, float]
    unknown: float
    observations: int
    existence: float
    last_seen: FrameStamp
    velocity: np.ndarray | None = None
    hits: int = 1
    misses: int = 0
    consecutive_misses: int = 0
    age: int = 1
    vanished_close: int = 0
    tracker_confidence: float | None = None
    state: MemoryState = MemoryState.VISIBLE
    occluder: str | None = None
    pending_termination: str | None = None
    last_partial: bool = False  # last observed box was truncated/covered
    ego_since_seen: np.ndarray = field(
        default_factory=lambda: np.zeros(2)
    )  # camera shift since last hit


class Tracker:
    """
    Stateful across frames of one episode; a new `episode_id` starts a fresh set of IDs.

    `terminable_classes` are the classes that can disappear for good when observed (see
    `collectable_classes`); without them only `terminate()` ends a track.
    """

    def __init__(
        self, config: TrackerConfig, terminable_classes: frozenset[str] = frozenset()
    ) -> None:
        config.validate()
        self.config = config
        self.terminable_classes = frozenset(terminable_classes)
        self._tracks: dict[str, _Track] = {}
        self._stamp: FrameStamp | None = None
        self._next_id = 0

    def terminate(self, track_id: str, reason: str) -> None:
        self._tracks[track_id].pending_termination = reason

    def update(
        self,
        frame: DetectionFrame,
        masks: np.ndarray | None = None,
        *,
        observable: bool = True,
        camera_shift: tuple[float, float] = (0.0, 0.0),
    ) -> TrackFrame:
        stamp = frame.stamp
        if self._stamp is None or stamp.episode_id != self._stamp.episode_id:
            self._tracks, self._next_id, dt = {}, 0, 0.0
        else:
            dt = stamp.timestamp_seconds - self._stamp.timestamp_seconds
            if stamp.frame_index <= self._stamp.frame_index or dt <= 0:
                raise ValueError(
                    f"frame stamps must strictly increase: {self._stamp} -> {stamp}"
                )
        self._stamp = stamp
        width, height = frame.image_width, frame.image_height
        shift = np.asarray(camera_shift, float)
        for track in self._tracks.values():
            track.age += 1
            motion = shift if track.velocity is None else shift + track.velocity * dt
            track.predicted = track.predicted + np.tile(motion, 2)
            track.ego_since_seen = track.ego_since_seen + shift

        lifecycle: dict[str, TrackLifecycle] = {}
        scores: dict[str, float] = {}
        detector_ids: dict[str, str] = {}
        visibility: dict[str, float | None] = {}
        if observable:
            objects = frame.objects
            det_masks = [
                (
                    None
                    if masks is None or o.instance_mask is None
                    else masks == o.instance_mask.instance_value
                )
                for o in objects
            ]
            matches = self._associate(objects, width)
            owner: dict[int, str] = {}
            for det_index, (track_id, score) in matches.items():
                owner[det_index] = track_id
                scores[track_id] = score
            for det_index, item in enumerate(objects):
                if det_index in owner:
                    continue
                duplicate = next(
                    (
                        owner[j]
                        for j in owner
                        if _iou(
                            _xyxy(objects[j].bounding_box), _xyxy(item.bounding_box)
                        )
                        >= self.config.relabel_iou
                    ),
                    None,
                )
                if (
                    duplicate is None
                ):  # NMS-free detectors can emit near-identical boxes: track them once
                    owner[det_index] = self._create(item, stamp)
                    lifecycle[owner[det_index]] = TrackLifecycle.CREATED
            duplicates = [i for i in range(len(objects)) if i not in owner]
            for det_index, item in enumerate(objects):
                if det_index in duplicates:
                    continue
                track = self._tracks[owner[det_index]]
                detector_ids[track.track_id] = item.detector_id
                visibility[track.track_id] = item.confidence
                full_box = (
                    track.predicted
                    if track.track_id in scores
                    else _xyxy(item.bounding_box)
                )
                cover, occluder = self._coverage(
                    full_box, det_index, objects, det_masks, owner, width, height
                )
                partial = (
                    _touches_edge(item.bounding_box, self.config.edge_margin_px)
                    or cover >= self.config.partial_coverage
                )
                if track.track_id in scores:
                    lifecycle[track.track_id] = (
                        TrackLifecycle.UPDATED
                        if track.state in _SEEN
                        else TrackLifecycle.REAPPEARED
                    )
                    self._hit(track, item, stamp, scores[track.track_id], partial)
                track.state = (
                    MemoryState.PARTIALLY_VISIBLE if partial else MemoryState.VISIBLE
                )
                track.last_partial = partial
                track.occluder = (
                    occluder
                    if partial and cover >= self.config.partial_coverage
                    else None
                )
            matched = set(owner.values())
            for track in self._tracks.values():
                if track.track_id not in matched:
                    lifecycle[track.track_id] = TrackLifecycle.MISSED
                    self._miss(track, objects, det_masks, owner, width, height)
        else:
            for track in self._tracks.values():
                lifecycle[track.track_id] = TrackLifecycle.MISSED
                track.state = MemoryState.UNOBSERVABLE
                track.existence *= self.config.unobservable_decay

        for track in self._tracks.values():
            if track.pending_termination is not None:
                track.state, track.existence = MemoryState.TERMINATED, 0.0
                lifecycle[track.track_id] = TrackLifecycle.TERMINATED
            elif (
                observable
                and track.track_id not in detector_ids
                and track.consecutive_misses >= self.config.min_misses_to_archive
                and track.existence < self.config.archive_existence
            ):
                lifecycle[track.track_id] = TrackLifecycle.ARCHIVED
        states = tuple(
            self._state(
                track,
                stamp,
                lifecycle[track.track_id],
                scores.get(track.track_id),
                detector_ids.get(track.track_id),
                visibility.get(track.track_id),
                height,
            )
            for track in self._tracks.values()
        )
        self._tracks = {
            key: track
            for key, track in self._tracks.items()
            if lifecycle[key]
            not in (TrackLifecycle.ARCHIVED, TrackLifecycle.TERMINATED)
        }
        return TrackFrame(SCHEMA_VERSION, stamp, observable, states, TRACKER_SOURCE)

    def _associate(
        self, objects: tuple[DetectorObject, ...], width: int
    ) -> dict[int, tuple[str, float]]:
        tracks = list(self._tracks.values())
        pairs = []
        for det_index, item in enumerate(objects):
            box = _xyxy(item.bounding_box)
            for track in tracks:
                overlap = _iou(track.predicted, box)
                similarity = _class_similarity(track.labels, item.semantic_distribution)
                if overlap >= self.config.min_iou and (
                    similarity >= self.config.min_class_similarity
                    or overlap >= self.config.relabel_iou
                ):
                    pairs.append(
                        (overlap * (0.5 + 0.5 * similarity), det_index, track.track_id)
                    )
        matches = _greedy(pairs)
        taken = {track_id for track_id, _ in matches.values()}
        pairs = []
        for det_index, item in enumerate(objects):
            if det_index in matches:
                continue
            centre = _centre(_xyxy(item.bounding_box))
            size = _diagonal(_xyxy(item.bounding_box))
            for track in tracks:
                last_size = _diagonal(_xyxy(track.box))
                if track.track_id in taken or max(
                    size, last_size
                ) > self.config.max_scale_change * max(min(size, last_size), 1.0):
                    continue
                similarity = _class_similarity(track.labels, item.semantic_distribution)
                distance = min(
                    np.linalg.norm(centre - _centre(_xyxy(track.box))),
                    np.linalg.norm(centre - _centre(track.predicted)),
                ) / max(track.size, 1.0)
                gate = max(
                    self.config.reid_distance,
                    self.config.reid_image_fraction * width / max(track.size, 1.0),
                )
                if similarity >= self.config.min_class_similarity and distance <= gate:
                    pairs.append(
                        (similarity * (1 - distance / gate), det_index, track.track_id)
                    )
        matches.update(_greedy(pairs))
        return matches

    def _create(self, item: DetectorObject, stamp: FrameStamp) -> str:
        track_id = f"{stamp.episode_id}:track-{self._next_id:04d}"
        self._next_id += 1
        box = _xyxy(item.bounding_box)
        distribution = item.semantic_distribution
        self._tracks[track_id] = _Track(
            track_id=track_id,
            box=item.bounding_box,
            mask=item.instance_mask,
            predicted=box,
            size=_diagonal(box),
            labels={e.label: e.probability for e in distribution.entries},
            unknown=distribution.unknown_probability,
            observations=1,
            existence=self.config.initial_existence * item.confidence,
            last_seen=stamp,
        )
        return track_id

    def _hit(
        self,
        track: _Track,
        item: DetectorObject,
        stamp: FrameStamp,
        score: float,
        partial: bool,
    ) -> None:
        config = self.config
        box = _xyxy(item.bounding_box)
        if (
            not partial
        ):  # a truncated box has a biased centre, so it would corrupt the velocity
            if not track.last_partial:
                elapsed = stamp.timestamp_seconds - track.last_seen.timestamp_seconds
                measured = (
                    _centre(box) - _centre(_xyxy(track.box)) - track.ego_since_seen
                ) / elapsed
                track.velocity = (
                    measured
                    if track.velocity is None
                    else (
                        config.velocity_smoothing * track.velocity
                        + (1 - config.velocity_smoothing) * measured
                    )
                )
            track.size = max(track.size, _diagonal(box))
        track.predicted = box
        weight = min(track.observations, config.class_memory)
        observed = {e.label: e.probability for e in item.semantic_distribution.entries}
        track.labels = {
            label: (track.labels.get(label, 0.0) * weight + observed.get(label, 0.0))
            / (weight + 1)
            for label in track.labels.keys() | observed.keys()
        }
        track.unknown = (
            track.unknown * weight + item.semantic_distribution.unknown_probability
        ) / (weight + 1)
        track.observations += 1
        track.existence += (1 - track.existence) * config.hit_gain * item.confidence
        track.tracker_confidence = (
            score
            if track.tracker_confidence is None
            else 0.5 * (track.tracker_confidence + score)
        )
        track.box, track.mask, track.last_seen = (
            item.bounding_box,
            item.instance_mask,
            stamp,
        )
        track.hits += 1
        track.consecutive_misses = track.vanished_close = 0
        track.ego_since_seen = np.zeros(2)

    def _miss(self, track, objects, det_masks, owner, width, height) -> None:
        config = self.config
        track.misses += 1
        track.consecutive_misses += 1
        inside = _inside_fraction(track.predicted, width, height)
        cover, occluder = self._coverage(
            track.predicted, None, objects, det_masks, owner, width, height
        )
        if (
            occluder is None
            and track.occluder in self._tracks
            and track.occluder not in owner.values()
        ):
            retained = self._tracks[
                track.occluder
            ]  # occluder itself missed: use its prediction
            if retained.predicted[3] >= track.predicted[3]:
                retained_cover = _box_coverage(
                    track.predicted, retained.predicted, width, height
                )
                if retained_cover > cover:
                    cover, occluder = retained_cover, retained.track_id
        area_fraction = _area(_xyxy(track.box)) / (width * height)
        # Left through the edge it was cut by (e.g. the agent turned): gone from view, not lost.
        left_frame = _touches_edge(track.box, config.edge_margin_px) and (
            track.consecutive_misses == 1 or track.state is MemoryState.OUT_OF_VIEW
        )
        if inside < config.out_of_view_fraction or left_frame:
            track.state, track.occluder, track.vanished_close = (
                MemoryState.OUT_OF_VIEW,
                None,
                0,
            )
            track.existence *= config.out_of_view_decay
        elif cover >= config.occlusion_coverage:
            track.state, track.occluder, track.vanished_close = (
                MemoryState.OCCLUDED,
                occluder,
                0,
            )
            track.existence *= config.occluded_decay
        else:
            terminable = sum(
                p
                for label, p in track.labels.items()
                if label in self.terminable_classes
            )
            close = (
                config.collection_area_fraction is not None
                and track.hits >= config.min_hits_to_terminate
                and area_fraction >= config.collection_area_fraction
                and inside >= 0.9
                and terminable > 0.5
            )
            track.vanished_close = track.vanished_close + 1 if close else 0
            if track.vanished_close >= config.termination_misses:
                track.pending_termination = "vanished_in_view_at_close_range"
            track.state, track.occluder = MemoryState.LOST, None
            track.existence *= config.lost_decay

    def _coverage(
        self, box, own_index, objects, det_masks, owner, width, height
    ) -> tuple[float, str | None]:
        x0, y0, x1, y1 = (int(round(v)) for v in _clip(box, width, height))
        if x1 <= x0 or y1 <= y0:
            return 0.0, None
        union = np.zeros((y1 - y0, x1 - x0), bool)
        best, best_owner = 0.0, None
        for index, item in enumerate(objects):
            other = item.bounding_box
            if (
                index == own_index or other.y_max < box[3]
            ):  # only nearer (lower bottom edge) objects occlude
                continue
            if det_masks[index] is not None:
                region = det_masks[index][y0:y1, x0:x1]
            else:
                region = np.zeros_like(union)
                region[
                    max(int(other.y_min) - y0, 0) : max(
                        int(math.ceil(other.y_max)) - y0, 0
                    ),
                    max(int(other.x_min) - x0, 0) : max(
                        int(math.ceil(other.x_max)) - x0, 0
                    ),
                ] = True
            share = float(region.mean())
            if share > best:
                best, best_owner = share, owner.get(index)
            union |= region
        return float(union.mean()), best_owner

    def _state(
        self, track, stamp, lifecycle, score, detector_id, confidence, height
    ) -> TrackState:
        speed = (
            None if track.velocity is None else float(np.linalg.norm(track.velocity))
        )
        motion = (
            MotionState.UNKNOWN
            if speed is None
            else (
                MotionState.MOVING
                if speed > self.config.moving_speed_px_s
                else MotionState.STATIONARY
            )
        )
        if track.state in _SEEN:
            visible: float | None = confidence
        elif track.state in (
            MemoryState.OCCLUDED,
            MemoryState.OUT_OF_VIEW,
            MemoryState.TERMINATED,
        ):
            visible = 0.0
        else:
            visible = None  # LOST / UNOBSERVABLE: whether it could be seen is unknown
        entries = tuple(
            ProbabilityEntry(label, p)
            for label, p in sorted(track.labels.items(), key=lambda kv: -kv[1])
            if p > 0
        )
        unknown = max(0.0, 1.0 - sum(e.probability for e in entries))
        return TrackState(
            record=TrackRecord(
                track.track_id,
                detector_id,
                stamp,
                lifecycle,
                score,
                track.age,
                track.misses,
                None,
                motion,
                TRACKER_SOURCE,
            ),
            memory=ObjectMemory(
                track.track_id,
                track.state,
                visible,
                min(max(track.existence, 0.0), 1.0),
                track.last_seen,
                stamp,
                track.age,
                MEMORY_SOURCE,
            ),
            semantic_distribution=CategoricalDistribution(entries, unknown),
            bounding_box=track.box,
            instance_mask=track.mask,
            image_velocity=(
                None
                if track.velocity is None
                else Vector2(*(float(v) for v in track.velocity))
            ),
            relative_depth=1.0 - track.box.y_max / height,
            hits=track.hits,
            tracker_confidence=track.tracker_confidence,
            occluder_track_id=(
                track.occluder if track.occluder in self._tracks else None
            ),
            termination_reason=(
                track.pending_termination
                if track.state is MemoryState.TERMINATED
                else None
            ),
        )


def _greedy(pairs: list[tuple[float, int, str]]) -> dict[int, tuple[str, float]]:
    # ponytail: greedy by score, not Hungarian; swap in scipy.optimize.linear_sum_assignment if
    # dense crowds of same-class objects start trading IDs.
    matches: dict[int, tuple[str, float]] = {}
    used: set[str] = set()
    for score, det_index, track_id in sorted(pairs, key=lambda p: (-p[0], p[1], p[2])):
        if det_index not in matches and track_id not in used:
            matches[det_index] = (track_id, min(score, 1.0))
            used.add(track_id)
    return matches


def _class_similarity(
    labels: dict[str, float], distribution: CategoricalDistribution
) -> float:
    """Cosine similarity over known classes; 1.0 when either side has no class evidence (unknown
    is not a contradiction)."""
    other = {e.label: e.probability for e in distribution.entries}
    norm = math.sqrt(
        sum(p * p for p in labels.values()) * sum(p * p for p in other.values())
    )
    return (
        1.0
        if norm == 0
        else sum(p * other.get(label, 0.0) for label, p in labels.items()) / norm
    )


def _xyxy(box: BoundingBox2D) -> np.ndarray:
    return np.array([box.x_min, box.y_min, box.x_max, box.y_max], float)


def _centre(box: np.ndarray) -> np.ndarray:
    return (box[:2] + box[2:]) / 2


def _diagonal(box: np.ndarray) -> float:
    return float(np.linalg.norm(box[2:] - box[:2]))


def _area(box: np.ndarray) -> float:
    return max(box[2] - box[0], 0.0) * max(box[3] - box[1], 0.0)


def _clip(box: np.ndarray, width: int, height: int) -> np.ndarray:
    return np.clip(box, 0, [width, height, width, height])


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = _area(np.concatenate([np.maximum(a[:2], b[:2]), np.minimum(a[2:], b[2:])]))
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def _inside_fraction(box: np.ndarray, width: int, height: int) -> float:
    area = _area(box)
    return _area(_clip(box, width, height)) / area if area > 0 else 0.0


def _box_coverage(box: np.ndarray, cover: np.ndarray, width: int, height: int) -> float:
    clipped = _clip(box, width, height)
    area = _area(clipped)
    inter = _area(
        np.concatenate(
            [np.maximum(clipped[:2], cover[:2]), np.minimum(clipped[2:], cover[2:])]
        )
    )
    return inter / area if area > 0 else 0.0


def _touches_edge(box: BoundingBox2D, margin: float) -> bool:
    return (
        box.x_min <= margin
        or box.y_min <= margin
        or box.x_max >= box.image_width - margin
        or box.y_max >= box.image_height - margin
    )
