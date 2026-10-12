"""
Offline identity evaluation of tracker output against capture instance IDs.

Evaluation only: the capture frames it takes are privileged ground truth, so this module is never
used on the runtime path. Each frame, tracks that were matched to a detection are paired greedily
with ground-truth objects (IoU >= `min_iou`, visibility >= `min_visibility`). Per ground-truth
instance it counts the distinct track IDs it received; an ID switch is a change of track ID
between consecutive matches, classified by what happened to the previous track.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping

from perception.contracts import BoundingBox2D, CaptureFrame, TrackFrame


def identity_metrics(
    tracks: Iterable[TrackFrame],
    ground_truth: Mapping[int, CaptureFrame],
    *,
    min_iou: float = 0.5,
    min_visibility: float = 0.2,
    min_matches: int = 10,
) -> dict:
    ids: dict[str, set[str]] = defaultdict(set)
    matches: Counter[str] = Counter()
    last: dict[str, str] = {}
    causes: Counter[str] = Counter()
    visible_objects = 0
    for frame in tracks:
        truth = ground_truth.get(frame.stamp.frame_index)
        if truth is None:
            continue
        alive = {t.record.track_id: t for t in frame.tracks}
        seen = [t for t in frame.tracks if t.record.detector_id is not None]
        objects = [
            o
            for o in truth.objects
            if o.bounding_box is not None
            and (o.visibility_fraction or 0.0) >= min_visibility
        ]
        visible_objects += len(objects)
        pairs = sorted(
            (
                (_iou(t.bounding_box, o.bounding_box), t.record.track_id, o.instance_id)
                for t in seen
                for o in objects
            ),
            reverse=True,
        )
        used_tracks, used_objects = set(), set()
        for overlap, track_id, instance in pairs:
            if overlap < min_iou or track_id in used_tracks or instance in used_objects:
                continue
            used_tracks.add(track_id)
            used_objects.add(instance)
            ids[instance].add(track_id)
            matches[instance] += 1
            previous = last.get(instance)
            if previous is not None and previous != track_id:
                old = alive.get(previous)
                if old is None:
                    causes["previous track archived or terminated"] += 1
                elif old.record.detector_id is not None:
                    causes["previous track took another detection"] += 1
                else:
                    causes[
                        f"previous track alive but unmatched ({old.memory.state.value})"
                    ] += 1
            last[instance] = track_id
    kept = [i for i in ids if matches[i] >= min_matches]
    counts = sorted(len(ids[i]) for i in kept)
    return {
        "instances": len(kept),
        "matched_observations": sum(matches[i] for i in kept),
        "visible_observations": visible_objects,
        "id_switches": sum(causes.values()),
        "ids_per_instance_median": statistics.median(counts) if counts else None,
        "ids_per_instance_max": counts[-1] if counts else None,
        "matched_frames_per_id": (
            sum(matches[i] for i in kept) / sum(counts) if counts else None
        ),
        "switch_causes": dict(causes.most_common()),
        "per_instance": {
            i: {"ids": len(ids[i]), "matched_frames": matches[i]} for i in kept
        },
    }


def _iou(a: BoundingBox2D, b: BoundingBox2D) -> float:
    width = max(0.0, min(a.x_max, b.x_max) - max(a.x_min, b.x_min))
    height = max(0.0, min(a.y_max, b.y_max) - max(a.y_min, b.y_min))
    inter = width * height
    union = (
        (a.x_max - a.x_min) * (a.y_max - a.y_min)
        + (b.x_max - b.x_min) * (b.y_max - b.y_min)
        - inter
    )
    return inter / union if union > 0 else 0.0
