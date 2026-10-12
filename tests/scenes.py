"""Synthetic scene fixtures (`tests/fixtures/scenes/*.json`) -> detections -> tracks -> scene graphs.

A scene is 256x256 with the Animal-AI horizon at row 128, so floor-contact distances are
meaningful. Each object has keyframed xyxy boxes (linear in between, held after the last one;
absent outside `from`..`until`), a class `label` and detector `score`. `occluded_by` clips the object's
detection to the part not covered by that nearer object and drops it when fully hidden;
`missing` lists [first, last] frame ranges where the detector misses the object (the frame is
still observable); `unobservable` lists [first, last] frame ranges passed to the tracker as
blackouts. Scene-graph fixtures live in `tests/fixtures/scenes/`, tracker-only scenarios in
`tests/fixtures/tracking_scenes/`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from perception.contracts import FrameStamp, SourceMetadata, SourceType
from perception.drawing import colour_for
from perception.inference.detector import Timing, build_detection
from perception.ontology import load_ontology
from perception.scene_graph import SceneGraphBuilder, SceneGraphConfig
from perception.tracking import Tracker, TrackerConfig, collectable_classes

ROOT = Path(__file__).resolve().parents[1]
SCENES = ROOT / "tests/fixtures/scenes"
TRACKING_SCENES = ROOT / "tests/fixtures/tracking_scenes"
SIZE, DT = 256, 0.06
SOURCE = SourceMetadata(SourceType.DETECTOR, "synthetic_scene", "1.0")


def scene_names(directory: Path = SCENES) -> list[str]:
    return sorted(p.stem for p in directory.glob("*.json"))


def load_scene(name: str, directory: Path = SCENES) -> dict:
    return json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))


def box_at(item: dict, frame: int) -> list[float] | None:
    if not item.get("from", 0) <= frame <= item.get("until", frame):
        return None
    keys = sorted((int(k), v) for k, v in item["keyframes"].items())
    if frame <= keys[0][0]:
        return keys[0][1]
    for (f0, b0), (f1, b1) in zip(keys, keys[1:]):
        if f0 <= frame <= f1:
            t = (frame - f0) / (f1 - f0)
            return [a + t * (b - a) for a, b in zip(b0, b1)]
    return keys[-1][1]


def _visible_part(box, cover):
    """`box` minus the horizontal span of `cover` (occluders here are taller than what they hide)."""
    if cover is None or cover[2] <= box[0] or cover[0] >= box[2]:
        return box
    left, right = [box[0], box[1], min(box[2], cover[0]), box[3]], [max(box[0], cover[2]), box[1], box[2], box[3]]
    part = max((left, right), key=lambda b: b[2] - b[0])
    return part if part[2] - part[0] >= 2 else None


def scene_image(scene: dict, index: int) -> np.ndarray:
    """Filled keyframed boxes, farthest first; detector dropouts do not erase scene objects."""
    image = np.full((SIZE, SIZE, 3), (150, 170, 190), np.uint8)
    image[SIZE // 2:] = (170, 160, 130)
    if any(a <= index <= b for a, b in scene.get("unobservable", [])):
        return np.zeros_like(image)
    objects = [(item, box_at(item, index)) for item in scene["objects"]]
    objects = sorted(((item, box) for item, box in objects if box is not None), key=lambda pair: pair[1][3])
    for item, box in objects:
        x0, y0, x1, y1 = (min(max(int(round(v)), 0), SIZE) for v in box)
        image[y0:y1, x0:x1] = colour_for(item["label"])
    return image


def scene_frames(scene: dict):
    """Yields `(Detection, observable)` per frame."""
    blackout = [range(a, b + 1) for a, b in scene.get("unobservable", [])]
    for index in range(scene["frames"]):
        observable = not any(index in r for r in blackout)
        candidates = []
        if observable:
            boxes = {item["name"]: box_at(item, index) for item in scene["objects"]}
            for item in scene["objects"]:
                box = boxes[item["name"]]
                if any(a <= index <= b for a, b in item.get("missing", [])):
                    continue
                if box is not None and item.get("occluded_by"):
                    box = _visible_part(box, boxes.get(item["occluded_by"]))
                if box is None:
                    continue
                mask = np.zeros((SIZE, SIZE), bool)
                x0, y0, x1, y1 = (int(round(v)) for v in box)
                mask[y0:y1, x0:x1] = True
                candidates.append((box, {item["label"]: item["score"]}, mask))
        stamp = FrameStamp(f"scene-{scene.get('name', 'fixture')}", index, index * DT)
        yield build_detection(stamp, SIZE, SIZE, candidates, SOURCE, Timing(0, 0, 0)), observable


def run_tracking_scene(name: str):
    """Returns (scene, track frames) for a `tests/fixtures/tracking_scenes/` scenario."""
    ontology = load_ontology(ROOT / "configs/ontology.json")
    tracker = Tracker(TrackerConfig.from_json_file(ROOT / "configs/tracker.json"), collectable_classes(ontology))
    scene = {**load_scene(name, TRACKING_SCENES), "name": name}
    return scene, [tracker.update(d.frame, d.masks, observable=o) for d, o in scene_frames(scene)]


def run_scene(name: str):
    """Returns (scene, track frames, snapshots, builder) using the repository configs."""
    ontology = load_ontology(ROOT / "configs/ontology.json")
    tracker = Tracker(TrackerConfig.from_json_file(ROOT / "configs/tracker.json"), collectable_classes(ontology))
    builder = SceneGraphBuilder(SceneGraphConfig.from_json_file(ROOT / "configs/scene_graph.json"), ontology)
    scene = {**load_scene(name), "name": name}
    tracks, snapshots = [], []
    for detection, observable in scene_frames(scene):
        frame = tracker.update(detection.frame, detection.masks, observable=observable)
        tracks.append(frame)
        snapshots.append(builder.update(frame, width=SIZE, height=SIZE))
    return scene, tracks, snapshots, builder
