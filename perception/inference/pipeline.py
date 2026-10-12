"""
Run any `Detector` over a frame stream, validate, serialize, and report latency/throughput.

Output layout under `output_root` (all optional parts are skipped when `output_root` is None):

    detections.jsonl                         one `DetectionFrame` per line, in stream order
    masks/<episode>/<frame:06d>.png          uint16 instance map (see `perception.inference.detector`)
    timing.jsonl                             per-frame stage timing in ms
    rgb/<episode>/<frame:06d>.png            input frames, if `save_frames` (replayable prerecorded)
    observations.jsonl                       `ObservationFrame` per line, if `save_frames` and the
                                             source supplies agent state
    tracks.jsonl                             one `TrackFrame` per line, if a `tracker` is given
    graphs.jsonl                             one `SSGSnapshot` per line, if a `graph_builder` is given
    graphs/<episode>/<frame:06d>.png         snapshot rendering, if `visualize` and a `graph_builder`
    overlays/<episode>/<frame:06d>.png       box/mask rendering (track IDs when tracking), if `visualize`
    summary.json                             counts, latency percentiles, throughput, tracking counts
"""

from __future__ import annotations

import itertools
import json
import statistics
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from perception.contracts import (
    SCHEMA_VERSION,
    DetectionFrame,
    ImageEncoding,
    ImageReference,
    ObservationFrame,
    SourceMetadata,
    SourceType,
    SSGSnapshot,
    TrackFrame,
    TrackLifecycle,
)
from perception.inference.detector import Detection, Detector, Frame
from perception.jsonio import write_json
from perception.ontology import Ontology
from perception.scene_graph import SceneGraphBuilder
from perception.scene_graph.render import render_snapshot
from perception.tracking import Tracker, action_camera_shift, is_blackout

OBSERVATION_SOURCE = SourceMetadata(
    SourceType.SENSOR, "animalai-observation-adapter", "1.0.0"
)


def run_inference(
    detector: Detector,
    frames: Iterable[Frame],
    output_root: str | Path | None = None,
    *,
    batch_size: int = 1,
    ontology: Ontology | None = None,
    visualize: bool = False,
    save_frames: bool = False,
    on_detection: Callable[[Frame, Detection], None] | None = None,
    tracker: Tracker | None = None,
    on_tracks: Callable[[Frame, TrackFrame], None] | None = None,
    graph_builder: SceneGraphBuilder | None = None,
    on_graph: Callable[[Frame, SSGSnapshot], None] | None = None,
) -> dict[str, Any]:
    if graph_builder is not None and tracker is None:
        raise ValueError("a graph_builder needs a tracker")
    root = Path(output_root) if output_root is not None else None
    if root is not None:
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"{root} is not empty; pick a new output directory")
        root.mkdir(parents=True, exist_ok=True)
    files = _open_outputs(
        root, save_frames, tracker is not None, graph_builder is not None
    )
    frame_ms: list[float] = []
    stage_ms: dict[str, list[float]] = {
        "preprocess": [],
        "inference": [],
        "postprocess": [],
        "tracking": [],
        "graph": [],
    }
    graph_counts = {
        "snapshots": 0,
        "macro_steps": 0,
        "relations": 0,
        "memory_only_nodes": 0,
    }
    counts = {"frames": 0, "empty_frames": 0, "detections": 0}
    lifecycle_counts = {event.value: 0 for event in TrackLifecycle}
    unobservable_frames = 0
    started = time.perf_counter()
    try:
        for batch in itertools.batched(frames, batch_size):
            batch_start = time.perf_counter()
            detections = detector.detect_batch(batch)
            per_frame_ms = (time.perf_counter() - batch_start) * 1000 / len(batch)
            if len(detections) != len(batch):
                raise RuntimeError(
                    f"{detector.name} returned {len(detections)} results for {len(batch)} frames"
                )
            for frame, detection in zip(batch, detections, strict=True):
                validate_detection(frame, detection, ontology)
                counts["frames"] += 1
                counts["detections"] += len(detection.frame.objects)
                counts["empty_frames"] += not detection.frame.objects
                frame_ms.append(per_frame_ms)
                for stage in ("preprocess", "inference", "postprocess"):
                    stage_ms[stage].append(getattr(detection.timing, f"{stage}_ms"))
                tracks = None
                if tracker is not None:
                    track_start = time.perf_counter()
                    observable = not is_blackout(frame.image, tracker.config)
                    shift = action_camera_shift(
                        frame.agent and frame.agent.recent_action,
                        frame.image.shape[1],
                        tracker.config,
                    )
                    tracks = tracker.update(
                        detection.frame,
                        detection.masks,
                        observable=observable,
                        camera_shift=shift,
                    )
                    stage_ms["tracking"].append(
                        (time.perf_counter() - track_start) * 1000
                    )
                    unobservable_frames += not observable
                    for item in tracks.tracks:
                        lifecycle_counts[item.record.lifecycle.value] += 1
                graph = None
                if graph_builder is not None:
                    graph_start = time.perf_counter()
                    height, width = frame.image.shape[:2]
                    graph = graph_builder.update(
                        tracks, width=width, height=height, agent=frame.agent
                    )
                    stage_ms["graph"].append((time.perf_counter() - graph_start) * 1000)
                    graph_counts["snapshots"] += 1
                    graph_counts["macro_steps"] += graph.is_macro_step
                    graph_counts["relations"] += len(graph.relations)
                    graph_counts["memory_only_nodes"] += sum(
                        n.memory_state.value not in ("visible", "partially_visible")
                        for n in graph.objects
                    )
                if root is not None:
                    _write(
                        root,
                        files,
                        frame,
                        detection,
                        per_frame_ms,
                        visualize,
                        save_frames,
                        tracks,
                        graph,
                    )
                if on_detection is not None:
                    on_detection(frame, detection)
                if on_tracks is not None and tracks is not None:
                    on_tracks(frame, tracks)
                if on_graph is not None and graph is not None:
                    on_graph(frame, graph)
    finally:
        for handle in files.values():
            handle.close()
    wall_seconds = time.perf_counter() - started
    summary = {
        "detector": {
            "name": detector.name,
            "version": detector.version,
            "device": detector.device,
        },
        "batch_size": batch_size,
        **counts,
        "latency_ms": {
            "per_frame_wall": _stats(frame_ms),
            **{k: _stats(v) for k, v in stage_ms.items()},
        },
        "throughput": {
            "detector_fps": (
                counts["frames"] / (sum(frame_ms) / 1000) if sum(frame_ms) else None
            ),
            "end_to_end_fps": counts["frames"] / wall_seconds if wall_seconds else None,
            "wall_seconds": wall_seconds,
        },
    }
    if tracker is not None:
        summary["tracking"] = {
            "unobservable_frames": unobservable_frames,
            "lifecycle_events": lifecycle_counts,
        }
    if graph_builder is not None:
        summary["scene_graph"] = graph_counts
    if root is not None:
        write_json(root / "summary.json", summary)
    return summary


def validate_detection(
    frame: Frame, detection: Detection, ontology: Ontology | None
) -> None:
    """Phase-1 contract checks the dataclass constructors cannot do on their own."""
    result = detection.frame
    height, width = frame.image.shape[:2]
    if result.stamp != frame.stamp:
        raise ValueError(f"detector changed the stamp: {frame.stamp} -> {result.stamp}")
    if (result.image_width, result.image_height) != (width, height):
        raise ValueError("detection frame size does not match the input frame")
    if detection.masks is not None and detection.masks.shape != (height, width):
        raise ValueError(
            f"instance map is {detection.masks.shape}, frame is {(height, width)}"
        )
    if (
        any(item.instance_mask is not None for item in result.objects)
        and detection.masks is None
    ):
        raise ValueError("objects reference an instance mask but none was produced")
    if ontology is not None:
        for item in result.objects:
            ontology.validate_distribution(item.semantic_distribution)


def _open_outputs(
    root: Path | None, save_frames: bool, tracking: bool, graphs: bool
) -> dict[str, Any]:
    if root is None:
        return {}
    names = (
        ["detections", "timing"]
        + (["observations"] if save_frames else [])
        + (["tracks"] if tracking else [])
        + (["graphs"] if graphs else [])
    )
    return {
        name: (root / f"{name}.jsonl").open("w", encoding="utf-8") for name in names
    }


def _write(
    root: Path,
    files: dict[str, Any],
    frame: Frame,
    detection: Detection,
    wall_ms: float,
    visualize: bool,
    save_frames: bool,
    tracks: TrackFrame | None,
    graph: SSGSnapshot | None,
) -> None:
    stamp = frame.stamp
    relative = f"{stamp.episode_id}/{stamp.frame_index:06d}.png"
    files["detections"].write(detection.frame.to_json(indent=None) + "\n")
    files["timing"].write(
        json.dumps(
            {
                "episode_id": stamp.episode_id,
                "frame_index": stamp.frame_index,
                "wall_ms": wall_ms,
                **{
                    k: getattr(detection.timing, k)
                    for k in ("preprocess_ms", "inference_ms", "postprocess_ms")
                },
            }
        )
        + "\n"
    )
    if tracks is not None:
        files["tracks"].write(tracks.to_json(indent=None) + "\n")
    if graph is not None:
        files["graphs"].write(graph.to_json(indent=None) + "\n")
        if visualize:
            render_snapshot(graph, frame.image).save(
                _parent(root / "graphs" / relative)
            )
    if detection.masks is not None:
        _save_png(root / "masks" / relative, detection.masks)
    if save_frames:
        _save_png(root / "rgb" / relative, frame.image)
        if frame.agent is not None:
            height, width = frame.image.shape[:2]
            observation = ObservationFrame(
                SCHEMA_VERSION,
                stamp,
                ImageReference(f"rgb/{relative}", width, height, ImageEncoding.PNG),
                None,
                frame.agent,
                OBSERVATION_SOURCE,
            )
            files["observations"].write(observation.to_json(indent=None) + "\n")
    if visualize:
        render_overlay(frame.image, detection, tracks).save(
            _parent(root / "overlays" / relative)
        )


def render_overlay(
    image: np.ndarray | None, detection: Detection, tracks: TrackFrame | None = None
) -> Image.Image:
    """Masks tinted by instance, boxes outlined, top label and confidence written above each box —
    prefixed by the track number, and coloured by track, when `tracks` is given.
    `image=None` draws a synthetic mask scene for image-independent mock detections."""
    track_of = (
        {}
        if tracks is None
        else {
            t.record.detector_id: t.record.track_id.rsplit("-", 1)[-1]
            for t in tracks.tracks
            if t.record.detector_id
        }
    )
    canvas = (
        image.astype(np.float32)
        if image is not None
        else np.full(
            (detection.frame.image_height, detection.frame.image_width, 3),
            235,
            np.float32,
        )
    )
    palette = np.array(
        [
            [230, 25, 75],
            [60, 180, 75],
            [255, 225, 25],
            [0, 130, 200],
            [245, 130, 48],
            [145, 30, 180],
            [70, 240, 240],
            [240, 50, 230],
        ],
        np.float32,
    )
    if detection.masks is not None:
        for value in np.unique(detection.masks[detection.masks > 0]):
            region = detection.masks == value
            canvas[region] = (
                0.5 * canvas[region] + 0.5 * palette[(int(value) - 1) % len(palette)]
            )
    overlay = Image.fromarray(canvas.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(overlay)
    for index, item in enumerate(detection.frame.objects):
        box = item.bounding_box
        track = track_of.get(item.detector_id)
        color = tuple(
            int(c) for c in palette[(int(track) if track else index) % len(palette)]
        )
        draw.rectangle(
            (
                box.x_min,
                box.y_min,
                max(box.x_min, box.x_max - 1),
                max(box.y_min, box.y_max - 1),
            ),
            outline=color,
        )
        entries = item.semantic_distribution.entries
        label = (
            f"{entries[0].label} {item.confidence:.2f}"
            if entries
            else f"? {item.confidence:.2f}"
        )
        if track:
            label = f"T{int(track)} {label}"
        text_width = draw.textbbox((0, 0), label)[2]
        text_x = max(0, min(box.x_min + 1, overlay.width - text_width - 1))
        draw.text((text_x, max(box.y_min - 10, 0)), label, fill=color)
    return overlay


def load_detections(path: str | Path) -> list[DetectionFrame]:
    with Path(path).open(encoding="utf-8") as handle:
        return [DetectionFrame.from_json(line) for line in handle if line.strip()]


def _save_png(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array).save(_parent(path))


def _parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "mean": statistics.fmean(ordered),
        "p50": ordered[len(ordered) // 2],
        "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "max": ordered[-1],
    }
