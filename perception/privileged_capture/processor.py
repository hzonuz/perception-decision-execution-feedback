from __future__ import annotations

import hashlib
import math
import re
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any

from perception.contracts import (
    SCHEMA_VERSION,
    AgentNode,
    CaptureFrame,
    MemoryState,
    NormalizedPoint,
    ObjectNode,
    SSGSnapshot,
    SourceMetadata,
    SourceType,
    Vector3,
)
from perception.jsonio import as_object as _as_object
from perception.jsonio import integer_field as _integer
from perception.jsonio import list_field as _list
from perception.jsonio import number_field as _number
from perception.jsonio import object_field as _object
from perception.jsonio import read_object as _read_object
from perception.jsonio import string_field as _string
from perception.jsonio import write_json, write_text as _write
from perception.ontology import Ontology, load_ontology


PRODUCER = "animal_ai_unity_capture"


def process_capture(
    capture_root: str | Path,
    ontology_path: str | Path = "configs/ontology.json",
) -> dict[str, Any]:
    root = Path(capture_root).resolve()
    ontology = load_ontology(ontology_path)
    raw_paths = sorted((root / "raw").glob("*/*.json"))
    if not raw_paths:
        raise ValueError(f"no raw capture records found under {root / 'raw'}")

    seen_frames: set[tuple[str, int]] = set()
    identity_classes: dict[tuple[str, str], str] = {}
    identity_frames: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
    ages: defaultdict[tuple[str, str], int] = defaultdict(int)
    unavailable: set[str] = set()
    captures: list[str] = []
    oracles: list[str] = []

    for raw_path in raw_paths:
        raw = _read_object(raw_path)
        if raw.get("format_version") != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported raw format_version {raw.get('format_version')!r}"
            )
        episode_id = _string(raw, "episode_id")
        frame_index = _integer(raw, "frame_index")
        frame_key = (episode_id, frame_index)
        if frame_key in seen_frames:
            raise ValueError(f"duplicate frame {episode_id}/{frame_index}")
        seen_frames.add(frame_key)
        unavailable.update(str(item) for item in raw.get("unavailable_fields", ()))

        for item in _list(raw, "objects"):
            instance_id = _string(item, "instance_id")
            exact_class = _string(item, "exact_simulator_class")
            identity_key = (episode_id, instance_id)
            previous = identity_classes.setdefault(identity_key, exact_class)
            if previous != exact_class:
                raise ValueError(
                    f"instance {instance_id!r} changed class from {previous!r} to {exact_class!r}"
                )
            identity_frames[identity_key].append(frame_index)
            ages[identity_key] += 1

        frame = _capture_frame(root, raw, ontology)
        ontology.validate_capture(frame)
        capture_relative = Path("annotations") / episode_id / f"{frame_index:06d}.json"
        _write(root / capture_relative, frame.to_json())
        captures.append(capture_relative.as_posix())

        oracle = _oracle_snapshot(frame, ages)
        ontology.validate_ssg(oracle)
        oracle_relative = Path("oracle") / episode_id / f"{frame_index:06d}.json"
        _write(root / oracle_relative, oracle.to_json())
        oracles.append(oracle_relative.as_posix())

    persistent = sorted(
        instance_id
        for (_, instance_id), frames in identity_frames.items()
        if len(frames) > 1
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "frames_validated": len(raw_paths),
        "capture_records": captures,
        "oracle_snapshots": oracles,
        "persistent_instance_ids": persistent,
        "stable_identity_verified": bool(persistent),
        "unavailable_fields": sorted(unavailable),
    }
    write_json(root / "capture_report.json", report)
    return report


def _capture_frame(root: Path, raw: dict[str, Any], ontology: Ontology) -> CaptureFrame:
    episode_id = _string(raw, "episode_id")
    arena_id = _string(raw, "arena_id")
    frame_index = _integer(raw, "frame_index")
    timestamp = _number(raw, "timestamp_seconds")
    stamp = {
        "episode_id": episode_id,
        "frame_index": frame_index,
        "timestamp_seconds": timestamp,
    }
    source = _source(
        str(raw.get("producer_version") or "unknown"),
        arena_id,
        raw.get("random_seed"),
    )

    rgb = _object(raw, "rgb")
    rgb_path = _string(rgb, "path")
    width = _integer(rgb, "width")
    height = _integer(rgb, "height")
    image_sha = _validate_png(root, rgb_path, width, height)
    mask = _object(raw, "instance_mask")
    if mask.get("encoding") != "rgb24_little_endian":
        raise ValueError(f"unsupported instance mask encoding {mask.get('encoding')!r}")
    _validate_png(
        root,
        _string(mask, "path"),
        _integer(mask, "width"),
        _integer(mask, "height"),
    )

    objects: list[dict[str, Any]] = []
    for item in _list(raw, "objects"):
        exact_class = _string(item, "exact_simulator_class")
        mapping = ontology.mapping_for(exact_class)
        if mapping is None:
            raise ValueError(f"unmapped simulator class {exact_class!r}")
        box = item.get("bounding_box")
        if box is not None:
            box = {
                **_as_object(box, "bounding_box"),
                "image_width": width,
                "image_height": height,
            }
        instance_value = item.get("instance_value")
        mask_reference = (
            {
                "path": _string(mask, "path"),
                "encoding": "png",
                "width": width,
                "height": height,
                "instance_value": instance_value,
            }
            if instance_value is not None
            else None
        )
        functional_types = [
            {
                "name": name,
                "belief": {
                    "probability": 1.0,
                    "source": source,
                    "uncertainty": None,
                },
            }
            for name in mapping.functional_types
        ]
        objects.append(
            {
                "instance_id": _string(item, "instance_id"),
                "exact_simulator_class": exact_class,
                "semantic_distribution": {
                    "entries": [{"label": mapping.semantic_class, "probability": 1.0}],
                    "unknown_probability": 0.0,
                },
                "functional_types": functional_types,
                "world_position": _object(item, "world_position"),
                "world_rotation": _object(item, "world_rotation"),
                "dimensions": _object(item, "dimensions"),
                "velocity": item.get("velocity"),
                "bounding_box": box,
                "instance_mask": mask_reference,
                "relative_depth": item.get("relative_depth"),
                "visibility_fraction": _clamped_visibility(
                    item.get("visibility_fraction")
                ),
                "occlusion_fraction": _complement(
                    _clamped_visibility(item.get("visibility_fraction"))
                ),
                "attributes": [
                    {
                        "name": "unity_tag",
                        "value": item.get("unity_tag"),
                        "confidence": 1.0,
                        "source": source,
                    }
                ],
                "source": source,
            }
        )

    agent = _object(raw, "agent")
    action_forward = agent.get("action_forward")
    action_rotate = agent.get("action_rotate")
    recent_action = (
        None
        if action_forward is None and action_rotate is None
        else {
            "forward": action_forward,
            "rotate": action_rotate,
            "name": None,
        }
    )
    lifecycle = [
        {
            "instance_id": _string(item, "instance_id"),
            "event": _string(item, "event"),
            "stamp": stamp,
            "reason": item.get("reason"),
            "source": source,
        }
        for item in _list(raw, "lifecycle_events")
    ]
    contacts = [
        {
            "subject_instance_id": _string(item, "subject_instance_id"),
            "object_instance_id": _string(item, "object_instance_id"),
            "stamp": stamp,
            "impulse": item.get("impulse"),
            "contact_point": item.get("contact_point"),
            "source": source,
        }
        for item in _list(raw, "contact_events")
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "arena_id": arena_id,
        "random_seed": raw.get("random_seed"),
        "stamp": stamp,
        "rgb": {
            "path": rgb_path,
            "width": width,
            "height": height,
            "encoding": "png",
            "sha256": image_sha,
        },
        "camera": raw.get("camera"),
        "agent": {
            "node_id": "agent",
            "world_position": agent.get("world_position"),
            "world_rotation": agent.get("world_rotation"),
            "velocity": agent.get("velocity"),
            "recent_action": recent_action,
            "health": agent.get("health"),
            "current_option": None,
            "option_status": None,
        },
        "objects": objects,
        "lifecycle_events": lifecycle,
        "contact_events": contacts,
        "source": source,
    }
    return CaptureFrame.from_dict(payload)


def _oracle_snapshot(
    frame: CaptureFrame,
    ages: dict[tuple[str, str], int],
) -> SSGSnapshot:
    source = frame.source
    object_nodes: list[ObjectNode] = []
    for item in frame.objects:
        box = item.bounding_box
        normalized_location = (
            NormalizedPoint(
                x=(box.x_min + box.x_max) / (2 * box.image_width),
                y=(box.y_min + box.y_max) / (2 * box.image_height),
            )
            if box is not None
            else None
        )
        bearing = None
        if (
            box is not None
            and frame.camera is not None
            and frame.camera.intrinsics is not None
        ):
            intrinsics = frame.camera.intrinsics
            center_x = (box.x_min + box.x_max) / 2
            bearing = math.degrees(math.atan2(center_x - intrinsics.cx, intrinsics.fx))

        distance = _distance(frame.agent.world_position, item.world_position)
        relative_velocity = _relative_velocity(frame.agent.velocity, item.velocity)
        if box is None:
            memory_state = MemoryState.OUT_OF_VIEW
        elif item.visibility_fraction is not None and item.visibility_fraction < 0.999:
            memory_state = MemoryState.PARTIALLY_VISIBLE
        else:
            memory_state = MemoryState.VISIBLE
        node_id = "obj_" + re.sub(r"[^A-Za-z0-9_]+", "_", item.instance_id)
        object_nodes.append(
            ObjectNode(
                node_id=node_id,
                track_id=item.instance_id,
                detector_id=None,
                exact_simulator_class=item.exact_simulator_class,
                semantic_distribution=item.semantic_distribution,
                functional_types=item.functional_types,
                bounding_box=box,
                instance_mask=item.instance_mask,
                normalized_location=normalized_location,
                bearing_degrees=bearing,
                relative_distance=distance,
                relative_velocity=relative_velocity,
                detector_confidence=None,
                visibility_probability=item.visibility_fraction,
                existence_probability=1.0,
                track_age_frames=ages[(frame.episode_id, item.instance_id)],
                last_seen=frame.stamp if box is not None else None,
                memory_state=memory_state,
                confidence=1.0,
                source=source,
                stamp=frame.stamp,
                attributes=item.attributes,
                uncertainty=None,
            )
        )

    return SSGSnapshot(
        schema_version=SCHEMA_VERSION,
        graph_id=f"{frame.episode_id}:frame-{frame.stamp.frame_index:06d}",
        episode_id=frame.episode_id,
        stamp=frame.stamp,
        is_macro_step=True,
        agent=AgentNode(
            node_id=frame.agent.node_id,
            state=frame.agent,
            stamp=frame.stamp,
            confidence=1.0,
            source=source,
            uncertainty=None,
        ),
        objects=tuple(object_nodes),
        relations=(),
        confidence=1.0,
        source=source,
        uncertainty=None,
    )


def _source(version: str, arena_id: str, seed: object) -> dict[str, Any]:
    return {
        "source": SourceType.SIMULATOR_ORACLE.value,
        "producer": PRODUCER,
        "producer_version": version,
        "privileged": True,
        "configuration_id": f"arena={arena_id};seed={seed}",
    }


def _distance(left: Vector3 | None, right: Vector3) -> float | None:
    if left is None:
        return None
    return math.sqrt(
        (right.x - left.x) ** 2 + (right.y - left.y) ** 2 + (right.z - left.z) ** 2
    )


def _relative_velocity(left: Vector3 | None, right: Vector3 | None) -> Vector3 | None:
    if left is None or right is None:
        return None
    return Vector3(right.x - left.x, right.y - left.y, right.z - left.z)


def _clamped_visibility(value: float | None) -> float | None:
    # visible_pixels / amodal_pixels occasionally exceeds 1.0 in the raw capture: a rare mask
    # colour collision or anti-aliasing edge in Unity's pixel count can inflate visible_pixels
    # past the true amodal silhouette. The measurement is unreliable past the boundary; clamp
    # rather than reject the object.
    if value is None:
        return None
    return max(0.0, min(1.0, value))


def _complement(value: float | None) -> float | None:
    return None if value is None else 1.0 - value


def _validate_png(root: Path, relative: str, width: int, height: int) -> str:
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"capture path escapes root: {relative!r}")
    try:
        payload = path.read_bytes()
    except FileNotFoundError as exc:
        raise ValueError(f"missing capture image {relative!r}") from exc
    if len(payload) < 24 or payload[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"invalid PNG {relative!r}")
    actual_width, actual_height = struct.unpack(">II", payload[16:24])
    if (actual_width, actual_height) != (width, height):
        raise ValueError(
            f"PNG dimensions for {relative!r} are {actual_width}x{actual_height}, "
            f"expected {width}x{height}"
        )
    return hashlib.sha256(payload).hexdigest()
