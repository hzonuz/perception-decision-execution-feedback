from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

from PIL import Image

from perception.contracts import CaptureFrame, ContractValidationError, ObjectAnnotation
from perception.dataset.geometry import instance_pixels, pixel_box, polygons
from perception.dataset.models import DatasetConfig, DatasetObject, Issue, Sample
from perception.ontology import Ontology


def load_samples(
    roots: list[Path],
    config: DatasetConfig,
    ontology: Ontology,
    class_ids: dict[str, int],
) -> tuple[list[Sample], list[Issue]]:
    samples: list[Sample] = []
    issues: list[Issue] = []
    seen_frames: set[tuple[str, int]] = set()
    for root in roots:
        paths = sorted((root / "annotations").glob("*/*.json"))
        if not paths:
            issues.append(
                Issue(
                    "error", "missing_annotations", f"no annotations found under {root}"
                )
            )
        for path in paths:
            try:
                frame = CaptureFrame.from_json_file(path)
            except (OSError, ValueError) as exc:
                issues.append(Issue("error", "malformed_annotation", f"{path}: {exc}"))
                continue
            try:
                ontology.validate_capture(frame)
            except ContractValidationError as exc:
                issues.append(
                    Issue(
                        "error",
                        (
                            "unknown_class"
                            if "unmapped simulator class" in str(exc)
                            else "malformed_annotation"
                        ),
                        f"{path}: {exc}",
                        frame.episode_id,
                        frame.stamp.frame_index,
                    )
                )
                continue
            frame_key = (frame.episode_id, frame.stamp.frame_index)
            if frame_key in seen_frames:
                issues.append(
                    Issue(
                        "error",
                        "duplicate_frame",
                        "duplicate episode/frame pair",
                        frame.episode_id,
                        frame.stamp.frame_index,
                    )
                )
                continue
            seen_frames.add(frame_key)

            image_path = _safe_source_path(root, frame.rgb.path)
            if image_path is None or not image_path.is_file():
                issues.append(
                    Issue(
                        "error",
                        "missing_image",
                        f"missing image {frame.rgb.path!r}",
                        frame.episode_id,
                        frame.stamp.frame_index,
                    )
                )
                continue
            try:
                with Image.open(image_path) as image:
                    if image.size != (frame.rgb.width, frame.rgb.height):
                        raise ValueError(
                            f"image is {image.width}x{image.height}, "
                            f"contract says {frame.rgb.width}x{frame.rgb.height}"
                        )
                    image.verify()
            except (OSError, ValueError) as exc:
                issues.append(
                    Issue(
                        "error",
                        "malformed_image",
                        str(exc),
                        frame.episode_id,
                        frame.stamp.frame_index,
                    )
                )
                continue

            if not frame.objects:
                issues.append(
                    Issue(
                        "warning" if config.allow_empty_annotations else "error",
                        "empty_annotation",
                        "frame has no annotated objects",
                        frame.episode_id,
                        frame.stamp.frame_index,
                    )
                )
            objects = _load_objects(root, frame, config, ontology, class_ids, issues)
            samples.append(Sample(root, path, frame, image_path, tuple(objects)))
    return samples, issues


def _load_objects(
    root: Path,
    frame: CaptureFrame,
    config: DatasetConfig,
    ontology: Ontology,
    class_ids: dict[str, int],
    issues: list[Issue],
) -> list[DatasetObject]:
    result: list[DatasetObject] = []
    mask_cache: dict[Path, Image.Image] = {}
    try:
        for annotation in frame.objects:
            mapping = ontology.mapping_for(annotation.exact_simulator_class)
            if mapping is None:
                issues.append(
                    Issue(
                        "error",
                        "unknown_class",
                        f"unmapped class {annotation.exact_simulator_class!r}",
                        frame.episode_id,
                        frame.stamp.frame_index,
                        annotation.instance_id,
                    )
                )
                continue
            class_name = (
                mapping.semantic_class
                if config.class_mode == "semantic"
                else ontology.resolve_simulator_class(annotation.exact_simulator_class)
            )
            if class_name not in class_ids:
                issues.append(
                    Issue(
                        "error",
                        "unknown_class",
                        f"class {class_name!r} is absent from class ordering",
                        frame.episode_id,
                        frame.stamp.frame_index,
                        annotation.instance_id,
                    )
                )
                continue
            if (
                annotation.instance_mask is None
                or annotation.instance_mask.instance_value is None
            ):
                issues.append(
                    Issue(
                        "warning",
                        "missing_mask",
                        "object is not visible or has no instance mask",
                        frame.episode_id,
                        frame.stamp.frame_index,
                        annotation.instance_id,
                    )
                )
                continue

            mask_path = _safe_source_path(root, annotation.instance_mask.path)
            if mask_path is None or not mask_path.is_file():
                issues.append(
                    Issue(
                        "error",
                        "missing_mask",
                        f"missing mask {annotation.instance_mask.path!r}",
                        frame.episode_id,
                        frame.stamp.frame_index,
                        annotation.instance_id,
                    )
                )
                continue
            try:
                if mask_path not in mask_cache:
                    mask_cache[mask_path] = Image.open(mask_path).convert("RGB")
                mask = mask_cache[mask_path]
                if mask.size != (frame.rgb.width, frame.rgb.height):
                    raise ValueError("mask dimensions do not match RGB")
                pixels = instance_pixels(mask, annotation.instance_mask.instance_value)
            except (OSError, ValueError) as exc:
                issues.append(
                    Issue(
                        "error",
                        "malformed_mask",
                        str(exc),
                        frame.episode_id,
                        frame.stamp.frame_index,
                        annotation.instance_id,
                    )
                )
                continue
            if not pixels:
                issues.append(
                    Issue(
                        "error",
                        "empty_mask",
                        "instance value has no pixels in the mask",
                        frame.episode_id,
                        frame.stamp.frame_index,
                        annotation.instance_id,
                    )
                )
                continue

            object_polygons = polygons(pixels)
            if not object_polygons or any(
                len(polygon) < 3 for polygon in object_polygons
            ):
                issues.append(
                    Issue(
                        "error",
                        "malformed_polygon",
                        "mask could not produce a polygon with at least three points",
                        frame.episode_id,
                        frame.stamp.frame_index,
                        annotation.instance_id,
                    )
                )
                continue
            if len(object_polygons) > 1:
                issues.append(
                    Issue(
                        "warning",
                        "multi_component_mask",
                        "YOLO label uses the largest visible mask component",
                        frame.episode_id,
                        frame.stamp.frame_index,
                        annotation.instance_id,
                    )
                )

            mask_box = pixel_box(pixels)
            if annotation.bounding_box is None:
                issues.append(
                    Issue(
                        "error",
                        "box_mask_disagreement",
                        "visible mask has no bounding box",
                        frame.episode_id,
                        frame.stamp.frame_index,
                        annotation.instance_id,
                    )
                )
                continue
            expected = (
                annotation.bounding_box.x_min,
                annotation.bounding_box.y_min,
                annotation.bounding_box.x_max,
                annotation.bounding_box.y_max,
            )
            if max(abs(left - right) for left, right in zip(expected, mask_box)) > (
                config.box_mask_tolerance_pixels
            ):
                issues.append(
                    Issue(
                        "error",
                        "box_mask_disagreement",
                        f"box={expected}, mask_box={mask_box}",
                        frame.episode_id,
                        frame.stamp.frame_index,
                        annotation.instance_id,
                    )
                )
                continue
            result.append(
                DatasetObject(
                    annotation=annotation,
                    class_name=class_name,
                    class_id=class_ids[class_name],
                    polygons=tuple(object_polygons),
                    area_pixels=len(pixels),
                    mask_box=mask_box,
                    motion=_motion(annotation, config.moving_speed_threshold),
                )
            )
    finally:
        for mask in mask_cache.values():
            mask.close()
    return result


def check_sequences(samples: list[Sample], issues: list[Issue]) -> None:
    by_episode: defaultdict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_episode[sample.frame.episode_id].append(sample)
    for episode_id, episode_samples in by_episode.items():
        ordered = sorted(episode_samples, key=lambda item: item.frame.stamp.frame_index)
        class_by_id: dict[str, str] = {}
        positions: defaultdict[str, list[int]] = defaultdict(list)
        for position, sample in enumerate(ordered):
            for item in sample.frame.objects:
                previous = class_by_id.setdefault(
                    item.instance_id, item.exact_simulator_class
                )
                if previous != item.exact_simulator_class:
                    issues.append(
                        Issue(
                            "error",
                            "broken_instance_id_sequence",
                            f"instance class changed from {previous!r} "
                            f"to {item.exact_simulator_class!r}",
                            episode_id,
                            sample.frame.stamp.frame_index,
                            item.instance_id,
                        )
                    )
                positions[item.instance_id].append(position)
        for instance_id, seen in positions.items():
            if seen != list(range(min(seen), max(seen) + 1)):
                issues.append(
                    Issue(
                        "error",
                        "broken_instance_id_sequence",
                        "instance disappears and later returns within one episode",
                        episode_id,
                        None,
                        instance_id,
                    )
                )


def check_split_integrity(
    samples: list[Sample],
    split_by_sample: dict[tuple[str, int], str],
    issues: list[Issue],
) -> None:
    episode_splits: defaultdict[str, set[str]] = defaultdict(set)
    for sample in samples:
        episode_splits[sample.frame.episode_id].add(
            split_by_sample[(sample.frame.episode_id, sample.frame.stamp.frame_index)]
        )
    for episode_id, splits in episode_splits.items():
        if len(splits) > 1:
            issues.append(
                Issue(
                    "error",
                    "invalid_split_overlap",
                    f"episode appears in multiple splits: {sorted(splits)}",
                    episode_id,
                )
            )


def _motion(annotation: ObjectAnnotation, threshold: float) -> str:
    if annotation.velocity is None:
        return "unknown"
    speed = math.sqrt(
        annotation.velocity.x**2 + annotation.velocity.y**2 + annotation.velocity.z**2
    )
    return "moving" if speed >= threshold else "stationary"


def _safe_source_path(root: Path, relative: str) -> Path | None:
    path = (root / relative).resolve()
    return path if path.is_relative_to(root) else None
