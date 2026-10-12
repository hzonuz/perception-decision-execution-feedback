from __future__ import annotations

import hashlib
import json
import math
import shutil
import statistics
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from perception.dataset.geometry import polygon_area
from perception.jsonio import write_json
from perception.dataset.models import DatasetConfig, Issue, SPLITS, Sample


def write_dataset(
    output: Path,
    samples: list[Sample],
    split_by_sample: dict[tuple[str, int], str],
    split_by_group: dict[str, str],
    class_names: list[str],
    config: DatasetConfig,
    export_coco: bool,
) -> None:
    for split in SPLITS:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
        (output / "metadata").mkdir(parents=True, exist_ok=True)

    split_samples: dict[str, list[Sample]] = {split: [] for split in SPLITS}
    metadata_files = {
        split: (output / "metadata" / f"{split}.jsonl").open("w", encoding="utf-8")
        for split in SPLITS
    }
    try:
        for sample in sorted(
            samples,
            key=lambda item: (item.frame.episode_id, item.frame.stamp.frame_index),
        ):
            split = split_by_sample[
                (sample.frame.episode_id, sample.frame.stamp.frame_index)
            ]
            split_samples[split].append(sample)
            image_destination = output / "images" / split / f"{sample.stem}.png"
            label_destination = output / "labels" / split / f"{sample.stem}.txt"
            shutil.copy2(sample.image_path, image_destination)
            _write_yolo_label(label_destination, sample)
            metadata_files[split].write(
                json.dumps(
                    _metadata_record(
                        sample, split, image_destination, label_destination
                    ),
                    sort_keys=True,
                )
                + "\n"
            )
    finally:
        for handle in metadata_files.values():
            handle.close()

    _write_data_yaml(output / "data.yaml", output, class_names)
    _write_split_manifest(output, split_by_group, split_samples, config)
    write_json(
        output / "statistics.json",
        _statistics(samples, split_by_sample, class_names, config),
    )
    _render_samples(output, split_samples, config)
    if export_coco:
        _write_coco(output, split_samples, class_names)


def validation_report(
    samples: list[Sample],
    issues: list[Issue],
    split_counts: dict[str, int],
    config: DatasetConfig,
) -> dict[str, Any]:
    errors = [issue.to_dict() for issue in issues if issue.severity == "error"]
    warnings = [issue.to_dict() for issue in issues if issue.severity == "warning"]
    return {
        "status": "failed" if errors else "passed",
        "frames_checked": len(samples),
        "objects_checked": sum(len(sample.frame.objects) for sample in samples),
        "split_counts": split_counts,
        "group_by": config.group_by,
        "checks": [
            "missing_image",
            "malformed_polygon",
            "box_mask_disagreement",
            "empty_annotation",
            "unknown_class",
            "invalid_split_overlap",
            "duplicate_frame",
            "broken_instance_id_sequence",
        ],
        "errors": errors,
        "warnings": warnings,
    }


def _write_yolo_label(path: Path, sample: Sample) -> None:
    lines: list[str] = []
    for item in sample.objects:
        polygon = max(item.polygons, key=polygon_area)
        coordinates = " ".join(
            f"{x / sample.frame.rgb.width:.6f} {y / sample.frame.rgb.height:.6f}"
            for x, y in polygon
        )
        lines.append(f"{item.class_id} {coordinates}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_data_yaml(path: Path, root: Path, class_names: list[str]) -> None:
    lines = [
        f"path: {root.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
        *(f"  {index}: {name}" for index, name in enumerate(class_names)),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_split_manifest(
    output: Path,
    split_by_group: dict[str, str],
    split_samples: dict[str, list[Sample]],
    config: DatasetConfig,
) -> None:
    write_json(
        output / "splits.json",
        {
            "group_by": config.group_by,
            "random_seed": config.random_seed,
            "ratios": config.ratios,
            "groups": {
                split: sorted(
                    group
                    for group, assigned in split_by_group.items()
                    if assigned == split
                )
                for split in SPLITS
            },
            "episodes": {
                split: sorted(
                    {sample.frame.episode_id for sample in split_samples[split]}
                )
                for split in SPLITS
            },
            "frames": {
                split: [
                    {
                        "episode_id": sample.frame.episode_id,
                        "frame_index": sample.frame.stamp.frame_index,
                    }
                    for sample in split_samples[split]
                ]
                for split in SPLITS
            },
        },
    )


def _statistics(
    samples: list[Sample],
    split_by_sample: dict[tuple[str, int], str],
    class_names: list[str],
    config: DatasetConfig,
) -> dict[str, Any]:
    class_counts = {name: 0 for name in class_names}
    size_values: list[float] = []
    size_buckets = {"small": 0, "medium": 0, "large": 0}
    visibility_values: list[float] = []
    visibility_buckets = _fraction_buckets()
    occlusion_values: list[float] = []
    occlusion_buckets = _fraction_buckets()
    motion = {"moving": 0, "stationary": 0, "unknown": 0}
    objects_per_frame: list[int] = []

    for sample in samples:
        objects_per_frame.append(len(sample.objects))
        image_area = sample.frame.rgb.width * sample.frame.rgb.height
        for item in sample.objects:
            class_counts[item.class_name] += 1
            fraction = item.area_pixels / image_area
            size_values.append(fraction)
            if fraction < config.small_area_fraction:
                size_buckets["small"] += 1
            elif fraction < config.large_area_fraction:
                size_buckets["medium"] += 1
            else:
                size_buckets["large"] += 1
            _add_fraction(
                item.annotation.visibility_fraction,
                visibility_values,
                visibility_buckets,
            )
            _add_fraction(
                item.annotation.occlusion_fraction,
                occlusion_values,
                occlusion_buckets,
            )
            motion[item.motion] += 1

    return {
        "frames": len(samples),
        "objects": sum(objects_per_frame),
        "frames_per_split": {
            split: sum(
                1
                for sample in samples
                if split_by_sample[
                    (sample.frame.episode_id, sample.frame.stamp.frame_index)
                ]
                == split
            )
            for split in SPLITS
        },
        "samples_per_class": class_counts,
        "object_size": {
            "area_fraction": _summary(size_values),
            "buckets": size_buckets,
            "thresholds": {
                "small_below": config.small_area_fraction,
                "large_at_or_above": config.large_area_fraction,
            },
        },
        "visibility": {
            "values": _summary(visibility_values),
            "buckets": visibility_buckets,
        },
        "occlusion": {
            "values": _summary(occlusion_values),
            "buckets": occlusion_buckets,
        },
        "motion": motion,
        "objects_per_frame": {
            "values": _summary([float(value) for value in objects_per_frame]),
            "histogram": {
                str(value): objects_per_frame.count(value)
                for value in sorted(set(objects_per_frame))
            },
        },
    }


def _metadata_record(
    sample: Sample,
    split: str,
    image_path: Path,
    label_path: Path,
) -> dict[str, Any]:
    return {
        "split": split,
        "episode_id": sample.frame.episode_id,
        "arena_id": sample.frame.arena_id,
        "random_seed": sample.frame.random_seed,
        "frame_index": sample.frame.stamp.frame_index,
        "timestamp_seconds": sample.frame.stamp.timestamp_seconds,
        "image": image_path.relative_to(image_path.parents[2]).as_posix(),
        "label": label_path.relative_to(label_path.parents[2]).as_posix(),
        "source_annotation": str(sample.annotation_path),
        "objects": [
            {
                "instance_id": item.annotation.instance_id,
                "exact_simulator_class": item.annotation.exact_simulator_class,
                "class_name": item.class_name,
                "class_id": item.class_id,
                "bounding_box": (
                    item.annotation.bounding_box.to_dict()
                    if item.annotation.bounding_box
                    else None
                ),
                "area_pixels": item.area_pixels,
                "visibility_fraction": item.annotation.visibility_fraction,
                "occlusion_fraction": item.annotation.occlusion_fraction,
                "motion": item.motion,
                "relative_depth": item.annotation.relative_depth,
            }
            for item in sample.objects
        ],
    }


def _render_samples(
    output: Path,
    split_samples: dict[str, list[Sample]],
    config: DatasetConfig,
) -> None:
    for split, samples in split_samples.items():
        for sample in samples[: config.visualization_samples_per_split]:
            with Image.open(sample.image_path) as source:
                image = source.convert("RGBA")
            overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            for item in sample.objects:
                color = _class_color(item.class_id)
                for polygon in item.polygons:
                    overlay_draw.polygon(polygon, fill=(*color, 80))
            image = Image.alpha_composite(image, overlay)
            draw = ImageDraw.Draw(image)
            captions: list[tuple[str, tuple[int, int, int]]] = []
            for item in sample.objects:
                color = _class_color(item.class_id)
                x_min, y_min, x_max, y_max = item.mask_box
                draw.rectangle((x_min, y_min, x_max, y_max), outline=color, width=1)
                visibility = _display_fraction(item.annotation.visibility_fraction)
                occlusion = _display_fraction(item.annotation.occlusion_fraction)
                captions.append(
                    (
                        f"{item.class_name} | {item.annotation.instance_id} | "
                        f"vis={visibility} | occ={occlusion}",
                        color,
                    )
                )

            scale = max(1, min(8, math.ceil(384 / image.width)))
            rendered = image.convert("RGB").resize(
                (image.width * scale, image.height * scale),
                Image.Resampling.NEAREST,
            )
            measure = ImageDraw.Draw(rendered)
            caption_width = max(
                (measure.textbbox((0, 0), text)[2] for text, _ in captions),
                default=0,
            )
            line_height = 12
            canvas = Image.new(
                "RGB",
                (
                    max(rendered.width, caption_width + 8),
                    rendered.height + line_height * len(captions),
                ),
                "black",
            )
            canvas.paste(rendered, (0, 0))
            caption_draw = ImageDraw.Draw(canvas)
            for index, (text, color) in enumerate(captions):
                caption_draw.text(
                    (4, rendered.height + index * line_height),
                    text,
                    fill=color,
                )
            destination = output / "visualizations" / split / f"{sample.stem}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(destination)


def _write_coco(
    output: Path,
    split_samples: dict[str, list[Sample]],
    class_names: list[str],
) -> None:
    categories = [
        {"id": index + 1, "name": name} for index, name in enumerate(class_names)
    ]
    annotation_id = 1
    for split, samples in split_samples.items():
        images: list[dict[str, Any]] = []
        annotations: list[dict[str, Any]] = []
        for image_id, sample in enumerate(samples, start=1):
            images.append(
                {
                    "id": image_id,
                    "file_name": f"../images/{split}/{sample.stem}.png",
                    "width": sample.frame.rgb.width,
                    "height": sample.frame.rgb.height,
                    "episode_id": sample.frame.episode_id,
                    "frame_index": sample.frame.stamp.frame_index,
                }
            )
            for item in sample.objects:
                x_min, y_min, x_max, y_max = item.mask_box
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": item.class_id + 1,
                        "segmentation": [
                            [coordinate for point in polygon for coordinate in point]
                            for polygon in item.polygons
                        ],
                        "area": item.area_pixels,
                        "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                        "iscrowd": 0,
                        "instance_id": item.annotation.instance_id,
                        "visibility_fraction": item.annotation.visibility_fraction,
                        "occlusion_fraction": item.annotation.occlusion_fraction,
                    }
                )
                annotation_id += 1
        write_json(
            output / "coco" / f"instances_{split}.json",
            {
                "info": {
                    "description": "Animal-AI segmentation export",
                    "version": "1.0.0",
                },
                "images": images,
                "annotations": annotations,
                "categories": categories,
            },
        )


def _fraction_buckets() -> dict[str, int]:
    return {
        "0.00-0.25": 0,
        "0.25-0.50": 0,
        "0.50-0.75": 0,
        "0.75-1.00": 0,
        "unknown": 0,
    }


def _add_fraction(
    value: float | None,
    values: list[float],
    buckets: dict[str, int],
) -> None:
    if value is None:
        buckets["unknown"] += 1
        return
    values.append(value)
    if value < 0.25:
        buckets["0.00-0.25"] += 1
    elif value < 0.5:
        buckets["0.25-0.50"] += 1
    elif value < 0.75:
        buckets["0.50-0.75"] += 1
    else:
        buckets["0.75-1.00"] += 1


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
    }


def _class_color(class_id: int) -> tuple[int, int, int]:
    digest = hashlib.sha256(str(class_id).encode()).digest()
    return 64 + digest[0] // 2, 64 + digest[1] // 2, 64 + digest[2] // 2


def _display_fraction(value: float | None) -> str:
    return "?" if value is None else f"{value:.2f}"
