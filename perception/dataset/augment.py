"""
Offline, simulation-appropriate geometric augmentation for a built YOLO dataset.

Only transforms with a real camera-geometry counterpart in this simulator are applied: a
horizontal mirror (a left-right flipped arena is an equally valid layout) and a zoom-crop (mimics
standing at a different distance/framing, which happens continuously during real play). No
photometric jitter — Unity renders every frame with fixed lighting and several semantic classes
are defined by exact pixel color (see CLAUDE.md's ontology section); randomizing hue/brightness
would fight that. Only the train split is touched; val/test are left untouched so evaluation
numbers stay comparable to the unaugmented dataset.

New images/labels are added into the existing `images/train/` and `labels/train/` directories
(no separate copy of the dataset), and their metadata is appended to `metadata/train.jsonl`.
`splits.json`/`statistics.json` are left as written by `build_dataset`/`merge_datasets` — they
still describe the pre-augmentation frame counts; the report this function returns is the record
of what augmentation added.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from perception.dataset.geometry import polygon_area
from perception.jsonio import write_json, write_text

PointTransform = Callable[[float, float, int, int], tuple[float, float]]
ImageTransform = Callable[[Image.Image], Image.Image]

_EPS = 1e-6


def augment_dataset(
    dataset_root: str | Path,
    *,
    fliplr: bool = True,
    zoom_max: float = 1.3,
    seed: int = 0,
    overwrite: bool = False,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root)
    report_path = dataset_root / "augmentation_report.json"
    if report_path.is_file() and not overwrite:
        return json.loads(report_path.read_text(encoding="utf-8"))
    if zoom_max <= 1.0:
        raise ValueError("zoom_max must be greater than 1.0")

    metadata_path = dataset_root / "metadata" / "train.jsonl"
    records = [
        json.loads(line)
        for line in metadata_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    rng = random.Random(seed)
    new_lines: list[str] = []
    counts = {"fliplr": 0, "zoom": 0}
    objects_kept = 0
    objects_dropped = 0

    for record in records:
        if record.get("augmented"):
            continue  # never augment an already-augmented variant
        label_lines = _read_label_lines(dataset_root / record["label"])
        if len(label_lines) != len(record["objects"]):
            raise ValueError(
                f"{record['label']}: {len(label_lines)} label lines but "
                f"{len(record['objects'])} metadata objects"
            )

        if fliplr:
            variant, kept, dropped = _write_variant(
                dataset_root,
                record,
                label_lines,
                "fliplr",
                point_transform=lambda x, y, w, h: (w - x, y),
                image_transform=lambda image: image.transpose(Image.FLIP_LEFT_RIGHT),
            )
            new_lines.append(json.dumps(variant, sort_keys=True))
            counts["fliplr"] += 1
            objects_kept += kept
            objects_dropped += dropped

        zoom = rng.uniform(1.05, zoom_max)
        offset_x = rng.uniform(0.0, 1.0)
        offset_y = rng.uniform(0.0, 1.0)
        point_transform, image_transform = _zoom_transforms(zoom, offset_x, offset_y)
        variant, kept, dropped = _write_variant(
            dataset_root,
            record,
            label_lines,
            "zoom",
            point_transform=point_transform,
            image_transform=image_transform,
        )
        new_lines.append(json.dumps(variant, sort_keys=True))
        counts["zoom"] += 1
        objects_kept += kept
        objects_dropped += dropped

    with metadata_path.open("a", encoding="utf-8") as handle:
        for line in new_lines:
            handle.write(line + "\n")

    report = {
        "source_frames": len(records),
        "variants_added": counts,
        "objects_kept": objects_kept,
        "objects_dropped": objects_dropped,
        "zoom_max": zoom_max,
        "seed": seed,
    }
    write_json(report_path, report)
    return report


def _read_label_lines(label_path: Path) -> list[tuple[str, list[tuple[float, float]]]]:
    lines = []
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        tokens = raw.split()
        class_id = tokens[0]
        coordinates = [float(value) for value in tokens[1:]]
        points = list(zip(coordinates[0::2], coordinates[1::2]))
        lines.append((class_id, points))
    return lines


def _zoom_transforms(
    zoom: float, offset_x: float, offset_y: float
) -> tuple[PointTransform, ImageTransform]:
    def crop_box(width: int, height: int) -> tuple[float, float, float, float]:
        crop_width = width / zoom
        crop_height = height / zoom
        left = offset_x * (width - crop_width)
        top = offset_y * (height - crop_height)
        return left, top, left + crop_width, top + crop_height

    def point_transform(
        x: float, y: float, width: int, height: int
    ) -> tuple[float, float]:
        left, top, right, bottom = crop_box(width, height)
        return (
            (x - left) / (right - left) * width,
            (y - top) / (bottom - top) * height,
        )

    def image_transform(image: Image.Image) -> Image.Image:
        width, height = image.size
        cropped = image.crop(crop_box(width, height))
        return cropped.resize((width, height), Image.BICUBIC)

    return point_transform, image_transform


def _write_variant(
    dataset_root: Path,
    record: dict[str, Any],
    label_lines: list[tuple[str, list[tuple[float, float]]]],
    suffix: str,
    *,
    point_transform: PointTransform,
    image_transform: ImageTransform,
) -> tuple[dict[str, Any], int, int]:
    split = record["split"]
    with Image.open(dataset_root / record["image"]) as source:
        image = source.convert("RGB")
    width, height = image.size
    transformed_image = image_transform(image)

    stem = Path(record["image"]).stem
    new_stem = f"{stem}__{suffix}"
    image_out = dataset_root / "images" / split / f"{new_stem}.png"
    label_out = dataset_root / "labels" / split / f"{new_stem}.txt"
    image_out.parent.mkdir(parents=True, exist_ok=True)
    transformed_image.save(image_out)

    kept_objects: list[dict[str, Any]] = []
    label_line_texts: list[str] = []
    dropped = 0
    for (class_id, points), item in zip(label_lines, record["objects"]):
        transformed_points = [
            point_transform(x * width, y * height, width, height) for x, y in points
        ]
        if any(
            nx < -_EPS or nx > width + _EPS or ny < -_EPS or ny > height + _EPS
            for nx, ny in transformed_points
        ):
            dropped += 1
            continue
        clamped = [
            (min(max(nx, 0.0), width), min(max(ny, 0.0), height))
            for nx, ny in transformed_points
        ]
        pixel_polygon = tuple((int(round(nx)), int(round(ny))) for nx, ny in clamped)
        xs = [point[0] for point in clamped]
        ys = [point[1] for point in clamped]
        new_item = dict(item)
        new_item["bounding_box"] = {
            "x_min": min(xs),
            "y_min": min(ys),
            "x_max": max(xs),
            "y_max": max(ys),
            "image_width": width,
            "image_height": height,
        }
        new_item["area_pixels"] = polygon_area(pixel_polygon)
        new_item["instance_id"] = f"{item['instance_id']}#{suffix}"
        kept_objects.append(new_item)
        coordinates = " ".join(
            f"{nx / width:.6f} {ny / height:.6f}" for nx, ny in clamped
        )
        label_line_texts.append(f"{class_id} {coordinates}")

    write_text(label_out, "\n".join(label_line_texts) if label_line_texts else "\n")

    new_record = dict(record)
    new_record["image"] = f"images/{split}/{new_stem}.png"
    new_record["label"] = f"labels/{split}/{new_stem}.txt"
    new_record["objects"] = kept_objects
    new_record["augmented"] = suffix
    return new_record, len(kept_objects), dropped
