from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from PIL import Image

from perception.dataset.converter import split_counts
from perception.dataset.models import SPLITS, DatasetValidationError
from perception.jsonio import write_json, write_text

_SOURCE_SPLIT_DIRECTORIES = ("train", "valid", "val", "test")
_AUGMENTATION_SUFFIX = re.compile(r"\.rf\.[0-9a-f]+$", re.IGNORECASE)
_FORMAT_SUFFIX = re.compile(r"_(png|jpg|jpeg|bmp)$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ExternalSource:
    name: str
    root: Path
    class_map: dict[str, str]
    group_pattern: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip() or "/" in self.name:
            raise ValueError("source name must be a non-empty path-safe string")
        if not self.class_map:
            raise ValueError(f"{self.name}: class_map must not be empty")
        if self.group_pattern is not None and not re.compile(self.group_pattern).groups:
            raise ValueError(f"{self.name}: group_pattern needs one capture group")


def merge_datasets(
    primary: str | Path,
    sources: Sequence[ExternalSource],
    output: str | Path,
    *,
    ratios: dict[str, float] | None = None,
    random_seed: int = 20260725,
) -> dict[str, Any]:
    primary_root = Path(primary).resolve()
    output = Path(output).resolve()
    ratios = ratios or {"train": 0.6, "val": 0.2, "test": 0.2}
    if output.exists() and any(output.iterdir()):
        raise DatasetValidationError(f"output directory is not empty: {output}")

    class_names = _primary_class_names(primary_root)
    class_ids = {name: index for index, name in enumerate(class_names)}
    for split in SPLITS:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    (output / "metadata").mkdir(parents=True, exist_ok=True)

    records: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    groups: dict[str, dict[str, str]] = {split: {} for split in SPLITS}
    report: dict[str, Any] = {"sources": {}}

    report["sources"]["capture"] = _copy_primary(primary_root, output, records, groups)
    for source in sources:
        report["sources"][source.name] = _copy_external(
            source,
            output,
            records,
            groups,
            class_ids,
            ratios,
            random_seed,
        )

    for split in SPLITS:
        write_text(
            output / "metadata" / f"{split}.jsonl",
            "\n".join(json.dumps(record, sort_keys=True) for record in records[split]),
        )
    write_text(
        output / "data.yaml",
        "\n".join(
            [
                f"path: {output.as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                *(f"  {index}: {name}" for index, name in enumerate(class_names)),
            ]
        ),
    )
    write_json(
        output / "splits.json",
        {
            "group_by": "episode",
            "random_seed": random_seed,
            "ratios": ratios,
            "episodes": {split: sorted(groups[split]) for split in SPLITS},
            "group_origin": {
                split: dict(sorted(groups[split].items())) for split in SPLITS
            },
        },
    )

    report.update(_verify(output, records, groups, class_names))
    write_json(output / "merge_report.json", report)
    return report


def _primary_class_names(primary_root: Path) -> list[str]:
    data_yaml = primary_root / "data.yaml"
    if not data_yaml.is_file():
        raise DatasetValidationError(
            f"primary dataset has no data.yaml: {primary_root}"
        )
    names = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))["names"]
    if isinstance(names, dict):
        return [
            str(name) for _, name in sorted(names.items(), key=lambda kv: int(kv[0]))
        ]
    return [str(name) for name in names]


def _copy_primary(
    primary_root: Path,
    output: Path,
    records: dict[str, list[dict[str, Any]]],
    groups: dict[str, dict[str, str]],
) -> dict[str, Any]:
    counts = {split: 0 for split in SPLITS}
    for split in SPLITS:
        metadata_path = primary_root / "metadata" / f"{split}.jsonl"
        if not metadata_path.is_file():
            raise DatasetValidationError(f"primary dataset is missing {metadata_path}")
        for line in metadata_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            image = primary_root / record["image"]
            label = primary_root / record["label"]
            if not image.is_file() or not label.is_file():
                raise DatasetValidationError(
                    f"primary dataset is missing {image} or {label}"
                )
            shutil.copy2(image, output / "images" / split / image.name)
            shutil.copy2(label, output / "labels" / split / label.name)
            record["source"] = "capture"
            record["mask_source"] = "polygon"
            for item in record["objects"]:
                item["mask_source"] = "polygon"
            records[split].append(record)
            groups[split][record["episode_id"]] = "capture"
            counts[split] += 1
    return {"kind": "capture_polygons", "images": counts, "root": str(primary_root)}


def _copy_external(
    source: ExternalSource,
    output: Path,
    records: dict[str, list[dict[str, Any]]],
    groups: dict[str, dict[str, str]],
    class_ids: dict[str, int],
    ratios: dict[str, float],
    random_seed: int,
) -> dict[str, Any]:
    root = Path(source.root).resolve()
    source_names = (
        _primary_class_names(root) if (root / "data.yaml").is_file() else None
    )
    if source_names is None:
        raise DatasetValidationError(f"{source.name}: no data.yaml under {root}")
    unmapped = [name for name in source_names if name not in source.class_map]
    if unmapped:
        raise DatasetValidationError(f"{source.name}: unmapped classes {unmapped}")
    unknown = [
        target for target in source.class_map.values() if target not in class_ids
    ]
    if unknown:
        raise DatasetValidationError(
            f"{source.name}: class_map targets are not primary classes: {unknown}"
        )

    pairs = _collect_pairs(root)
    if not pairs:
        raise DatasetValidationError(
            f"{source.name}: no image/label pairs under {root}"
        )
    pattern = re.compile(source.group_pattern) if source.group_pattern else None
    by_group: dict[str, list[tuple[Path, Path]]] = {}
    for image, label in pairs:
        by_group.setdefault(_group_key(image, pattern), []).append((image, label))

    assignment = _assign_groups(sorted(by_group), ratios, random_seed, source.name)
    counts = {split: 0 for split in SPLITS}
    objects = {"polygon": 0, "box_rectangle": 0}
    for group in sorted(by_group):
        split = assignment[group]
        episode_id = f"{source.name}/{group}"
        groups[split][episode_id] = source.name
        for frame_index, (image, label) in enumerate(sorted(by_group[group])):
            stem = f"{source.name}__{image.stem}"
            image_name = f"{stem}{image.suffix.lower()}"
            shutil.copy2(image, output / "images" / split / image_name)
            with Image.open(image) as handle:
                width, height = handle.size
            items, lines = _convert_label(
                label, source_names, source.class_map, class_ids
            )
            write_text(output / "labels" / split / f"{stem}.txt", "\n".join(lines))
            for item in items:
                objects[item["mask_source"]] += 1
                box = item.pop("normalized_box")
                item["bounding_box"] = {
                    "x_min": box[0] * width,
                    "y_min": box[1] * height,
                    "x_max": box[2] * width,
                    "y_max": box[3] * height,
                    "image_width": width,
                    "image_height": height,
                }
                item["area_pixels"] = (
                    (box[2] - box[0]) * width * (box[3] - box[1]) * height
                )
                item["instance_id"] = f"{episode_id}#{frame_index}:{item['index']}"
            records[split].append(
                {
                    "split": split,
                    "source": source.name,
                    "mask_source": "mixed",
                    "episode_id": episode_id,
                    "arena_id": group,
                    "random_seed": None,
                    "frame_index": frame_index,
                    "timestamp_seconds": float(frame_index),
                    "image": f"images/{split}/{image_name}",
                    "label": f"labels/{split}/{stem}.txt",
                    "source_annotation": str(label),
                    "objects": items,
                }
            )
            counts[split] += 1
    return {
        "kind": "external_yolo",
        "root": str(root),
        "images": counts,
        "groups": len(by_group),
        "objects_by_mask_source": objects,
        "class_map": source.class_map,
    }


def _collect_pairs(root: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for name in _SOURCE_SPLIT_DIRECTORIES:
        images = root / name / "images"
        labels = root / name / "labels"
        if not images.is_dir() or not labels.is_dir():
            continue
        for image in sorted(images.iterdir()):
            if image.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp"):
                continue
            label = labels / f"{image.stem}.txt"
            if label.is_file():
                pairs.append((image, label))
    return pairs


def _group_key(image: Path, pattern: re.Pattern[str] | None) -> str:
    stem = (
        _FORMAT_SUFFIX.sub("", _AUGMENTATION_SUFFIX.sub("", image.stem)) or image.stem
    )
    if pattern is None:
        return stem
    match = pattern.search(stem)
    if match is None:
        raise DatasetValidationError(f"group_pattern did not match image stem {stem!r}")
    return match.group(1)


def _assign_groups(
    group_names: list[str],
    ratios: dict[str, float],
    random_seed: int,
    salt: str,
) -> dict[str, str]:
    ordered = sorted(
        group_names,
        key=lambda group: hashlib.sha256(
            f"{random_seed}:{salt}:{group}".encode()
        ).digest(),
    )
    counts = split_counts(len(ordered), ratios)
    assignment: dict[str, str] = {}
    offset = 0
    for split in SPLITS:
        for group in ordered[offset : offset + counts[split]]:
            assignment[group] = split
        offset += counts[split]
    return assignment


def _convert_label(
    label: Path,
    source_names: list[str],
    class_map: dict[str, str],
    class_ids: dict[str, int],
) -> tuple[list[dict[str, Any]], list[str]]:
    items: list[dict[str, Any]] = []
    lines: list[str] = []
    for index, raw in enumerate(label.read_text(encoding="utf-8").splitlines()):
        parts = raw.split()
        if not parts:
            continue
        source_id = int(parts[0])
        if not 0 <= source_id < len(source_names):
            raise DatasetValidationError(f"{label}: class id {source_id} out of range")
        class_name = class_map[source_names[source_id]]
        class_id = class_ids[class_name]
        values = [min(1.0, max(0.0, float(value))) for value in parts[1:]]
        if len(values) == 4:
            center_x, center_y, width, height = values
            x_min, x_max = center_x - width / 2, center_x + width / 2
            y_min, y_max = center_y - height / 2, center_y + height / 2
            polygon = [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]
            mask_source = "box_rectangle"
        elif len(values) >= 6 and len(values) % 2 == 0:
            polygon = values
            x_min, x_max = min(values[0::2]), max(values[0::2])
            y_min, y_max = min(values[1::2]), max(values[1::2])
            mask_source = "polygon"
        else:
            raise DatasetValidationError(
                f"{label}: line {index} is not a box or polygon"
            )
        lines.append(f"{class_id} " + " ".join(f"{value:.6f}" for value in polygon))
        items.append(
            {
                "index": index,
                "class_id": class_id,
                "class_name": class_name,
                "exact_simulator_class": source_names[source_id],
                "mask_source": mask_source,
                "normalized_box": (x_min, y_min, x_max, y_max),
                "visibility_fraction": None,
                "occlusion_fraction": None,
                "relative_depth": None,
                "motion": None,
            }
        )
    return items, lines


def _verify(
    output: Path,
    records: dict[str, list[dict[str, Any]]],
    groups: dict[str, dict[str, str]],
    class_names: list[str],
) -> dict[str, Any]:
    for left in SPLITS:
        for right in SPLITS:
            if left < right and set(groups[left]) & set(groups[right]):
                raise DatasetValidationError(
                    f"group leakage between {left} and {right}: "
                    f"{sorted(set(groups[left]) & set(groups[right]))}"
                )
    for split in SPLITS:
        if not records[split]:
            raise DatasetValidationError(f"{split} split is empty")
        for record in records[split]:
            if not (output / record["image"]).is_file():
                raise DatasetValidationError(f"missing image {record['image']}")
            if not (output / record["label"]).is_file():
                raise DatasetValidationError(f"missing label {record['label']}")

    histogram: dict[str, dict[str, int]] = {}
    mask_sources: dict[str, int] = {}
    for split in SPLITS:
        counts = dict.fromkeys(class_names, 0)
        for record in records[split]:
            for item in record["objects"]:
                counts[item["class_name"]] += 1
                mask_sources[item["mask_source"]] = (
                    mask_sources.get(item["mask_source"], 0) + 1
                )
        histogram[split] = {name: value for name, value in counts.items() if value}
    return {
        "class_names": class_names,
        "images": {split: len(records[split]) for split in SPLITS},
        "groups": {split: len(groups[split]) for split in SPLITS},
        "objects_by_class": histogram,
        "objects_by_mask_source": mask_sources,
        "leakage": "none",
        "output_root": str(output),
    }
