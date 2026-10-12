from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from perception.jsonio import sanitize


def validate_dataset(
    data_yaml: str | Path,
) -> tuple[Path, list[str], list[dict[str, Any]]]:
    data_yaml = Path(data_yaml)
    if not data_yaml.is_file():
        raise FileNotFoundError(f"dataset YAML not found: {data_yaml}")

    raw = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("names"), dict):
        raise ValueError("dataset YAML must contain an indexed names mapping")
    root = Path(raw["path"])
    root = (
        (data_yaml.parent / root).resolve()
        if not root.is_absolute()
        else root.resolve()
    )
    names = [
        str(name)
        for _, name in sorted(raw["names"].items(), key=lambda item: int(item[0]))
    ]
    splits = json.loads((root / "splits.json").read_text(encoding="utf-8"))
    episode_sets = [
        set(splits["episodes"][split]) for split in ("train", "val", "test")
    ]
    if any(
        episode_sets[left] & episode_sets[right]
        for left, right in ((0, 1), (0, 2), (1, 2))
    ):
        raise ValueError("dataset has episode leakage across splits")
    if any(not episode_set for episode_set in episode_sets):
        raise ValueError(
            "train, val, and test must each contain an independent episode"
        )
    records = [
        json.loads(line)
        for line in (root / "metadata/test.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("held-out test metadata is empty")
    for record in records:
        if not (root / record["image"]).is_file():
            raise FileNotFoundError(f"test image is missing: {record['image']}")
    return root, names, records


def ultralytics_metrics(metrics: Any) -> dict[str, Any]:
    return sanitize(
        {
            "box": {
                "precision": metrics.box.mp,
                "recall": metrics.box.mr,
                "map50": metrics.box.map50,
                "map50_95": metrics.box.map,
            },
            "mask": {
                "precision": metrics.seg.mp,
                "recall": metrics.seg.mr,
                "map50": metrics.seg.map50,
                "map50_95": metrics.seg.map,
            },
            "per_class": metrics.summary(),
        }
    )
