"""Leakage-safe conversion from capture contracts to segmentation datasets."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

from perception.dataset.export import (
    validation_report,
    write_dataset,
    write_json,
)
from perception.dataset.models import (
    SPLITS,
    DatasetConfig,
    DatasetValidationError,
    Issue,
    Sample,
)
from perception.dataset.validation import (
    check_sequences,
    check_split_integrity,
    load_samples,
)
from perception.ontology import Ontology, load_ontology


def build_dataset(
    capture_roots: list[str | Path],
    output_root: str | Path,
    config_path: str | Path = "configs/dataset.json",
    ontology_path: str | Path = "configs/ontology.json",
    *,
    export_coco: bool | None = None,
) -> dict[str, Any]:
    """Validate, split, convert, summarize, and render a capture dataset."""
    config = DatasetConfig.from_json_file(config_path)
    ontology = load_ontology(ontology_path)
    output = Path(output_root).resolve()
    if output.exists():
        if not output.is_dir():
            raise NotADirectoryError(f"output path is not a directory: {output}")
        if any(output.iterdir()):
            raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    class_names = _class_names(config, ontology)
    class_ids = {name: index for index, name in enumerate(class_names)}
    samples, issues = load_samples(
        [Path(root).resolve() for root in capture_roots],
        config,
        ontology,
        class_ids,
    )
    split_by_group = _split_groups(samples, config)
    split_by_sample = {
        (sample.frame.episode_id, sample.frame.stamp.frame_index): split
        for group, split in split_by_group.items()
        for sample in samples
        if _group_value(sample, config.group_by) == group
    }
    check_sequences(samples, issues)
    check_split_integrity(samples, split_by_sample, issues)

    split_counts = {
        split: sum(1 for assigned in split_by_sample.values() if assigned == split)
        for split in SPLITS
    }
    for split, count in split_counts.items():
        if count == 0:
            issues.append(
                Issue(
                    "warning",
                    "empty_split",
                    f"{split} is empty; more independent groups are required",
                )
            )

    report = validation_report(samples, issues, split_counts, config)
    write_json(output / "validation_report.json", report)
    errors = [issue for issue in issues if issue.severity == "error"]
    if errors:
        raise DatasetValidationError(
            f"dataset validation failed with {len(errors)} error(s); "
            f"see {output / 'validation_report.json'}"
        )

    coco_enabled = config.export_coco if export_coco is None else export_coco
    write_dataset(
        output,
        samples,
        split_by_sample,
        split_by_group,
        class_names,
        config,
        coco_enabled,
    )
    return {
        "output_root": str(output),
        "frames": len(samples),
        "objects": sum(len(sample.objects) for sample in samples),
        "split_counts": split_counts,
        "class_names": class_names,
        "validation_status": report["status"],
        "warnings": len(report["warnings"]),
        "coco_exported": coco_enabled,
    }


def _split_groups(samples: list[Sample], config: DatasetConfig) -> dict[str, str]:
    groups = sorted({_group_value(sample, config.group_by) for sample in samples})
    groups.sort(
        key=lambda group: hashlib.sha256(
            f"{config.random_seed}:{group}".encode()
        ).digest()
    )
    counts = _split_counts(len(groups), config.ratios)
    assignment: dict[str, str] = {}
    offset = 0
    for split in SPLITS:
        for group in groups[offset : offset + counts[split]]:
            assignment[group] = split
        offset += counts[split]
    return assignment


def _split_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    counts = {split: 0 for split in SPLITS}
    active = [split for split in SPLITS if ratios[split] > 0]
    if not total or not active:
        return counts
    if total < len(active):
        for split in sorted(
            active, key=lambda name: (-ratios[name], SPLITS.index(name))
        )[:total]:
            counts[split] = 1
        return counts

    for split in active:
        counts[split] = 1
    remaining = total - len(active)
    desired = {
        split: remaining * ratios[split] / sum(ratios[name] for name in active)
        for split in active
    }
    for split in active:
        counts[split] += math.floor(desired[split])
    unassigned = total - sum(counts.values())
    order = sorted(
        active,
        key=lambda name: (
            -(desired[name] - math.floor(desired[name])),
            SPLITS.index(name),
        ),
    )
    for split in order[:unassigned]:
        counts[split] += 1
    return counts


def _group_value(sample: Sample, group_by: str) -> str:
    if group_by == "episode":
        return sample.frame.episode_id
    if group_by == "arena":
        return sample.frame.arena_id
    return str(sample.frame.random_seed)


def _class_names(config: DatasetConfig, ontology: Ontology) -> list[str]:
    if config.class_mode == "semantic":
        return [item.name for item in ontology.semantic_classes]
    return [item.exact_class for item in ontology.simulator_mappings]
