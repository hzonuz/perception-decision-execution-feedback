from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw, ImageFont

_SPLITS = ("train", "val", "test")


def build_class_samples(
    dataset_root: str | Path,
    output: str | Path | None = None,
    *,
    padding: int = 12,
    min_size: int = 96,
    target_area_fraction: float = 0.12,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root)
    output = Path(output) if output is not None else dataset_root / "class_samples"
    names = yaml.safe_load((dataset_root / "data.yaml").read_text(encoding="utf-8"))[
        "names"
    ]

    best: dict[str, dict[str, Any]] = {}
    for split in _SPLITS:
        metadata_path = dataset_root / "metadata" / f"{split}.jsonl"
        if not metadata_path.is_file():
            continue
        for line in metadata_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            for item in record["objects"]:
                box = item.get("bounding_box")
                if not box:
                    continue
                rank = _rank(item, box, target_area_fraction)
                name = item["class_name"]
                if name not in best or rank > best[name]["rank"]:
                    best[name] = {
                        "rank": rank,
                        "image": dataset_root / record["image"],
                        "box": box,
                        "mask_source": item.get("mask_source", "?"),
                        "split": split,
                    }

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    for name, entry in best.items():
        _write_sample(
            output / f"{name}.png", name, entry, padding=padding, min_size=min_size
        )

    class_names = set(names.values())
    return {
        "output": str(output),
        "written": sorted(best),
        "missing": sorted(class_names - set(best)),
    }


def _rank(
    item: dict[str, Any], box: dict[str, Any], target_area_fraction: float
) -> tuple:
    area_fraction = item["area_pixels"] / (box["image_width"] * box["image_height"])
    clipped = (
        box["x_min"] <= 1
        or box["y_min"] <= 1
        or box["x_max"] >= box["image_width"] - 1
        or box["y_max"] >= box["image_height"] - 1
    )
    return (
        not clipped,
        item.get("mask_source") == "polygon",
        -abs(area_fraction - target_area_fraction),
    )


def _write_sample(
    path: Path, name: str, entry: dict[str, Any], *, padding: int, min_size: int
) -> None:
    box = entry["box"]
    with Image.open(entry["image"]) as source:
        left = max(0, int(box["x_min"]) - padding)
        top = max(0, int(box["y_min"]) - padding)
        right = min(box["image_width"], int(box["x_max"]) + padding)
        bottom = min(box["image_height"], int(box["y_max"]) + padding)
        crop = source.convert("RGB").crop((left, top, right, bottom))

    scale = max(1, min_size // max(crop.width, crop.height, 1))
    if scale > 1:
        crop = crop.resize((crop.width * scale, crop.height * scale), Image.NEAREST)

    label = f"{name}  [{entry['mask_source']}, {entry['split']}]"
    min_canvas_width = 210  # wide enough for the longest class name at a readable size
    canvas_width = max(crop.width, min_canvas_width)
    banner_height = 22
    canvas = Image.new("RGB", (canvas_width, crop.height + banner_height), (24, 24, 24))
    canvas.paste(crop, ((canvas_width - crop.width) // 2, banner_height))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (4, 3), label, fill=(255, 255, 255), font=_fitted_font(label, canvas_width - 8)
    )
    canvas.save(path)


def _fitted_font(
    text: str, max_width: int, start_size: int = 15, min_size: int = 9
) -> ImageFont.ImageFont:
    size = start_size
    font = ImageFont.load_default()
    while size > min_size:
        try:
            font = ImageFont.truetype(
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf", size
            )
        except OSError:
            return ImageFont.load_default()
        if font.getlength(text) <= max_width:
            return font
        size -= 1
    return font


def build_contact_sheet(
    samples_dir: str | Path,
    output: str | Path | None = None,
    *,
    columns: int = 7,
    cell_size: tuple[int, int] = (220, 190),
) -> Path:
    samples_dir = Path(samples_dir)
    output = (
        Path(output)
        if output is not None
        else samples_dir.parent / "class_samples_contact_sheet.png"
    )
    samples = sorted(samples_dir.glob("*.png"))
    if not samples:
        raise FileNotFoundError(f"no samples found in {samples_dir}")

    cell_width, cell_height = cell_size
    rows = (len(samples) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), (12, 12, 12))
    for index, path in enumerate(samples):
        with Image.open(path) as sample:
            sample.thumbnail((cell_width - 6, cell_height - 6))
            x = (index % columns) * cell_width + (cell_width - sample.width) // 2
            y = (index // columns) * cell_height + (cell_height - sample.height) // 2
            sheet.paste(sample, (x, y))
    sheet.save(output)
    return output
