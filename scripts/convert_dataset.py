#!/usr/bin/env python3
"""Build a validated YOLO segmentation dataset from capture roots."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from perception.dataset import build_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/dataset.json"))
    parser.add_argument("--ontology", type=Path, default=Path("configs/ontology.json"))
    parser.add_argument(
        "--coco",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the configuration's optional COCO export setting.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = build_dataset(
        args.capture_root,
        args.output,
        args.config,
        args.ontology,
        export_coco=args.coco,
    )
    logging.info("%s", json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
