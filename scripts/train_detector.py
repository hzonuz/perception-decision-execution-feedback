#!/usr/bin/env python3
"""Train and evaluate the YOLO segmentation detector."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from perception.detection import run_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/detector.json"))
    parser.add_argument("--data", help="Override the dataset YAML path.")
    parser.add_argument("--name", help="Override the experiment name.")
    parser.add_argument("--resume", help="Resume an incomplete Ultralytics checkpoint.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    output = run_experiment(
        args.config,
        data=args.data,
        experiment_name=args.name,
        resume_checkpoint=args.resume,
    )
    logging.info("experiment saved to %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
