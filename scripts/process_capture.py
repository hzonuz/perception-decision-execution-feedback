#!/usr/bin/env python3
"""Validate raw Unity capture and emit capture/oracle contracts."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from perception.privileged_capture import process_capture


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, default=Path("configs/ontology.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    report = process_capture(args.capture_root, args.ontology)
    logging.info("%s", json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
