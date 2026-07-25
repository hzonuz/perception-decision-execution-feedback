#!/usr/bin/env python3
"""Validate capture and SSG JSON files against the shared contracts."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from perception.contracts import CaptureFrame, SSGSnapshot
from perception.ontology import load_ontology


LOGGER = logging.getLogger("perception.contract_validation")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ontology", type=Path, default=Path("configs/ontology.json"))
    parser.add_argument("--capture", action="append", type=Path, default=[])
    parser.add_argument("--ssg", action="append", type=Path, default=[])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    ontology = load_ontology(args.ontology)
    LOGGER.info("ontology_validated path=%s version=%s", args.ontology, ontology.ontology_version)

    for path in args.capture:
        frame = CaptureFrame.from_json_file(path)
        ontology.validate_capture(frame)
        LOGGER.info(
            "capture_validated path=%s episode=%s frame=%d objects=%d",
            path,
            frame.episode_id,
            frame.stamp.frame_index,
            len(frame.objects),
        )

    for path in args.ssg:
        graph = SSGSnapshot.from_json_file(path)
        ontology.validate_ssg(graph)
        LOGGER.info(
            "ssg_validated path=%s graph=%s frame=%d nodes=%d relations=%d",
            path,
            graph.graph_id,
            graph.stamp.frame_index,
            1 + len(graph.objects),
            len(graph.relations),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
