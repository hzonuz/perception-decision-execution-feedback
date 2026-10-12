from __future__ import annotations
import argparse
import json
import sys

from perception.inference.detector import MockDetector
from perception.inference.pipeline import run_inference
from perception.inference.sources import live_frames, prerecorded_frames
from perception.ontology import load_ontology
from perception.scene_graph import SceneGraphBuilder, SceneGraphConfig
from perception.tracking import Tracker, TrackerConfig, collectable_classes

DEFAULT_WEIGHTS = "runs/detector/notebook_yolo26n_seg/weights/best.pt"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m perception.inference",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--frames", help="directory of prerecorded images (one episode)"
    )
    source.add_argument(
        "--live", action="store_true", help="step a live Animal-AI environment"
    )
    parser.add_argument("--detector", choices=["yolo", "mock"], default="yolo")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda:0, ...")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--conf", type=float, default=0.25, help="detector confidence threshold"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="seeded, CPU unless --device is set",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ontology",
        default="configs/ontology.json",
        help="validate labels against it; '' skips",
    )
    parser.add_argument(
        "--output",
        help="write detections/masks/timing/summary here (must be new or empty)",
    )
    parser.add_argument(
        "--visualize", action="store_true", help="write overlays/ (needs --output)"
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="write rgb/ + observations.jsonl (needs --output)",
    )
    parser.add_argument(
        "--track", action="store_true", help="run the tracker too; writes tracks.jsonl"
    )
    parser.add_argument("--tracker-config", default="configs/tracker.json")
    parser.add_argument(
        "--graph",
        action="store_true",
        help="build scene graphs too (implies --track); graphs.jsonl",
    )
    parser.add_argument("--graph-config", default="configs/scene_graph.json")
    parser.add_argument(
        "--frame-interval",
        type=float,
        default=0.06,
        help="prerecorded seconds per frame",
    )
    parser.add_argument(
        "--observations",
        help="observations.jsonl from a live --save-frames run (replays actions)",
    )
    live = parser.add_argument_group("live")
    live.add_argument(
        "--env", help="Animal-AI binary (required with --live; nothing is downloaded)"
    )
    live.add_argument("--arena", default="", help="arena YAML")
    live.add_argument("--episodes", type=int, default=1)
    live.add_argument("--max-steps", type=int, default=500)
    live.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="camera size; the detector trained on 256",
    )
    live.add_argument("--decision-period", type=int, default=3)
    live.add_argument("--port", type=int, default=5005)
    args = parser.parse_args(argv)
    if (args.visualize or args.save_frames) and not args.output:
        parser.error("--visualize and --save-frames need --output")
    if args.live and not args.env:
        parser.error("--live needs --env (path to the Animal-AI binary)")

    ontology = load_ontology(args.ontology) if args.ontology else None
    if args.detector == "mock":
        names = sorted(ontology.semantic_names) if ontology else ["object"]
        detector = MockDetector(names, seed=args.seed)
    else:
        from perception.inference.yolo import YoloDetector

        detector = YoloDetector(
            args.weights,
            device=args.device,
            image_size=args.image_size,
            confidence_threshold=args.conf,
            deterministic=args.deterministic,
            seed=args.seed,
        )

    tracker = graph_builder = None
    if args.graph and not ontology:
        parser.error("--graph needs --ontology")
    if args.track or args.graph:
        terminable = collectable_classes(ontology) if ontology else frozenset()
        tracker = Tracker(TrackerConfig.from_json_file(args.tracker_config), terminable)
    if args.graph:
        graph_builder = SceneGraphBuilder(
            SceneGraphConfig.from_json_file(args.graph_config), ontology
        )
    options = dict(
        batch_size=args.batch_size,
        ontology=ontology,
        visualize=args.visualize,
        save_frames=args.save_frames,
        tracker=tracker,
        graph_builder=graph_builder,
    )
    if args.frames:
        summary = run_inference(
            detector,
            prerecorded_frames(
                args.frames,
                frame_interval=args.frame_interval,
                observations=args.observations,
            ),
            args.output,
            **options,
        )
    else:
        from animalai.environment import AnimalAIEnvironment

        env = AnimalAIEnvironment(
            file_name=args.env,
            arenas_configurations=args.arena,
            seed=args.seed,
            base_port=args.port,
            useCamera=True,
            resolution=args.resolution,
            useRayCasts=False,
            decisionPeriod=args.decision_period,
        )
        try:
            frames = live_frames(
                env,
                episodes=args.episodes,
                max_steps=args.max_steps,
                seed=args.seed,
                decision_period=args.decision_period,
            )
            summary = run_inference(detector, frames, args.output, **options)
        finally:
            env.close()
    json.dump(summary, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
