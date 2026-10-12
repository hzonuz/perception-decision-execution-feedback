from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from perception.contracts import DetectionFrame, FrameStamp, ObservationFrame, SourceMetadata, SourceType
from perception.inference import (
    Frame,
    MockDetector,
    Timing,
    live_frames,
    load_detections,
    prerecorded_frames,
    render_overlay,
    run_inference,
)
from perception.inference.__main__ import main
from perception.inference.detector import build_detection
from perception.ontology import load_ontology

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "runs/detector/notebook_yolo26n_seg/weights/best.pt"
ONTOLOGY = load_ontology(ROOT / "configs/ontology.json")
CLASSES = sorted(ONTOLOGY.semantic_names)
SOURCE = SourceMetadata(SourceType.DETECTOR, "unit-test", "1.0")


def frames(count: int = 6, episode: str = "ep", size: tuple[int, int] = (24, 32)) -> list[Frame]:
    rng = np.random.default_rng(0)
    return [
        Frame(FrameStamp(episode, i, i * 0.06), rng.integers(0, 255, (*size, 3), dtype=np.uint8))
        for i in range(count)
    ]


def has_yolo() -> bool:
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        return False
    return WEIGHTS.is_file()


class BuildDetectionTests(unittest.TestCase):
    def test_fully_covered_mask_has_no_reference(self) -> None:
        mask = np.ones((10, 10), bool)
        detection = build_detection(
            FrameStamp("ep", 0, 0.0), 10, 10,
            [((0, 0, 10, 10), {"Wall": 0.9}, mask), ((0, 0, 10, 10), {"GoodGoal": 0.5}, mask)],
            SOURCE, Timing(0, 0, 0),
        )
        self.assertTrue((detection.masks == 1).all())
        self.assertIsNotNone(detection.frame.objects[0].instance_mask)
        self.assertIsNone(detection.frame.objects[1].instance_mask)

    def test_overlap_goes_to_higher_confidence_and_scores_are_renormalized(self) -> None:
        low = np.zeros((10, 10), bool)
        low[0:6, 0:6] = True
        high = np.zeros((10, 10), bool)
        high[4:10, 4:10] = True
        detection = build_detection(
            FrameStamp("ep", 0, 0.0), 10, 10,
            [((0, 0, 6, 6), {"Wall": 0.4}, low), ((4, 4, 10, 10), {"GoodGoal": 0.9, "BadGoal": 0.6}, high)],
            SOURCE, Timing(0, 0, 0),
        )
        first, second = detection.frame.objects
        self.assertEqual(first.semantic_distribution.entries[0].label, "GoodGoal")  # sorted by confidence
        self.assertAlmostEqual(first.semantic_distribution.entries[0].probability, 0.6)  # 0.9 / 1.5
        self.assertEqual(first.semantic_distribution.unknown_probability, 0.0)
        self.assertAlmostEqual(second.semantic_distribution.unknown_probability, 0.6)
        self.assertEqual(detection.masks[5, 5], 1)  # overlap pixel belongs to the confident object
        self.assertEqual(detection.masks[0, 0], 2)
        self.assertEqual(second.instance_mask.instance_value, 2)

    def test_degenerate_and_out_of_frame_boxes(self) -> None:
        detection = build_detection(
            FrameStamp("ep", 0, 0.0), 10, 10,
            [((3, 3, 3, 8), {"Wall": 0.9}, None), ((-5, -5, 20, 4), {"Wall": 0.5}, None)],
            SOURCE, Timing(0, 0, 0),
        )
        (only,) = detection.frame.objects
        box = only.bounding_box
        self.assertEqual((box.x_min, box.y_min, box.x_max, box.y_max), (0, 0, 10, 4))
        self.assertIsNone(only.instance_mask)
        self.assertIsNone(detection.masks)


class MockDetectorTests(unittest.TestCase):
    def test_synthetic_overlay_matches_mock_masks(self) -> None:
        for detection in MockDetector(CLASSES, seed=3).detect_batch(frames(12)):
            overlay = np.asarray(render_overlay(None, detection))
            self.assertEqual(overlay.shape, (24, 32, 3))
            if detection.masks is not None:
                self.assertTrue(np.all(np.any(overlay[detection.masks > 0] != 235, axis=1)))
            if not detection.frame.objects:
                self.assertTrue((overlay == 235).all())

    def test_deterministic_and_independent_of_batching(self) -> None:
        inputs = frames(12)
        one_by_one = [MockDetector(CLASSES, seed=3).detect(f).frame for f in inputs]
        batched = [d.frame for d in MockDetector(CLASSES, seed=3).detect_batch(inputs[::-1])][::-1]
        self.assertEqual(one_by_one, batched)
        self.assertNotEqual(one_by_one, [MockDetector(CLASSES, seed=4).detect(f).frame for f in inputs])
        self.assertTrue(any(not f.objects for f in one_by_one), "some frames should be empty")
        self.assertTrue(any(len(o.semantic_distribution.entries) > 1 for f in one_by_one for o in f.objects))


class PipelineTests(unittest.TestCase):
    def detectors(self):
        yield MockDetector(CLASSES, seed=1)
        if has_yolo():
            from perception.inference.yolo import YoloDetector

            yield YoloDetector(WEIGHTS, deterministic=True)

    def test_same_downstream_interface_for_every_detector(self) -> None:
        for detector in self.detectors():
            with self.subTest(detector=detector.name), tempfile.TemporaryDirectory() as tmp:
                seen = []
                summary = run_inference(
                    detector, frames(5), tmp, batch_size=2, ontology=ONTOLOGY, visualize=True, save_frames=True,
                    on_detection=lambda frame, detection: seen.append((frame.stamp, detection.frame)),
                )
                written = load_detections(Path(tmp) / "detections.jsonl")
                self.assertEqual(written, [frame for _, frame in seen])
                self.assertEqual([f.stamp for f in written], [stamp for stamp, _ in seen])
                self.assertEqual(summary["frames"], 5)
                self.assertEqual(summary["detections"], sum(len(f.objects) for f in written))
                self.assertGreater(summary["throughput"]["detector_fps"], 0)
                self.assertEqual(set(summary["latency_ms"]["per_frame_wall"]), {"mean", "p50", "p95", "max"})
                self.assertEqual(json.loads((Path(tmp) / "summary.json").read_text())["frames"], 5)
                self.assertEqual(len((Path(tmp) / "timing.jsonl").read_text().splitlines()), 5)
                self.assertEqual(len(list((Path(tmp) / "overlays").rglob("*.png"))), 5)
                for frame in written:
                    for item in frame.objects:
                        if item.instance_mask is not None:
                            instance_map = np.asarray(Image.open(Path(tmp) / item.instance_mask.path))
                            self.assertTrue((instance_map == item.instance_mask.instance_value).any())

    def test_yolo_deterministic_mode_repeats_exactly(self) -> None:
        if not has_yolo():
            self.skipTest("ultralytics or trained weights not available")
        from perception.inference.yolo import YoloDetector

        inputs = [f for f in prerecorded_frames(self.dataset_frames())][:4]
        first = [d.frame for d in YoloDetector(WEIGHTS, deterministic=True).detect_batch(inputs)]
        second = [d.frame for d in YoloDetector(WEIGHTS, deterministic=True).detect_batch(inputs)]
        self.assertEqual(first, second)
        self.assertTrue(any(f.objects for f in first), "trained model should detect something on real frames")
        # batch size 1 differs from batched only by float rounding: same objects, same labels
        single = [YoloDetector(WEIGHTS, deterministic=True).detect(f).frame for f in inputs]
        for batched, alone in zip(first, single, strict=True):
            self.assertEqual([o.semantic_distribution.entries[0].label for o in batched.objects],
                             [o.semantic_distribution.entries[0].label for o in alone.objects])
            for a, b in zip(batched.objects, alone.objects, strict=True):
                self.assertAlmostEqual(a.bounding_box.x_min, b.bounding_box.x_min, places=2)

    def dataset_frames(self) -> Path:
        images = sorted((ROOT / "datasets/animalai_merged/images/val").glob("*.png"))[:4]
        if not images:
            self.skipTest("no dataset images")
        tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        for path in images:
            (tmp / path.name).write_bytes(path.read_bytes())
        return tmp

    def test_refuses_non_empty_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "keep.txt").write_text("x")
            with self.assertRaises(FileExistsError):
                run_inference(MockDetector(CLASSES), frames(1), tmp)

    def test_blank_frames_produce_empty_valid_detections(self) -> None:
        summary = run_inference(MockDetector(CLASSES, max_objects=0), frames(3), ontology=ONTOLOGY)
        self.assertEqual((summary["frames"], summary["empty_frames"], summary["detections"]), (3, 3, 0))


class SourceTests(unittest.TestCase):
    def test_prerecorded_frames_are_ordered_rgb(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for name, color in (("b.png", (0, 255, 0)), ("a.png", (255, 0, 0)), ("notes.txt", None)):
                if color is None:
                    (Path(tmp) / name).write_text("ignored")
                else:
                    Image.new("RGB", (8, 6), color).save(Path(tmp) / name)
            loaded = list(prerecorded_frames(tmp, episode_id="rec", frame_interval=0.1))
        self.assertEqual([f.stamp for f in loaded], [FrameStamp("rec", 0, 0.0), FrameStamp("rec", 1, 0.1)])
        self.assertEqual(tuple(loaded[0].image[0, 0]), (255, 0, 0))
        self.assertEqual(loaded[0].image.shape, (6, 8, 3))

    def test_live_frames_parse_observations_and_stop_at_terminal(self) -> None:
        env = FakeEnv(steps_until_terminal=3)
        loaded = list(live_frames(env, episodes=2, max_steps=10, seed=5))
        self.assertEqual([f.stamp.episode_id for f in loaded], ["live-5-000"] * 4 + ["live-5-001"] * 4)
        self.assertEqual([f.stamp.frame_index for f in loaded[:4]], [0, 1, 2, 3])
        self.assertAlmostEqual(loaded[2].stamp.timestamp_seconds, 0.12)
        self.assertEqual(env.resets, 1)
        agent = loaded[1].agent
        self.assertEqual(agent.health, 1.0)
        self.assertEqual((agent.world_position.x, agent.world_position.z), (20.0, 20.0))
        self.assertIsNone(loaded[0].agent.recent_action)
        self.assertIsNotNone(agent.recent_action)
        self.assertEqual(loaded[0].image.dtype, np.uint8)

        with tempfile.TemporaryDirectory() as tmp:
            run_inference(MockDetector(CLASSES), live_frames(FakeEnv(2), max_steps=5), tmp, save_frames=True)
            observations = [ObservationFrame.from_json(line)
                            for line in (Path(tmp) / "observations.jsonl").read_text().splitlines()]
            self.assertEqual(len(observations), 3)
            self.assertTrue((Path(tmp) / observations[0].rgb.path).is_file())
            # recorded live frames replay through prerecorded mode
            replay = list(prerecorded_frames(Path(tmp) / "rgb" / "live-0-000"))
            self.assertEqual(len(replay), 3)
            self.assertIsNone(replay[1].agent)
            replay = list(prerecorded_frames(Path(tmp) / "rgb" / "live-0-000",
                                             observations=Path(tmp) / "observations.jsonl"))
            self.assertEqual([f.stamp for f in replay], [o.stamp for o in observations])
            self.assertEqual(replay[1].agent, observations[1].agent)
            self.assertIsNotNone(replay[1].agent.recent_action)


class CliTests(unittest.TestCase):
    def test_prerecorded_mock_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            images = Path(tmp) / "frames"
            images.mkdir()
            for i in range(3):
                Image.new("RGB", (16, 16), (i * 40, 0, 0)).save(images / f"{i:03d}.png")
            out = io.StringIO()
            with redirect_stdout(out):
                main(["--frames", str(images), "--detector", "mock", "--output", str(Path(tmp) / "out"),
                      "--ontology", str(ROOT / "configs/ontology.json"), "--visualize"])
            self.assertEqual(json.loads(out.getvalue())["frames"], 3)
            self.assertEqual(len(load_detections(Path(tmp) / "out" / "detections.jsonl")), 3)


class YoloConversionTests(unittest.TestCase):
    def test_same_anchor_rows_fold_into_one_multiclass_object(self) -> None:
        try:
            from perception.inference.yolo import detection_from_result
        except ImportError:
            self.skipTest("ultralytics not installed")
        mask = np.zeros((1, 16, 16), np.float32)
        mask[0, 4:10, 4:10] = 1.0
        result = SimpleNamespace(
            orig_shape=(16, 16),
            boxes=SimpleNamespace(xyxy=[[4.0, 4.0, 10.0, 10.0]] * 2, conf=[0.7, 0.2], cls=[0, 1]),
            masks=SimpleNamespace(data=np.concatenate([mask, mask])),
            names={0: "GoodGoal", 1: "GoodGoalMulti"},
            speed={"preprocess": 1.0, "inference": 2.0, "postprocess": 3.0},
        )
        detection = detection_from_result(result, FrameStamp("ep", 2, 0.12), SOURCE)
        (item,) = detection.frame.objects
        self.assertEqual([e.label for e in item.semantic_distribution.entries], ["GoodGoal", "GoodGoalMulti"])
        self.assertAlmostEqual(item.semantic_distribution.unknown_probability, 0.1)
        self.assertEqual(item.confidence, 0.7)
        self.assertEqual(int(detection.masks.sum()), 36)
        self.assertEqual(detection.timing.total_ms, 6.0)
        self.assertEqual(DetectionFrame.from_json(detection.frame.to_json()), detection.frame)


class BoundaryTests(unittest.TestCase):
    def test_runtime_package_never_loads_privileged_capture(self) -> None:
        code = ("import sys, perception.inference, perception.inference.__main__, perception.inference.yolo; "
                "print('perception.privileged_capture' in sys.modules)")
        output = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertEqual(output.stdout.strip(), "False")


class FakeEnv:
    """Minimal ML-Agents surface: camera obs + 7-float vector obs, terminal after N steps."""

    behavior_specs = {"AnimalAI?team=0": None}

    def __init__(self, steps_until_terminal: int) -> None:
        self.limit = steps_until_terminal
        self.step_count = 0
        self.resets = 0

    def reset(self) -> None:
        self.step_count = 0
        self.resets += 1

    def get_steps(self, behavior: str):
        obs = [np.full((1, 8, 8, 3), 0.5, np.float32),
               np.array([[1.0, 0.0, 0.0, 0.1 * self.step_count, 20.0, 0.0, 20.0]], np.float32)]
        present, empty = _Steps(obs, 1), _Steps(obs, 0)
        return (empty, present) if self.step_count >= self.limit else (present, empty)

    def set_actions(self, behavior: str, action) -> None:
        self.last_action = action

    def step(self) -> None:
        self.step_count += 1


class _Steps(SimpleNamespace):
    def __init__(self, obs, count: int) -> None:
        super().__init__(obs=obs, count=count)

    def __len__(self) -> int:
        return self.count


if __name__ == "__main__":
    unittest.main()
