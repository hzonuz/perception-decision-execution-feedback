"""Animal-AI demonstration: trained YOLO + tracker over a held-out (test-split) captured episode.

Uses only the episode's RGB frames — the runtime path. Skipped when weights or captures are absent.
"""

from __future__ import annotations

import itertools
import tempfile
import unittest
from pathlib import Path

from perception.contracts import MemoryState, TrackFrame, TrackLifecycle

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "runs/detector/notebook_yolo26n_seg/weights/best.pt"
EPISODE = ROOT / "captures/curriculum_run/rgb/curriculum-0108"


@unittest.skipUnless(WEIGHTS.is_file() and EPISODE.is_dir(), "trained weights or demo capture not available")
class AnimalAITrackingDemo(unittest.TestCase):
    def test_detect_and_track_a_captured_episode(self) -> None:
        from perception.inference import prerecorded_frames, run_inference
        from perception.inference.yolo import YoloDetector
        from perception.ontology import load_ontology
        from perception.tracking import Tracker, TrackerConfig, collectable_classes

        ontology = load_ontology(ROOT / "configs/ontology.json")
        tracker = Tracker(TrackerConfig.from_json_file(ROOT / "configs/tracker.json"), collectable_classes(ontology))
        frames = itertools.islice(prerecorded_frames(EPISODE), 120)
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_inference(YoloDetector(WEIGHTS, deterministic=True), frames, tmp,
                                    ontology=ontology, tracker=tracker)
            tracks = [TrackFrame.from_json(line) for line in (Path(tmp) / "tracks.jsonl").read_text().splitlines()]

        self.assertEqual(len(tracks), 120)
        self.assertGreater(summary["tracking"]["lifecycle_events"]["reappeared"], 0)
        self.assertLess(summary["latency_ms"]["tracking"]["p50"], 20.0)
        lifetimes: dict[str, set[int]] = {}
        states: set[MemoryState] = set()
        for frame in tracks:
            for state in frame.tracks:
                lifetimes.setdefault(state.record.track_id, set()).add(frame.stamp.frame_index)
                states.add(state.memory.state)
        self.assertGreaterEqual(max(len(v) for v in lifetimes.values()), 60, "some identity persists for half the clip")
        self.assertTrue({MemoryState.VISIBLE, MemoryState.OUT_OF_VIEW, MemoryState.LOST} <= states)
        # a track that went unseen and came back under the same ID
        self.assertTrue(any(
            s.record.lifecycle is TrackLifecycle.REAPPEARED and s.hits >= 10 for f in tracks for s in f.tracks
        ))


if __name__ == "__main__":
    unittest.main()
