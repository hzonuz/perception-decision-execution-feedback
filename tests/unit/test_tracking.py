"""Deterministic synthetic scenarios for the tracker (200x160 frames, 0.06 s per frame)."""

from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from perception.contracts import (
    BoundingBox2D,
    ContractValidationError,
    FrameStamp,
    MemoryState,
    PrimitiveAction,
    SourceMetadata,
    SourceType,
    TrackFrame,
    TrackLifecycle,
)
from perception.inference.detector import Timing, build_detection
from perception.ontology import load_ontology
from perception.tracking import Tracker, TrackerConfig, action_camera_shift, collectable_classes, is_blackout

ROOT = Path(__file__).resolve().parents[2]
CONFIG = TrackerConfig.from_json_file(ROOT / "configs/tracker.json")
W, H, DT = 200, 160, 0.06
SOURCE = SourceMetadata(SourceType.DETECTOR, "synthetic", "1.0")


def rect_mask(box):
    mask = np.zeros((H, W), bool)
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    mask[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = True
    return mask


def detections(index, objects, episode="synthetic"):
    """objects: (box, {label: score}) with masks drawn as the box; off-frame parts are clipped."""
    candidates = [(box, scores, rect_mask(box)) for box, scores in objects]
    return build_detection(FrameStamp(episode, index, index * DT), W, H, candidates, SOURCE, Timing(0, 0, 0))


def run(scenes, config=CONFIG, observable=lambda i: True, terminable=frozenset({"GoodGoal"}), shift=lambda i: (0.0, 0.0)):
    tracker = Tracker(config, terminable)
    out = []
    for index, objects in enumerate(scenes):
        detection = detections(index, objects)
        frame = tracker.update(detection.frame, detection.masks, observable=observable(index), camera_shift=shift(index))
        assert TrackFrame.from_json(frame.to_json()) == frame  # every frame conforms to the contract
        out.append(frame)
    return out


def by_id(frame):
    return {t.record.track_id: t for t in frame.tracks}


def goal(x, y, size=20, score=0.9, label="GoodGoal"):
    return ((x, y, x + size, y + size), {label: score})


class TrackerScenarioTests(unittest.TestCase):
    def test_two_same_class_objects_crossing_keep_their_ids(self) -> None:
        # A (farther: 20 px, bottom 110) moves right; B (nearer: 28 px, bottom 116) moves left and
        # completely hides A while A's box lies inside B's.
        scenes = []
        for i in range(30):
            ax, bx = 40 + 4 * i, 152 - 4 * i
            hidden = 0 <= ax - bx <= 8
            scenes.append(([] if hidden else [goal(ax, 90)]) + [goal(bx, 88, size=28)])
        frames = run(scenes)
        first = frames[0].tracks
        a_id = min(first, key=lambda t: t.bounding_box.x_min).record.track_id
        b_id = max(first, key=lambda t: t.bounding_box.x_min).record.track_id
        last = by_id(frames[-1])
        self.assertEqual(set(last), {a_id, b_id}, "no new IDs after crossing")
        self.assertGreater(last[a_id].bounding_box.x_min, last[b_id].bounding_box.x_min)  # A ended on the right
        occluded = [by_id(f)[a_id] for f in frames if by_id(f)[a_id].memory.state is MemoryState.OCCLUDED]
        self.assertTrue(occluded, "A should be occluded while B covers it")
        self.assertTrue(all(t.occluder_track_id == b_id for t in occluded))
        self.assertTrue(any(by_id(f)[a_id].record.lifecycle is TrackLifecycle.REAPPEARED for f in frames))

    def test_temporary_detector_dropout_keeps_the_track(self) -> None:
        scenes = [[goal(90, 70)] if not 10 <= i < 14 else [] for i in range(20)]
        frames = run(scenes)
        (track_id,) = by_id(frames[0])
        for i in range(10, 14):
            state = by_id(frames[i])[track_id]
            self.assertIs(state.memory.state, MemoryState.LOST)
            self.assertIsNone(state.memory.visibility_probability, "unexplained miss: visibility unknown")
        self.assertIs(by_id(frames[14])[track_id].record.lifecycle, TrackLifecycle.REAPPEARED)
        self.assertEqual(set(by_id(frames[-1])), {track_id})
        existence = [by_id(f)[track_id].memory.existence_probability for f in frames]
        for i in range(11, 14):
            self.assertAlmostEqual(existence[i], existence[i - 1] * CONFIG.lost_decay)
        self.assertGreater(existence[-1], existence[13])

    def test_object_leaving_the_camera_and_coming_back(self) -> None:
        scenes = [[goal(140 + 10 * i, 70)] for i in range(8)]  # exits right edge by frame 5
        scenes += [[] for _ in range(10)]
        scenes += [[goal(185 - 5 * i, 70)] for i in range(5)]  # agent turns back: re-enters on the right
        frames = run(scenes)
        (track_id,) = by_id(frames[0])
        out = [i for i, f in enumerate(frames) if track_id in by_id(f)
               and by_id(f)[track_id].memory.state is MemoryState.OUT_OF_VIEW]
        self.assertGreaterEqual(len(out), 10)
        for i in out[1:]:
            self.assertAlmostEqual(by_id(frames[i])[track_id].memory.existence_probability,
                                   by_id(frames[i - 1])[track_id].memory.existence_probability * CONFIG.out_of_view_decay)
            self.assertEqual(by_id(frames[i])[track_id].memory.visibility_probability, 0.0)
        back = by_id(frames[18])[track_id]
        self.assertIs(back.record.lifecycle, TrackLifecycle.REAPPEARED)
        self.assertEqual(set(by_id(frames[-1])), {track_id})

    def test_object_passing_behind_a_wall(self) -> None:
        wall = (90, 40, 130, 140)
        scenes = []
        for i in range(25):
            x0, x1 = 20 + 6 * i, 40 + 6 * i  # goal y 60-80: bottom edge above the wall's, so farther
            visible = [(x0, x1)] if x1 <= wall[0] or x0 >= wall[2] else (
                [(x0, wall[0])] if x0 < wall[0] else [(wall[2], x1)] if x1 > wall[2] else [])
            objects = [(wall, {"Wall": 0.95})]
            objects += [((a, 60, b, 80), {"GoodGoal": 0.9}) for a, b in visible if b - a >= 2]
            scenes.append(objects)
        frames = run(scenes)
        ids = {t.semantic_distribution.entries[0].label: t.record.track_id for t in frames[0].tracks}
        goal_id, wall_id = ids["GoodGoal"], ids["Wall"]
        states = [by_id(f)[goal_id].memory.state for f in frames]
        self.assertIn(MemoryState.PARTIALLY_VISIBLE, states)
        hidden = [by_id(f)[goal_id] for f in frames if by_id(f)[goal_id].memory.state is MemoryState.OCCLUDED]
        self.assertTrue(hidden)
        self.assertTrue(all(t.occluder_track_id == wall_id for t in hidden), "wall retained as occluder")
        self.assertEqual(set(by_id(frames[-1])), {goal_id, wall_id}, "same ID on the far side")
        self.assertIs(by_id(frames[-1])[goal_id].memory.state, MemoryState.VISIBLE)

    def test_blackout_is_unobservable_not_absent(self) -> None:
        scenes = [[goal(40, 70), goal(120, 70, label="BadGoal")] if not 5 <= i < 25 else [] for i in range(30)]
        frames = run(scenes, observable=lambda i: not 5 <= i < 25)
        before = by_id(frames[4])
        for frame in frames[5:25]:
            self.assertFalse(frame.observable)
            self.assertEqual(set(by_id(frame)), set(before))
            for track_id, state in by_id(frame).items():
                self.assertIs(state.memory.state, MemoryState.UNOBSERVABLE)
                self.assertIsNone(state.memory.visibility_probability)
                self.assertEqual(state.memory.existence_probability, before[track_id].memory.existence_probability)
                self.assertEqual(state.record.missed_frames, 0)
        after = by_id(frames[25])
        self.assertEqual(set(after), set(before))
        self.assertTrue(all(s.record.lifecycle is TrackLifecycle.REAPPEARED for s in after.values()))

        # the same 20 empty frames while observable are real misses: both tracks decay and are archived
        observable_gap = run(scenes)
        self.assertTrue(set(before).isdisjoint(by_id(observable_gap[-1])))

    def test_object_termination_when_collected(self) -> None:
        scenes = [[((60 - 4 * i, 50 - 3 * i, 110 + 4 * i, 100 + 3 * i), {"GoodGoal": 0.9})] for i in range(6)]
        scenes += [[] for _ in range(4)]
        frames = run(scenes)
        (track_id,) = by_id(frames[0])
        terminal = [f for f in frames if track_id in by_id(f)
                    and by_id(f)[track_id].memory.state is MemoryState.TERMINATED]
        self.assertEqual(len(terminal), 1, "reported exactly once")
        state = by_id(terminal[0])[track_id]
        self.assertIs(state.record.lifecycle, TrackLifecycle.TERMINATED)
        self.assertEqual(state.memory.existence_probability, 0.0)
        self.assertEqual(state.termination_reason, "vanished_in_view_at_close_range")
        self.assertEqual(terminal[0].stamp.frame_index, 6 + CONFIG.termination_misses - 1)
        self.assertEqual(frames[-1].tracks, ())

        # a non-collectable object that vanishes at close range is lost (and decays), never terminated
        walls = run([[((x0, y0, x1, y1), {"Wall": 0.9})] for (x0, y0, x1, y1), _ in (s[0] for s in scenes[:6])]
                    + [[] for _ in range(4)])
        self.assertNotIn(MemoryState.TERMINATED, {t.memory.state for f in walls for t in f.tracks})
        self.assertIs(walls[7].tracks[0].memory.state, MemoryState.LOST)

    def test_explicit_termination_evidence(self) -> None:
        tracker = Tracker(CONFIG)
        first = tracker.update(detections(0, [goal(50, 50)]).frame)
        track_id = first.tracks[0].record.track_id
        tracker.terminate(track_id, "reward_collected")
        (state,) = tracker.update(detections(1, [goal(50, 50)]).frame).tracks
        self.assertEqual((state.memory.state, state.termination_reason), (MemoryState.TERMINATED, "reward_collected"))

    def test_false_positive_decays_and_is_archived(self) -> None:
        scenes = [[goal(90, 70, score=0.4)]] + [[] for _ in range(6)]
        frames = run(scenes)
        (track_id,) = by_id(frames[0])
        self.assertAlmostEqual(by_id(frames[0])[track_id].memory.existence_probability, CONFIG.initial_existence * 0.4)
        self.assertIn(track_id, by_id(frames[1]), "one missed frame never deletes a track")
        alive = [f for f in frames if track_id in by_id(f)]
        existence = [by_id(f)[track_id].memory.existence_probability for f in alive]
        for previous, current in zip(existence, existence[1:]):
            self.assertAlmostEqual(current, previous * CONFIG.lost_decay)
        self.assertIs(by_id(alive[-1])[track_id].record.lifecycle, TrackLifecycle.ARCHIVED)
        self.assertLess(existence[-1], CONFIG.archive_existence)
        self.assertEqual(frames[-1].tracks, ())

        slower = run(scenes, replace(CONFIG, lost_decay=0.9))
        self.assertGreater(sum(track_id in by_id(f) for f in slower), len(alive), "decay rate comes from config")


class TrackerBehaviourTests(unittest.TestCase):
    def test_camera_rotation_is_compensated_by_the_action_shift(self) -> None:
        # two static same-class goals; the agent stands still, then abruptly turns left twice, so
        # the scene jumps right 30 px/frame — constant-velocity prediction alone cannot see it coming
        offsets = [0, 0, 0, 30, 60]
        scenes = [[goal(10 + dx, 70), goal(40 + dx, 70)] for dx in offsets]
        pan = lambda i: (30.0, 0.0) if i >= 3 else (0.0, 0.0)
        def left_goal_end_x(frames):  # where the track born as the left goal ended up
            left = min(frames[0].tracks, key=lambda t: t.bounding_box.x_min).record.track_id
            return by_id(frames[-1])[left].bounding_box.x_min

        compensated = run(scenes, shift=pan)
        self.assertEqual(left_goal_end_x(compensated), 70)
        for state in compensated[-1].tracks:
            self.assertAlmostEqual(state.image_velocity.x, 0.0, places=6, msg="camera motion is not object motion")
        self.assertNotEqual(left_goal_end_x(run(scenes)), 70, "without the shift the IDs swap")
        step = CONFIG.rotation_shift_fraction * W
        self.assertEqual(action_camera_shift(PrimitiveAction(0, 2), W, CONFIG), (step, 0.0))
        self.assertEqual(action_camera_shift(PrimitiveAction(1, 1), W, CONFIG), (-step, 0.0))
        self.assertEqual(action_camera_shift(None, W, CONFIG), (0.0, 0.0))

    def test_duplicate_detections_make_one_track(self) -> None:
        frames = run([[goal(50, 50), ((50.5, 50, 70.5, 70), {"GoodGoal": 0.8})]])
        self.assertEqual(len(frames[0].tracks), 1)

    def test_cross_class_detection_does_not_steal_a_track(self) -> None:
        frames = run([[goal(50, 50)], [((52, 52, 90, 90), {"Wall": 0.9})]])
        goal_id = frames[0].tracks[0].record.track_id
        goal_state, wall_state = by_id(frames[1])[goal_id], next(t for t in frames[1].tracks if t.record.track_id != goal_id)
        self.assertIs(wall_state.record.lifecycle, TrackLifecycle.CREATED)
        self.assertIs(goal_state.record.lifecycle, TrackLifecycle.MISSED)
        self.assertEqual(goal_state.semantic_distribution.entries[0].label, "GoodGoal")
        # the wall's bottom edge is lower, so it is nearer and hides the goal
        self.assertEqual((goal_state.memory.state, goal_state.occluder_track_id), (MemoryState.OCCLUDED, wall_state.record.track_id))

    def test_collectable_classes_come_from_the_ontology(self) -> None:
        classes = collectable_classes(load_ontology(ROOT / "configs/ontology.json"))
        self.assertIn("GoodGoal", classes)
        self.assertNotIn("Wall", classes)

    def test_contract_rejects_inconsistent_track_state(self) -> None:
        state = run([[goal(50, 50)]])[0].tracks[0]
        with self.assertRaises(ContractValidationError):
            replace(state, hits=5)  # hits + misses > age
        with self.assertRaises(ContractValidationError):
            replace(state, termination_reason="collected")  # reason without TERMINATED state

    def test_class_belief_updates_across_observations(self) -> None:
        scenes = [[((80, 60, 100, 80), {"GoodGoal": 0.6, "GoodGoalMulti": 0.3})]]
        scenes += [[((80, 60, 100, 80), {"GoodGoalMulti": 0.8})] for _ in range(6)]
        frames = run(scenes)
        first = frames[0].tracks[0].semantic_distribution.entries[0].label
        last = frames[-1].tracks[0].semantic_distribution
        self.assertEqual(first, "GoodGoal")
        self.assertEqual(last.entries[0].label, "GoodGoalMulti")
        self.assertEqual(len(frames[-1].tracks), 1, "a class change is not a new track")

    def test_velocity_and_motion_state(self) -> None:
        frames = run([[goal(20 + 5 * i, 70)] for i in range(5)])
        velocity = frames[-1].tracks[0].image_velocity
        self.assertAlmostEqual(velocity.x, 5 / DT, places=6)
        self.assertAlmostEqual(velocity.y, 0.0, places=6)
        self.assertEqual(frames[-1].tracks[0].record.motion_state.value, "moving")
        self.assertIsNone(frames[0].tracks[0].image_velocity)

    def test_episode_reset_and_stamp_order(self) -> None:
        tracker = Tracker(CONFIG)
        tracker.update(detections(3, [goal(10, 10)]).frame)
        with self.assertRaises(ValueError):
            tracker.update(detections(3, [goal(10, 10)]).frame)
        fresh = tracker.update(detections(0, [goal(10, 10)], episode="next").frame)
        self.assertEqual(fresh.tracks[0].record.track_id, "next:track-0000")
        self.assertIs(fresh.tracks[0].record.lifecycle, TrackLifecycle.CREATED)

    def test_blackout_detection_and_config_validation(self) -> None:
        self.assertTrue(is_blackout(np.zeros((8, 8, 3), np.uint8), CONFIG))
        self.assertFalse(is_blackout(np.full((8, 8, 3), 120, np.uint8), CONFIG))
        with self.assertRaises(ValueError):
            replace(CONFIG, min_misses_to_archive=1).validate()
        with self.assertRaises(ValueError):
            replace(CONFIG, lost_decay=1.5).validate()


if __name__ == "__main__":
    unittest.main()


class TrackingFixtureTests(unittest.TestCase):
    """The JSON scenarios shown in notebooks/06_tracking.ipynb: expected state per track and frame
    (V visible, p partial, O occluded, > out of view, ? lost, # unobservable, X terminated, . absent)."""

    EXPECTED = {
        "behind_wall": ["VVVVVVVVVVVVVVVVVVVVVVVVVVVVVV", "VVVVVVVVppOOOOOOOpVVVVVVVVVVVV"],
        "blackout": ["VVVVVV##########VVVVVV", "VVVVVV##########VVVVVV"],
        "crossing": ["VVVVVVVVVVVVVVVVVVVVVVVVVVVVVV", "VVVVVVVVVVVVppOOpVVVVVVVVVVVVV"],
        "detector_dropout": ["VVVVVVVV????VVVVVVVV"],
        "false_positive": ["VVVVVVVVVVVV", "...V?????..."],
        "leaving_view": ["VVVVVVpp>>>>>>>>>>ppppVVVV"],
        "termination": ["VVVVVV?X.."],
    }
    CODES = {"visible": "V", "partially_visible": "p", "occluded": "O", "out_of_view": ">", "lost": "?",
             "unobservable": "#", "terminated": "X"}

    def test_scene_image_hides_goal_behind_wall(self) -> None:
        from perception.drawing import colour_for
        from tests.scenes import SIZE, run_tracking_scene, scene_image

        scene, _ = run_tracking_scene("behind_wall")
        goal_colour, wall_colour = colour_for("GoodGoal"), colour_for("Wall")
        for index in (0, 8, 11, 14, 17, 18, 29):
            image = scene_image(scene, index)
            self.assertEqual(image.shape, (SIZE, SIZE, 3))
            self.assertEqual(image.dtype, np.uint8)
            self.assertTrue((image[80:150, 100:160] == wall_colour).all())
            self.assertEqual(bool(np.all(image == goal_colour, axis=2).any()), index not in (11, 14))
        scene, _ = run_tracking_scene("detector_dropout")
        self.assertTrue(np.array_equal(scene_image(scene, 0), scene_image(scene, 8)))
        scene, _ = run_tracking_scene("blackout")
        self.assertTrue((scene_image(scene, 6) == 0).all())

    def test_state_sequences(self) -> None:
        from tests.scenes import TRACKING_SCENES, run_tracking_scene, scene_names

        self.assertEqual(sorted(self.EXPECTED), scene_names(TRACKING_SCENES))
        for name, expected in self.EXPECTED.items():
            _, frames = run_tracking_scene(name)
            ids = sorted({t.record.track_id for f in frames for t in f.tracks})
            rows = ["".join(self.CODES[by_id(f)[i].memory.state.value] if i in by_id(f) else "." for f in frames)
                    for i in ids]
            with self.subTest(scenario=name):
                self.assertEqual(rows, expected)


class IdentityEvaluationTests(unittest.TestCase):
    def ground_truth(self, scene):
        from tests.helpers import make_capture_frame
        from tests.scenes import box_at

        template = make_capture_frame()
        annotation = template.objects[0]
        size = dict(rgb=replace(template.rgb, width=256, height=256), camera=replace(
            template.camera, intrinsics=replace(template.camera.intrinsics, width=256, height=256, cx=128.0, cy=128.0)))
        truth = {}
        for index in range(scene["frames"]):
            objects = []
            for item in scene["objects"]:
                box = box_at(item, index)
                hidden = item["name"] == "A" and abs(box_at(scene["objects"][1], index)[0] - box[0]) <= 10
                objects.append(replace(annotation, instance_id=f"gt-{item['name']}", instance_mask=None,
                                       bounding_box=BoundingBox2D(*box, 256, 256), visibility_fraction=0.0 if hidden else 1.0))
            truth[index] = replace(template, objects=tuple(objects), lifecycle_events=(), **size)
        return truth

    def test_identity_metrics_count_switches(self) -> None:
        from perception.tracking.evaluation import identity_metrics
        from tests.scenes import run_tracking_scene

        scene, frames = run_tracking_scene("crossing")
        truth = self.ground_truth(scene)
        clean = identity_metrics(frames, truth)
        self.assertEqual((clean["instances"], clean["id_switches"], clean["ids_per_instance_max"]), (2, 0, 1))
        # swap the two IDs halfway: every GT instance now sees two IDs and one switch
        a, b = sorted({t.record.track_id for f in frames for t in f.tracks})
        swap = {a: b, b: a}
        swapped = [f if i < 20 else replace(f, tracks=tuple(
            replace(t, record=replace(t.record, track_id=swap[t.record.track_id]),
                    memory=replace(t.memory, track_id=swap[t.record.track_id])) for t in f.tracks))
            for i, f in enumerate(frames)]
        broken = identity_metrics(swapped, truth)
        self.assertEqual((broken["id_switches"], broken["ids_per_instance_max"]), (2, 2))
        self.assertEqual(broken["switch_causes"], {"previous track took another detection": 2})
