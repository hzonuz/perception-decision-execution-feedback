from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from perception.contracts import (
    GraphHistory,
    MemoryState,
    Relation,
    RelationPredicate as P,
    SourceMetadata,
    SourceType,
    SSGSnapshot,
)
from perception.ontology import load_ontology
from perception.scene_graph import SceneGraphBuilder, SceneGraphConfig, to_asp
from perception.scene_graph.relations import _p_more
from perception.tracking import Tracker, TrackerConfig
from tests.scenes import ROOT, SIZE, box_at, load_scene, run_scene, scene_frames, scene_names

CONFIG = SceneGraphConfig.from_json_file(ROOT / "configs/scene_graph.json")
ONTOLOGY = load_ontology(ROOT / "configs/ontology.json")
FIXTURES = ("basic", "lava", "numerosity", "object_permanence", "ramp", "spatial_elimination")


def named(snapshots, scene):
    """fixture object name -> node id, matched on the first frame each track appears."""
    out = {}
    for snapshot in snapshots:
        for node in snapshot.objects:
            label = node.semantic_distribution.entries[0].label
            box = node.bounding_box
            for item in scene["objects"]:
                if item["label"] == label and item["name"] not in out and node.node_id not in out.values():
                    x = (box.x_min + box.x_max) / 2
                    target = box_at(item, snapshot.stamp.frame_index)
                    if target and target[0] - 4 <= x <= target[2] + 4:
                        out[item["name"]] = node.node_id
    return out


def edges(snapshot, predicate, subject=None, obj=None):
    return [r for r in snapshot.relations if r.predicate is predicate
            and (subject is None or r.subject_node_id == subject) and (obj is None or r.object_node_id == obj)]


def node(snapshot, node_id):
    return next((n for n in snapshot.objects if n.node_id == node_id), None)


class FixtureGraphTests(unittest.TestCase):
    runs = {}

    @classmethod
    def setUpClass(cls) -> None:
        cls.runs = {name: run_scene(name) for name in FIXTURES}

    def scene(self, name):
        scene, tracks, snapshots, builder = self.runs[name]
        return scene, snapshots, named(snapshots, scene), builder

    def test_every_fixture_exists_and_its_graphs_round_trip(self) -> None:
        self.assertEqual(set(FIXTURES), set(scene_names()))
        for name in FIXTURES:
            _, snapshots, ids, builder = self.scene(name)
            with self.subTest(fixture=name):
                self.assertEqual(len(ids), len(load_scene(name)["objects"]), "every fixture object got a node")
                for snapshot in snapshots:
                    self.assertEqual(SSGSnapshot.from_json(snapshot.to_json()), snapshot)
                    self.assertEqual(snapshot.graph_id, f"{snapshot.episode_id}:frame-{snapshot.stamp.frame_index:06d}")
                    self.assertTrue(all(r.probability > 0 for r in snapshot.relations), "nothing asserted false")
                    for r in snapshot.relations:  # memory facts only ever describe an unseen subject
                        if r.predicate in (P.LAST_SEEN_BEHIND, P.LIKELY_OCCLUDER, P.PREVIOUSLY_VISIBLE, P.BELIEVED_PRESENT):
                            self.assertNotIn(node(snapshot, r.subject_node_id).memory_state,
                                             (MemoryState.VISIBLE, MemoryState.PARTIALLY_VISIBLE))
                history = builder.history()
                self.assertEqual(GraphHistory.from_json(history.to_json()), history)
                macro = builder.history(macro_only=True).snapshots
                self.assertEqual([s.stamp.frame_index for s in macro], list(range(0, len(snapshots), CONFIG.macro_every_frames)))
                # stable IDs: a node id always refers to the same track
                pairs = {(n.node_id, n.track_id) for s in snapshots for n in s.objects}
                self.assertEqual(len(pairs), len({p[0] for p in pairs}))

    def test_basic_approach(self) -> None:
        _, snapshots, ids, _ = self.scene("basic")
        last = snapshots[-1]
        self.assertTrue(edges(last, P.MOVING_TOWARD, ids["goal"], "agent"))
        self.assertGreater(edges(last, P.NEAR, ids["goal"], "agent")[0].probability, 0.9)
        self.assertTrue(edges(last, P.LEFT_OF, ids["wall"], ids["goal"]))
        self.assertTrue(edges(last, P.RIGHT_OF, ids["goal"], ids["wall"]))
        self.assertFalse(edges(last, P.DANGER_BETWEEN) + edges(last, P.PATH_BLOCKED_BY))
        self.assertEqual(node(last, ids["goal"]).memory_state, MemoryState.VISIBLE)

    def test_lava_between_agent_and_goal(self) -> None:
        _, snapshots, ids, _ = self.scene("lava")
        (danger,) = edges(snapshots[-1], P.DANGER_BETWEEN, ids["goal"])
        self.assertEqual(danger.object_node_id, ids["lava"], "the off-path HotZone is not between")
        self.assertGreater(danger.probability, 0.6)
        self.assertLessEqual(danger.uncertainty.lower_bound, danger.probability)

    def test_ramp_platform(self) -> None:
        _, snapshots, ids, _ = self.scene("ramp")
        last = snapshots[-1]
        self.assertTrue(edges(last, P.ON, ids["goal"], ids["platform"]))
        self.assertTrue(edges(last, P.SUPPORTS, ids["platform"], ids["goal"]))
        adjacent = {frozenset((r.subject_node_id, r.object_node_id)) for r in edges(last, P.ADJACENT)}
        self.assertIn(frozenset((ids["platform"], ids["ramp"])), adjacent)
        self.assertFalse(edges(last, P.ON, ids["ramp"]), "the ramp is beside the platform, not on it")
        self.assertFalse(edges(last, P.ON, ids["platform"]), "a wall does not rest on things")

    def test_numerosity_counts_memory_only_goals(self) -> None:
        _, snapshots, ids, _ = self.scene("numerosity")
        last = snapshots[-1]
        hidden = node(last, ids["l2"])
        self.assertIs(hidden.memory_state, MemoryState.OCCLUDED)
        self.assertEqual(next(a.value for a in hidden.attributes if a.name == "observation"), "memory_only")
        (more,) = edges(last, P.MORE_GOALS_LEFT, "agent")
        self.assertGreater(more.probability, 0.9)
        self.assertAlmostEqual(more.symbolic_value, 3.0, delta=0.2)
        self.assertLess(sum(r.probability for r in edges(last, P.MORE_GOALS_RIGHT)), 0.05)

    def test_spatial_elimination_picks_the_right_screen(self) -> None:
        _, snapshots, ids, _ = self.scene("spatial_elimination")
        behind = [r for s in snapshots for r in edges(s, P.LAST_SEEN_BEHIND, ids["goal"])]
        self.assertTrue(behind)
        self.assertEqual({r.object_node_id for r in behind}, {ids["right_screen"]})
        self.assertEqual({r.object_node_id for s in snapshots for r in edges(s, P.LIKELY_OCCLUDER, ids["goal"])},
                         {ids["right_screen"]})

    def test_object_permanence_keeps_the_hidden_goal(self) -> None:
        _, snapshots, ids, _ = self.scene("object_permanence")
        goal, wall = ids["goal"], ids["wall"]
        self.assertTrue(all(node(s, goal) is not None for s in snapshots), "goal node survives the disappearance")
        hidden = [s for s in snapshots if node(s, goal).memory_state is MemoryState.OCCLUDED]
        self.assertGreaterEqual(len(hidden), 5)
        for snapshot in hidden:
            n = node(snapshot, goal)
            self.assertIs(n.source.source, SourceType.TEMPORAL_MEMORY)
            self.assertIsNone(n.detector_confidence)
            (behind,) = edges(snapshot, P.LAST_SEEN_BEHIND, goal)
            self.assertEqual(behind.object_node_id, wall)
            self.assertTrue(edges(snapshot, P.BELIEVED_PRESENT, goal))
            self.assertTrue(edges(snapshot, P.PREVIOUSLY_VISIBLE, goal))
        ages = [edges(s, P.LAST_SEEN_BEHIND, goal)[0].age_frames for s in hidden]
        self.assertEqual(ages, list(range(len(hidden))), "age counts frames since it went behind")
        back = next(s for s in snapshots[snapshots.index(hidden[-1]):] if node(s, goal).memory_state is MemoryState.VISIBLE)
        (reappeared,) = [r for s in snapshots for r in edges(s, P.REAPPEARED_AFTER, goal)]
        self.assertGreaterEqual(reappeared.symbolic_value, len(hidden))
        self.assertFalse(edges(back, P.LAST_SEEN_BEHIND, goal), "no longer behind anything once seen again")


class GraphSemanticsTests(unittest.TestCase):
    def build(self, name, config=CONFIG, unobservable=(), scene=None):
        tracker = Tracker(TrackerConfig.from_json_file(ROOT / "configs/tracker.json"))
        builder = SceneGraphBuilder(config, ONTOLOGY)
        scene = {**(scene or load_scene(name)), "name": name, "unobservable": list(unobservable)}
        return [builder.update(tracker.update(d.frame, d.masks, observable=o), width=SIZE, height=SIZE)
                for d, o in scene_frames(scene)]

    def test_stale_relations_decay_and_expire_per_config(self) -> None:
        snapshots = self.build("object_permanence")
        stale = [(s.stamp.frame_index, r) for s in snapshots for r in s.relations
                 if any(e.startswith("stale_frames=") for e in r.evidence) and r.predicate is not P.LAST_SEEN_BEHIND]
        self.assertTrue(stale)
        for index, relation in stale:
            frames = int(next(e for e in relation.evidence if e.startswith("stale_frames=")).split("=")[1])
            self.assertLessEqual(frames, CONFIG.ttl(relation.predicate))
            self.assertLess(relation.stamp.frame_index, index, "a carried edge keeps its evidence stamp")
        none = self.build("object_permanence", replace(CONFIG, relation_ttl_frames={"default": 0, "last_seen_behind": 300}))
        self.assertFalse([r for s in none for r in s.relations if r.predicate is not P.LAST_SEEN_BEHIND
                          and any(e.startswith("stale_frames=") for e in r.evidence)])

    def test_memory_facts_end_when_the_subject_is_seen_again(self) -> None:
        # the wall stops being detected, then the goal comes into view
        scene = load_scene("object_permanence")
        scene["objects"][0]["until"] = 15  # gone while the goal is still hidden behind it
        snapshots = self.build("occluder_lost", scene=scene)
        goal = next(n.node_id for n in snapshots[0].objects if n.semantic_distribution.entries[0].label == "GoodGoal")
        self.assertTrue([s for s in snapshots if edges(s, P.LAST_SEEN_BEHIND, goal)])
        for snapshot in snapshots:
            if node(snapshot, goal).memory_state in (MemoryState.VISIBLE, MemoryState.PARTIALLY_VISIBLE):
                self.assertFalse(edges(snapshot, P.LAST_SEEN_BEHIND, goal) + edges(snapshot, P.LIKELY_OCCLUDER, goal))

    def test_blackout_keeps_nodes_as_memory(self) -> None:
        snapshots = self.build("basic", unobservable=[(4, 7)])
        for snapshot in snapshots[4:8]:
            self.assertEqual(len(snapshot.objects), 2)
            self.assertEqual({n.memory_state for n in snapshot.objects}, {MemoryState.UNOBSERVABLE})
            fresh = [r for r in snapshot.relations if not any(e.startswith("stale_frames=") for e in r.evidence)
                     and r.predicate not in (P.PREVIOUSLY_VISIBLE, P.BELIEVED_PRESENT, P.MORE_GOALS_LEFT, P.MORE_GOALS_RIGHT)]
            self.assertEqual(fresh, [], "nothing is observed in a blackout")
        self.assertEqual({n.node_id for n in snapshots[3].objects}, {n.node_id for n in snapshots[8].objects})

    def test_unknown_distance_is_omitted_not_false(self) -> None:
        # a wall whose box reaches the bottom edge (floor contact not visible) and a goal at the horizon
        scene = {"frames": 3, "objects": [
            {"name": "wall", "label": "Wall", "score": 0.95, "keyframes": {"0": [0, 150, 60, 256]}},
            {"name": "goal", "label": "GoodGoal", "score": 0.9, "keyframes": {"0": [120, 110, 136, 129]}},
            {"name": "near_goal", "label": "GoodGoal", "score": 0.9, "keyframes": {"0": [180, 130, 200, 150]}}]}
        last = self.build("unknown_distance", scene=scene)[-1]
        by_label = {n.bounding_box.x_min: n for n in last.objects}
        wall, horizon_goal, near_goal = by_label[0], by_label[120], by_label[180]
        for unknown in (wall, horizon_goal):
            self.assertIsNone(unknown.relative_distance)
            self.assertFalse([r for r in last.relations if unknown.node_id in (r.subject_node_id, r.object_node_id)
                              and r.predicate in (P.NEAR, P.FAR, P.ADJACENT)], "unknown distance: omitted")
        self.assertIsNotNone(near_goal.relative_distance)
        self.assertTrue(edges(last, P.NEAR, near_goal.node_id, "agent"))
        self.assertTrue(edges(last, P.LEFT_OF, wall.node_id, near_goal.node_id), "image relations still hold")

    def test_unknown_class_gives_unknown_functional_belief(self) -> None:
        builder = SceneGraphBuilder(CONFIG, ONTOLOGY)
        state = run_scene("basic")[1][0].tracks[0]
        from perception.contracts import CategoricalDistribution, ProbabilityEntry
        vague = replace(state, semantic_distribution=CategoricalDistribution((ProbabilityEntry("Wall", 0.3),), 0.7))
        beliefs = builder._beliefs(vague)
        point, lower, upper = beliefs["barrier"]
        self.assertIsNone(point, "30% known mass is below min_known_class_mass")
        self.assertEqual((lower, upper), (0.3, 1.0))
        self.assertNotIn("collectable", beliefs, "no evidence for it: omitted, not 0")

    def test_asp_export(self) -> None:
        snapshot = next(s for s in run_scene("object_permanence")[2] if any(n.memory_state is MemoryState.OCCLUDED for n in s.objects))
        facts = to_asp(snapshot)
        self.assertIn("memory_only(", facts)
        self.assertIn("visible(", facts)
        self.assertIn("rel(last_seen_behind,", facts)
        self.assertRegex(facts, r"val\(previously_visible,obj_\w+,\d+,\d+\)\.")
        body = [line for line in facts.splitlines() if line.startswith(("rel(", "val("))]
        self.assertEqual(len(body), len(snapshot.relations))
        self.assertEqual(len([line for line in to_asp(snapshot, min_probability=0.99).splitlines() if line.startswith("rel(")]),
                         len([r for r in snapshot.relations if r.object_node_id and r.probability >= 0.99]))

    def test_learned_model_plugs_in(self) -> None:
        class Fake:
            name = "fake"

            def __init__(self, source):
                self.source = source

            def predict(self, context):
                n = context.nodes[0]
                return [Relation(n.node_id, P.SAFE_TO_INTERACT, None, True, 0.7, self.source, context.stamp, 0)]

        tracks = run_scene("basic")[1][0]
        learned = SourceMetadata(SourceType.LEARNED_RELATION_MODEL, "fake", "0")
        snapshot = SceneGraphBuilder(CONFIG, ONTOLOGY, (Fake(learned),)).update(tracks, width=SIZE, height=SIZE)
        self.assertEqual([r.probability for r in edges(snapshot, P.SAFE_TO_INTERACT)], [0.7])
        with self.assertRaises(ValueError):
            SceneGraphBuilder(CONFIG, ONTOLOGY, (Fake(SourceMetadata(SourceType.TASK_RULE, "x", "0")),)).update(
                tracks, width=SIZE, height=SIZE)

    def test_goal_count_probability_and_config_validation(self) -> None:
        self.assertEqual(_p_more([1.0, 1.0], [1.0]), (1.0, 0.0))
        np.testing.assert_allclose(_p_more([0.5], [0.5]), (0.25, 0.25))
        with self.assertRaises(ValueError):
            replace(CONFIG, relation_ttl_frames={"default": 5, "not_a_predicate": 1}).validate()
        with self.assertRaises(ValueError):
            replace(CONFIG, near_distance=20.0).validate()


if __name__ == "__main__":
    unittest.main()
