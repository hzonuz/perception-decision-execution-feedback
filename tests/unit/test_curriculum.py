from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from dataclasses import replace

from perception.curriculum import (
    AGENT_START,
    UNSPAWNABLE_CLASSES,
    Item,
    build_curriculum,
    competition_blocks,
    infeasible,
    merged_document,
    mistagged_fields,
    select_competition_configs,
    validate,
)
from perception.ontology import load_ontology


ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "animal-ai-unity/Assets/Resources/competition_sample.yaml"
BASE = ROOT / "competition_configurations/competition_sample_base.yaml"
COMPETITION = ROOT / "competition_configurations"


class _ArenaLoader(yaml.SafeLoader):
    """Animal-AI's YAML dialect uses tags the C# reader understands; map them to plain mappings."""


for tag in ("!ArenaConfig", "!Arena", "!Item", "!Vector3", "!RGB"):
    _ArenaLoader.add_constructor(
        tag, lambda loader, node: loader.construct_mapping(node, deep=True)
    )


def _exact_classes() -> set[str]:
    ontology = load_ontology(ROOT / "configs/ontology.json")
    return {mapping.exact_class for mapping in ontology.simulator_mappings}


class CurriculumTests(unittest.TestCase):
    def test_every_spawnable_class_is_spawned_and_placement_is_legal(self) -> None:
        classes = _exact_classes()

        report = validate(build_curriculum(classes), classes)

        spawnable = classes - UNSPAWNABLE_CLASSES
        counts = report["instances_per_class"]
        self.assertEqual(set(counts), classes)
        self.assertTrue(all(counts[name] for name in spawnable))
        self.assertGreater(report["arenas"], 50)

    def test_unspawnable_classes_are_never_placed(self) -> None:
        """A class the project cannot render or register would silently produce an empty arena."""
        classes = _exact_classes()

        counts = validate(build_curriculum(classes), classes)["instances_per_class"]

        self.assertTrue(UNSPAWNABLE_CLASSES <= classes)
        for name in UNSPAWNABLE_CLASSES:
            self.assertEqual(counts[name], 0, name)

    def test_task_arenas_alone_cover_every_spawnable_class(self) -> None:
        """The 20 hand-designed arenas must spawn every usable object without the generated tiers."""
        classes = _exact_classes()

        tasks = [arena for arena in build_curriculum(classes) if arena.tier == "tasks"]
        covered = set().union(*(arena.classes for arena in tasks))

        self.assertEqual(len(tasks), 20)
        self.assertEqual(covered, classes - UNSPAWNABLE_CLASSES)

    def test_generation_is_deterministic(self) -> None:
        classes = _exact_classes()

        first = build_curriculum(classes, seed=7)
        second = build_curriculum(classes, seed=7)
        third = build_curriculum(classes, seed=8)

        self.assertEqual(first, second)
        self.assertNotEqual(first, third)

    def test_unmapped_class_is_rejected(self) -> None:
        classes = _exact_classes()
        arenas = build_curriculum(classes)

        with self.assertRaises(ValueError):
            validate(arenas, classes - {"Wall"})
        with self.assertRaises(ValueError):
            build_curriculum(classes - {"Ramp"})

    def test_infeasible_arena_is_rejected(self) -> None:
        """Two solid objects in one spot means the second never spawns, so refuse to emit it."""
        classes = _exact_classes()
        arenas = build_curriculum(classes)
        broken = replace(
            arenas[0],
            items=(
                Item(name="Agent", positions=(AGENT_START,), rotations=(0.0,)),
                Item(name="Wall", positions=((20.0, 0.0, 15.0),), sizes=((3.0, 3.0, 3.0),)),
                Item(name="GoodGoal", positions=((20.0, 0.0, 15.0),), sizes=((2.0, 2.0, 2.0),)),
            ),
        )

        self.assertEqual(infeasible(arenas[0]), [])
        self.assertTrue(infeasible(broken))
        with self.assertRaises(ValueError):
            validate([broken], classes)

    def test_reward_inside_a_hazard_is_rejected(self) -> None:
        """A goal standing in a DeathZone cannot be reached, so the arena teaches nothing."""
        arena = replace(
            build_curriculum(_exact_classes())[0],
            items=(
                Item(name="Agent", positions=(AGENT_START,), rotations=(0.0,)),
                Item(name="DeathZone", positions=((20.0, 0.0, 20.0),), sizes=((10.0, 0.0, 10.0),)),
                Item(name="GoodGoal", positions=((20.0, 0.0, 20.0),), sizes=((2.0, 2.0, 2.0),)),
            ),
        )

        self.assertTrue(any("unreachable" in reason for reason in infeasible(arena)))

    def test_merged_unity_config_parses_and_renumbers(self) -> None:
        document = _expected_sample()

        parsed = yaml.load(document, Loader=_ArenaLoader)
        indices = sorted(parsed["arenas"])

        self.assertEqual(indices, list(range(len(indices))))
        for arena in parsed["arenas"].values():
            names = [item["name"] for item in arena["items"]]
            # Official arenas may omit the agent (Animal-AI then spawns it at random); what is
            # never legal is spawning two of them.
            self.assertLessEqual(names.count("Agent"), 1)

    def test_no_empty_arena_survives_the_merge(self) -> None:
        """An arena with nothing but the agent costs a capture episode and yields no labels."""
        parsed = yaml.load(_expected_sample(), Loader=_ArenaLoader)

        empty = [
            index
            for index, arena in parsed["arenas"].items()
            if not [item for item in arena["items"] if item["name"] != "Agent"]
        ]

        self.assertEqual(empty, [])

    def test_no_goal_is_recoloured(self) -> None:
        """Recolouring a goal makes GoodGoal and BadGoal visually identical in the labels."""
        goals = {
            "GoodGoal", "GoodGoalBounce", "GoodGoalMulti", "GoodGoalMultiBounce", "BadGoal",
            "BadGoalBounce", "BadGoalMulti", "DecoyGoal", "DecoyGoalBounce", "DecoyGoalMulti",
            "GrowGoal", "ShrinkGoal", "RipenGoal", "RipenGoalEpEnding", "DecayGoal",
            "DecayGoalEpEnding",
        }
        parsed = yaml.load(_expected_sample(), Loader=_ArenaLoader)

        recoloured = [
            item["name"]
            for arena in parsed["arenas"].values()
            for item in arena["items"]
            if item["name"] in goals and item.get("colors")
        ]

        self.assertEqual(recoloured, [])

    def test_imported_competition_arenas_are_verbatim(self) -> None:
        blocks = competition_blocks(select_competition_configs(COMPETITION, 40))

        merged = {line.strip() for line in _expected_sample().splitlines() if line.strip()}

        self.assertGreaterEqual(len(blocks), 35)
        for _, body in blocks:
            for line in body:
                if line.strip():
                    self.assertIn(line.strip(), merged)

    def test_merged_config_has_no_tag_unity_would_reject(self) -> None:
        """`colors` must hold !RGB and `positions`/`sizes` !Vector3.

        YamlDotNet throws `InvalidCastException: Object must implement IConvertible` otherwise, and
        one bad arena aborts the whole file — no arena builds, no agent spawns, and Unity reports
        "Cameras not initialized properly".
        """
        body = _expected_sample().splitlines()

        self.assertEqual(mistagged_fields(body), [])

    def test_official_arena_holding_only_unspawnable_objects_is_skipped(self) -> None:
        """Such an arena would render blank, costing a capture episode for no annotations."""
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "only-tunnel.yml"
            config.write_text(
                "!ArenaConfig\narenas:\n  0: !Arena\n    pass_mark: 0\n    t: 250\n    items:\n"
                "    - !Item\n      name: Agent\n      positions:\n"
                "      - !Vector3 {x: 20, y: 0, z: 4}\n      rotations: [0]\n"
                "    - !Item\n      name: CylinderTunnel\n      positions:\n"
                "      - !Vector3 {x: 20, y: 0, z: 16}\n",
                encoding="utf-8",
            )

            self.assertEqual(competition_blocks([config]), [])

    def test_malformed_official_arenas_are_skipped(self) -> None:
        broken = COMPETITION / "3-4-1.yml"  # tags its wall colours !Vector3

        self.assertTrue(mistagged_fields(broken.read_text(encoding="utf-8").splitlines()))
        self.assertEqual(competition_blocks([broken]), [])

    def test_checked_in_unity_sample_matches_the_generator(self) -> None:
        """`competition_sample.yaml` is generated; regenerate it instead of hand-editing."""
        self.assertEqual(SAMPLE.read_text(encoding="utf-8"), _expected_sample())


def _expected_sample() -> str:
    classes = _exact_classes()
    return merged_document(
        BASE.read_text(encoding="utf-8"),
        build_curriculum(classes),
        competition_blocks(select_competition_configs(COMPETITION, 40)),
    )


if __name__ == "__main__":
    unittest.main()
