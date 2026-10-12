"""
Generate Animal-AI arena configurations that cover the whole ontology.

The captured dataset is class-imbalanced: goals and walls dominate, while ramps, tunnels, decoys,
movable blocks, transparent walls, and the spawner/signboard family are rare or absent. These
arenas exist to fix that — every exact simulator class appears at several distances, in confusable
pairs, under occlusion, and in cluttered scenes.

Output is the same YAML dialect as `competition_configurations/*.yml`: one `!ArenaConfig` document
per file, plus a merged multi-arena file for Unity's `Resources/competition_sample.yaml`, which is
what `AAI3EnvironmentManager.configFile` loads during capture.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ARENA_SIZE = 40.0
AGENT_START = (
    20.0,
    0.0,
    4.0,
)  # rotation 0 faces +z, so everything is placed ahead of the agent
NEAR, MID, FAR = 11.0, 19.0, 30.0
MIN_SEPARATION = 0.6  # metres of clear space required between two object bounding boxes

# Classes the checked-in Unity project cannot actually spawn. Putting one of these in an arena
# produces an object that is invisible, uncollidable, or missing altogether — which is how an
# arena ends up looking empty. Verified against this checkout:
#
#   CylinderTunnel, CylinderTunnelTransparent
#       Their mesh is `Assets/Materials/Prefabs/Cylinder.blend`. Unity can only import `.blend`
#       files when Blender is installed locally, and it is not, so the mesh imports empty: the
#       renderer has zero bounds, the MeshCollider has no geometry, and the capture records
#       `dimensions: {0,0,0}` with no mask. Install Blender, reimport, and drop these two from
#       this set to get tunnel data back.
#   RampOld
#       Not listed in `AAI3Arena.allPrefabs`, so `prefabs.GetList().Find(...)` returns null and
#       the spawnable is skipped in silence. It also carries no `Prefab` component, so it would
#       throw on `GetComponent<IPrefab>().SetColor(...)` if it were registered. `Ramp` is the
#       supported prefab; AAI keeps this one only as legacy geometry.
UNSPAWNABLE_CLASSES = frozenset(
    {"CylinderTunnel", "CylinderTunnelTransparent", "RampOld"}
)

# One representative per group, used to build pairs and clutter.
GOALS = (
    "GoodGoal",
    "GoodGoalMulti",
    "GoodGoalBounce",
    "GrowGoal",
    "RipenGoal",
    "ShrinkGoal",
)
BAD_GOALS = ("BadGoal", "BadGoalBounce", "BadGoalMulti")
DECOYS = ("DecoyGoal", "DecoyGoalBounce", "DecoyGoalMulti")
BLOCKS = ("HeavyBlock", "LightBlock", "HollowBox", "JBlock", "LBlock", "UBlock")
TUNNELS = ("CylinderTunnel", "CylinderTunnelTransparent")
RAMPS = ("Ramp", "RampOld")
ZONES = ("DeathZone", "HotZone")
SPAWNERS = (
    "SpawnerTree",
    "SpawnerDispenserTall",
    "SpawnerContainerShort",
    "SpawnerButton",
)
HAZARD_CLASSES = frozenset(ZONES)
GOAL_LIKE_CLASSES = frozenset(
    GOALS + BAD_GOALS + DECOYS + ("DecayGoal", "DecayGoalEpEnding", "RipenGoalEpEnding")
)

# Items whose spawn needs more than position/rotation/size/colour. Copied from the Animal-AI
# reference config `Assets/Resources/test_configs/every_object_spawn_test.yaml`.
EXTRA_PARAMETERS: dict[str, tuple[str, ...]] = {
    "SignBoard": ("symbolNames:", "- default"),
    "SpawnerButton": (
        "moveDurations: [0.1]",
        "resetDurations: [1.0]",
        'rewardNames: ["GoodGoal"]',
        "rewardWeights: [100]",
        "spawnProbability: 1.0",
        "maxRewardCounts: [-1]",
    ),
    "SpawnerDispenserTall": (
        "initialValues: [2.5]",
        "finalValues: [0.5]",
        "timesBetweenSpawns: [0.0]",
        "doorDelays: [0.0]",
        "spawnCounts: [10]",
    ),
    "SpawnerContainerShort": (
        "initialValues: [2.5]",
        "finalValues: [0.5]",
        "timesBetweenSpawns: [0.5]",
        "doorDelays: [1.0]",
        "spawnCounts: [1]",
    ),
    "SpawnerTree": (
        "initialValues: [2.5]",
        "finalValues: [0.5]",
        "ripenTimes: [4.0]",
        "timesBetweenSpawns: [0.5]",
        "spawnCounts: [10]",
    ),
    "DataZone": (
        "zoneVisibility: true",
        'triggerZoneID: ["curriculum zone"]',
    ),
}

# Default footprint per class, in Unity units.
SIZES: dict[str, tuple[float, float, float]] = {
    "Wall": (3.0, 3.0, 1.0),
    "WallTransparent": (3.0, 3.0, 1.0),
    "CylinderTunnel": (4.0, 4.0, 4.0),
    "CylinderTunnelTransparent": (4.0, 4.0, 4.0),
    "Ramp": (4.0, 2.0, 6.0),
    "RampOld": (4.0, 2.0, 6.0),
    "DeathZone": (5.0, 0.0, 5.0),
    "HotZone": (5.0, 0.0, 5.0),
    "DataZone": (5.0, 0.0, 5.0),
    "SignBoard": (2.0, 2.0, 2.0),
}
DEFAULT_SIZE = (2.0, 2.0, 2.0)


@dataclass(frozen=True, slots=True)
class Item:
    name: str
    positions: tuple[tuple[float, float, float], ...]
    rotations: tuple[float, ...] = ()
    sizes: tuple[tuple[float, float, float], ...] = ()
    extra: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.positions:
            raise ValueError(f"{self.name}: needs at least one position")
        for group, values in (("rotations", self.rotations), ("sizes", self.sizes)):
            if values and len(values) != len(self.positions):
                raise ValueError(
                    f"{self.name}: {group} must match the number of positions"
                )


@dataclass(frozen=True, slots=True)
class Arena:
    key: str
    tier: str
    goal: str
    items: tuple[Item, ...]
    random_seed: int
    time_limit: int = 250
    pass_mark: int = 0
    notes: tuple[str, ...] = field(default=())

    @property
    def classes(self) -> set[str]:
        return {item.name for item in self.items if item.name != "Agent"}


def build_curriculum(exact_classes: set[str], seed: int = 20260725) -> list[Arena]:
    unknown = (
        set(GOALS + BAD_GOALS + DECOYS + BLOCKS + TUNNELS + RAMPS + ZONES + SPAWNERS)
        | {"Wall", "WallTransparent", "SignBoard", "DataZone"}
    ) - exact_classes
    if unknown:
        raise ValueError(
            f"curriculum references classes the ontology does not map: {sorted(unknown)}"
        )

    spawnable = sorted(exact_classes - UNSPAWNABLE_CLASSES)
    arenas: list[Arena] = []
    arenas += _tier_tasks()
    arenas += _tier_solo(spawnable)
    arenas += _tier_pairs()
    arenas += _tier_occlusion()
    arenas += _tier_structures()
    arenas += _tier_clutter(spawnable, seed)
    return arenas


def _agent() -> Item:
    return Item(name="Agent", positions=(AGENT_START,), rotations=(0.0,))


def _size(name: str, scale: float = 1.0) -> tuple[float, float, float]:
    x, y, z = SIZES.get(name, DEFAULT_SIZE)
    return round(x * scale, 2), round(y * scale, 2), round(z * scale, 2)


def _object(name: str, positions, scale: float = 1.0, rotation: float = 0.0) -> Item:
    return Item(
        name=name,
        positions=tuple(positions),
        rotations=tuple(rotation for _ in positions),
        sizes=tuple(_size(name, scale) for _ in positions),
        extra=EXTRA_PARAMETERS.get(name, ()),
    )


def _wall(positions, sizes, rotations=None) -> Item:
    positions = tuple(positions)
    return Item(
        name="Wall",
        positions=positions,
        rotations=tuple(rotations or (0.0,) * len(positions)),
        sizes=tuple(sizes),
    )


def _tier_tasks() -> list[Arena]:
    specifications: list[tuple[str, str, list[Item]]] = [
        (
            "y-maze-choice",
            "corridor forks; reward on one side, punishment on the other",
            [
                _wall(
                    ((16.0, 0.0, 12.0), (24.0, 0.0, 12.0), (20.0, 0.0, 26.0)),
                    ((1.0, 3.0, 14.0), (1.0, 3.0, 14.0), (1.0, 3.0, 14.0)),
                ),
                _object("SignBoard", ((20.0, 0.0, 8.0),)),
                _object("GoodGoal", ((14.0, 0.0, 32.0),)),
                _object("BadGoal", ((26.0, 0.0, 32.0),)),
            ],
        ),
        (
            "detour-around-transparent",
            "the goal is visible through glass; the opening is off to the side",
            [
                Item(
                    name="WallTransparent",
                    positions=((16.0, 0.0, 16.0), (26.0, 0.0, 16.0)),
                    rotations=(0.0, 0.0),
                    sizes=((12.0, 4.0, 1.0), (8.0, 4.0, 1.0)),
                ),
                _wall(((6.0, 0.0, 16.0),), ((6.0, 4.0, 1.0),)),
                _object("GoodGoal", ((20.0, 0.0, 24.0),)),
                _object("DecoyGoal", ((30.0, 0.0, 10.0),)),
            ],
        ),
        (
            "ramp-to-platform-goal",
            "the reward sits on a raised surface reachable only by ramp",
            [
                _object("Ramp", ((20.0, 0.0, 13.0),), 1.6),
                _wall(((20.0, 0.0, 22.0),), ((10.0, 3.0, 8.0),)),
                _object("GoodGoalMulti", ((20.0, 3.2, 22.0),)),
                _object("DeathZone", ((10.0, 0.0, 18.0),)),
            ],
        ),
        (
            "push-block-over-hotzone",
            "a movable block can bridge the heated floor",
            [
                _object("LightBlock", ((17.0, 0.0, 11.0),), 1.4),
                _object("HeavyBlock", ((23.0, 0.0, 11.0),), 1.4),
                _object("HotZone", ((20.0, 0.0, 18.0),), 1.6),
                _object("GoodGoal", ((20.0, 0.0, 28.0),)),
            ],
        ),
        (
            "gap-is-the-only-door",
            "a solid wall with one narrow gap in it",
            [
                _wall(
                    ((11.5, 0.0, 18.0), (28.5, 0.0, 18.0)),
                    ((15.0, 4.0, 1.0), (15.0, 4.0, 1.0)),
                ),
                _object("GoodGoalBounce", ((20.0, 0.0, 28.0),)),
                _object("BadGoalBounce", ((12.0, 0.0, 26.0),)),
            ],
        ),
        (
            "decoy-behind-glass",
            "the decoy sits behind glass; the real reward is reachable",
            [
                Item(
                    name="WallTransparent",
                    positions=((16.0, 0.0, 14.0), (16.0, 0.0, 19.0)),
                    rotations=(0.0, 0.0),
                    sizes=((7.0, 3.0, 1.0), (7.0, 3.0, 1.0)),
                ),
                _object("DecoyGoalMulti", ((16.0, 0.0, 16.5),), 0.8),
                _object("GoodGoal", ((28.0, 0.0, 24.0),)),
                _wall(((20.0, 0.0, 31.0),), ((16.0, 3.0, 1.0),)),
            ],
        ),
        (
            "decoy-field",
            "one real reward among look-alikes",
            [
                _object("DecoyGoal", ((12.0, 0.0, 16.0),)),
                _object("DecoyGoalBounce", ((20.0, 0.0, 20.0),)),
                _object("DecoyGoalMulti", ((28.0, 0.0, 16.0),)),
                _object("GoodGoal", ((20.0, 0.0, 30.0),)),
            ],
        ),
        (
            "hazard-gauntlet",
            "two lethal patches flank a safe corridor to the reward",
            [
                _object("DeathZone", ((12.0, 0.0, 18.0),), 1.4),
                _object("DeathZone", ((28.0, 0.0, 18.0),), 1.4),
                _object("GoodGoal", ((20.0, 0.0, 26.0),)),
                _wall(((20.0, 0.0, 33.0),), ((20.0, 3.0, 1.0),)),
            ],
        ),
        (
            "hotzone-shortcut-or-detour",
            "a costly straight line versus a safe walk around",
            [
                _object("HotZone", ((20.0, 0.0, 18.0),), 2.0),
                _wall(
                    ((11.0, 0.0, 18.0), (29.0, 0.0, 18.0)),
                    ((1.0, 3.0, 12.0), (1.0, 3.0, 12.0)),
                ),
                _object("GrowGoal", ((20.0, 0.0, 30.0),)),
                _object("BadGoalMulti", ((6.0, 0.0, 24.0),)),
            ],
        ),
        (
            "spawner-tree-harvest",
            "rewards drop from a tree into a container",
            [
                _object("SpawnerTree", ((20.0, 0.0, 20.0),), 1.2),
                _object("SpawnerContainerShort", ((14.0, 0.0, 14.0),)),
                _object("SignBoard", ((26.0, 0.0, 14.0),)),
                _wall(((20.0, 0.0, 30.0),), ((14.0, 3.0, 1.0),)),
            ],
        ),
        (
            "press-button-for-reward",
            "the button dispenses a reward elsewhere in the room",
            [
                _object("SpawnerButton", ((20.0, 0.0, 12.0),)),
                _object("SpawnerDispenserTall", ((28.0, 0.0, 20.0),)),
                _wall(
                    ((12.0, 0.0, 19.0), (20.0, 0.0, 28.0)),
                    ((1.0, 3.0, 14.0), (14.0, 3.0, 1.0)),
                ),
                _object("SignBoard", ((14.0, 0.0, 12.0),)),
            ],
        ),
        (
            "ripen-and-wait",
            "unripe now, valuable later — timing over greed",
            [
                _object("RipenGoal", ((14.0, 0.0, 18.0),)),
                _object("RipenGoalEpEnding", ((26.0, 0.0, 18.0),)),
                _object("DecayGoal", ((20.0, 0.0, 26.0),)),
                _wall(((20.0, 0.0, 33.0),), ((18.0, 3.0, 1.0),)),
            ],
        ),
        (
            "decay-race",
            "value bleeds away with distance travelled",
            [
                _object("DecayGoalEpEnding", ((20.0, 0.0, 14.0),)),
                _object("GoodGoal", ((12.0, 0.0, 30.0),)),
                _object("GoodGoal", ((28.0, 0.0, 30.0),)),
                _object("HotZone", ((20.0, 0.0, 24.0),)),
            ],
        ),
        (
            "grow-or-shrink",
            "two rewards change size in opposite directions",
            [
                _object("GrowGoal", ((13.0, 0.0, 20.0),)),
                _object("ShrinkGoal", ((27.0, 0.0, 20.0),)),
                Item(
                    name="WallTransparent",
                    positions=((20.0, 0.0, 20.0),),
                    rotations=(0.0,),
                    sizes=((1.0, 4.0, 14.0),),
                ),
                _object("GoodGoalMultiBounce", ((20.0, 0.0, 32.0),)),
            ],
        ),
        (
            "block-shape-maze",
            "awkwardly shaped obstacles between agent and reward",
            [
                _object("JBlock", ((14.0, 0.0, 14.0),), 1.3),
                _object("LBlock", ((26.0, 0.0, 14.0),), 1.3),
                _object("UBlock", ((20.0, 0.0, 21.0),), 1.3),
                _object("HollowBox", ((14.0, 0.0, 26.0),), 1.3),
                _object("GoodGoal", ((26.0, 0.0, 30.0),)),
            ],
        ),
        (
            "multi-goal-harvest",
            "collect many small rewards while avoiding the punishing ones",
            [
                _object(
                    "GoodGoalMulti",
                    ((12.0, 0.0, 14.0), (20.0, 0.0, 22.0), (28.0, 0.0, 14.0)),
                    0.8,
                ),
                _object("GoodGoalMultiBounce", ((16.0, 0.0, 28.0),), 0.8),
                _object("BadGoalMulti", ((24.0, 0.0, 28.0),), 0.8),
                _wall(((20.0, 0.0, 34.0),), ((22.0, 3.0, 1.0),)),
            ],
        ),
        (
            "bounce-alley",
            "bouncing rewards in a narrow corridor",
            [
                _wall(
                    ((14.0, 0.0, 20.0), (26.0, 0.0, 20.0)),
                    ((1.0, 3.0, 22.0), (1.0, 3.0, 22.0)),
                ),
                _object("GoodGoalBounce", ((20.0, 0.0, 16.0),)),
                _object("BadGoalBounce", ((20.0, 0.0, 24.0),)),
                _object("DecoyGoalBounce", ((20.0, 0.0, 30.0),)),
            ],
        ),
        (
            "datazone-crossing",
            "a marked region between the agent and the reward",
            [
                _object("DataZone", ((20.0, 0.0, 18.0),), 1.6),
                _wall(
                    ((12.0, 0.0, 26.0), (28.0, 0.0, 26.0)),
                    ((10.0, 3.0, 1.0), (10.0, 3.0, 1.0)),
                ),
                _object("GoodGoal", ((20.0, 0.0, 32.0),)),
                _object("SignBoard", ((20.0, 0.0, 11.0),)),
            ],
        ),
        (
            "ramp-lookout",
            "climb to see over the walls before choosing",
            [
                # The ramp abuts the platform without overlapping it: Animal-AI spawns items in
                # order and refuses a spot already taken, so an overlapping platform would simply
                # never appear.
                _object("Ramp", ((20.0, 0.0, 12.2),), 1.6),
                _wall(
                    ((20.0, 0.0, 21.0), (20.0, 0.0, 30.0)),
                    ((10.0, 3.0, 8.0), (26.0, 5.0, 1.0)),
                ),
                _object("GoodGoal", ((11.0, 0.0, 26.0),)),
                _object("BadGoal", ((29.0, 0.0, 26.0),)),
            ],
        ),
        (
            "grand-tour",
            "one arena holding a bit of everything",
            [
                _object("WallTransparent", ((8.0, 0.0, 12.0),)),
                _object("SignBoard", ((32.0, 0.0, 12.0),)),
                _object("Ramp", ((8.0, 0.0, 24.0),)),
                _object("UBlock", ((32.0, 0.0, 24.0),)),
                _object("HeavyBlock", ((14.0, 0.0, 18.0),), 0.8),
                _object("LightBlock", ((26.0, 0.0, 18.0),), 0.8),
                _object("HollowBox", ((20.0, 0.0, 12.0),), 0.8),
                _object("DeathZone", ((6.0, 0.0, 32.0),)),
                _object("HotZone", ((34.0, 0.0, 32.0),)),
                _object("SpawnerDispenserTall", ((14.0, 0.0, 32.0),)),
                _object("SpawnerTree", ((26.0, 0.0, 32.0),)),
                _object("GoodGoal", ((20.0, 0.0, 26.0),)),
                _object("BadGoal", ((20.0, 0.0, 34.0),)),
                _object("DecoyGoal", ((20.0, 0.0, 20.0),), 0.7),
            ],
        ),
    ]
    return [
        Arena(
            key=f"task-{index + 1:02d}-{key}",
            tier="tasks",
            goal=goal,
            items=tuple([_agent(), *items]),
            random_seed=20260700 + index,
        )
        for index, (key, goal, items) in enumerate(specifications)
    ]


def _tier_solo(classes: list[str]) -> list[Arena]:
    """One class per arena at three distances, so scale is learned without class confusion."""
    arenas = []
    for index, name in enumerate(classes):
        positions = ((20.0, 0.0, NEAR), (14.0, 0.0, MID), (26.0, 0.0, FAR))
        scales = (1.0, 0.8, 0.6)
        items = [_agent()] + [
            _object(name, (position,), scale)
            for position, scale in zip(positions, scales, strict=True)
        ]
        arenas.append(
            Arena(
                key=f"solo-{index + 1:02d}-{name.lower()}",
                tier="solo",
                goal=f"{name} alone at near/mid/far range",
                items=tuple(items),
                random_seed=20260800 + index,
            )
        )
    return arenas


def _tier_pairs() -> list[Arena]:
    """Classes that look alike, side by side, so the detector must separate them."""
    groups = [
        ("goal-vs-decoy", ("GoodGoal", "DecoyGoal", "GoodGoalMulti", "DecoyGoalMulti")),
        ("goal-vs-bad", ("GoodGoal", "BadGoal", "GoodGoalBounce", "BadGoalBounce")),
        ("wall-vs-transparent", ("Wall", "WallTransparent", "Wall", "WallTransparent")),
        ("ramp-vs-wall", ("Ramp", "Wall", "Ramp", "Wall")),
        ("zone-pair", ("DeathZone", "HotZone")),
        ("block-family", ("HeavyBlock", "LightBlock", "HollowBox", "UBlock")),
        ("block-shapes", ("JBlock", "LBlock", "HollowBox", "HeavyBlock")),
        (
            "spawner-family",
            ("SpawnerTree", "SpawnerDispenserTall", "SpawnerContainerShort"),
        ),
        ("goal-ripeness", ("RipenGoal", "DecayGoal", "GrowGoal", "ShrinkGoal")),
        (
            "episode-ending-goals",
            ("RipenGoalEpEnding", "DecayGoalEpEnding", "GoodGoal"),
        ),
        (
            "multi-goals",
            ("GoodGoalMulti", "GoodGoalMultiBounce", "BadGoalMulti", "DecoyGoalBounce"),
        ),
    ]
    arenas = []
    for index, (key, names) in enumerate(groups):
        spread = [12.0, 18.0, 24.0, 28.0][: len(names)]
        items = [_agent()] + [
            _object(name, ((x, 0.0, MID if position % 2 == 0 else MID + 5.0),))
            for position, (name, x) in enumerate(zip(names, spread, strict=True))
        ]
        arenas.append(
            Arena(
                key=f"pair-{index + 1:02d}-{key}",
                tier="pairs",
                goal="side-by-side classes that are easy to confuse: "
                + ", ".join(names),
                items=tuple(items),
                random_seed=20260900 + index,
            )
        )
    return arenas


def _tier_occlusion() -> list[Arena]:
    """Partial visibility: the case the visibility/occlusion slices in evaluation measure."""
    specifications = [
        (
            "transparent-in-front",
            [
                _object("WallTransparent", ((20.0, 0.0, 12.0),), 2.0),
                _object("GoodGoal", ((20.0, 0.0, 17.0),)),
                _object("BadGoal", ((15.0, 0.0, 17.0),)),
            ],
        ),
        (
            "wall-half-hides-goal",
            [
                _object("Wall", ((18.0, 0.0, 14.0),), 1.5),
                _object("GoodGoal", ((19.5, 0.0, 18.0),)),
                _object("GoodGoalMulti", ((24.0, 0.0, 22.0),)),
            ],
        ),
        (
            "block-in-front-of-decoy",
            [
                _object("HeavyBlock", ((20.0, 0.0, 13.0),), 1.5),
                _object("DecoyGoal", ((20.0, 0.0, 17.0),)),
                _object("LightBlock", ((25.0, 0.0, 15.0),)),
            ],
        ),
        (
            "goal-behind-glass-slit",
            [
                Item(
                    name="WallTransparent",
                    positions=((15.0, 0.0, 15.0), (25.0, 0.0, 15.0)),
                    rotations=(0.0, 0.0),
                    sizes=((8.0, 3.0, 1.0), (8.0, 3.0, 1.0)),
                ),
                _object("GoodGoal", ((20.0, 0.0, 19.0),), 0.6),
                _object("DecoyGoal", ((28.0, 0.0, 25.0),)),
            ],
        ),
        (
            "goal-beside-ublock",
            [
                _object("UBlock", ((18.0, 0.0, 15.0),), 1.6),
                _object("GoodGoal", ((21.5, 0.0, 15.0),), 0.7),
                _object("HollowBox", ((26.0, 0.0, 20.0),)),
            ],
        ),
        (
            "crowded-same-class",
            [
                _object(
                    "GoodGoal",
                    (
                        (17.0, 0.0, 13.0),
                        (20.0, 0.0, 14.0),
                        (23.0, 0.0, 13.5),
                        (20.0, 0.0, 19.0),
                    ),
                ),
                _object("Wall", ((20.0, 0.0, 24.0),), 2.0),
            ],
        ),
    ]
    return [
        Arena(
            key=f"occlusion-{index + 1:02d}-{key}",
            tier="occlusion",
            goal="partial visibility and overlap",
            items=tuple([_agent(), *items]),
            random_seed=20261000 + index,
        )
        for index, (key, items) in enumerate(specifications)
    ]


def _tier_structures() -> list[Arena]:
    """Scenes with vertical structure: ramps onto raised surfaces, corridors, hazard layouts."""
    specifications = [
        (
            "ramp-to-raised-wall",
            [
                _object("Ramp", ((20.0, 0.0, 11.4),), 1.5),
                Item(
                    name="Wall",
                    positions=((20.0, 0.0, 20.0),),
                    rotations=(0.0,),
                    sizes=((8.0, 3.0, 8.0),),
                ),
                _object("GoodGoal", ((20.0, 3.2, 20.0),)),
            ],
        ),
        (
            "hazard-corridor",
            [
                Item(
                    name="Wall",
                    positions=((14.0, 0.0, 18.0), (26.0, 0.0, 18.0)),
                    rotations=(0.0, 0.0),
                    sizes=((1.0, 3.0, 16.0), (1.0, 3.0, 16.0)),
                ),
                _object("DeathZone", ((20.0, 0.0, 16.0),)),
                _object("GoodGoal", ((20.0, 0.0, 26.0),)),
            ],
        ),
        (
            "hotzone-detour",
            [
                _object("HotZone", ((20.0, 0.0, 15.0),), 1.6),
                _object("Ramp", ((12.0, 0.0, 20.0),)),
                _object("GoodGoalMulti", ((28.0, 0.0, 24.0),)),
                _object("BadGoal", ((20.0, 0.0, 28.0),)),
            ],
        ),
        (
            "walled-gate",
            [
                Item(
                    name="Wall",
                    positions=((11.0, 0.0, 16.0), (29.0, 0.0, 16.0)),
                    rotations=(0.0, 0.0),
                    sizes=((14.0, 4.0, 1.0), (14.0, 4.0, 1.0)),
                ),
                _object("HollowBox", ((20.0, 0.0, 16.0),), 1.2),
                _object("GoodGoal", ((20.0, 0.0, 24.0),)),
            ],
        ),
        (
            "spawner-room",
            [
                _object("SpawnerTree", ((14.0, 0.0, 18.0),)),
                _object("SpawnerDispenserTall", ((26.0, 0.0, 18.0),)),
                _object("SignBoard", ((20.0, 0.0, 24.0),)),
                _object("SpawnerButton", ((20.0, 0.0, 12.0),)),
            ],
        ),
        (
            "datazone-and-signs",
            [
                _object("DataZone", ((20.0, 0.0, 16.0),)),
                _object("SignBoard", ((14.0, 0.0, 22.0),)),
                _object("SignBoard", ((26.0, 0.0, 22.0),)),
                _object("SpawnerContainerShort", ((20.0, 0.0, 26.0),)),
            ],
        ),
        (
            "block-stack-and-ramp",
            [
                _object("HeavyBlock", ((15.0, 0.0, 14.0),), 1.4),
                _object("LightBlock", ((25.0, 0.0, 14.0),), 1.4),
                _object("Ramp", ((20.0, 0.0, 20.0),), 1.3),
                _object("GoodGoal", ((20.0, 0.0, 27.0),)),
            ],
        ),
        (
            "transparent-maze",
            [
                Item(
                    name="WallTransparent",
                    positions=((16.0, 0.0, 14.0), (24.0, 0.0, 14.0), (20.0, 0.0, 22.0)),
                    rotations=(0.0, 0.0, 90.0),
                    sizes=((6.0, 3.0, 1.0), (6.0, 3.0, 1.0), (8.0, 3.0, 1.0)),
                ),
                _object("GoodGoal", ((20.0, 0.0, 18.0),)),
                _object("DecoyGoal", ((12.0, 0.0, 24.0),)),
            ],
        ),
    ]
    return [
        Arena(
            key=f"structure-{index + 1:02d}-{key}",
            tier="structures",
            goal="vertical structure, corridors, and hazard layouts",
            items=tuple([_agent(), *items]),
            random_seed=20261100 + index,
        )
        for index, (key, items) in enumerate(specifications)
    ]


def _tier_clutter(classes: list[str], seed: int, count: int = 14) -> list[Arena]:
    """Many classes per frame at mixed range — where small and distant objects come from."""
    generator = random.Random(seed)
    arenas = []
    for index in range(count):
        chosen = generator.sample(classes, k=generator.randint(6, 9))
        items = [_agent()]
        placed: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = [
            (AGENT_START, (1.0, 1.0, 1.0))
        ]
        for name in chosen:
            # Rejection-sample a free spot. Animal-AI retries a *specified* position 20 times
            # unchanged and then gives up, so an overlapping object simply never spawns; keeping
            # the boxes apart here is what guarantees the arena contains what it claims to.
            for _ in range(200):
                x = round(generator.uniform(6.0, 34.0), 1)
                z = round(generator.uniform(9.0, 34.0), 1)
                scale = round(generator.choice((0.5, 0.7, 1.0, 1.3)), 2)
                position, size = (x, 0.0, z), _size(name, scale)
                if not any(
                    _boxes_overlap(position, size, other_position, other_size)
                    for other_position, other_size in placed
                ):
                    placed.append((position, size))
                    items.append(_object(name, (position,), scale))
                    break
        arenas.append(
            Arena(
                key=f"clutter-{index + 1:02d}",
                tier="clutter",
                goal="mixed classes at mixed range and scale",
                items=tuple(items),
                random_seed=20261200 + index,
            )
        )
    return arenas


def _boxes_overlap(
    left_position: tuple[float, float, float],
    left_size: tuple[float, float, float],
    right_position: tuple[float, float, float],
    right_size: tuple[float, float, float],
    separation: float = MIN_SEPARATION,
) -> bool:
    """True when two spawn boxes are closer than `separation` on every axis.

    Positions are box centres in x/z and the *base* in y, matching Animal-AI: `Prefab.AdjustY`
    lifts an object by half its height, so `y` is where the object sits, not its centre.
    """
    for axis, (extent_left, extent_right) in enumerate(
        zip(left_size, right_size, strict=True)
    ):
        if axis == 1:
            low_left, high_left = left_position[1], left_position[1] + max(
                extent_left, 0.01
            )
            low_right, high_right = right_position[1], right_position[1] + max(
                extent_right, 0.01
            )
        else:
            low_left = left_position[axis] - extent_left / 2
            high_left = left_position[axis] + extent_left / 2
            low_right = right_position[axis] - extent_right / 2
            high_right = right_position[axis] + extent_right / 2
        # Separated on this axis when one box ends at least `separation` before the other starts.
        if high_left + separation <= low_right or high_right + separation <= low_left:
            return False
    return True


def _placements(
    arena: Arena,
) -> list[tuple[str, tuple[float, float, float], tuple[float, float, float]]]:
    placements = []
    for item in arena.items:
        sizes = item.sizes or ((1.0, 1.0, 1.0),) * len(item.positions)
        for position, size in zip(item.positions, sizes, strict=True):
            placements.append((item.name, position, size))
    return placements


def infeasible(arena: Arena) -> list[str]:
    reasons: list[str] = []
    placements = _placements(arena)
    objects = [entry for entry in placements if entry[0] != "Agent"]

    unspawnable = sorted(
        {name for name, _, _ in objects if name in UNSPAWNABLE_CLASSES}
    )
    if unspawnable:
        reasons.append(f"cannot spawn in this project: {unspawnable}")
    if not objects:
        reasons.append("no objects")

    for index, (name, position, size) in enumerate(objects):
        for other_name, other_position, other_size in objects[index + 1 :]:
            if _boxes_overlap(
                position, size, other_position, other_size, separation=0.0
            ):
                reasons.append(
                    f"{name} at {position[0]},{position[2]} overlaps "
                    f"{other_name} at {other_position[0]},{other_position[2]}"
                )
        if _boxes_overlap(position, size, AGENT_START, (1.0, 1.0, 1.0), separation=0.0):
            reasons.append(
                f"{name} at {position[0]},{position[2]} blocks the agent spawn"
            )
        if name in GOAL_LIKE_CLASSES:
            for hazard_name, hazard_position, hazard_size in objects:
                if hazard_name in HAZARD_CLASSES and _boxes_overlap(
                    (position[0], 0.0, position[2]),
                    (size[0], 0.01, size[2]),
                    (hazard_position[0], 0.0, hazard_position[2]),
                    (hazard_size[0], 0.01, hazard_size[2]),
                    separation=0.0,
                ):
                    reasons.append(
                        f"{name} at {position[0]},{position[2]} stands inside "
                        f"{hazard_name} — unreachable without dying"
                    )
    return reasons


def arena_yaml(arena: Arena, index: int = 0, indent: str = "  ") -> str:
    lines = [
        f"{indent}{index}: !Arena",
        f"{indent * 2}pass_mark: {arena.pass_mark}",
        f"{indent * 2}t: {arena.time_limit}",
        f"{indent * 2}randomSeed: {arena.random_seed}",
        f"{indent * 2}items:",
    ]
    for item in arena.items:
        lines.append(f"{indent * 2}- !Item")
        lines.append(f"{indent * 3}name: {item.name}")
        lines.append(f"{indent * 3}positions:")
        lines += [
            f"{indent * 3}- !Vector3 {_vector(position)}" for position in item.positions
        ]
        if item.rotations:
            lines.append(
                f"{indent * 3}rotations: [{', '.join(_number(r) for r in item.rotations)}]"
            )
        if item.sizes:
            lines.append(f"{indent * 3}sizes:")
            lines += [f"{indent * 3}- !Vector3 {_vector(size)}" for size in item.sizes]
        lines += [f"{indent * 3}{line}" for line in item.extra]
    return "\n".join(lines)


def arena_document(arena: Arena) -> str:
    return "\n".join(
        [
            f"# {arena.key} — {arena.goal}",
            f"# tier: {arena.tier}; classes: {', '.join(sorted(arena.classes))}",
            "!ArenaConfig",
            "arenas:",
            arena_yaml(arena, 0),
            "",
        ]
    )


def select_competition_configs(directory: str | Path, count: int = 40) -> list[Path]:
    """An even spread of official `<level>-<task>-<variant>.yml` arenas across all ten levels."""
    pattern = re.compile(r"^(\d+)-(\d+)-(\d+)$")
    candidates = sorted(
        (path for path in Path(directory).glob("*.yml") if pattern.match(path.stem)),
        key=lambda path: tuple(int(part) for part in pattern.match(path.stem).groups()),
    )
    firsts = [path for path in candidates if path.stem.endswith("-1")]
    if not firsts:
        return []
    return firsts[:: max(1, len(firsts) // count)][:count]


def competition_blocks(paths: list[str | Path]) -> list[tuple[str, list[str]]]:
    """
    Lift arenas out of official `competition_configurations/*.yml` files **verbatim**.

    The bodies are spliced as text, never re-emitted, so colours, symbol names, spawner parameters
    and everything else this module does not model survive untouched. Arenas holding nothing but
    the agent are skipped, as are any that recolour a goal — a recoloured goal makes GoodGoal and
    BadGoal visually identical, which would poison the detector labels.

    Arenas that Unity's YAML reader would reject are skipped as well: a single malformed arena
    aborts deserialization of the whole file, and with no config the arena never builds, the agent
    never spawns, and Unity reports "Cameras not initialized properly". Fifteen of the official
    configs tag `colors:` entries `!Vector3` instead of `!RGB`, which trips exactly that.
    """
    blocks: list[tuple[str, list[str]]] = []
    for path in paths:
        path = Path(path)
        lines = path.read_text(encoding="utf-8").splitlines()
        for name, body in _split_arenas(path.stem, lines):
            if not _has_objects(body, ignore=UNSPAWNABLE_CLASSES):
                continue
            if _recolours_goal(body):
                continue
            if mistagged_fields(body):
                continue
            blocks.append((name, body))
    return blocks


def mistagged_fields(body: list[str]) -> list[tuple[int, str, str]]:
    """`positions`/`sizes` must hold `!Vector3` and `colors` must hold `!RGB`, or Unity throws."""
    expected = {"positions": "!Vector3", "sizes": "!Vector3", "colors": "!RGB"}
    problems: list[tuple[int, str, str]] = []
    field_name: str | None = None
    for number, line in enumerate(body, 1):
        stripped = line.strip()
        if stripped.endswith(":") and not stripped.startswith("-"):
            field_name = stripped[:-1]
            continue
        if not stripped.startswith("- !"):
            if stripped and not stripped.startswith("-"):
                field_name = None
            continue
        tag = (
            stripped.split(None, 1)[1].split(None, 1)[0]
            if " " in stripped[2:]
            else stripped[2:]
        )
        tag = tag.split("{", 1)[0].strip()
        if tag == "!Item":
            field_name = None
            continue
        if field_name in expected and tag != expected[field_name]:
            problems.append((number, field_name, tag))
    return problems


def merged_document(
    base: str,
    arenas: list[Arena],
    imported: list[tuple[str, list[str]]] | None = None,
) -> str:
    sections: list[list[str]] = []
    header = [
        line for line in base.rstrip("\n").splitlines() if not _is_arena_header(line)
    ]
    for index, (_, body) in enumerate(
        _split_arenas("base", base.rstrip("\n").splitlines())
    ):
        if _has_objects(body):
            sections.append(body)
    kept_base = len(sections)

    for _, body in imported or []:
        sections.append(body)

    document = [line for line in header if line.strip() and not line.startswith("  ")]
    numbered: list[str] = []
    for index, body in enumerate(sections):
        numbered.append(f"  {index}: !Arena")
        numbered += body
    for offset, arena in enumerate(arenas):
        numbered.append(arena_yaml(arena, kept_base + len(imported or []) + offset))
    return "\n".join([*document, *numbered, ""])


def write_curriculum(directory: str | Path, arenas: list[Arena]) -> list[Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for stale in directory.glob("*.yml"):
        stale.unlink()
    written = []
    for arena in arenas:
        path = directory / f"{arena.key}.yml"
        path.write_text(arena_document(arena), encoding="utf-8")
        written.append(path)
    return written


def validate(arenas: list[Arena], exact_classes: set[str]) -> dict[str, Any]:
    coverage: dict[str, int] = dict.fromkeys(sorted(exact_classes), 0)
    keys: set[str] = set()
    for arena in arenas:
        if arena.key in keys:
            raise ValueError(f"duplicate arena key {arena.key}")
        keys.add(arena.key)
        agents = [item for item in arena.items if item.name == "Agent"]
        if len(agents) != 1:
            raise ValueError(
                f"{arena.key}: needs exactly one Agent, found {len(agents)}"
            )
        for item in arena.items:
            if item.name != "Agent" and item.name not in exact_classes:
                raise ValueError(
                    f"{arena.key}: {item.name} is not an ontology-mapped class"
                )
            if item.name != "Agent":
                coverage[item.name] += len(item.positions)
            for x, y, z in item.positions:
                if not (0.0 <= x <= ARENA_SIZE and 0.0 <= z <= ARENA_SIZE and y >= 0.0):
                    raise ValueError(
                        f"{arena.key}: {item.name} position {(x, y, z)} is outside the arena"
                    )
        reasons = infeasible(arena)
        if reasons:
            raise ValueError(f"{arena.key} is infeasible: {reasons}")
    expected = exact_classes - UNSPAWNABLE_CLASSES
    missing = sorted(name for name in expected if coverage[name] == 0)
    if missing:
        raise ValueError(f"classes never spawned: {missing}")
    return {
        "arenas": len(arenas),
        "instances": sum(coverage.values()),
        "per_tier": {
            tier: sum(1 for arena in arenas if arena.tier == tier)
            for tier in dict.fromkeys(arena.tier for arena in arenas)
        },
        "instances_per_class": coverage,
        "excluded_classes": sorted(UNSPAWNABLE_CLASSES & exact_classes),
    }


GOAL_CLASSES = frozenset(
    GOALS + BAD_GOALS + DECOYS + ("DecayGoal", "DecayGoalEpEnding", "RipenGoalEpEnding")
)


def _is_arena_header(line: str) -> bool:
    stripped = line.strip()
    return stripped.endswith(": !Arena") and stripped[:-8].strip().isdigit()


def _split_arenas(source: str, lines: list[str]) -> list[tuple[str, list[str]]]:
    arenas: list[tuple[str, list[str]]] = []
    body: list[str] | None = None
    for line in lines:
        if _is_arena_header(line):
            body = []
            arenas.append((f"{source}#{len(arenas)}", body))
        elif body is not None:
            body.append(line)
    return [(name, _trim(body)) for name, body in arenas]


def _trim(body: list[str]) -> list[str]:
    while body and not body[-1].strip():
        body.pop()
    return body


def _has_objects(body: list[str], ignore: frozenset[str] = frozenset()) -> bool:
    names = [line.split("name:", 1)[1].strip() for line in body if "name:" in line]
    return any(name != "Agent" and name not in ignore for name in names)


def _recolours_goal(body: list[str]) -> bool:
    current = None
    for line in body:
        if "name:" in line:
            current = line.split("name:", 1)[1].strip()
        elif current in GOAL_CLASSES and line.strip().startswith("colors:"):
            return True
    return False


def _vector(value: tuple[float, float, float]) -> str:
    x, y, z = (_number(component) for component in value)
    return f"{{x: {x}, y: {y}, z: {z}}}"


def _number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(round(float(value), 2))
