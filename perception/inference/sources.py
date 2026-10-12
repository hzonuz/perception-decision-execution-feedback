"""
Frame sources: prerecorded images for debugging, and the live Animal-AI observation stream.

Timestamps are simulation time, not wall-clock, so runs are reproducible: Animal-AI's fixed
timestep is 0.02 s (`ProjectSettings/TimeManager.asset`) and the agent decides every
`decision_period` steps, so one observation is `0.02 * decision_period` seconds.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from mlagents_envs.base_env import ActionTuple

from perception.contracts import (
    AgentState,
    FrameStamp,
    ObservationFrame,
    PrimitiveAction,
    Vector3,
)
from perception.inference.detector import Frame

FIXED_TIMESTEP_SECONDS = 0.02
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def prerecorded_frames(
    directory: str | Path,
    *,
    episode_id: str | None = None,
    frame_interval: float = 0.06,
    observations: str | Path | None = None,
) -> Iterator[Frame]:
    directory = Path(directory)
    paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise FileNotFoundError(f"no {sorted(IMAGE_SUFFIXES)} images in {directory}")
    episode_id = episode_id or directory.name
    recorded = {}
    if observations is not None:
        with Path(observations).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    item = ObservationFrame.from_json(line)
                    if item.rgb is not None:
                        recorded[Path(item.rgb.path).parts[-2:]] = item
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
        item = recorded.get((directory.name, path.name))
        if item is None:
            yield Frame(FrameStamp(episode_id, index, index * frame_interval), rgb)
        else:
            yield Frame(item.stamp, rgb, item.agent)


def live_frames(
    env: Any,
    *,
    episodes: int = 1,
    max_steps: int = 500,
    seed: int = 0,
    decision_period: int = 3,
    run_id: str = "live",
) -> Iterator[Frame]:
    rng = np.random.default_rng(seed)
    behavior = next(iter(env.behavior_specs))
    dt = FIXED_TIMESTEP_SECONDS * decision_period
    for episode in range(episodes):
        if episode:
            env.reset()
        episode_id = f"{run_id}-{seed}-{episode:03d}"
        action = None
        for step in range(max_steps):
            decision, terminal = env.get_steps(behavior)
            steps = terminal if len(terminal) else decision
            if not len(steps):
                break
            yield Frame(
                FrameStamp(episode_id, step, step * dt),
                *_parse_observation(steps.obs, action),
            )
            if len(terminal):
                break
            action = rng.integers(0, 3, size=2)
            env.set_actions(
                behavior, ActionTuple(discrete=np.array([action], dtype=np.int32))
            )
            env.step()


def _parse_observation(
    obs: list[np.ndarray], action: np.ndarray | None
) -> tuple[np.ndarray, AgentState]:
    camera, vector = obs[0][0], obs[-1][0]
    if camera.ndim != 3 or camera.shape[-1] != 3:
        raise ValueError(
            f"expected an (H, W, 3) RGB camera observation, got {camera.shape}"
        )
    if vector.shape != (7,):
        raise ValueError(f"expected 7 vector observations, got {vector.shape}")
    image = (
        camera
        if camera.dtype == np.uint8
        else (np.clip(camera, 0.0, 1.0) * 255).round().astype(np.uint8)
    )
    agent = AgentState(
        node_id="agent",
        world_position=Vector3(*(float(v) for v in vector[4:7])),
        world_rotation=None,  # not in the Animal-AI observation
        velocity=Vector3(*(float(v) for v in vector[1:4])),
        recent_action=(
            None if action is None else PrimitiveAction(int(action[0]), int(action[1]))
        ),
        health=float(vector[0]),
    )
    return image, agent
