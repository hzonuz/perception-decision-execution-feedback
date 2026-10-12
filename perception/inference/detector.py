"""
Detector-neutral runtime interface.

Coordinate conventions (every detector must follow them; `DetectionFrame` enforces the ranges):

- Pixel coordinates are in the *original* frame, not the model input: origin at the top-left
  corner, x to the right, y down, continuous edge coordinates, so a box spans
  `[x_min, x_max) x [y_min, y_max)` with `0 <= x_min < x_max <= width`.
- `normalized_location` is the box centre divided by `(width, height)`, in `[0, 1]`.
- Instance masks are one `uint16` instance map per frame at the original resolution: pixel value
  `k + 1` belongs to `objects[k]`, `0` is background. It is saved as
  `masks/<episode_id>/<frame_index:06d>.png` under the output root, and each object's
  `MaskReference` points there with `instance_value = k + 1`. Where instances overlap, the
  higher-confidence one owns the pixel; an object with no pixels has `instance_mask=None`.
- `semantic_distribution` holds the detector's class probabilities; mass the detector does not
  assign to a known class is `unknown_probability`, never silently given to a class.

A detector sees one frame at a time: no tracking, no temporal memory, no relations. Batching is a
throughput optimization only: a frame's output must not depend on the rest of its batch beyond
floating-point rounding (YOLO's batched kernels move boxes by ~1e-5 px versus batch size 1).
Deterministic mode repeats bit-for-bit for the same inputs, weights, device and batch size.
"""

from __future__ import annotations

import hashlib
import math
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from perception.contracts import (
    SCHEMA_VERSION,
    AgentState,
    BoundingBox2D,
    CategoricalDistribution,
    DetectionFrame,
    DetectorObject,
    FrameStamp,
    MaskEncoding,
    MaskReference,
    NormalizedPoint,
    ProbabilityEntry,
    SourceMetadata,
    SourceType,
    Uncertainty,
)


@dataclass(frozen=True, slots=True)
class Frame:
    """One RGB observation: `image` is `(H, W, 3)` uint8 in RGB order."""

    stamp: FrameStamp
    image: np.ndarray
    agent: AgentState | None = None

    def __post_init__(self) -> None:
        if (
            self.image.ndim != 3
            or self.image.shape[2] != 3
            or self.image.dtype != np.uint8
        ):
            raise ValueError(
                f"frame image must be (H, W, 3) uint8, got {self.image.shape} {self.image.dtype}"
            )


@dataclass(frozen=True, slots=True)
class Timing:
    """Per-frame milliseconds. For a batch, each frame gets the batch total divided evenly."""

    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float

    @property
    def total_ms(self) -> float:
        return self.preprocess_ms + self.inference_ms + self.postprocess_ms


@dataclass(frozen=True, slots=True)
class Detection:
    frame: DetectionFrame
    masks: np.ndarray | None  # (H, W) uint16 instance map, see the module docstring
    timing: Timing


class Detector(ABC):
    name: str
    version: str
    device: str
    class_names: tuple[str, ...]

    @abstractmethod
    def detect_batch(self, frames: Sequence[Frame]) -> list[Detection]:
        pass

    def detect(self, frame: Frame) -> Detection:
        return self.detect_batch([frame])[0]


def resolve_device(device: str = "auto", *, deterministic: bool = False) -> str:
    if device != "auto":
        return device
    if deterministic:
        return "cpu"
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def mask_path(stamp: FrameStamp) -> str:
    return f"masks/{stamp.episode_id}/{stamp.frame_index:06d}.png"


def build_detection(
    stamp: FrameStamp,
    width: int,
    height: int,
    candidates: Sequence[
        tuple[tuple[float, float, float, float], dict[str, float], np.ndarray | None]
    ],
    source: SourceMetadata,
    timing: Timing,
) -> Detection:
    kept = []
    for box, scores, mask in candidates:
        x_min, x_max = (min(max(float(v), 0.0), width) for v in (box[0], box[2]))
        y_min, y_max = (min(max(float(v), 0.0), height) for v in (box[1], box[3]))
        scores = {
            label: min(max(float(p), 0.0), 1.0) for label, p in scores.items() if p > 0
        }
        if x_min < x_max and y_min < y_max and scores:
            kept.append(((x_min, y_min, x_max, y_max), scores, mask))
    kept.sort(key=lambda item: -max(item[1].values()))

    instance_map = (
        np.zeros((height, width), np.uint16)
        if any(m is not None for *_, m in kept)
        else None
    )
    for index in reversed(
        range(len(kept))
    ):  # highest confidence painted last, so it wins overlaps
        mask = kept[index][2]
        if mask is not None:
            instance_map[mask.astype(bool)] = index + 1

    objects = []
    for index, ((x_min, y_min, x_max, y_max), scores, mask) in enumerate(kept):
        total = sum(scores.values())
        scale = 1.0 / total if total > 1.0 else 1.0
        entries = tuple(
            ProbabilityEntry(label=label, probability=p * scale)
            for label, p in sorted(scores.items(), key=lambda kv: -kv[1])
        )
        unknown = max(0.0, 1.0 - sum(entry.probability for entry in entries))
        entropy = -sum(
            p * math.log(p)
            for p in [e.probability for e in entries] + [unknown]
            if p > 0
        )
        has_pixels = (
            instance_map is not None
            and mask is not None
            and bool((instance_map == index + 1).any())
        )
        objects.append(
            DetectorObject(
                detector_id=f"det_{index:04d}",
                semantic_distribution=CategoricalDistribution(
                    entries=entries, unknown_probability=unknown
                ),
                bounding_box=BoundingBox2D(x_min, y_min, x_max, y_max, width, height),
                instance_mask=(
                    MaskReference(
                        mask_path(stamp), MaskEncoding.PNG, width, height, index + 1
                    )
                    if has_pixels
                    else None
                ),
                normalized_location=NormalizedPoint(
                    (x_min + x_max) / (2 * width), (y_min + y_max) / (2 * height)
                ),
                confidence=max(scores.values()),
                uncertainty=Uncertainty(entropy=entropy),
                source=source,
            )
        )
    frame = DetectionFrame(SCHEMA_VERSION, stamp, width, height, tuple(objects), source)
    return Detection(frame=frame, masks=instance_map, timing=timing)


class MockDetector(Detector):
    name = "mock_detector"
    version = "1.0.0"

    def __init__(
        self, class_names: Sequence[str], *, seed: int = 0, max_objects: int = 4
    ) -> None:
        self.class_names = tuple(class_names)
        self.seed = seed
        self.max_objects = max_objects
        self.device = "cpu"
        self.source = SourceMetadata(
            SourceType.DETECTOR,
            self.name,
            self.version,
            configuration_id=f"seed={seed}",
        )

    def detect_batch(self, frames: Sequence[Frame]) -> list[Detection]:
        return [self._detect_one(frame) for frame in frames]

    def _detect_one(self, frame: Frame) -> Detection:
        start = time.perf_counter()
        height, width = frame.image.shape[:2]
        key = f"{self.seed}:{frame.stamp.episode_id}:{frame.stamp.frame_index}".encode()
        rng = np.random.default_rng(
            int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
        )
        candidates = []
        for _ in range(int(rng.integers(0, self.max_objects + 1))):
            x0, x1 = sorted(rng.uniform(0, width, 2))
            y0, y1 = sorted(rng.uniform(0, height, 2))
            labels = rng.choice(
                len(self.class_names), size=min(2, len(self.class_names)), replace=False
            )
            weights = rng.dirichlet(
                np.ones(len(labels) + 1)
            )  # last share stays unknown
            mask = np.zeros((height, width), bool)
            mask[int(y0) : int(math.ceil(y1)), int(x0) : int(math.ceil(x1))] = True
            candidates.append(
                (
                    (x0, y0, x1, y1),
                    {self.class_names[i]: float(w) for i, w in zip(labels, weights)},
                    mask,
                )
            )
        elapsed = (time.perf_counter() - start) * 1000
        return build_detection(
            frame.stamp,
            width,
            height,
            candidates,
            self.source,
            Timing(0.0, elapsed, 0.0),
        )
