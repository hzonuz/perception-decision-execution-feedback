from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from perception.contracts.base import ContractValidationError, JsonContract


SCHEMA_VERSION = "1.0.0"
JsonScalar = str | int | float | bool | None


class SourceType(str, Enum):
    CONFIGURATION = "configuration"
    SENSOR = "sensor"
    PROPRIOCEPTION = "proprioception"
    DETECTOR = "detector"
    TRACKER = "tracker"
    TEMPORAL_MEMORY = "temporal_memory"
    GEOMETRY = "geometry"
    LEARNED_RELATION_MODEL = "learned_relation_model"
    TASK_RULE = "task_rule"
    SIMULATOR_ORACLE = "simulator_oracle"


class MemoryState(str, Enum):
    TENTATIVE = "tentative"
    VISIBLE = "visible"
    PARTIALLY_VISIBLE = "partially_visible"
    OCCLUDED = "occluded"
    OUT_OF_VIEW = "out_of_view"
    UNOBSERVABLE = "unobservable"
    BELIEVED_PRESENT = "believed_present"
    TERMINATED = "terminated"
    LOST = "lost"  # observable, in view, unexplained miss: unlike TERMINATED, no evidence of absence


class MotionState(str, Enum):
    UNKNOWN = "unknown"
    MOVING = "moving"
    STATIONARY = "stationary"


@dataclass(frozen=True, slots=True)
class FrameStamp(JsonContract):
    episode_id: str
    frame_index: int
    timestamp_seconds: float

    def __post_init__(self) -> None:
        _require_nonempty(self.episode_id, "episode_id")
        _require_nonnegative_int(self.frame_index, "frame_index")
        _require_nonnegative_finite(self.timestamp_seconds, "timestamp_seconds")


@dataclass(frozen=True, slots=True)
class SourceMetadata(JsonContract):
    source: SourceType
    producer: str
    producer_version: str | None = None
    privileged: bool = False
    configuration_id: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty(self.producer, "producer")
        if self.source is SourceType.SIMULATOR_ORACLE and not self.privileged:
            raise ContractValidationError(
                "simulator_oracle sources must be marked privileged"
            )
        if self.privileged and self.source is not SourceType.SIMULATOR_ORACLE:
            raise ContractValidationError(
                "only simulator_oracle sources may be privileged"
            )


@dataclass(frozen=True, slots=True)
class Uncertainty(JsonContract):
    standard_deviation: float | None = None
    entropy: float | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    calibration_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("standard_deviation", "entropy"):
            value = getattr(self, name)
            if value is not None:
                _require_nonnegative_finite(value, name)
        if self.lower_bound is not None:
            _require_finite(self.lower_bound, "lower_bound")
        if self.upper_bound is not None:
            _require_finite(self.upper_bound, "upper_bound")
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.lower_bound > self.upper_bound
        ):
            raise ContractValidationError("lower_bound must not exceed upper_bound")


@dataclass(frozen=True, slots=True)
class ProbabilityEntry(JsonContract):
    label: str
    probability: float

    def __post_init__(self) -> None:
        _require_nonempty(self.label, "label")
        _require_probability(self.probability, "probability")


@dataclass(frozen=True, slots=True)
class CategoricalDistribution(JsonContract):
    entries: tuple[ProbabilityEntry, ...]
    unknown_probability: float

    def __post_init__(self) -> None:
        _require_probability(self.unknown_probability, "unknown_probability")
        labels = [entry.label for entry in self.entries]
        if len(labels) != len(set(labels)):
            raise ContractValidationError("distribution labels must be unique")
        total = self.unknown_probability + sum(
            entry.probability for entry in self.entries
        )
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ContractValidationError(
                f"distribution probabilities must sum to 1, got {total}"
            )

    @classmethod
    def unknown(cls) -> "CategoricalDistribution":
        return cls(entries=(), unknown_probability=1.0)

    @classmethod
    def certain(cls, label: str) -> "CategoricalDistribution":
        return cls(
            entries=(ProbabilityEntry(label=label, probability=1.0),),
            unknown_probability=0.0,
        )


@dataclass(frozen=True, slots=True)
class BernoulliBelief(JsonContract):
    """A probability of true; ``None`` explicitly means unknown."""

    probability: float | None
    source: SourceMetadata
    uncertainty: Uncertainty | None = None

    def __post_init__(self) -> None:
        if self.probability is not None:
            _require_probability(self.probability, "probability")


@dataclass(frozen=True, slots=True)
class NamedBelief(JsonContract):
    name: str
    belief: BernoulliBelief

    def __post_init__(self) -> None:
        _require_nonempty(self.name, "name")


@dataclass(frozen=True, slots=True)
class AttributeValue(JsonContract):
    name: str
    value: JsonScalar
    confidence: float | None
    source: SourceMetadata

    def __post_init__(self) -> None:
        _require_nonempty(self.name, "name")
        if self.confidence is not None:
            _require_probability(self.confidence, "confidence")
        if isinstance(self.value, float):
            _require_finite(self.value, "value")


def _require_nonempty(value: str, name: str) -> None:
    if not value or not value.strip():
        raise ContractValidationError(f"{name} must be non-empty")


def _require_finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ContractValidationError(f"{name} must be finite")


def _require_nonnegative_finite(value: float, name: str) -> None:
    _require_finite(value, name)
    if value < 0:
        raise ContractValidationError(f"{name} must be non-negative")


def _require_nonnegative_int(value: int, name: str) -> None:
    if value < 0:
        raise ContractValidationError(f"{name} must be non-negative")


def _require_probability(value: float, name: str) -> None:
    _require_finite(value, name)
    if not 0.0 <= value <= 1.0:
        raise ContractValidationError(f"{name} must be in [0, 1]")
