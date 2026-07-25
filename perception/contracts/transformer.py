"""High-level policy boundary contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from perception.contracts.base import ContractValidationError, JsonContract
from perception.contracts.common import SCHEMA_VERSION, JsonScalar, _require_nonempty, _require_probability
from perception.contracts.ssg import GraphHistory, SSGSnapshot


class OptionResult(str, Enum):
    UNKNOWN = "unknown"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True, slots=True)
class DecisionInput(JsonContract):
    schema_version: str
    graph: SSGSnapshot
    graph_history: GraphHistory
    previous_option: str | None
    previous_option_result: OptionResult
    previous_high_level_reward: float | None
    perception_confidence: float
    elapsed_option_steps: int

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ContractValidationError(f"unsupported schema_version {self.schema_version!r}")
        if self.graph.episode_id != self.graph_history.episode_id:
            raise ContractValidationError("graph and history must belong to the same episode")
        if self.graph_history.snapshots and self.graph_history.snapshots[-1] != self.graph:
            raise ContractValidationError("current graph must be the last history snapshot")
        _require_probability(self.perception_confidence, "perception_confidence")
        if self.elapsed_option_steps < 0:
            raise ContractValidationError("elapsed_option_steps must be non-negative")


@dataclass(frozen=True, slots=True)
class DecisionArgument(JsonContract):
    name: str
    value: JsonScalar

    def __post_init__(self) -> None:
        _require_nonempty(self.name, "name")


@dataclass(frozen=True, slots=True)
class DecisionOutput(JsonContract):
    option: str
    target_object_id: str | None
    arguments: tuple[DecisionArgument, ...]
    confidence: float

    def __post_init__(self) -> None:
        _require_nonempty(self.option, "option")
        names = [item.name for item in self.arguments]
        if len(names) != len(set(names)):
            raise ContractValidationError("decision argument names must be unique")
        _require_probability(self.confidence, "confidence")


@dataclass(frozen=True, slots=True)
class ActionMaskEntry(JsonContract):
    option: str
    enabled: bool | None
    confidence: float | None
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_nonempty(self.option, "option")
        if self.confidence is not None:
            _require_probability(self.confidence, "confidence")
        if self.enabled is None and self.confidence is not None:
            raise ContractValidationError("unknown action-mask state cannot carry confidence")

