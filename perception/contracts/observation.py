from __future__ import annotations

from dataclasses import dataclass

from perception.contracts.base import ContractValidationError, JsonContract
from perception.contracts.capture import AgentState
from perception.contracts.common import (
    SCHEMA_VERSION,
    FrameStamp,
    SourceMetadata,
    SourceType,
    _require_finite,
)
from perception.contracts.geometry import ImageReference


@dataclass(frozen=True, slots=True)
class ObservationFrame(JsonContract):
    schema_version: str
    stamp: FrameStamp
    rgb: ImageReference | None
    raycast_values: tuple[float, ...] | None
    agent: AgentState
    source: SourceMetadata

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ContractValidationError(
                f"unsupported schema_version {self.schema_version!r}"
            )
        if self.rgb is None and self.raycast_values is None:
            raise ContractValidationError(
                "observation must contain RGB or raycast values"
            )
        if self.raycast_values is not None:
            for index, value in enumerate(self.raycast_values):
                _require_finite(value, f"raycast_values[{index}]")
        if self.source.source is not SourceType.SENSOR or self.source.privileged:
            raise ContractValidationError(
                "observations require a non-privileged sensor source"
            )
