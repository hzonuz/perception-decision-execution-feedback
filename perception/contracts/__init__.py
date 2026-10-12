"""Versioned public data contracts for all perception modules."""

from perception.contracts.base import ContractValidationError, JsonContract
from perception.contracts.capture import (
    AgentState,
    CaptureFrame,
    ContactEvent,
    LifecycleEvent,
    LifecycleEventType,
    ObjectAnnotation,
    PrimitiveAction,
)
from perception.contracts.common import (
    SCHEMA_VERSION,
    AttributeValue,
    BernoulliBelief,
    CategoricalDistribution,
    FrameStamp,
    MemoryState,
    MotionState,
    NamedBelief,
    ProbabilityEntry,
    SourceMetadata,
    SourceType,
    Uncertainty,
)
from perception.contracts.detection import DetectionFrame, DetectorObject
from perception.contracts.geometry import (
    BoundingBox2D,
    CameraCalibration,
    CameraExtrinsics,
    CameraIntrinsics,
    ImageEncoding,
    ImageReference,
    MaskEncoding,
    MaskReference,
    NormalizedPoint,
    Quaternion,
    Vector2,
    Vector3,
)
from perception.contracts.observation import ObservationFrame
from perception.contracts.relations import Relation, RelationPredicate
from perception.contracts.ssg import AgentNode, GraphHistory, ObjectNode, SSGSnapshot
from perception.contracts.tracking import ObjectMemory, TrackFrame, TrackLifecycle, TrackRecord, TrackState
from perception.contracts.transformer import (
    ActionMaskEntry,
    DecisionArgument,
    DecisionInput,
    DecisionOutput,
    OptionResult,
)

__all__ = [name for name in globals() if not name.startswith("_")]
