from __future__ import annotations

import unittest

from perception.contracts import (
    SCHEMA_VERSION,
    ActionMaskEntry,
    BoundingBox2D,
    CategoricalDistribution,
    ContractValidationError,
    DecisionInput,
    DetectionFrame,
    DetectorObject,
    FrameStamp,
    GraphHistory,
    MaskEncoding,
    MaskReference,
    MotionState,
    NormalizedPoint,
    ObservationFrame,
    OptionResult,
    SourceMetadata,
    SourceType,
    TrackLifecycle,
    TrackRecord,
)
from tests.helpers import make_oracle_ssg


class DetectorContractTests(unittest.TestCase):
    def test_model_neutral_detection_round_trip(self) -> None:
        source = SourceMetadata(SourceType.DETECTOR, "fake-detector", "1.0")
        item = DetectorObject(
            detector_id="frame-8:det-0",
            semantic_distribution=CategoricalDistribution.certain("positive_goal"),
            bounding_box=BoundingBox2D(2.0, 3.0, 10.0, 12.0, 16, 16),
            instance_mask=MaskReference("masks/mask-0.png", MaskEncoding.PNG, 16, 16, 1),
            normalized_location=NormalizedPoint(0.375, 0.46875),
            confidence=0.8,
            uncertainty=None,
            source=source,
        )
        frame = DetectionFrame(
            schema_version=SCHEMA_VERSION,
            stamp=FrameStamp("episode-1", 8, 0.8),
            image_width=16,
            image_height=16,
            objects=(item,),
            source=source,
        )

        self.assertEqual(DetectionFrame.from_json(frame.to_json()), frame)


class ObservationContractTests(unittest.TestCase):
    def test_non_privileged_observation_round_trip(self) -> None:
        from perception.contracts import ImageEncoding, ImageReference
        from tests.helpers import fixture_agent

        observation = ObservationFrame(
            schema_version=SCHEMA_VERSION,
            stamp=FrameStamp("episode-1", 0, 0.0),
            rgb=ImageReference("rgb/episode-1/000000.png", 64, 64, ImageEncoding.PNG),
            raycast_values=None,
            agent=fixture_agent(),
            source=SourceMetadata(SourceType.SENSOR, "animalai-observation-adapter", "1.0"),
        )

        self.assertEqual(ObservationFrame.from_json(observation.to_json()), observation)


class TrackingContractTests(unittest.TestCase):
    def test_track_identity_is_episode_scoped_and_serializable(self) -> None:
        record = TrackRecord(
            track_id="episode-1:track-4",
            detector_id="frame-8:det-0",
            stamp=FrameStamp("episode-1", 8, 0.8),
            lifecycle=TrackLifecycle.UPDATED,
            association_confidence=0.75,
            age_frames=5,
            missed_frames=0,
            relative_velocity=None,
            motion_state=MotionState.UNKNOWN,
            source=SourceMetadata(SourceType.TRACKER, "fake-tracker", "1.0"),
        )

        self.assertEqual(TrackRecord.from_json(record.to_json()), record)


class HistoryAndPolicyContractTests(unittest.TestCase):
    def test_history_rejects_duplicate_frame(self) -> None:
        graph = make_oracle_ssg()
        with self.assertRaisesRegex(ContractValidationError, "strictly increasing"):
            GraphHistory(SCHEMA_VERSION, graph.episode_id, (graph, graph))

    def test_decision_input_round_trip(self) -> None:
        graph = make_oracle_ssg()
        history = GraphHistory(SCHEMA_VERSION, graph.episode_id, (graph,))
        decision_input = DecisionInput(
            schema_version=SCHEMA_VERSION,
            graph=graph,
            graph_history=history,
            previous_option=None,
            previous_option_result=OptionResult.UNKNOWN,
            previous_high_level_reward=None,
            perception_confidence=1.0,
            elapsed_option_steps=0,
        )

        self.assertEqual(DecisionInput.from_json(decision_input.to_json()), decision_input)

    def test_unknown_action_mask_is_not_disabled(self) -> None:
        entry = ActionMaskEntry(
            option="climb",
            enabled=None,
            confidence=None,
            reasons=("climbability_not_observed",),
        )

        self.assertIsNone(entry.enabled)


if __name__ == "__main__":
    unittest.main()
