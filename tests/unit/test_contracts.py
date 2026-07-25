from __future__ import annotations

import unittest

from perception.contracts import (
    AgentNode,
    CategoricalDistribution,
    ContractValidationError,
    FrameStamp,
    MemoryState,
    ObjectMemory,
    ObjectNode,
    ProbabilityEntry,
    SSGSnapshot,
    SourceMetadata,
    SourceType,
)
from tests.helpers import fixture_agent, fixture_stamp, make_capture_frame, make_oracle_ssg, oracle_source


class ProbabilityContractTests(unittest.TestCase):
    def test_unknown_is_not_false_or_unknown_object(self) -> None:
        distribution = CategoricalDistribution.unknown()

        self.assertEqual(distribution.entries, ())
        self.assertEqual(distribution.unknown_probability, 1.0)

    def test_probabilities_must_sum_to_one(self) -> None:
        with self.assertRaisesRegex(ContractValidationError, "sum to 1"):
            CategoricalDistribution(
                entries=(ProbabilityEntry("positive_goal", 0.7),),
                unknown_probability=0.2,
            )


class SerializationTests(unittest.TestCase):
    def test_capture_round_trip_is_lossless(self) -> None:
        frame = make_capture_frame()

        decoded = type(frame).from_json(frame.to_json())

        self.assertEqual(decoded, frame)

    def test_unknown_json_field_is_rejected(self) -> None:
        data = fixture_stamp().to_dict()
        data["invented"] = True

        with self.assertRaisesRegex(ContractValidationError, "unexpected fields"):
            FrameStamp.from_dict(data)

    def test_validation_error_keeps_path_message_and_cause(self) -> None:
        data = oracle_source().to_dict()
        data["source"] = "not-a-source"

        with self.assertRaises(ContractValidationError) as raised:
            SourceMetadata.from_dict(data)

        error = raised.exception
        self.assertEqual(error.path, "SourceMetadata.source")
        self.assertIn("expected one of", error.message)
        self.assertIsInstance(error.__cause__, ValueError)


class PartialObservabilityTests(unittest.TestCase):
    def test_missed_detection_can_preserve_existence(self) -> None:
        memory = ObjectMemory(
            track_id="track-1",
            state=MemoryState.OCCLUDED,
            visibility_probability=0.0,
            existence_probability=0.85,
            last_seen=FrameStamp("episode-1", 4, 0.4),
            updated_at=FrameStamp("episode-1", 5, 0.5),
            age_frames=6,
            source=SourceMetadata(
                source=SourceType.TEMPORAL_MEMORY,
                producer="memory-test",
            ),
        )

        self.assertEqual(memory.visibility_probability, 0.0)
        self.assertGreater(memory.existence_probability or 0.0, 0.0)

    def test_unknown_probabilities_remain_none(self) -> None:
        memory = ObjectMemory(
            track_id="track-1",
            state=MemoryState.UNOBSERVABLE,
            visibility_probability=None,
            existence_probability=None,
            last_seen=None,
            updated_at=FrameStamp("episode-1", 0, 0.0),
            age_frames=1,
            source=SourceMetadata(
                source=SourceType.TEMPORAL_MEMORY,
                producer="memory-test",
            ),
        )

        decoded = ObjectMemory.from_json(memory.to_json())
        self.assertIsNone(decoded.visibility_probability)
        self.assertIsNone(decoded.existence_probability)


class PrivilegeBoundaryTests(unittest.TestCase):
    def test_simulator_source_requires_privileged_flag(self) -> None:
        with self.assertRaisesRegex(ContractValidationError, "must be marked privileged"):
            SourceMetadata(
                source=SourceType.SIMULATOR_ORACLE,
                producer="bad-source",
                privileged=False,
            )

    def test_normal_node_cannot_expose_exact_simulator_class(self) -> None:
        oracle_node = make_oracle_ssg().objects[0]
        data = oracle_node.to_dict()
        data["source"] = SourceMetadata(
            source=SourceType.DETECTOR,
            producer="detector-test",
        ).to_dict()

        with self.assertRaisesRegex(ContractValidationError, "privileged oracle node"):
            ObjectNode.from_dict(data)


class GraphValidationTests(unittest.TestCase):
    def test_dangling_relation_is_rejected(self) -> None:
        graph = make_oracle_ssg()
        relation_data = graph.relations[0].to_dict()
        relation_data["object_node_id"] = "missing-node"
        graph_data = graph.to_dict()
        graph_data["relations"] = [relation_data]

        with self.assertRaisesRegex(ContractValidationError, "relation object"):
            SSGSnapshot.from_dict(graph_data)

    def test_agent_state_id_must_match_node_id(self) -> None:
        with self.assertRaisesRegex(ContractValidationError, "must match"):
            AgentNode(
                node_id="different-agent",
                state=fixture_agent(),
                stamp=fixture_stamp(),
                confidence=1.0,
                source=oracle_source(),
            )


if __name__ == "__main__":
    unittest.main()
