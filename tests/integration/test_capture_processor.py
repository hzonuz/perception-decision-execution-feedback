from __future__ import annotations

import json
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

from perception.contracts import CaptureFrame, SSGSnapshot
from perception.privileged_capture import process_capture


ROOT = Path(__file__).resolve().parents[2]


class CaptureProcessorIntegrationTests(unittest.TestCase):
    def test_two_raw_frames_validate_and_keep_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            capture_root = Path(temporary)
            for frame_index in (0, 1):
                rgb_path = Path("rgb/capture-test") / f"{frame_index:06d}.png"
                mask_path = Path("masks/capture-test") / f"{frame_index:06d}.png"
                _write_png(capture_root / rgb_path, 4, 4, (64, 64, 64))
                _write_png(capture_root / mask_path, 4, 4, (1, 0, 0))
                raw = _raw_frame(frame_index, rgb_path.as_posix(), mask_path.as_posix())
                raw_path = capture_root / "raw/capture-test" / f"{frame_index:06d}.json"
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_text(json.dumps(raw), encoding="utf-8")

            report = process_capture(capture_root, ROOT / "configs/ontology.json")

            self.assertTrue(report["stable_identity_verified"])
            self.assertEqual(
                report["persistent_instance_ids"],
                ["capture-test:instance-000001"],
            )
            first = CaptureFrame.from_json_file(
                capture_root / "annotations/capture-test/000000.json"
            )
            second = CaptureFrame.from_json_file(
                capture_root / "annotations/capture-test/000001.json"
            )
            oracle = SSGSnapshot.from_json_file(
                capture_root / "oracle/capture-test/000001.json"
            )
            self.assertEqual(first.objects[0].instance_id, second.objects[0].instance_id)
            self.assertEqual(oracle.objects[0].track_age_frames, 2)
            self.assertTrue(oracle.source.privileged)


def _raw_frame(frame_index: int, rgb_path: str, mask_path: str) -> dict[str, object]:
    vector = {"x": 0.0, "y": 0.0, "z": 0.0}
    rotation = {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
    return {
        "format_version": "1.0.0",
        "producer_version": "0.1.0",
        "episode_id": "capture-test",
        "arena_id": "0",
        "random_seed": 7,
        "frame_index": frame_index,
        "timestamp_seconds": frame_index * 0.02,
        "rgb": {"path": rgb_path, "width": 4, "height": 4},
        "instance_mask": {
            "path": mask_path,
            "width": 4,
            "height": 4,
            "encoding": "rgb24_little_endian",
        },
        "camera": {
            "camera_id": "AgentCamMid",
            "intrinsics": {
                "width": 4,
                "height": 4,
                "fx": 3.0,
                "fy": 3.0,
                "cx": 2.0,
                "cy": 2.0,
                "near_clip": 0.1,
                "far_clip": 100.0,
            },
            "extrinsics": {"world_position": vector, "world_rotation": rotation},
        },
        "agent": {
            "world_position": vector,
            "world_rotation": rotation,
            "velocity": vector,
            "action_forward": 1,
            "action_rotate": 0,
            "health": 100.0,
        },
        "objects": [
            {
                "instance_id": "capture-test:instance-000001",
                "instance_value": 1,
                "exact_simulator_class": "GoodGoal",
                "unity_tag": "Untagged",
                "world_position": {"x": 0.0, "y": 0.0, "z": 3.0},
                "world_rotation": rotation,
                "dimensions": {"x": 1.0, "y": 1.0, "z": 1.0},
                "velocity": None,
                "bounding_box": {
                    "x_min": 1.0,
                    "y_min": 1.0,
                    "x_max": 3.0,
                    "y_max": 3.0,
                },
                "relative_depth": 3.0,
                "visibility_fraction": 1.0,
                "occlusion_fraction": 0.0,
            }
        ],
        "lifecycle_events": (
            [
                {
                    "instance_id": "capture-test:instance-000001",
                    "event": "spawned",
                    "reason": "first_observed_in_simulator",
                }
            ]
            if frame_index == 0
            else []
        ),
        "contact_events": [],
        "unavailable_fields": ["collision/contact events"],
    }


def _write_png(path: Path, width: int, height: int, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = b"".join(b"\x00" + bytes(color) * width for _ in range(height))

    def chunk(name: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + name
            + payload
            + struct.pack(">I", zlib.crc32(name + payload) & 0xFFFFFFFF)
        )

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


if __name__ == "__main__":
    unittest.main()
