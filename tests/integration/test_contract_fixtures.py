from __future__ import annotations

import json
import subprocess
import sys
import unittest
from importlib.metadata import version
from pathlib import Path

from perception.contracts import CaptureFrame, SSGSnapshot
from perception.ontology import load_ontology
from animalai.download import get_binary_version


ROOT = Path(__file__).resolve().parents[2]
ONTOLOGY_PATH = ROOT / "configs" / "ontology.json"
CAPTURE_PATH = ROOT / "tests" / "fixtures" / "contracts" / "capture_frame.json"
SSG_PATH = ROOT / "tests" / "fixtures" / "contracts" / "oracle_ssg.json"
UNITY_ROOT = ROOT / "animal-ai-unity"


class ContractFixtureIntegrationTests(unittest.TestCase):
    def test_capture_fixture_validates(self) -> None:
        ontology = load_ontology(ONTOLOGY_PATH)
        frame = CaptureFrame.from_json_file(CAPTURE_PATH)

        ontology.validate_capture(frame)
        self.assertEqual(frame.to_dict(), json.loads(CAPTURE_PATH.read_text(encoding="utf-8")))

    def test_oracle_fixture_validates(self) -> None:
        ontology = load_ontology(ONTOLOGY_PATH)
        graph = SSGSnapshot.from_json_file(SSG_PATH)

        ontology.validate_ssg(graph)
        self.assertTrue(graph.source.privileged)
        self.assertEqual(graph.to_dict(), json.loads(SSG_PATH.read_text(encoding="utf-8")))

    def test_ontology_covers_checked_in_spawnable_prefabs(self) -> None:
        ontology = load_ontology(ONTOLOGY_PATH)
        infrastructure = {"AAI3Agent", "AAI3Arena", "SpawnedObjectsHolder", "UI ProgressBar"}
        prefab_names = {
            path.stem
            for path in (UNITY_ROOT / "Assets").rglob("*.prefab")
            if "ML-Agents" not in path.parts and "TextMesh Pro" not in path.parts
        }
        mapped = {item.exact_class for item in ontology.simulator_mappings}

        self.assertEqual(prefab_names - infrastructure, mapped)

    def test_compatibility_record_matches_checked_in_baseline(self) -> None:
        compatibility = json.loads(
            (ROOT / "configs" / "compatibility.json").read_text(encoding="utf-8")
        )
        project_version = (UNITY_ROOT / "ProjectSettings" / "ProjectVersion.txt").read_text(
            encoding="utf-8"
        )
        manifest = json.loads(
            (UNITY_ROOT / "Packages" / "manifest.json").read_text(encoding="utf-8")
        )
        head = subprocess.run(
            ["git", "-C", str(UNITY_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        self.assertIn(f"m_EditorVersion: {compatibility['unity_editor']}", project_version)
        self.assertEqual(
            manifest["dependencies"]["com.unity.ml-agents"],
            compatibility["unity_ml_agents"],
        )
        self.assertEqual(head, compatibility["unity_source_commit"])
        self.assertEqual(version("animalai"), compatibility["animalai_python"])
        self.assertEqual(get_binary_version(), compatibility["animalai_binary_release"])
        self.assertEqual(version("mlagents-envs-aai"), compatibility["mlagents_envs_aai_python"])
        self.assertEqual(
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            compatibility["python"],
        )


if __name__ == "__main__":
    unittest.main()
