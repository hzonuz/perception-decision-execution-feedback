from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

from IPython.core.inputtransformer2 import TransformerManager


NOTEBOOKS = sorted((Path(__file__).resolve().parents[2] / "notebooks").glob("*.ipynb"))
_TRANSFORMER = TransformerManager()


class NotebookTests(unittest.TestCase):
    """The notebooks are the pipeline entry points; keep them loadable and parseable."""

    def test_notebooks_exist(self) -> None:
        self.assertEqual(
            [path.name for path in NOTEBOOKS],
            [
                "01_process_captures.ipynb",
                "02_build_yolo26_dataset.ipynb",
                "03_augment_dataset.ipynb",
                "04_train_yolo26_detector.ipynb",
                "05_runtime_inference.ipynb",
                "06_tracking.ipynb",
                "07_scene_graphs.ipynb",
            ],
        )

    def test_code_cells_parse(self) -> None:
        for path in NOTEBOOKS:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] == "code":
                    with self.subTest(notebook=path.name, cell=index):
                        source = "".join(cell["source"])
                        # IPython magics (%autoreload, etc.) aren't valid Python syntax on their
                        # own; transform_cell rewrites them the same way Jupyter does before running.
                        ast.parse(_TRANSFORMER.transform_cell(source))


if __name__ == "__main__":
    unittest.main()
