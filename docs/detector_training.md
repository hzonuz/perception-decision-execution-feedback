# Detector training

Detector training uses Ultralytics YOLO instance segmentation behind a command-line
entry point. It does not add tracking, memory, relations, or SSG logic.

## Verified environment

The implementation was smoke-tested on:

- macOS 15.1.1, Apple M1 (8 GPU cores, 16 GB RAM);
- Python 3.14.4;
- torch 2.13.0;
- torchvision 0.28.0;
- Ultralytics 8.4.102;
- CUDA unavailable;
- torch MPS built but unavailable, so the smoke run used CPU.

`requirements-ml.txt` pins the ML packages without changing the
base dependency file. Every run also writes all installed package
versions and hardware/backend availability to `environment.json`.

## Configuration

`configs/detector.json` controls the dataset, output directory, experiment
name, random seed, epochs, image size, batch size, worker count, device,
evaluation thresholds, and augmentations.

The model filename selects the scale (`yolo11n-seg.yaml`, `yolo11s-seg.yaml`,
and so on). The committed smoke configuration builds the nano model from its
package-local YAML and therefore downloads no weights. To initialize from
pretrained weights, set `pretrained_checkpoint` to a local checkpoint or an
official Ultralytics model name. The latter may download weights. To resume an
interrupted run, set `resume_checkpoint` or pass `--resume CHECKPOINT`.

## Reproduce the smoke run

From a clean checkout:

```bash
venv/bin/python -m pip install -r requirements-ml.txt

PYTHONPATH=. venv/bin/python -c \
  "from pathlib import Path; from tests.unit.test_dataset_converter import _materialize_fixture; _materialize_fixture(Path('captures/detector_fixture'))"

PYTHONPATH=. venv/bin/python scripts/convert_dataset.py \
  --capture-root captures/detector_fixture \
  --output datasets/detector_smoke \
  --config configs/dataset.json \
  --ontology configs/ontology.json

PYTHONPATH=. venv/bin/python scripts/train_detector.py \
  --config configs/detector.json
```

The fixture has three independent episodes. The dataset converter assigns one episode and
two frames to each of train, validation, and test, with no episode overlap.
The current real smoke capture has only one episode and must not be
used to claim held-out performance.

The one-epoch run is a wiring check using random initialization; its metrics
are not evidence of detector quality. Train on a representative multi-arena
capture with pretrained initialization before drawing research conclusions.

## Outputs

`runs/detector/<experiment>/` contains:

- Ultralytics training artifacts and `weights/best.pt`;
- `evaluation/` with held-out test plots;
- `metrics.json` with box/mask mAP, per-class precision/recall, small-object
  recall, visibility/occlusion slices, crowded same-class recall, and
  confidence calibration;
- `error_analysis/` with false positives, false negatives, class confusion,
  transparent-object failures, small/distant failures, and rendered
  predictions;
- `artifacts/inference.pt`, the optimizer-stripped lightweight checkpoint;
- `inference/detection_frame.json`, a validated shared `DetectionFrame`;
- `manifest.json` and `environment.json` for reproducibility.

Empty evaluation slices are serialized with `null` recall rather than being
reported as false or zero. Detector output is explicitly non-privileged and
does not import the privileged capture package.

## Tests

```bash
PYTHONPATH=. venv/bin/python -m unittest \
  tests.unit.test_detector_training -v

PYTHONPATH=. venv/bin/python -m unittest discover -s tests -v
```

The detector unit test uses a small fake result object. CI does not instantiate
a model, access the network, or download weights.
