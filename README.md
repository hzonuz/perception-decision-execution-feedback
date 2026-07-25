# Perception–Decision–Execution–Feedback

Research code for a learned, temporally consistent perception subsystem for
Animal-AI. The intended runtime pipeline is:

```text
RGB observation
  -> instance segmentation
  -> multi-object tracking
  -> temporal object memory
  -> relation inference
  -> probabilistic Symbolic Scene Graph (SSG)
  -> high-level decision-transformer input
```

The repository is intentionally being built in gated milestones.

## Project documents

- [Repository audit](docs/repository_audit.md)
- [Perception architecture and implementation plan](docs/perception_architecture.md)
- [Verified compatibility baseline](docs/compatibility.md)
- [SSG and shared-contract schema](docs/ssg_schema.md)
- [Privileged capture](docs/capture.md)
- [Dataset format](docs/dataset_format.md)
- [Detector training and evaluation](docs/detector_training.md)

## Current status

The contracts, capture, dataset, and detector milestones are complete. The repository contains typed shared contracts, an
opt-in privileged Unity capture path, leakage-safe YOLO/COCO dataset
conversion, and configurable YOLO instance-segmentation training and
object-level evaluation. Tracking and temporal memory have not started.

## Python environment

The verified environment uses Python 3.14.4, Animal-AI 6.1.2, and
`mlagents-envs-aai` 1.0.0. Use the project virtual environment and install the
pinned dependencies from `requirements.txt`:

```bash
python3.14 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Do not substitute the unrelated `mlagents` trainer package or upgrade the
verified Unity/Python baseline without an explicit compatibility review.

## Validate contracts

Run all tests:

```bash
PYTHONPATH=. venv/bin/python -m unittest discover -s tests -v
```

Validate external JSON records:

```bash
PYTHONPATH=. venv/bin/python scripts/validate_contracts.py \
  --capture tests/fixtures/contracts/capture_frame.json \
  --ssg tests/fixtures/contracts/oracle_ssg.json
```
