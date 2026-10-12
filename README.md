# Perception–Decision–Execution–Feedback

Research code for a learned, temporally consistent perception subsystem for Animal-AI:

```text
RGB observation -> instance segmentation -> multi-object tracking -> temporal object memory
  -> relation inference -> probabilistic Symbolic Scene Graph (SSG) -> decision-transformer input
```

Built so far: shared contracts, ontology, privileged Unity capture, dataset conversion and merge,
YOLO26 detector training, runtime detector interface + inference pipeline, multi-object tracking
with temporal object memory, relation inference + probabilistic temporal SSG construction. Not
started: the decision-transformer policy — check `perception/` before assuming a module exists.

## Setup

```bash
python3.14 -m venv venv && source venv/bin/activate
pip install -e ".[ml]"        # base deps + torch/ultralytics/jupyter; pyproject.toml is the only manifest
jupyter lab notebooks/
```

Do not install the unrelated `mlagents` trainer (Python 3.10 only), and do not move the
Python/Unity/Animal-AI baseline in `configs/compatibility.json` without a compatibility review —
integration tests check it against the interpreter, the Unity checkout, and its manifests.

## Notebooks

| Notebook | Does |
|---|---|
| [01_process_captures](notebooks/01_process_captures.ipynb) | raw Unity capture → validated `CaptureFrame` + oracle `SSGSnapshot`, then re-validates everything written |
| [02_build_yolo26_dataset](notebooks/02_build_yolo26_dataset.ipynb) | captures → leakage-safe YOLO segmentation dataset, then merges the Roboflow export into `datasets/animalai_merged` |
| [03_augment_dataset](notebooks/03_augment_dataset.ipynb) | optional: flipped/zoomed variants of every train frame, in place |
| [04_train_yolo26_detector](notebooks/04_train_yolo26_detector.ipynb) | trains and evaluates the YOLO26 detector, writes the run record |
| [05_runtime_inference](notebooks/05_runtime_inference.ipynb) | the detector interface step by step: detections with class probabilities, masks, serialized output, YOLO frame grid, batching/determinism, empty frames, full prerecorded run, latency sweep, optional live mode |
| [06_tracking](notebooks/06_tracking.ipynb) | the tracker on seven synthetic scenarios and a captured episode (state timelines, reappearances), plus identity evaluation against capture ground truth |
| [07_scene_graphs](notebooks/07_scene_graphs.ipynb) | relations and scene graphs on the six task fixtures and on random captured episodes every `STRIDE` frames: renders, node-link diagrams, relation timelines, ASP facts, export/re-import |

Each notebook explains its own cells and ends by running its phase's tests. Logic lives in
`perception/`; notebooks only orchestrate and show results. 05-07 need no training step: open and run.

## Runtime inference

`perception.inference` runs any `Detector` (YOLO adapter or the deterministic mock) over prerecorded
images or the live Animal-AI stream and writes validated `DetectionFrame`s, instance masks, timing,
and a latency/throughput summary. Conventions are in `perception/inference/detector.py`.

```bash
python -m perception.inference --frames <image dir> --output runs/inference/dbg --visualize
python -m perception.inference --frames <image dir> --detector mock     # same outputs/interface, no torch
python -m perception.inference --live --env <AAI binary> --arena <arena.yml> --output runs/inference/live --save-frames --track
```

`--track` adds `perception.tracking` (writes `tracks.jsonl`): stable IDs, class belief, velocity,
visibility state, existence probability, likely occluder. Its state machine and tuning are in
`perception/tracking/tracker.py` and `configs/tracker.json`. `--graph` adds `perception.scene_graph`
(writes `graphs.jsonl`, and snapshot renders with `--visualize`): an `SSGSnapshot` per frame with
the agent, visible and memory-only object nodes, and probabilistic spatial/temporal/task relations;
`to_asp()` exports ASP-style facts. Relation definitions are in
`perception/scene_graph/relations.py`, thresholds and expiry in `configs/scene_graph.json`.

## Rules that matter

- **Privileged vs runtime.** Simulator identity, masks, and oracle graphs are for offline
  annotation, training targets, and evaluation only. `perception.privileged_capture` and the Unity
  `AnimalAI.PrivilegedCapture` assembly are never imported by inference code. A `simulator_oracle`
  source must be privileged; nothing else may be.
- **Unknown is not false.** `null` means unknown or unmeasured, a probability of `0.0` is evidence
  of absence, and an omitted label is not false. Never collapse them.
- **Identity is episode-scoped.** Track and simulator instance IDs never cross an episode reset;
  frame stamps are strictly increasing within an episode.
- **Splits are grouped before frames are materialized**, so no episode or arena appears in two
  splits. There are dedicated leakage tests.

## Layout

```text
perception/     contracts, ontology, jsonio, privileged_capture, dataset, detection, inference, tracking, scene_graph
notebooks/      the pipeline (see table above)
configs/        ontology, dataset, detector, tracker, scene_graph, compatibility baseline
tests/          unit + integration; run: PYTHONPATH=. venv/bin/python -m unittest discover -s tests
captures/ datasets/ runs/    generated, gitignored
```
