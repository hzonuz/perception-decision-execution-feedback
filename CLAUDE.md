# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code for a learned, temporally consistent perception subsystem for Animal-AI. The
intended runtime pipeline:

```
RGB observation -> instance segmentation -> multi-object tracking -> temporal object memory
  -> relation inference -> probabilistic Symbolic Scene Graph (SSG) -> decision-transformer input
```

The repository is built in gated milestones. The contracts, capture, dataset, detector, runtime
detector/inference, tracking/temporal-memory, and relations/SSG-construction milestones are
complete; the decision-transformer policy has not started. Don't assume those modules exist — check `perception/` before referencing a
module described only as a plan.

**There is no `docs/` directory.** `README.md` holds the summary and the project-wide rules; each
notebook documents its own cells. When behaviour changes, update the owning notebook
(or README) in the same commit. Keep it short — no restating the same rule in three places.

## The core invariant: privileged vs. runtime boundary

This is the single most important architectural rule in the repo. Privileged simulator state
(exact object identity, masks, oracle graphs) is for offline annotation, training targets, and
evaluation only — it must never be reachable from normal inference:

- Unity capture code lives in the `AnimalAI.PrivilegedCapture` assembly
  (`animal-ai-unity/Assets/PerceptionCapture/`); Python processing lives in
  `perception.privileged_capture`.
- Neither is imported by the public `perception` package or any future inference/runtime module.
- A `simulator_oracle` provenance source must be marked `privileged=True`; no other source may be.
- Exact simulator class names on graph object nodes are rejected by contract validation unless
  the node is privileged (see `perception/ontology.py::Ontology.validate_ssg`).
- Unity only creates the runtime capture controller when capture is explicitly requested:
  the `--perception-capture-output` flag, the `PERCEPTION_CAPTURE_OUTPUT` environment
  variable, or a `perception_capture.json` at the repo root (gitignored, for Editor Play).
  Without one of those, no capture code runs.

When adding code, keep this one-way dependency intact: privileged/offline code may import shared
contracts and ontology; runtime/inference code must never import `perception.privileged_capture`
or Unity oracle output.

## Environment

Python 3.14.4 only (`requires-python == "3.14.*"` in `pyproject.toml`, `.python-version` pins it).
A `venv/` already exists in the repo root — use it directly rather than the host interpreter,
which may resolve to an unrelated Python.

```bash
source venv/bin/activate     # or invoke venv/bin/python directly
pip install -e ".[ml]"       # pyproject.toml is the ONLY dependency manifest
```

`pyproject.toml` holds the pinned base dependencies and the `ml` extra (torch/torchvision/
ultralytics/jupyterlab). There are no `requirements*.txt` files — don't add them back.

Do not install the unrelated `mlagents` trainer package (Python 3.10 only — incompatible), and
do not upgrade the animalai/Unity/ML-Agents baseline in place without an explicit compatibility
review. `configs/compatibility.json` is the machine-readable baseline and is checked by the
integration tests against the actual Unity source, manifests, interpreter, and installed
distributions.

## Commands

Run the full test suite:

```bash
PYTHONPATH=. venv/bin/python -m unittest discover -s tests -v
```

Run a single test module / class / test:

```bash
PYTHONPATH=. venv/bin/python -m unittest tests.unit.test_contracts -v
PYTHONPATH=. venv/bin/python -m unittest tests.unit.test_contracts.SomeTestCase.test_something -v
```

Runtime inference (prerecorded images or live Animal-AI; `--detector mock` needs no torch):

```bash
PYTHONPATH=. venv/bin/python -m perception.inference --frames <image dir> --output runs/inference/<name> --visualize
PYTHONPATH=. venv/bin/python -m perception.inference --frames captures/curriculum_run/rgb/curriculum-0108 --graph --output runs/inference/<name> --visualize
```

Run the pipeline (capture processing, dataset conversion, detector training) from the notebooks —
there is no `scripts/` directory:

```bash
venv/bin/jupyter lab notebooks/   # 01 -> 02 -> 03 -> 04 build and train; 05-07 show each runtime phase
```

- `notebooks/01_process_captures.ipynb` — raw Unity capture → validated `CaptureFrame` +
  oracle `SSGSnapshot`, then ontology/contract validation of everything written.
- `notebooks/02_build_yolo26_dataset.ipynb` — capture(s) → Ultralytics/YOLO26 segmentation
  dataset (output dir must not already exist/be non-empty; set `OVERWRITE = True` to rebuild),
  then merges external YOLO exports (`roboflow_dataset/`) into `datasets/animalai_merged`.
- `notebooks/03_augment_dataset.ipynb` — optional but recommended: adds flipped/zoomed variants
  of every train frame in place via `perception.dataset.augment_dataset`, driven by
  `configs/augmentation.json`.
- `notebooks/04_train_yolo26_detector.ipynb` — train/evaluate the YOLO26 detector from
  `configs/detector.json`.
- `notebooks/05_runtime_inference.ipynb`, `06_tracking.ipynb`, `07_scene_graphs.ipynb` — one per
  runtime phase: every stage run in its own cell with its intermediate results shown (tables,
  renders, timelines, latency), ending with that phase's tests. 06's evaluation section reads
  privileged capture annotations (identity ground truth, recorded actions) — offline only. 07 shows
  captured episodes every `STRIDE` (100) frames. Outputs go to new timestamped folders under
  `runs/inference/` and `runs/scene_graph/`. The notebooks were generated by a script, not by hand:
  edit cells directly now.

Each notebook only orchestrates `perception` modules; do not put logic in a notebook. For a
scripted run use the library directly: `perception.privileged_capture.process_capture(...)`,
`perception.dataset.build_dataset(...)` / `merge_datasets(...)` / `augment_dataset(...)`,
`perception.detection.validate_dataset(...)`. There is deliberately no `run_experiment` wrapper —
one copy of the training orchestration, and it lives in notebook 04.

Exact end-to-end reproduction commands (Unity smoke capture, arena fixtures, ML environment) are in
the corresponding notebook's opening markdown — follow those verbatim rather than improvising
flags, since seeds/fixtures/arena files are pinned there.

## Architecture

### `perception/contracts/`

The shared, versioned data model (schema version tracked in `perception/contracts/common.py`).
Every public record is a **frozen, slotted dataclass** mixing in `JsonContract`
(`perception/contracts/base.py`), which provides strict `to_dict`/`to_json`/`from_dict`/`from_json`
round trips via reflection over dataclass fields and type hints — there is no Pydantic/other
schema library dependency by design.
Decoding rejects missing required fields, unknown fields, invalid enum values, non-finite numbers,
out-of-range probabilities, duplicate identities, and dangling graph relations.

Key semantic rule used throughout: **`null` means unknown/unmeasured; a probability of `0.0` is
positive evidence of false/absence; an omitted label is not implicitly false.** Don't collapse
"unknown" into a default/false value anywhere in contract or downstream code.

Module split within `contracts/`: `base.py` (JSON mixin/codec), `common.py` (distributions,
beliefs, provenance, schema version), `geometry.py` (vectors, boxes, camera calibration),
`capture.py` (privileged `CaptureFrame`/`ObjectAnnotation`), `observation.py` (non-privileged
`ObservationFrame`), `detection.py` (detector-neutral `DetectionFrame`/`DetectorObject`),
`tracking.py` (`TrackRecord`/`ObjectMemory`, composed with box/mask/class/velocity/hits/occluder
into `TrackState`, one `TrackFrame` per frame), `relations.py` (`Relation`, `RelationPredicate`
vocabulary), `ssg.py` (`SSGSnapshot`, `AgentNode`, `ObjectNode`), `transformer.py`
(`DecisionInput`/`DecisionOutput`/`ActionMaskEntry`, policy-facing boundary).

Identity/time rules: track and simulator instance IDs are scoped to an episode and never carry
across episode resets; `FrameStamp` (episode, frame index, timestamp) must be strictly
increasing within an episode.

### `perception/ontology.py`

Loads/validates `configs/ontology.json`: semantic classes, functional (multi-label) types,
simulator-exact-class-to-semantic mappings, simulator prefab aliases, and the relation predicate
vocabulary (which must exactly match `RelationPredicate` in `contracts/relations.py` — schema and
ontology versions are checked independently). `Ontology.validate_capture` / `validate_ssg` enforce
that every object's exact simulator class is mapped and its semantic/functional labels are known.

`configs/dataset.json`'s `class_mode` is `"semantic"`. Every `simulator_mappings` entry's
`semantic_class` is 1:1 with its own exact class (`Wall`→`"Wall"`, `DeathZone`→`"DeathZone"`, ...)
**except** five deliberate merges, verified against real rendered pixels, not guessed:
- `GoodGoal` = GoodGoal, GoodGoalBounce (RGB ~189,224,138, pixel-identical). The merged semantic
  class takes the name of one of its members — same for `GoodGoalMulti`/`BadGoal`/`DecoyGoal`
  below — so it is not 1:1 with "GoodGoal the exact simulator class" even though the strings match.
- `GoodGoalMulti` = GoodGoalMulti, GoodGoalMultiBounce, GrowGoal, ShrinkGoal (RGB ~245,221,104,
  pixel-identical — yes, all four; a still frame cannot tell a growing goal from a shrinking one).
- `BadGoal` = BadGoal, BadGoalBounce (RGB ~241,107,109, pixel-identical). `BadGoalMulti` is a
  distinct flat orange (RGB ~255,117,0) and is kept out of this group.
- `DecoyGoal` = DecoyGoal, DecoyGoalBounce, DecoyGoalMulti (RGB ~212,220,228, pixel-identical).
  Bad and Decoy goals are red vs. grey — not confusable with each other, so they stay in separate
  merge groups rather than one combined "negative goal" class.
- `jul_blocks` = LBlock, JBlock, UBlock (constantly confused with `Wall` and each other at oblique
  angles). `HeavyBlock`, `HollowBox`, and `LightBlock` are deliberately kept out of this group —
  don't fold them in without being asked.
Re-verify colors from `perception.dataset.build_class_samples` output (not raw prefab material
values — lighting/shading changes what the detector actually sees) before changing this grouping.
Merging doesn't touch `exact_simulator_class`, which capture always preserves; when a class_id is
merged, `perception.detection.evaluate_predictions`'s `recall_by_exact_class` reports recall per
raw class within the group, so a starved rare sibling doesn't hide behind the merged class's own
recall.

### `perception/privileged_capture/`

Processes raw Unity capture output (`captures/<episode>/{rgb,masks,raw}/...`) into validated
`CaptureFrame` and oracle `SSGSnapshot` JSON under `annotations/` and `oracle/`. Stable instance
IDs come from Unity object instance handles, exported as episode-scoped monotonic IDs; Unity
object *names* are never used as identity. See `notebooks/01_process_captures.ipynb` for the output
layout and the visible/amodal-mask visibility-fraction computation.

### `perception/dataset/`

Converts validated capture records into a leakage-safe YOLO instance-segmentation dataset
(`converter.py` orchestrates, `export.py` writes YOLO/COCO, `geometry.py` handles polygon/mask
math, `validation.py` runs the integrity report, `models.py` holds internal dataset records,
`merge.py` folds external YOLO exports in).
Grouping/split assignment (by episode, arena, or seed — configured in `configs/dataset.json`)
happens **before** frames are materialized, and episodes are checked post-assignment to guarantee
no group crosses a train/val/test split. Don't change split logic without preserving that
guarantee — there are dedicated leakage tests for it.

### `perception/dataset/merge.py`

Merges external YOLO datasets (Roboflow exports) into a capture-derived dataset while keeping the
capture dataset's canonical class indices. Every external class needs an explicit `class_map`
entry — unmapped classes raise, they are never dropped silently. External images are re-grouped by
`ExternalSource.group_pattern` (applied after Roboflow's `.rf.<hash>` suffix is stripped) so
augmented copies and same-arena frames cannot straddle a split. Box-only labels are widened to
rectangular polygons and tagged `mask_source="box_rectangle"`; capture polygons stay
`mask_source="polygon"`. Keep that tag — mask quality claims are only valid on the polygon slice.

### `perception/dataset/augment.py`

Offline geometric augmentation, driven by `configs/augmentation.json` and run from
`notebooks/03_augment_dataset.ipynb`. Adds flipped/zoomed variants of every train frame into the
same `images/train/`, `labels/train/`, `metadata/train.jsonl` in place — val/test are never
touched. Only transforms with a real camera-geometry counterpart in this simulator are used
(horizontal mirror, zoom-crop); no hue/saturation/brightness/noise augmentation, since Unity
renders every frame with fixed lighting and several semantic classes are defined by exact pixel
color (see the ontology section above) — photometric jitter would fight that. An object cut by a
zoom crop is dropped from that variant rather than clipped, so no partial/fabricated polygon is
ever written; a kept object's `visibility_fraction`/`occlusion_fraction` stay valid since the
crop+resize scales the whole frame uniformly. Idempotent: re-running without `overwrite=True` is a
no-op once `<dataset_root>/augmentation_report.json` exists.

### `perception/jsonio.py`

The single copy of strict JSON field coercion (`string_field`, `integer_field`, `number_field`,
`object_field`, `list_field`, `boolean_field`, `read_object`), plus `sanitize`/`write_json`/
`write_text`. Dataset, detection, and capture processing all import it — do not re-add a local
copy of these helpers.

### `perception/detection/`

Reusable detector pieces only, driven by `configs/detector.json`: `config.py` (validated
`DetectorConfig`), `training.py` (`validate_dataset` split/leakage gate, `ultralytics_metrics`),
`evaluation.py` (object-level sliced metrics and error analysis), `results.py` (Ultralytics result
→ evaluation predictions; the runtime `DetectionFrame` conversion is in `perception.inference.yolo`). The train/evaluate loop itself lives in notebook 03. Adds no
tracking/memory/relation/SSG logic.
Output detections use the detector-neutral `DetectionFrame` contract — never YOLO-native types —
so a future detector can be swapped in behind the same interface. Training explicitly does not
import `perception.privileged_capture`.

### `perception/inference/`

Runtime detection, independent of YOLO. `detector.py` holds the `Detector` ABC (`detect_batch`,
`detect`), `Frame` (stamp + RGB uint8 image + optional non-privileged `AgentState`), `Detection`
(`DetectionFrame` + one uint16 instance map + `Timing`), the shared `build_detection` contract
assembly, and `MockDetector` (seeded per frame, so output never depends on order or batching). The
module docstring is the single statement of the coordinate/mask/probability conventions — change
them there. `yolo.py` is the Ultralytics adapter (imported on demand; feeds BGR as Ultralytics
expects, `retina_masks=True`, folds same-anchor multi-class rows into one multi-label
distribution). `sources.py`: `prerecorded_frames` (a directory of images, never sidecar capture
JSON) and `live_frames` (steps an `AnimalAIEnvironment`, random seeded actions for now; timestamps
are simulation time, 0.02 s × decision period). `pipeline.py`: `run_inference` batches, validates
against contracts + ontology, writes the output layout in its docstring, and reports latency
percentiles and throughput. CLI: `python -m perception.inference --help`. A detector must stay
stateless across frames — tracking/memory/relations belong downstream, fed via `on_detection`.

### `perception/tracking/`

Multi-object tracker + temporal object memory, separate from detection: consumes any
`DetectionFrame` (+ instance map) and emits `TrackFrame`s, independent of the SSG. Tuning lives in
`configs/tracker.json` (`TrackerConfig`, no defaults in code). The `tracker.py` module docstring is
the single statement of the state machine (VISIBLE / PARTIALLY_VISIBLE / OCCLUDED / OUT_OF_VIEW /
UNOBSERVABLE / LOST / TERMINATED) and the per-state existence decay — change it there. Rules that
tests pin down: one miss never deletes a track (`min_misses_to_archive >= 2`); an unobservable
(blackout) frame counts no miss and does not decay existence — unobservable is not absent; LOST
(unexplained miss) is distinct from TERMINATED (observed disappearance, only for ontology
`collectable` classes or explicit `terminate()`). Association is class-gated (no cross-class
stealing below `relabel_iou`). Ego-motion is an explicit `camera_shift` input: the pipeline derives
it from the agent's own `recent_action` via the `rotation_shift_fraction` calibration (measured on
the curriculum captures at ~0.145 s/frame — re-measure for another decision period). Prerecorded
frames carry actions only when replayed with the live run's `observations.jsonl`.

Known ceiling, measured offline against capture identities: on held-out episodes a unique-class
object keeps one ID for ~20-35 matched frames; crowds of identical goals under fast turns still trade
IDs. Dominant cause is re-association after large motion and detector recall, not memory decay.

### `perception/scene_graph/`

Relation engine + SSG builder over tracker output, independent of the detector. All thresholds,
probabilities, TTLs and the camera model live in `configs/scene_graph.json` (`SceneGraphConfig`, no
code defaults). `relations.py`'s docstring is the one statement of frames of reference and edge
semantics; `builder.py`'s of node kinds, stale-edge carrying/expiry, ages and macro steps. Rules
the tests pin down: relations are only ever asserted (probability > 0), never set false, and are
omitted when inputs are unknown (e.g. floor-contact distance when a box bottom is cut or at the
horizon); memory-only nodes stay distinct (`temporal_memory` source, `observation=memory_only`
attribute, last observed box); node IDs derive from track IDs; an edge between two visible nodes
is never carried stale, and `last_seen_behind`/`likely_occluder` end when their subject is seen
again. `last_seen_behind` is only created when a node vanishes *in view* (OCCLUDED/LOST), never by
leaving the frame. `on` needs a subject that can rest on things (`on_subject_types`) — without that,
far arena walls seen above near ones read as "on". The floor-contact distance was checked against
capture depths (median error <= 10% from 5 to 20 units, unreliable under ~5). The `RelationModel`
protocol is the extension point for a learned model (`learned_relation_model` source enforced).
Synthetic task fixtures are `tests/fixtures/scenes/*.json`, loaded by `tests/scenes.py`.

### `notebooks/`

Seven notebooks: `01_process_captures`, `02_build_yolo26_dataset`, `03_augment_dataset`
(optional), `04_train_yolo26_detector`; then `05_runtime_inference`, `06_tracking`,
`07_scene_graphs`, which show each runtime phase's results (not pipeline steps). Their plots come
from `perception/tracking/render.py` (`render_tracks`, `plot_track_timeline`) and
`perception/scene_graph/render.py` (`render_snapshot`, `draw_graph`, `plot_relation_timeline`);
shared drawing helpers are in `perception/drawing.py`; identity scoring is
`perception/tracking/evaluation.py`. Tracker scenarios are `tests/fixtures/tracking_scenes/*.json`. They assemble `perception` modules but must not contain
core logic — put new logic in `perception/`, not in a cell. Each opens with a table of what its
cells do; keep that table current instead of adding prose. `tests/unit/test_notebooks.py` keeps
the set and its cells parseable; the notebooks are not executed by the suite.

`runs/detector/notebook_yolo26n_seg` is the user's finished 50-epoch training run. Never delete or
overwrite anything under `runs/`.

### `animal-ai-unity/`

A vendored/nested Unity project checkout of Animal-AI (has its own `.git`, ignored by the parent
repo's tracking except where explicitly needed). Privileged capture C# lives under
`Assets/PerceptionCapture/` (assembly `AnimalAI.PrivilegedCapture`); its own tests are under
`Assets/Tests/{EditMode,Playmode}/`. Treat it as a separate project boundary — verified
Unity editor/ML-Agents package versions are recorded in `configs/compatibility.json`, not
rediscovered from this checkout.

Deliberate local changes to the vendored project, all verified against real capture data — keep
them when updating the checkout:

- `Assets/Prefabs/AAI3Agent.prefab`: Rigidbody `m_CollisionDetection: 2` (ContinuousDynamic, was
  Discrete). With Discrete detection, destroying a goal that overlaps the agent fires a
  depenetration impulse (project cap 10 u/s) that pushes the mass-100 agent straight through the
  thin floor collider; it then falls out of the world for the rest of the episode.
- `Assets/Scripts/ArenaBuilders.cs`: `Build()` clears `_badGoalsMultiSpawned` as well as
  `_goodGoalsMultiSpawned`. Stale entries inflate `numberOfGoals` on every `BadGoalMulti`, so
  `numberOfGoalsCollected >= numberOfGoals` never holds and the arena cannot be completed.
- `Assets/PerceptionCapture/GroundTruthCaptureExporter.cs`: `visibility_fraction` is clamped to
  `[0,1]`, and frames where the agent is outside the arena volume are skipped.

Three ontology-mapped classes cannot be spawned by this checkout — see `UNSPAWNABLE_CLASSES` in
`perception/curriculum.py` for the evidence and how to re-enable them. `CylinderTunnel` and
`CylinderTunnelTransparent` take their mesh from `Cylinder.blend`, which Unity cannot import
without a local Blender install, so they render with zero bounds and no collider; `RampOld` is
absent from `AAI3Arena.allPrefabs` and carries no `Prefab` component, so it never spawns at all.
Arenas built around them look empty.

## Testing conventions

- `tests/unit/` — deterministic logic tests (contracts, dataset converter, detector training
  wiring, ontology/contract cross-module rules).
- `tests/integration/` — cross-module and capture-processor tests, including the compatibility
  checks against `animal-ai-unity` and installed package versions.
- `tests/fixtures/` — small, reviewable, non-privileged JSON/dataset fixtures (e.g.
  `tests/fixtures/dataset/spec.json` drives a materialized tiny PNG capture used by dataset
  tests without committing generated datasets).
- `tests/helpers.py` — canonical fixture builders (`make_capture_frame`, `make_oracle_ssg`, etc.)
  for constructing valid contract instances in tests; prefer these over hand-rolling dataclasses.

## Generated artifacts

`datasets/`, `runs/`, `captures/`, and similar generated/large directories are gitignored by
design (see `.gitignore`) — reproducibility comes from configs/seeds/scripts, not committed
outputs. Don't try to commit contents of these directories; small synthetic fixtures under
`tests/fixtures/` are the exception and are intentionally version-controlled.
