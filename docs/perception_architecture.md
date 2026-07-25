# Perception architecture and milestone-based implementation plan

Status: Milestone 0 architecture decision record
Baseline: greenfield repository; simulator and dependency versions unresolved

## Scope and invariants

The subsystem will turn ordinary Animal-AI RGB observations and available
non-privileged agent state into a probabilistic temporal Symbolic Scene Graph
(SSG). Privileged simulator state is restricted to offline annotation, oracle
graph generation, training targets, debugging, and evaluation. It must never be
reachable from the normal inference dependency graph.

The design has the following invariants:

- YOLO instance segmentation is the first detector implementation, but detector
  outputs cross a model-neutral interface so another detector can replace it.
- Detection, tracking, memory, relation inference, graph construction, and
  policy adaptation are separate modules with typed boundaries.
- Missing evidence is represented as unknown or as a probability distribution,
  never silently converted to false.
- A missed detection updates visibility and existence beliefs; it does not
  immediately delete an object.
- Every node and relation carries provenance, confidence/uncertainty, and
  temporal metadata appropriate to its source.
- Online perception updates may occur every frame. Immutable policy snapshots
  are emitted only when requested at macro-step boundaries.
- Dataset grouping and split assignment happen at arena, procedural-seed, or
  episode level before frames are materialized into train/validation/test sets.
- Reproducibility metadata includes random seeds, configuration fingerprints,
  package versions, model identifiers, and dataset/split revisions.

## Runtime and privileged-data boundary

```text
NORMAL INFERENCE

Animal-AI observation + permitted proprioception
              |
              v
     Observation adapter
              |
              v
 Detector protocol <- YOLO segmentation adapter
              |
              v
          Tracker
              |
              v
     Temporal memory
              |
              v
       Relation engine
              |
              v
      SSG state/history
              |
              v
 Transformer adapter + action-mask derivation

PRIVILEGED, OFFLINE OR EVALUATION ONLY

Simulator state -> annotation exporter -> dataset/training targets
Simulator state -> oracle SSG -------> comparison/evaluation
```

The privileged side may depend on shared ontology and serialization contracts.
The runtime side must not import Unity annotation or oracle providers. Packaging
and dependency tests should enforce this one-way boundary.

## Proposed repository layout

Because the repository is greenfield, the suggested layout can be adopted with
minor changes. Directories should be created only when their milestone begins; an
empty scaffold would imply interfaces that have not yet been verified.

```text
perception/
  contracts/              # Versioned external and inter-module data models
  dataset/                # Offline records, grouping, split and conversion code
  detection/              # Detector protocol and model-specific adapters
  tracking/               # Association and persistent identity
  memory/                 # Visibility/existence belief and object permanence
  relations/              # Spatial, temporal and task-derived inference
  ssg/                    # Graph state, history and serialization
  transformer_adapter/    # Policy tensors/tokens and action masks
  evaluation/             # Metrics and oracle/predicted comparison
  visualization/          # Debug rendering with uncertainty/provenance
unity/
  perception_capture/     # Privileged annotation export integration
configs/                  # Human-readable, validated runtime/training config
scripts/                  # Thin, compositional command entry points
tests/
  unit/                    # Deterministic logic tests
  integration/             # Cross-module synthetic and simulator fixtures
  fixtures/                # Small, reviewable, non-privileged fixtures
docs/
```

Top-level scripts should assemble modules but contain no core logic. Modules
must receive configuration and clocks explicitly and must not use mutable
process-global state.

## Module boundaries

### Observation adapter

Normalizes the verified Animal-AI observation contract into a timestamped RGB
frame plus available proprioception. It is the only runtime module aware of the
environment wrapper API. The exact adapter cannot be selected until the actual
environment interface is present.

### Detection

Defines a detector-neutral request/result boundary. A result contains
frame-local instances, semantic score distributions, boxes, masks where
available, confidence, model provenance, and coordinate metadata. The first
adapter will wrap a version-pinned YOLO instance-segmentation model. Tracker and
downstream modules consume only the neutral result, not YOLO-native objects.

### Tracking

Associates detections across frames and owns stable track identity. Association
may use geometry, masks, class compatibility, motion, and later learned
appearance embeddings, but none of those are required to leak into the graph
contract. It reports association confidence and explicit lifecycle events.

### Temporal memory

Owns persistent object belief under partial observability. It distinguishes at
least tentative, visible, partially visible, occluded, out-of-view,
unobservable, believed-present, and terminated states. Missed detections decay
visibility/existence according to evidence and configuration. Removal requires
an explicit termination rule; elapsed time alone should normally archive a
belief rather than assert non-existence.

### Relation engine

Consumes current tracks, memory, image geometry, permitted agent state, and
history. Each relation producer is isolated by source: geometric, tracker,
temporal-memory, learned, or task-derived. Producers emit candidates with
evidence and calibrated probability; they do not mutate graph storage
directly. Inverse and exclusivity constraints must preserve unknown states.

### SSG state and history

Combines the explicit agent node, current object beliefs, relations, provenance,
and uncertainty into versioned snapshots. The mutable per-frame builder is
internal; exported snapshots are immutable values. A bounded history supports
temporal reasoning, JSON serialization, optional ASP-like facts, and macro-step
policy snapshots.

### Transformer adapter and action masks

Converts one or more graph snapshots plus prior-option context into compact,
deterministic tensors or tokens. It derives masks from probabilistic graph
evidence using configurable thresholds and records why an option was enabled or
disabled. Ordering, truncation, padding, vocabulary versions, and uncertainty
encoding must be deterministic and serialized with the model configuration.

The policy-facing contract will combine the current graph, bounded graph
history, previous option/result/reward, aggregate perception confidence, and
elapsed option steps. Policy output validation is a separate responsibility:
the requested option, target track ID, arguments, and confidence are checked
against the exact snapshot used for the decision. Initial mask rules include
requiring climbable evidence for `climb`, platform/support evidence for
`balance`, barrier/occluder evidence for `explore`, interactable/collectable
evidence for `interact`, and hazard plus task-target evidence for `avoid`.

### Oracle and evaluation

Privileged simulator records produce oracle annotations and oracle SSGs only in
offline/evaluation code. Comparison aligns predicted identities and graph facts
without making oracle IDs available to inference. Metrics should cover
detection/segmentation, tracking, calibration, persistence, node attributes,
relations, graph-level quality, and eventual decision utility.

## Planned ontology direction

Milestone 1 will keep exact simulator identity separate from generalization. An
object can retain its exact simulator class while also carrying a probability
distribution over initial semantic concepts:

```text
positive_goal, negative_goal, wall, transparent_wall, ramp, platform,
hazard, movable_block, tunnel, decoy, unknown_object
```

Functional concepts are multi-label beliefs rather than a single exclusive
class:

```text
collectable, harmful, climbable, traversable, support, occluder, barrier,
movable, possible_tool
```

Visibility and motion concepts include `visible`, `partially_visible`,
`occluded`, `out_of_view`, `unobservable`, `moving`, `stationary`,
`believed_present`, and `terminated`. Their definitions must distinguish a
measured state from an inferred memory state.

The initial relation vocabulary is grouped by producer so that adding a learned
relation model does not replace deterministic evidence:

- Perceptual/spatial: `left_of`, `right_of`, `above`, `below`, `near`, `far`,
  `overlaps`, `adjacent`, `in_front_of`, `behind`, `on`, `supports`, and
  `occludes`.
- Temporal/memory: `same_track`, `moving_toward`, `moving_away`,
  `previously_visible`, `disappeared_after`, `last_seen_behind`,
  `believed_present`, and `reappeared_after`.
- Task-derived: `goal_on_platform`, `danger_between`, `path_blocked_by`,
  `likely_occluder`, `more_goals_left`, `more_goals_right`,
  `safe_to_interact`, and `likely_reachable`.

The ontology is versioned independently from the serialized graph schema.
Unknown concepts can retain their raw detector/simulator label and attributes
without being forced into a known class.

## Information and uncertainty model

Milestone 1 will define exact schemas. The contracts must support these semantics:

- Categorical values use probability distributions or an explicit unknown
  state, not a default class standing for missing data.
- Optional measurements distinguish `not measured`, `not observable`, and
  `measured with uncertainty` where behavior differs.
- Boolean-like semantic facts use a tri-state/probabilistic representation;
  absent graph edges do not imply false unless the schema explicitly declares a
  closed-world field.
- Every object belief includes stable track ID, frame-local detector ID when
  observed, class probabilities, functional attributes, image geometry,
  estimated bearing/distance/velocity where supported, detector confidence,
  visibility and existence probabilities, age, last-seen time, memory state,
  attributes, and provenance.
- The explicit agent node includes only actually available proprioception and
  controller context, with unknown values represented explicitly.
- Every relation includes subject, predicate, object or symbolic value,
  probability, source, observation time, age, optional evidence, and optional
  uncertainty/calibration data.
- Raw masks should be referenced or encoded compactly rather than duplicated in
  every historical graph snapshot.

Confidence values from different modules are not assumed to be interchangeable.
The schema must retain source-specific scores and calibration metadata before a
later milestone defines fusion rules.

## Time and identity semantics

All frame updates require a monotonic frame index and simulator or monotonic
timestamp. Episode ID is part of identity scope: track IDs never carry across
episode resets. Dataset records additionally retain arena and procedural-seed
grouping keys.

The graph clock and detector clock are explicit. A macro-step snapshot names the
latest included frame and time, so the policy cannot mistake a stale belief for
a fresh observation. Track age, time-since-seen, relation age, and history
retention are derived from those clocks rather than wall-clock calls hidden in
modules.

## Configuration and reproducibility

Configuration will be separated by concern (ontology, capture, detector,
tracker, relations, and SSG) and validated at process boundaries. Each run
records resolved configuration, seed set, package/runtime versions, source
revision, detector artifact checksum, ontology/schema version, and dataset/split
identifier. Defaults must be explicit and testable.

Model weights, generated datasets, recordings, run directories, and large
exports are ignored by default. Small synthetic fixtures and metadata manifests
remain version-controlled.

## Testing strategy

- Contract round-trip and validation tests for unknown values, distributions,
  temporal fields, schema versions, and provenance.
- Deterministic unit tests for association costs, lifecycle transitions,
  probability decay/fusion, relation rules, snapshot ordering, and action masks.
- Synthetic integration sequences covering missed detections, occlusion,
  out-of-view motion, reappearance, identity ambiguity, episode reset, and
  macro-step sampling.
- Privilege-boundary tests that fail if runtime packages import oracle/capture
  modules or consume fields marked simulator-only.
- Dataset leakage tests proving that no arena/seed/episode group appears in more
  than one split.
- Serialization compatibility and golden-fixture tests, with migrations when a
  contract version changes.
- Simulator integration tests only after the real versions and launch contract
  are verified; they remain separate from fast unit tests.

## Milestone plan and gates

Each milestone stops at its exit gate. Later modules may be described but are not
implemented early.

### Milestone 0 — Repository audit and architecture plan

Deliverables are this audit, architecture plan, repository entry point, and
artifact ignore policy. Exit gate: actual repository state and all unknown
version/interface facts are recorded without invented APIs.

### Milestone 1 — Ontology and shared data contracts

First record the verified compatibility matrix and reproducible Python project
environment. Then implement the versioned ontology and typed, validated
contracts for observations, detections, tracks, memory beliefs, agent state,
relations, SSG snapshots/history, provenance, uncertainty, and timestamps.
Include JSON round trips, schema/version policy, deterministic synthetic
fixtures, and unit tests. Do not add detector inference or Unity capture.

Exit gate: all required information can represent unknown values; object and
relation records retain confidence/source/time; fixtures validate cleanly; no
runtime module imports privileged types.

### Milestone 2 — Unity annotation exporter and oracle data capture

Using only verified Unity and Animal-AI APIs, export RGB plus privileged
instance masks/IDs, exact simulator class, pose/geometry, episode/arena/seed,
camera, timing, and termination metadata. Map exact simulator classes to the
general ontology without discarding the exact class. Keep capture data outside
normal inference.

Exit gate: a deterministic small episode can be captured, validated against the
Milestone 1 schema, and replayed without using privileged fields as detector input.

### Milestone 3 — Dataset conversion, splitting, validation, and visualization

Convert captures to the pinned YOLO segmentation format and a canonical record
format. Assign splits exclusively by arena, procedural seed, or episode before
frame expansion. Add leakage detection, distribution reports, mask/box checks,
class mapping reports, and visual overlays.

Exit gate: a versioned small dataset passes validation, has zero group leakage,
and can be visually audited from its manifest.

### Milestone 4 — YOLO instance-segmentation training and evaluation

Pin a compatible YOLO implementation only after the environment matrix is
known. Add configuration-driven training, resume behavior, fixed seeds, artifact
metadata, segmentation/detection and calibration metrics, per-class analysis,
and arena/seed/episode-held-out evaluation.

Exit gate: a reproducible baseline produces a registered model artifact and an
evaluation report without committing weights or generated run data.

### Milestone 5 — Runtime detector abstraction and inference pipeline

Implement the detector protocol, YOLO adapter, preprocessing/postprocessing,
batch/single-frame inference, device selection, validation, and timing. Ensure
downstream code sees neutral contracts and prove replaceability with a fake
detector integration test.

Exit gate: RGB-only inference produces validated neutral detections; runtime
imports no privileged capture/oracle code; fake and YOLO adapters satisfy the
same contract tests.

### Milestone 6 — Multi-object tracking and temporal object memory

Implement configurable association, lifecycle management, stable IDs, motion
estimation, missed-detection handling, re-identification hooks, and probabilistic
visibility/existence memory. Evaluate identity and permanence under synthetic
occlusion and real held-out episodes.

Exit gate: tracks persist through configured missed observations, uncertainty
changes are explicit, episode resets clear identity scope, and metrics/logs are
reproducible.

### Milestone 7 — Relation engine and probabilistic SSG construction

Implement independent relation producers, evidence-aware probability handling,
agent/object nodes, graph snapshots/history, JSON serialization, optional
ASP-like export, transformer tokens/tensors, and explainable action masks.

Exit gate: deterministic sequences yield valid per-frame and macro-step graphs;
unknown is preserved; every node/edge carries required metadata; history and
adapter outputs are stable under serialization.

### Milestone 8 — Oracle/predicted SSG comparison

Build oracle SSGs from privileged captures in evaluation-only code, align them
with predicted graphs, and report calibrated node, identity, attribute,
relation, persistence, and graph-level metrics. Include failure slices for
occlusion and partial observability.

Exit gate: evaluation proves oracle data cannot enter normal inference and
produces reproducible, stratified comparison reports.

## Decisions intentionally deferred

The following choices require evidence absent from the repository and must not
be made during Milestone 0:

- Python version, environment manager, package/build backend, and data-model
  library/version.
- Unity/ML-Agents integration classes and process-launch strategy.
- Animal-AI observation, action, reset, terminal, and arena parser adapters.
- YOLO package/version, tensor APIs, device policy, and export formats.
- Tracker algorithm and relation thresholds.
- Distance/depth estimation method, because the available observations and
  camera calibration are unknown.
- Storage format for dense masks and long graph histories.

## Milestone 1 entry condition

Milestone 1 may begin after the compatibility evidence listed in the repository
audit is available. If only version declarations are supplied, they must be
verified against actual manifests/source before environment-specific APIs are
implemented.
