# Shared contracts and SSG schema

Schema version: 1.0.0

Ontology version: 1.0.0

## Contract rules

All public records are frozen, slotted dataclasses in `perception.contracts`.
They provide strict JSON serialization and reject missing required fields,
unknown fields, invalid enum values, non-finite numbers, out-of-range
probabilities, duplicate identities, and dangling graph relations.

JSON is the external interchange format. Paths inside capture records are
dataset-relative. Dense RGB and mask bytes remain external files so graph
history does not duplicate image data.

## Unknown versus false

The schema uses three distinct representations:

- `null` means a measurement or belief is unknown or unavailable;
- a probability of `0.0` is evidence for false or absence;
- an omitted semantic label is not interpreted as false.

`CategoricalDistribution.unknown_probability` reserves explicit epistemic
mass. This is separate from the `unknown_object` semantic class: an object may
be confidently classified as `unknown_object`, or its semantic class may be
unknown. `BernoulliBelief.probability=null` likewise means unknown and is not
equivalent to `0.0`.

## Provenance boundary

Every source has a source category, producer, optional producer/configuration
version, and privilege flag. A `simulator_oracle` source must be privileged;
other sources cannot be marked privileged. Capture records require an oracle
source. Exact simulator class names on graph object nodes are rejected unless
the node is privileged.

The contracts themselves contain no Unity imports. A future runtime detector
can consume the shared types without importing capture or oracle execution
code. Milestone 2 must place Unity access behind a separate assembly/package and a
separate Python capture package.

## Time and identity

`FrameStamp` contains episode ID, zero-based frame index, and non-negative
timestamp. Track and simulator instance identities are scoped to an episode.
Graph history cannot cross episode boundaries and frames must be strictly
increasing.

Capture objects require a stable string `instance_id`; Unity object names are
not identity fields. Milestone 2 will assign IDs when objects are registered and
retain them until an explicit lifecycle event terminates the instance.

## Capture records

`CaptureFrame` is the Milestone 2 output target. It includes:

- episode, arena, optional seed, frame, and timestamp;
- RGB image reference and optional camera calibration;
- agent state and recent primitive action;
- object annotations with exact simulator class, generalized semantic
  distribution, functional beliefs, world pose/dimensions/velocity, optional
  box/mask/depth, visibility/occlusion, attributes, and source;
- lifecycle and contact events.

Unavailable camera, mask, depth, velocity, visibility, occlusion, action, or
contact data is explicitly `null` or an empty event list. A Milestone 2 exporter
must not fabricate values to satisfy optional fields.

`ObservationFrame` is the non-privileged counterpart used by normal inference.
It accepts the configured RGB and/or raycast observations plus permitted agent
state and requires a non-privileged sensor source. It has no simulator-object
fields.

## Detector, track, and memory records

`DetectionFrame` and `DetectorObject` form the detector-neutral boundary. They
contain no YOLO-native types. A detector object has a frame-local ID, semantic
distribution, box, optional mask, normalized location, confidence,
uncertainty, and model source.

`TrackRecord` owns stable identity and association evidence. `ObjectMemory`
separates visibility probability from existence probability. The lifecycle
includes missed and reappeared states; a missed detection can therefore have
zero visibility with positive existence.

## SSG nodes and relations

Every `SSGSnapshot` contains one explicit `AgentNode`, zero or more
`ObjectNode` records, and probabilistic `Relation` records.

An object node can represent stable and detector-local IDs, semantic and
functional beliefs, image geometry, mask reference, bearing, relative distance
and velocity, detector confidence, visibility/existence, track age, last-seen
time, memory state, arbitrary typed attributes, confidence, uncertainty,
source, and frame stamp.

Relations contain subject, one object node or symbolic value, a versioned
predicate, probability, source, frame stamp, age, evidence, and uncertainty.
Graph construction rejects references to nodes absent from the snapshot.

## Policy boundary

`DecisionInput` combines the current graph, same-episode graph history,
previous option/result/reward, aggregate perception confidence, and elapsed
option steps. `DecisionOutput` contains an option, optional target ID,
arguments, and confidence. `ActionMaskEntry.enabled=null` explicitly means the
mask state is unknown rather than disabled.

## Validation

Run the deterministic test suite:

```bash
PYTHONPATH=. venv/bin/python -m unittest discover -s tests -v
```

Validate JSON files at an external boundary:

```bash
PYTHONPATH=. venv/bin/python scripts/validate_contracts.py \
  --ontology configs/ontology.json \
  --capture path/to/capture-frame.json \
  --ssg path/to/oracle-ssg.json
```

Schema changes require a version increment, fixture updates, compatibility
tests, and an explicit migration policy. Adding ontology concepts without a
contract shape change increments the ontology version independently.
