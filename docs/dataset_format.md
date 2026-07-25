# Dataset format

The dataset converter turns validated `CaptureFrame` records into an
episode-safe instance-segmentation dataset. It does not use or expose
privileged simulator state at inference time.

## Build

```bash
venv/bin/python scripts/convert_dataset.py \
  --capture-root captures/capture_smoke \
  --output datasets/dataset_smoke \
  --config configs/dataset.json \
  --ontology configs/ontology.json
```

The output directory must be empty. This prevents stale labels from silently
surviving a rebuild.

`configs/dataset.json` controls:

- grouping by `episode`, `arena`, or `seed`;
- deterministic random seed and train/validation/test ratios;
- semantic or exact-simulator class labels;
- box/mask tolerance and motion/size thresholds;
- optional COCO export;
- the number of rendered examples.

Groups are deterministically hashed with the configured seed and assigned as
whole units. Episodes are checked after assignment and may never cross split
boundaries.

## YOLO instance-segmentation output

```text
data.yaml
images/{train,val,test}/...
labels/{train,val,test}/...
metadata/{train,val,test}.jsonl
splits.json
statistics.json
validation_report.json
visualizations/{train,val,test}/...
coco/instances_{train,val,test}.json
```

Each YOLO label row is:

```text
class_id x1 y1 x2 y2 ... xn yn
```

Polygon coordinates are normalized to `[0, 1]` and contain at least three
points. This follows the current Ultralytics instance-segmentation dataset
format: <https://docs.ultralytics.com/datasets/segment/>.

The generated `data.yaml` contains an absolute dataset root so it can be
passed directly to Ultralytics. Regenerate the dataset after moving it.

`metadata/*.jsonl` preserves episode, arena, seed, instance identity,
visibility, occlusion, relative depth, motion, and source annotation paths for
detector stratified evaluation.

## COCO export

COCO export is enabled by default and may be disabled with `--no-coco`.
Categories use one-based IDs while YOLO uses zero-based IDs. Disconnected
visible mask components are retained as multiple COCO polygons; the YOLO
label uses the largest component because its text format has one polygon per
object.

## Integrity report

Conversion checks:

- missing or malformed images/masks;
- malformed polygons;
- box/mask disagreement;
- empty annotations;
- unknown ontology classes;
- split overlap;
- duplicate frames;
- broken instance-ID sequences.

Unknown values stay unknown in metadata. Empty annotations are allowed by
default and reported as warnings.

The checked-in deterministic CI specification is
`tests/fixtures/dataset/spec.json`. Tests materialize its tiny PNG capture in a
temporary directory and verify all three group-disjoint splits without
committing generated datasets.
