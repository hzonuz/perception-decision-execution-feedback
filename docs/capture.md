# Privileged capture

Privileged capture is isolated from normal inference:

- Unity capture code lives in the `AnimalAI.PrivilegedCapture` assembly.
- Python processing lives in `perception.privileged_capture`.
- Neither is imported by the public `perception` package or any inference module.
- Unity creates the runtime capture controller only when
  `--perception-capture-output` is supplied.

The implementation was verified against Animal-AI source commit
`a2ca7550e3c32a16be268b44fe4ae60028d011c6`, Unity `6000.0.75f1`, and
Unity ML-Agents `4.0.0`. It uses the existing ML-Agents
`CameraSensorComponent.Camera`, `TrainingArena.arenaID`,
`TrainingArena.ArenaConfig.randomSeed`, and spawned `Prefab` components.

## Output

Each selected frame produces:

- `rgb/<episode>/<frame>.png`;
- `masks/<episode>/<frame>.png`, a visible RGB24 instance map whose integer
  value is `R + (G << 8) + (B << 16)`;
- `raw/<episode>/<frame>.json`, direct privileged Unity state;
- `annotations/<episode>/<frame>.json`, a validated shared `CaptureFrame`;
- `oracle/<episode>/<frame>.json`, a validated oracle `SSGSnapshot`.

Stable IDs are allocated from Unity object instance handles and exported as
episode-scoped monotonic IDs. Names are used only to preserve the exact
simulator class, never as identity.

The visible mask is rendered with normal scene depth. A second target-only
render estimates the amodal pixel count, so visibility is
`visible_pixels / amodal_pixels` and occlusion is its complement.

## Reproduce the deterministic smoke capture

From the repository root:

```bash
mkdir -p captures/capture_smoke
PERCEPTION_CAPTURE_OUTPUT="$PWD/captures/capture_smoke" \
  "/Applications/Unity/Unity.app/Contents/MacOS/Unity" \
  -batchmode \
  -projectPath "$PWD/animal-ai-unity" \
  -runTests \
  -testPlatform PlayMode \
  -testFilter PerceptionCapturePlayModeTests.CaptureTwoFramesKeepsStableSimulatorIdentity \
  -testResults "$PWD/captures/capture_unity_results.xml" \
  -logFile "$PWD/captures/capture_unity.log"

venv/bin/python scripts/process_capture.py \
  --capture-root captures/capture_smoke \
  --ontology configs/ontology.json

venv/bin/python scripts/validate_contracts.py \
  --capture captures/capture_smoke/annotations/capture-smoke-episode/000000.json \
  --capture captures/capture_smoke/annotations/capture-smoke-episode/000001.json \
  --ssg captures/capture_smoke/oracle/capture-smoke-episode/000001.json
```

The Animal-AI arena fixture is
`Assets/Resources/test_configs/perception_capture.yaml` and fixes the
seed to `20260725`.

For an opt-in runtime session, pass:

```text
--perception-capture-output <directory>
--perception-capture-width 84
--perception-capture-height 84
--perception-capture-every 1
--perception-capture-max-frames 100
--perception-capture-episode <episode-id>
```

## Explicitly unavailable fields

- Camera distortion coefficients: Unity's configured perspective camera does
  not expose a calibrated distortion model.
- Object physics velocity: `null` for objects without a `Rigidbody`.
- Collision/contact events: not exported because Animal-AI exposes only the
  agent's last private contact point; changing collision behavior was outside
  the minimally invasive capture boundary.

Oracle snapshots contain privileged nodes and exact state but intentionally no
derived relation labels. Spatial/temporal relation inference belongs to
Milestone 7.
