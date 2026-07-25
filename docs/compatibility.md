# Compatibility baseline

Verification date: 2026-07-22

## Selected baseline

| Component | Verified version/revision | Repository evidence |
| --- | --- | --- |
| Python | 3.14.4 | Project `venv` and `.python-version` |
| Animal-AI Python API | 6.1.2 | Installed distribution metadata |
| Animal-AI downloadable binary | 4.3.1 | `animalai.download.get_binary_version()` |
| ML-Agents Python environment fork | `mlagents-envs-aai` 1.0.0 | Installed distribution metadata |
| Animal-AI Unity source | `a2ca7550e3c32a16be268b44fe4ae60028d011c6` | Nested source checkout HEAD |
| Unity editor | 6000.0.75f1, revision `26349cd2a5c8` | `ProjectSettings/ProjectVersion.txt` and installed editor application |
| Unity ML-Agents package | 4.0.0 | `Packages/manifest.json` |
| Unity Test Framework | 1.6.0 | `Packages/manifest.json` |

The Unity source commit message identifies the environment version as 4.3.1.
The Unity repository README still lists older editor and ML-Agents versions;
the checked-in project and package manifests are treated as authoritative.

Machine-readable values are in `configs/compatibility.json`. The integration
tests compare that record with the actual Unity source checkout, manifests,
Python interpreter, and installed distributions.

## Verified interfaces

### Python environment interface

`animalai.environment.AnimalAIEnvironment` subclasses the ML-Agents
`UnityEnvironment`. Its constructor passes the requested process seed and
configures RGB camera, optional raycast, resolution, engine timing, and arena
YAML side channels. Its `get_obs_dict` method exposes:

- camera observation when enabled;
- raycast observation when enabled;
- health;
- agent-local velocity;
- agent world position.

The normal observation interface does not expose simulator object transforms,
stable identities, masks, or class metadata. Those remain Milestone 2 privileged
capture concerns.

### Action interface

`TrainingAgent` defines two discrete branches. Both accept values 0, 1, and 2:

- forward branch: no movement, forward, backward;
- rotation branch: no rotation, right, left.

The `PrimitiveAction` contract accepts these values or `null` when an action was
not available. It does not infer an action from velocity.

### Arena format

The checked-in YamlDotNet model defines a tagged `!ArenaConfig` containing an
integer-keyed arena dictionary and global behavior flags. Each `!Arena`
supports time limit, pass mark, items, blackout steps, `randomSeed`, and arena
merging. Each `!Item` preserves the exact prefab name plus lists of positions,
rotations, sizes, colors, and object-specific parameters.

The ontology copies the ten legacy prefab aliases from the source
`AliasMapper`. Arena ID and per-arena seed have explicit capture fields.

### Exact simulator objects

The checked-in spawnable prefab inventory contains 36 non-infrastructure
classes. Every class has an ontology mapping. The integration suite fails if
that inventory and mapping diverge. The four excluded prefabs are the agent,
arena container, spawned-object holder, and UI progress bar.

Some generalized functional labels necessarily interpret component behavior
(for example, whether a block is a possible tool). Exact simulator class names
are never discarded, and Milestone 2 observations may refine these ontology
mappings without changing object identity.

## Dependency policy

Milestone 1 contracts use only the Python standard library. This avoids adding a
Pydantic or YAML runtime before it is needed. The ontology uses JSON so both
unknown keys and types can be validated without an extra parser dependency.

`requirements.txt` pins the Animal-AI client and its verified transitive
environment dependencies. It intentionally does not include the unrelated
Unity ML-Agents trainer. Milestone 4 must use a separate, reviewed ML dependency
set rather than upgrading this baseline in place.
