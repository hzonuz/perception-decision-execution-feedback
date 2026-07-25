# Milestone 0 repository audit

> Historical note: this records the intentionally empty Milestone 0 state. The
> subsequently supplied baseline is recorded in
> [compatibility.md](compatibility.md).

Audit date: 2026-07-21
Repository state: greenfield

## Executive result

The working tree contained only Git metadata when this audit began. The
`master` branch had no commits, source tree, dependency manifest, environment
lock file, Unity project, tests, fixtures, or generated assets. The user
confirmed that this empty state is intentional.

Consequently, the project versions and simulator interfaces requested by the
audit cannot be inferred from repository evidence. This document records those
facts explicitly instead of selecting convenient or current APIs that might be
incompatible with the intended Animal-AI baseline.

## Evidence

| Audit item | Repository evidence | Result | Required follow-up |
| --- | --- | --- | --- |
| Animal-AI version | No package manifest, source tree, release tag, or environment binary | Unknown | Add the intended source/release and record its exact revision and distribution version. |
| Unity version | No `ProjectSettings/ProjectVersion.txt` or Unity project | Unknown | Add the intended Unity project or declare the exact editor version used to build the environment. |
| ML-Agents version | No Unity `Packages/manifest.json`, Python lock file, or installed project environment | Unknown | Record both the Unity package and Python package versions; verify their compatibility with the Animal-AI baseline. |
| Python project environment | No `pyproject.toml`, lock file, requirements file, or virtual environment declaration | Unknown | Select a Python version only after checking the Animal-AI/ML-Agents compatibility constraints, then commit a reproducible environment specification. |
| Observation interface | No Python wrapper, environment adapter, or Unity behavior definition | Unknown | Inspect the actual reset/step return types, RGB shape/dtype/range, camera ordering, action format, and terminal/truncation behavior before implementing an adapter. |
| Arena configuration format | No arena files, schemas, parser, or examples | Unknown | Inspect the baseline parser and representative arenas before defining capture or annotation tooling. |
| Test infrastructure | No test runner configuration or test files | Absent | Establish the compatible runner and deterministic fixtures during Milestone 1. |

## Host environment observations

These observations describe the developer machine, not a project environment,
and therefore are **not** approved dependency choices:

- `python` resolves to `/opt/homebrew/anaconda3/bin/python` and reports Python
  3.12.4 with pip 24.0.
- That interpreter currently exposes PyTorch 2.8.0, pytest 8.4.2, and Pydantic
  2.13.4.
- `animalai`, `mlagents`, `mlagents-envs`, and `ultralytics` were not reported by
  `python -m pip show`.
- `python3` resolves to
  `/Library/Frameworks/Python.framework/Versions/3.8/bin/python3` and reports the
  pre-release Python 3.8.0a4 with pip 19.0.3.

The two different interpreter resolutions are an additional reason to require
an explicit project environment and to invoke tools through that environment,
not through an unqualified host command.

## Compatibility evidence required before Milestone 1 API selection

The following facts must be committed or otherwise made available for
inspection:

1. Animal-AI distribution version and exact source revision, if locally
   modified.
2. Unity editor version from the actual project.
3. `com.unity.ml-agents` package version and any relevant transitive Unity
   packages.
4. Python `mlagents`/`mlagents-envs` versions used by Animal-AI.
5. Supported Python range and the intended OS/runtime environment.
6. One minimal arena configuration and the code that parses it.
7. A minimal observation trace or test fixture from the real environment,
   including array metadata and action/episode semantics.

Until these are known, architecture descriptions remain implementation-neutral
and no Unity, ML-Agents, environment, YOLO, Pydantic, or packaging API should be
assumed.

## Commands used for the audit

All commands were read-only:

```text
pwd
git status --short
rg --files
ls -la
git rev-parse --show-toplevel
git log -5 --oneline --decorate --stat
git status --branch --porcelain=v2
git remote -v
git branch -avv
command -v python
command -v python3
python --version
python -m pip --version
python -m pip show animalai mlagents mlagents-envs torch ultralytics pytest pydantic
python3 --version
python3 -m pip --version
python3 -m pip show animalai mlagents mlagents-envs torch ultralytics pytest pydantic
find . -maxdepth 3 -not -path './.git/*' -print
```

An attempted `git ls-remote` could not resolve GitHub inside the sandbox. No
remote content was fetched and the user subsequently confirmed that the
repository is intentionally empty.
