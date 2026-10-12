from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def as_object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def read_object(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON record {path}: {exc}") from exc
    return as_object(value, str(path))


def object_field(value: dict[str, Any], name: str) -> dict[str, Any]:
    if name not in value:
        raise ValueError(f"missing required field {name!r}")
    return as_object(value[name], name)


def list_field(value: dict[str, Any], name: str) -> list[dict[str, Any]]:
    items = value.get(name)
    if not isinstance(items, list):
        raise ValueError(f"{name} must be an array")
    return [as_object(item, f"{name}[{index}]") for index, item in enumerate(items)]


def string_field(value: dict[str, Any], name: str) -> str:
    item = value.get(name)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return item


def integer_field(value: dict[str, Any], name: str) -> int:
    item = value.get(name)
    if type(item) is not int:
        raise ValueError(f"{name} must be an integer")
    return item


def number_field(value: dict[str, Any], name: str) -> float:
    item = value.get(name)
    if type(item) not in (int, float) or not math.isfinite(item):
        raise ValueError(f"{name} must be a finite number")
    return float(item)


def boolean_field(value: dict[str, Any], name: str) -> bool:
    item = value.get(name)
    if type(item) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return item


def sanitize(value: Any) -> Any:
    """Make numpy/torch scalars and non-finite floats JSON-safe; non-finite becomes null."""
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [sanitize(item) for item in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: str | Path, payload: Any) -> None:
    write_text(
        path,
        json.dumps(sanitize(payload), indent=2, sort_keys=True, allow_nan=False),
    )


def write_text(path: str | Path, payload: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        payload if payload.endswith("\n") else payload + "\n", encoding="utf-8"
    )
