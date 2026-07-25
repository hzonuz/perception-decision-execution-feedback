from __future__ import annotations

import json
import types
from collections.abc import Mapping
from dataclasses import MISSING, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints


class ContractValidationError(ValueError):
    def __init__(self, message: object, path: str | None = None):
        self.message = str(message)
        self.path = path
        super().__init__(self.message)

    def __str__(self) -> str:
        return f"{self.path}: {self.message}" if self.path else self.message


ContractT = TypeVar("ContractT", bound="JsonContract")


class JsonContract:
    """Mixin providing strict JSON round trips for dataclass contracts."""

    def to_dict(self) -> dict[str, Any]:
        encoded = _encode(self)
        if not isinstance(encoded, dict):
            raise TypeError("contract root must encode to an object")
        return encoded

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )

    @classmethod
    def from_dict(cls: type[ContractT], data: Mapping[str, Any]) -> ContractT:
        decoded = _decode(cls, data, path=cls.__name__)
        if not isinstance(decoded, cls):
            raise ContractValidationError(f"could not decode {cls.__name__}")
        return decoded

    @classmethod
    def from_json(cls: type[ContractT], payload: str) -> ContractT:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ContractValidationError(f"invalid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ContractValidationError("contract JSON root must be an object")
        return cls.from_dict(data)

    @classmethod
    def from_json_file(cls: type[ContractT], path: str | Path) -> ContractT:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


def _encode(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _encode(getattr(value, field.name)) for field in fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple | list):
        return [_encode(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _encode(item) for key, item in value.items()}
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise TypeError(f"unsupported contract value: {type(value).__name__}")


def _decode(expected: Any, value: Any, *, path: str) -> Any:
    if expected is Any:
        return value

    origin = get_origin(expected)
    args = get_args(expected)

    if origin in (Union, types.UnionType):
        errors: list[str] = []
        for option in args:
            try:
                return _decode(option, value, path=path)
            except ContractValidationError as exc:
                errors.append(str(exc))
        raise ContractValidationError(
            f"value does not match any allowed type ({'; '.join(errors)})", path=path
        )

    if origin is tuple:
        if not isinstance(value, list | tuple):
            raise ContractValidationError("expected an array", path=path)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(
                _decode(args[0], item, path=f"{path}[{index}]")
                for index, item in enumerate(value)
            )
        if len(value) != len(args):
            raise ContractValidationError(
                f"expected {len(args)} array items", path=path
            )
        return tuple(
            _decode(item_type, item, path=f"{path}[{index}]")
            for index, (item_type, item) in enumerate(zip(args, value, strict=True))
        )

    if origin is list:
        if not isinstance(value, list):
            raise ContractValidationError(f"expected an array", path=path)
        item_type = args[0] if args else Any
        return [
            _decode(item_type, item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]

    if origin in (dict, Mapping):
        if not isinstance(value, Mapping):
            raise ContractValidationError(f"expected an object", path=path)
        key_type, value_type = args if args else (Any, Any)
        return {
            _decode(key_type, key, path=f"{path}.<key>"): _decode(
                value_type, item, path=f"{path}.{key}"
            )
            for key, item in value.items()
        }

    if isinstance(expected, type) and issubclass(expected, Enum):
        try:
            return expected(value)
        except (TypeError, ValueError) as exc:
            allowed = ", ".join(repr(item.value) for item in expected)
            raise ContractValidationError(
                f"expected one of {allowed}", path=path
            ) from exc

    if isinstance(expected, type) and is_dataclass(expected):
        if not isinstance(value, Mapping):
            raise ContractValidationError(f"expected an object", path=path)
        field_by_name = {field.name: field for field in fields(expected)}
        extra = set(value) - set(field_by_name)
        if extra:
            raise ContractValidationError(
                f"unexpected fields {sorted(extra)}", path=path
            )
        hints = get_type_hints(expected)
        kwargs: dict[str, Any] = {}
        for name, field in field_by_name.items():
            if name in value:
                kwargs[name] = _decode(hints[name], value[name], path=f"{path}.{name}")
            elif field.default is MISSING and field.default_factory is MISSING:
                raise ContractValidationError(
                    f"missing required field {name!r}", path=path
                )
        try:
            return expected(**kwargs)
        except ContractValidationError:
            raise
        except (TypeError, ValueError) as exc:
            raise ContractValidationError(exc, path=path) from exc

    if expected is None or expected is type(None):
        if value is not None:
            raise ContractValidationError("expected null", path=path)
        return None
    if expected is bool:
        if type(value) is not bool:
            raise ContractValidationError("expected a boolean", path=path)
        return value
    if expected is int:
        if type(value) is not int:
            raise ContractValidationError("expected an integer", path=path)
        return value
    if expected is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ContractValidationError("expected a number", path=path)
        return float(value)
    if expected is str:
        if not isinstance(value, str):
            raise ContractValidationError("expected a string", path=path)
        return value
    if isinstance(expected, type) and isinstance(value, expected):
        return value
    raise ContractValidationError(f"unsupported target type {expected!r}", path=path)
