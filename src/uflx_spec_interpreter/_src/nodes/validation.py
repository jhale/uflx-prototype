"""Structural construction checks, in the order of the language chapters."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from .expressions import AbstractExpression
    from .indices import AbstractExtent, GatheredIndex, Index
    from .spaces import AbstractFunctionSpace


def _construction_arguments(args: tuple[object, ...]) -> None:
    from .base import Base

    for value in args:
        if isinstance(value, Base):
            continue
        if type(value) is tuple:
            _construction_arguments(cast("tuple[object, ...]", value))
        elif type(value) is float:
            if not math.isfinite(value):
                raise ValueError("construction floats must be finite")
        elif type(value) not in (int, str):
            raise TypeError(f"invalid construction argument type: {type(value).__name__}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _extent(n: int | AbstractExtent) -> None:
    if isinstance(n, int):
        _require(n >= 0, "extent must be nonnegative")


def _merge_indices(
    *groups: tuple[tuple[Index, int | AbstractExtent], ...],
) -> tuple[tuple[Index, int | AbstractExtent], ...]:
    out: dict[Index, int | AbstractExtent] = {}
    for group in groups:
        for i, n in group:
            _require(i not in out or out[i] == n, "one index has inconsistent extents")
            out[i] = n
    return tuple(out.items())


def _entries(
    shape: tuple[int | AbstractExtent, ...], indices: tuple[int | Index | GatheredIndex, ...]
) -> tuple[tuple[Index, int | AbstractExtent], ...]:
    from . import Index

    if len(indices) != len(shape):
        raise IndexError("entries must cover exactly the rank")
    direct: list[tuple[Index, int | AbstractExtent]] = []
    gathered: list[tuple[Index, int | AbstractExtent]] = []
    for i, n in zip(indices, shape, strict=True):
        if isinstance(i, int):
            if type(n) is not int or not 0 <= i < n:
                raise IndexError("integer entry is outside the axis")
        elif isinstance(i, Index):
            direct.append((i, n))
        else:
            if i.extent != n:
                raise IndexError("gather extent differs from the axis")
            gathered.extend(i.free_indices)
    return _merge_indices(tuple(direct), tuple(gathered))


def _admissible(
    V: AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...],
    value: AbstractExpression | tuple[AbstractExpression, ...],
) -> None:
    from . import ScalarType

    if isinstance(V, tuple):
        if not isinstance(value, tuple):
            raise ValueError("argument tuple does not match")
        _require(len(value) == len(V), "argument tuple does not match")
        for W, g in zip(V, value, strict=True):
            _admissible(W, g)
        return
    if isinstance(value, tuple):
        raise TypeError("expression argument required")
    _require(value.value_shape == V.value_shape, "argument value shape does not match")
    _require(value.argument_spaces == V.argument_spaces, "argument spaces do not match")
    _require(
        value.scalar_type in (ScalarType.real, V.scalar_type), "argument scalar type does not match"
    )
    _require(value.domain in (None, V.domain), "argument domain does not match")


def _scalar_operand(f: AbstractExpression, nonlinear: bool = True) -> None:
    _require(f.value_shape == (), "operation requires a scalar")
    if nonlinear:
        _require(not f.argument_spaces, "nonlinear operand has argument spaces")


def _matrix_operand(f: AbstractExpression, rectangular: bool = False) -> None:
    _require(not f.argument_spaces, "nonlinear operand has argument spaces")
    _require(len(f.value_shape) == 2, "matrix operand must have rank two")
    a, b = f.value_shape
    _require(
        a == b
        or (
            rectangular
            and isinstance(a, int)
            and isinstance(b, int)
            and b < a
            and f.scalar_type == "real"
        ),
        "matrix dimensions or scalar type are invalid",
    )
