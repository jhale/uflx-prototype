"""Structural construction checks, in the order of the language chapters."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .expressions import AbstractExpression
    from .indices import AbstractExtent, GatheredIndex, Index
    from .spaces import AbstractFunctionSpace


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def extent(n: int | AbstractExtent) -> None:
    if isinstance(n, int):
        require(n >= 0, "extent must be nonnegative")


def merge_indices(
    *groups: tuple[tuple[Index, int | AbstractExtent], ...],
) -> tuple[tuple[Index, int | AbstractExtent], ...]:
    out: dict[Index, int | AbstractExtent] = {}
    for group in groups:
        for i, n in group:
            require(i not in out or out[i] == n, "one index has inconsistent extents")
            out[i] = n
    return tuple(out.items())


def entries(
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
    return merge_indices(tuple(direct), tuple(gathered))


def admissible(
    V: AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...],
    value: AbstractExpression | tuple[AbstractExpression, ...],
) -> None:
    from . import ScalarType

    if isinstance(V, tuple):
        if not isinstance(value, tuple):
            raise ValueError("argument tuple does not match")
        require(len(value) == len(V), "argument tuple does not match")
        for W, g in zip(V, value, strict=True):
            admissible(W, g)
        return
    if isinstance(value, tuple):
        raise TypeError("expression argument required")
    require(value.value_shape == V.value_shape, "argument value shape does not match")
    require(value.argument_spaces == V.argument_spaces, "argument spaces do not match")
    require(
        value.scalar_type in (ScalarType.real, V.scalar_type), "argument scalar type does not match"
    )
    require(value.domain in (None, V.domain), "argument domain does not match")
