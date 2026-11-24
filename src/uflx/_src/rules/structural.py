# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Movement through tensor construction and binding."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from ..algorithms.names import derived_index, derived_indices
from ..algorithms.substitution import freshen
from ..algorithms.traversal import map_values
from ..nodes import (
    BroadcastTo,
    ComponentTensor,
    Coordinate,
    DirectionalDerivative,
    Form,
    Gradient,
    Indexed,
    IndexSum,
    PullBack,
    PushForward,
)
from ..nodes.static import field_terminal
from ..typing.nodes import Construction
from .coordinates import chained, piola_law

if TYPE_CHECKING:
    from ..nodes.domains import AbstractDomain
    from ..nodes.expressions import AbstractExpression
    from ..nodes.indices import Index


def commute(
    f: AbstractExpression,
    transform: Callable[[AbstractExpression], AbstractExpression],
    fields: tuple[str, ...],
    avoid: Sequence[Construction],
    independent: Callable[[AbstractExpression], bool],
    at: Callable[[AbstractExpression], AbstractExpression] = lambda e: e,
) -> AbstractExpression:
    """Move an operation through selected operands, keeping other values fixed."""
    from ..nodes import AbstractExpression
    from ..nodes.structure import nodes_in

    unchanged = tuple(name for name in f.value_fields if name not in fields)
    if any(
        isinstance(value, AbstractExpression) and not independent(value)
        for value in nodes_in(f, unchanged)
    ):
        raise TypeError(f"cannot commute through dependent operands of {type(f).__name__}")
    fresh = freshen(f, avoid)
    unchanged = tuple(name for name in fresh.value_fields if name not in fields)
    return map_values(map_values(fresh, at, unchanged), transform, fields)


def structural_fields(expression: AbstractExpression) -> tuple[str, ...] | None:
    """ST01. Value operands of indexing and binding constructions."""
    match expression:
        case BroadcastTo():
            return ("value",)
        case Indexed() | ComponentTensor() | IndexSum() | Form() | DirectionalDerivative():
            return ("expression",)
        case _:
            return None


def application_retained(f: AbstractExpression) -> bool:
    """ST03 and ST04. Retain applications of transformed forms and library terminals."""
    from ..nodes import Function, Gradient, PullBack
    from ..nodes.static import field_terminal

    return isinstance(f, Function | Gradient | PullBack) or field_terminal(f)


def retain_gradient(
    f: AbstractExpression,
    P: AbstractDomain,
    partial: Callable[[AbstractExpression, AbstractDomain, Index], AbstractExpression],
) -> AbstractExpression:
    """ST08. Reconstruct the gradient from component derivatives."""
    j = derived_index("uflx.apply_gradients", f, P, "j")
    axes = derived_indices("uflx.apply_gradients", (f, P), len(f.value_shape))
    df = partial(f, P, j)
    return ComponentTensor(Indexed(df, axes) if axes != () else df, (*axes, j))


def gradient_remains(f: AbstractExpression, P: AbstractDomain) -> bool:
    """ST12. Retain whole gradients of fields and mapped values."""
    identity = isinstance(f, Coordinate) and P.dimension == P.ambient_coordinate_domain.dimension
    leaf = field_terminal(f) or isinstance(f, PullBack | Gradient | PushForward)
    return (
        f.domain is not None
        and (leaf or piola_law(f) is not None)
        and not identity
        and not chained(f)
    )


def gradient_component(f: AbstractExpression, P: AbstractDomain, j: Index) -> AbstractExpression:
    """ST09. Select a component of a retained gradient; also its partial type."""
    from ..tensors import component_tensor, indexed

    axes = derived_indices("uflx.apply_gradients", (f, P), len(f.value_shape))
    return component_tensor(indexed(Gradient(f, P), (*axes, j)), axes)
