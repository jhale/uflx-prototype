"""Movement through tensor construction and binding."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from ..names import _derived_index, _derived_indices
from ..nodes import (
    Action,
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
from ..static import _field_terminal, _shape, _support
from ..substitution import _freshen
from ..traversal import _map_values
from .coordinates import _chained, _piola_law

if TYPE_CHECKING:
    from ..nodes.base import AbstractExpression
    from ..nodes.domains import AbstractDomain
    from ..nodes.indices import Index
    from ..types import _Construction


__all__ = [
    "_action_output",
    "_action_retained",
    "_action_structure",
    "_commute",
    "_gradient",
    "_gradient_component",
    "_gradient_remains",
    "_structural",
]


def _commute(
    f: AbstractExpression,
    transform: Callable[[AbstractExpression], AbstractExpression],
    fields: tuple[str, ...],
    avoid: Sequence[_Construction],
    at: Callable[[AbstractExpression], AbstractExpression] = lambda e: e,
) -> AbstractExpression:
    """Freshen bindings before moving an operation through selected operands."""
    fresh = _freshen(f, avoid)
    unchanged = tuple(name for name in fresh.value_fields if name not in fields)
    return _map_values(_map_values(fresh, at, unchanged), transform, fields)


def _structural(
    f: AbstractExpression,
    transform: Callable[[AbstractExpression], AbstractExpression],
    avoid: Sequence[_Construction],
    at: Callable[[AbstractExpression], AbstractExpression] = lambda e: e,
) -> AbstractExpression | None:
    """ST01. Commute through indexing and construction, respecting scopes."""
    match f:
        case BroadcastTo():
            fields = ("value",)
        case Indexed() | ComponentTensor() | IndexSum() | Form() | DirectionalDerivative():
            fields = ("expression",)
        case _:
            return None
    return _commute(f, transform, fields, avoid, at)


def _action_retained(f: AbstractExpression) -> bool:
    """ST03-ST05. Applications of transformed forms and terminals remain."""
    from ..nodes import Function, Gradient, PullBack
    from ..static import _field_terminal

    return isinstance(f, Function | Gradient | PullBack | Action) or _field_terminal(f)


def _action_structure(
    f: AbstractExpression,
    p: int,
    g: AbstractExpression | tuple[AbstractExpression, ...],
    act: Callable[
        [AbstractExpression, int, AbstractExpression | tuple[AbstractExpression, ...]],
        AbstractExpression,
    ],
) -> AbstractExpression | None:
    """ST02-ST05. Select the product slot or retain the application."""
    from ..arithmetic import multiply
    from ..nodes import Product
    from ..static import _arity

    match f:
        case Product(a, b) if p < _arity(a):
            return multiply(act(a, p, g), b)
        case Product(a, b):
            return multiply(a, act(b, p - _arity(a), g))
        case _ if _action_retained(f):
            return Action(f, p, g)
        case _:
            return None


def _gradient(
    f: AbstractExpression,
    P: AbstractDomain,
    partial: Callable[[AbstractExpression, AbstractDomain, Index], AbstractExpression],
) -> AbstractExpression:
    """ST08. Reconstruct the gradient from component derivatives."""
    j = _derived_index("uflx.apply_gradients", f, P, "j")
    axes = _derived_indices("uflx.apply_gradients", (f, P), len(_shape(f)))
    df = partial(f, P, j)
    return ComponentTensor(Indexed(df, axes) if axes else df, (*axes, j))


def _gradient_remains(f: AbstractExpression, P: AbstractDomain) -> bool:
    """ST12. Retain whole gradients of fields and mapped values."""
    identity = isinstance(f, Coordinate) and P.dimension == P.ambient_coordinate_domain.dimension
    leaf = _field_terminal(f) or isinstance(f, PullBack | Gradient | PushForward)
    return (
        _support(f) is not None
        and (leaf or _piola_law(f) is not None)
        and not identity
        and not _chained(f)
    )


def _gradient_component(f: AbstractExpression, P: AbstractDomain, j: Index) -> AbstractExpression:
    """ST09. Select a component of a retained gradient; also its partial type."""
    from ..tensors import component_tensor, indexed

    axes = _derived_indices("uflx.apply_gradients", (f, P), len(_shape(f)))
    return component_tensor(indexed(Gradient(f, P), (*axes, j)), axes)


def _action_output(
    f: Action, transform: Callable[[AbstractExpression], AbstractExpression]
) -> AbstractExpression:
    """ST07, ST10-ST11. Transform the output while keeping the argument fixed."""
    return Action(transform(f.form), f.position, f.argument)
