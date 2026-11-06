from __future__ import annotations

from typing import TYPE_CHECKING

from .algorithms.names import derived_indices
from .arithmetic import multiply
from .nodes import (
    ComponentTensor,
    Coordinate,
    Curl,
    DirectionalDerivative,
    Divergence,
    Gradient,
    Indexed,
    IndexSum,
    Integral,
)
from .tensors import permutation_tensor

if TYPE_CHECKING:
    from .nodes.domains import AbstractDomain
    from .nodes.expressions import AbstractExpression
    from .nodes.forms import Function


def coordinate(P: AbstractDomain) -> Coordinate:
    return Coordinate(P)


def gradient(f: AbstractExpression, P: AbstractDomain) -> AbstractExpression:
    return Gradient(f, P)


def divergence(u: AbstractExpression, P: AbstractDomain) -> AbstractExpression:
    return Divergence(u, P)


def curl(u: AbstractExpression, P: AbstractDomain) -> AbstractExpression:
    return Curl(u, P)


def directional_derivative(
    f: AbstractExpression,
    x: Function | tuple[Function, ...],
    a: AbstractExpression | tuple[AbstractExpression, ...],
    h: AbstractExpression | tuple[AbstractExpression, ...],
) -> DirectionalDerivative:
    """Construct the full real differential, including complex and mixed variables.

    Bind directions explicitly with form. Complex projections are separate
    compositions; geometry dependencies are exposed during derivative evaluation.
    """
    return DirectionalDerivative(f, x, a, h)


def integral(f: AbstractExpression, P: AbstractDomain) -> AbstractExpression:
    return Integral(f, P)


def trace_of_gradient(f: AbstractExpression, P: AbstractDomain) -> AbstractExpression:
    # The value of Divergence(f, P), the last axis of f contracted with the axis of the gradient.
    *lead, k = derived_indices("uflx.divergence", (f, P), len(f.value_shape))
    entry = IndexSum(Indexed(Gradient(f, P), (*lead, k, k)), (k,))
    return ComponentTensor(entry, tuple(lead)) if lead else entry


def rotation_of_gradient(f: AbstractExpression, P: AbstractDomain) -> AbstractExpression:
    # The value of Curl(f, P), the sum of epsilon[..., j, k] * grad(f)[k, j] over j and k, with
    # the permutation tensor of rank n, so a vector for n = 3 and a scalar for n = 2.
    n = P.ambient_coordinate_domain.dimension
    *out, j, k = derived_indices("uflx.curl", (f, P), n)
    orientation = Indexed(permutation_tensor(n), (*out, j, k))
    gradient_entry = Indexed(Gradient(f, P), (k, j))
    entry = IndexSum(multiply(orientation, gradient_entry), (j, k))
    return ComponentTensor(entry, tuple(out)) if out else entry
