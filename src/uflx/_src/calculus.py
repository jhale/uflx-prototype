# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

from __future__ import annotations

from typing import TYPE_CHECKING, overload

from .algorithms.names import derived_indices
from .arithmetic import multiply
from .nodes import (
    Coordinate,
    Curl,
    DirectionalDerivative,
    Divergence,
    Gradient,
    Indexed,
    IndexSum,
    Integral,
)
from .tensors import component_tensor, permutation_tensor
from .typing import Argument

if TYPE_CHECKING:
    from .nodes.domains import AbstractDomain
    from .nodes.expressions import AbstractExpression
    from .nodes.forms import Function


def coordinate(domain: AbstractDomain) -> Coordinate:
    return Coordinate(domain)


def gradient(expression: AbstractExpression, domain: AbstractDomain) -> AbstractExpression:
    return Gradient(expression, domain)


def divergence(expression: AbstractExpression, domain: AbstractDomain) -> AbstractExpression:
    return Divergence(expression, domain)


def curl(expression: AbstractExpression, domain: AbstractDomain) -> AbstractExpression:
    return Curl(expression, domain)


@overload
def directional_derivative(
    expression: AbstractExpression,
    variable: Argument[Function],
    direction: Argument[AbstractExpression],
) -> DirectionalDerivative: ...


@overload
def directional_derivative(
    expression: AbstractExpression,
    variable: Argument[Function],
    point: Argument[AbstractExpression],
    direction: Argument[AbstractExpression],
) -> DirectionalDerivative: ...


def directional_derivative(
    expression: AbstractExpression,
    variable: Argument[Function],
    *values: Argument[AbstractExpression],
    **keywords: Argument[AbstractExpression],
) -> DirectionalDerivative:
    """Take the directional derivative at the variable itself or at an explicit point."""
    if len(values) == 1 and len(keywords) == 0:
        return DirectionalDerivative(expression, variable, variable, values[0])
    if values == () and keywords.keys() == {"direction"}:
        return DirectionalDerivative(expression, variable, variable, keywords["direction"])

    def at(
        point: Argument[AbstractExpression],
        direction: Argument[AbstractExpression],
    ) -> DirectionalDerivative:
        return DirectionalDerivative(expression, variable, point, direction)

    return at(*values, **keywords)


def integral(
    expression: AbstractExpression, integration_domain: AbstractDomain
) -> AbstractExpression:
    return Integral(expression, integration_domain)


def trace_of_gradient(f: AbstractExpression, P: AbstractDomain) -> AbstractExpression:
    # The value of Divergence(f, P), the last axis of f contracted with the axis of the gradient.
    *lead, k = derived_indices("uflx.divergence", (f, P), len(f.value_shape))
    entry = IndexSum(Indexed(Gradient(f, P), (*lead, k, k)), (k,))
    return component_tensor(entry, tuple(lead))


def rotation_of_gradient(f: AbstractExpression, P: AbstractDomain) -> AbstractExpression:
    # The value of Curl(f, P), the sum of epsilon[..., j, k] * grad(f)[k, j] over j and k, with
    # the permutation tensor of rank n, so a vector for n = 3 and a scalar for n = 2.
    n = P.ambient_coordinate_domain.dimension
    *out, j, k = derived_indices("uflx.curl", (f, P), n)
    orientation = Indexed(permutation_tensor(n), (*out, j, k))
    gradient_entry = Indexed(Gradient(f, P), (k, j))
    entry = IndexSum(multiply(orientation, gradient_entry), (j, k))
    return component_tensor(entry, tuple(out))
