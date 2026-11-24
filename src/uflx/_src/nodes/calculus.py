# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, final

from .arguments import Argument, ArgumentSpaces, argument_leaves, map_argument
from .caching import cached
from .expressions import AbstractExpression, BindingExpression
from .scalar_types import NumericalScalarType, ScalarType, numerical_type

if TYPE_CHECKING:
    from .domains import AbstractDomain
    from .forms import Function
    from .indices import AbstractExtent, Index
    from .topology import AbstractTopology


@final
class DirectionalDerivative(BindingExpression["Function"]):
    value_fields = ("expression", "point", "direction")
    identifier = "uflx.DirectionalDerivative"
    # The variable, the point and the direction are a function and two expressions, or tuples.
    expression: AbstractExpression
    variable: Argument[Function]
    point: Argument[AbstractExpression]
    direction: Argument[AbstractExpression]
    scope = ("expression", "variable")
    binding_fields = ("variable",)

    @property
    @cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return self.expression.value_shape

    @property
    @cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        edges = (
            self.expression,
            *argument_leaves(self.point),
            *argument_leaves(self.direction),
        )
        from .static import free_indices

        return free_indices(*edges)

    @property
    @cached
    def argument_spaces(self) -> ArgumentSpaces:
        return self.expression.argument_spaces

    @property
    @cached
    def scalar_type(self) -> NumericalScalarType:
        return numerical_type(self.expression.scalar_type)

    @property
    @cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.expression.domain

    def _validate(self) -> None:
        from ..algorithms.substitution import occurs
        from .validation import admissible, require

        xs = argument_leaves(self.variable)
        require(len(set(xs)) == len(xs), "derivative variables must be distinct")
        require(
            not any(occurs(x, variable.space) for x in xs for variable in xs),
            "derivative variable spaces must be independent of the variables",
        )
        spaces = map_argument(lambda x: x.space, self.variable)
        admissible(spaces, self.point)
        admissible(spaces, self.direction)


@final
class Coordinate(AbstractExpression):
    value_fields = ()
    is_terminal = True
    identifier = "uflx.Coordinate"

    _domain: AbstractDomain

    @property
    def domain(self) -> AbstractDomain:
        return self._domain

    @property
    @cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return (self.domain.ambient_coordinate_domain.dimension,)

    @property
    @cached
    def scalar_type(self) -> Literal["real"]:
        return ScalarType.real

    @property
    @cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return ()

    @property
    @cached
    def argument_spaces(self) -> ArgumentSpaces:
        return ()

    def _validate(self) -> None:
        _validate_calculus(self)


@final
class Gradient(AbstractExpression):
    value_fields = ("expression",)
    identifier = "uflx.Gradient"

    expression: AbstractExpression
    _domain: AbstractDomain

    @property
    def domain(self) -> AbstractDomain:
        return self._domain

    @property
    @cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return (*self.expression.value_shape, self.domain.ambient_coordinate_domain.dimension)

    @property
    @cached
    def scalar_type(self) -> NumericalScalarType:
        return numerical_type(self.expression.scalar_type)

    @property
    @cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return self.expression.free_indices

    @property
    @cached
    def argument_spaces(self) -> ArgumentSpaces:
        return self.expression.argument_spaces

    def _validate(self) -> None:
        _validate_calculus(self)


@final
class Divergence(AbstractExpression):
    value_fields = ("expression",)
    identifier = "uflx.Divergence"
    # The trace of the gradient over the last axis of the value shape and the axis of the gradient.

    expression: AbstractExpression
    _domain: AbstractDomain

    @property
    def domain(self) -> AbstractDomain:
        return self._domain

    @property
    @cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return self.expression.value_shape[:-1]

    @property
    @cached
    def scalar_type(self) -> NumericalScalarType:
        return numerical_type(self.expression.scalar_type)

    @property
    @cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return self.expression.free_indices

    @property
    @cached
    def argument_spaces(self) -> ArgumentSpaces:
        return self.expression.argument_spaces

    def _validate(self) -> None:
        _validate_calculus(self)


@final
class Curl(AbstractExpression):
    value_fields = ("expression",)
    identifier = "uflx.Curl"
    # The curl of a vector field, contracted with the permutation tensor of rank n.

    expression: AbstractExpression
    _domain: AbstractDomain

    @property
    def domain(self) -> AbstractDomain:
        return self._domain

    @property
    @cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        # The axes of the permutation tensor that the curl leaves, n - 2 of them.
        n = self.domain.ambient_coordinate_domain.dimension
        return (n,) * (n - 2)

    @property
    @cached
    def scalar_type(self) -> NumericalScalarType:
        return numerical_type(self.expression.scalar_type)

    @property
    @cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return self.expression.free_indices

    @property
    @cached
    def argument_spaces(self) -> ArgumentSpaces:
        return self.expression.argument_spaces

    def _validate(self) -> None:
        _validate_calculus(self)


@final
class Integral(AbstractExpression):
    value_fields = ("expression",)
    identifier = "uflx.Integral"
    expression: AbstractExpression
    integration_domain: AbstractDomain

    @property
    @cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return self.expression.value_shape

    @property
    @cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        # Mesh integration binds its own cell index, never an integrand index.
        return self.expression.free_indices

    @property
    @cached
    def scalar_type(self) -> NumericalScalarType:
        return numerical_type(self.expression.scalar_type)

    @property
    @cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return None

    @property
    @cached
    def argument_spaces(self) -> ArgumentSpaces:
        return self.expression.argument_spaces

    def _validate(self) -> None:
        _validate_calculus(self)


def _validate_calculus(e: Coordinate | Gradient | Divergence | Curl | Integral) -> None:
    from .meshes import Mesh
    from .validation import require

    P = e.integration_domain if isinstance(e, Integral) else e.domain
    if isinstance(e, Coordinate):
        return
    f = e.expression
    domains = (
        (None, P, P.ambient_domain)
        if isinstance(e, Integral) and isinstance(P, Mesh)
        else (None, P)
    )
    require(f.domain in domains, "expression and domain are not a valid pair")
    dim = P.ambient_coordinate_domain.dimension
    if isinstance(e, Divergence):
        require(
            bool(f.value_shape) and f.value_shape[-1] == dim,
            "divergence requires a last axis in the ambient frame",
        )
    if isinstance(e, Curl):
        require(
            dim in (2, 3) and f.value_shape == (dim,),
            "curl requires a vector in dimension two or three",
        )
